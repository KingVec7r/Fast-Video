#!/usr/bin/env python3
"""
Benchmark Fast Video model across 8–1024 frames with varying mm_max_compress_loop.

Compares against the metrics from delay_memory_test.py (peak GPU memory,
first-token latency, subsequent-token latency, total latency).

Configs tested: mm_max_compress_loop ∈ {1, 2, 4, 8, 16}
Frames: 8, 16, 24, ..., 1024 (step 8)
Output: 101 tokens (greedy decoding)

Results saved to model_test/fast_video_benchmark_results.json
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# ── Paths & constants ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FRAMES_DIR = BASE_DIR / "video_frames"
OUTPUT_JSON = BASE_DIR / "fast_video_benchmark_results.json"
FRAME_COUNTS = list(range(8, 1025, 8))  # 8, 16, 24, ..., 1024
MAX_NEW_TOKENS = 101
PROMPT = "Describe this video in detail."

MODEL_PATH = str(BASE_DIR.parent / "checkpoints" / "checkpoints_video_f32")
COMPRESS_LOOPS = [16, 1024]
IMAGE_SIZE = 512


# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def sample_frames(n: int) -> List[Image.Image]:
    """Sample n frames uniformly from the video_frames directory."""
    all_frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if not all_frames:
        raise FileNotFoundError(f"No frames found in {FRAMES_DIR}")
    avail = len(all_frames)
    idx = np.linspace(0, avail - 1, min(n, avail), dtype=int)
    frames = [Image.open(all_frames[i]).convert("RGB") for i in idx]
    while len(frames) < n:
        needed = n - len(frames)
        extra_idx = np.linspace(0, avail - 1, min(needed, avail), dtype=int)
        frames += [Image.open(all_frames[i]).convert("RGB") for i in extra_idx]
    return frames[:n]


def count_params(model) -> Tuple[int, float]:
    n = sum(p.numel() for p in model.parameters())
    return n, n / 1e9


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_fast_video_model(
    model_path: str,
    mm_max_compress_loop: int,
    gpu_id: int = 0,
) -> Tuple[Any, AutoTokenizer, Any]:
    """Load the Fast Video model with a specific mm_max_compress_loop."""
    device = f"cuda:{gpu_id}"

    print(f"  Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    print(f"  Loading model from {model_path} ...")
    from fast_onevision.model import FastOneVisionForCausalLM

    model = FastOneVisionForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
        trust_remote_code=True,
    ).eval()

    model.config.mm_max_compress_loop = mm_max_compress_loop
    compressor = model.get_compressor()
    if compressor is not None:
        compressor.max_encode_loop = mm_max_compress_loop

    image_processor = model.get_vision_tower().image_processor
    print(f"  mm_max_compress_loop = {mm_max_compress_loop}")
    return model, tokenizer, image_processor


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt formatting (Qwen3-style for Fast Video)
# ═══════════════════════════════════════════════════════════════════════════════

def format_prompt(question: str, num_frames: int) -> str:
    """Build prompt with <|video_pad|> placeholders — 1 per frame."""
    video_pads = "<|video_pad|>" * num_frames
    vision_block = f"<|vision_start|>{video_pads}<|vision_end|>"
    return (
        f"<|im_start|>user\n{vision_block}\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def preprocess_frames(
    frames: List[Image.Image],
    image_processor,
    image_size: int = 512,
) -> torch.Tensor:
    """Preprocess frames through the vision tower's image processor."""
    frames_pil = [Image.fromarray(np.array(f)) for f in frames]
    out = image_processor(
        frames_pil,
        size=image_size,
        crop_size=image_size,
        return_tensors='pt',
    )
    return out['pixel_values']  # [T, C, H, W]


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark for a single (model, frame_count) pair
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_one(
    model,
    tokenizer: AutoTokenizer,
    image_processor,
    n_frames: int,
) -> Dict[str, Any]:
    """Run memory + latency measurement for a single frame count.

    Measurement strategy (same as delay_memory_test.py):
      1. Warmup with 1 token (primes CUDA context, not measured).
      2. reset_peak_memory_stats() — baseline = model + visual tensors on GPU.
      3. Generate MAX_NEW_TOKENS tokens — captures peak memory.
      4. First-token latency measured separately afterwards.

    Returns dict with all metrics.
    """
    frames = sample_frames(n_frames)
    pixel_values = preprocess_frames(frames, image_processor, IMAGE_SIZE)
    prompt = format_prompt(PROMPT, n_frames)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')

    # Move to GPU
    pixel_values = pixel_values.to(dtype=model.dtype, device=model.device)  # [T, C, H, W]
    input_ids = input_ids.to(model.device)

    input_seq_len = input_ids.shape[-1]

    gen_kwargs = {
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # ── Warmup (1 token) ─────────────────────────────────────────────────
    with torch.inference_mode():
        model.generate(
            input_ids=input_ids,
            images=pixel_values,
            max_new_tokens=1,
            **gen_kwargs,
        )
    torch.cuda.synchronize()

    # ── Peak memory + total latency ──────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        model.generate(
            input_ids=input_ids,
            images=pixel_values,
            max_new_tokens=MAX_NEW_TOKENS,
            **gen_kwargs,
        )
    end.record()
    torch.cuda.synchronize()
    total_s = start.elapsed_time(end) / 1000.0
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

    # ── First-token latency ──────────────────────────────────────────────
    s1 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    s1.record()
    with torch.inference_mode():
        model.generate(
            input_ids=input_ids,
            images=pixel_values,
            max_new_tokens=1,
            **gen_kwargs,
        )
    e1.record()
    torch.cuda.synchronize()
    first_s = s1.elapsed_time(e1) / 1000.0

    subsequent_avg = (total_s - first_s) / (MAX_NEW_TOKENS - 1) if MAX_NEW_TOKENS > 1 else 0

    return {
        "num_frames": n_frames,
        "first_token_s": round(first_s, 4),
        "total_s": round(total_s, 4),
        "subsequent_avg_s": round(subsequent_avg, 4),
        "peak_mem_gib": round(peak_mem, 4),
        "mem_reserved_gib": round(mem_reserved, 4),
        "input_seq_len": input_seq_len,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main benchmark loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark_for_loop(mm_max_compress_loop: int, gpu_id: int = 0) -> List[Dict]:
    """Run full benchmark for one mm_max_compress_loop value."""
    print(f"\n{'='*60}")
    print(f"  Fast Video — mm_max_compress_loop = {mm_max_compress_loop}")
    print(f"{'='*60}")

    model, tokenizer, image_processor = load_fast_video_model(
        MODEL_PATH, mm_max_compress_loop, gpu_id
    )
    total_n, total_b = count_params(model)
    print(f"  Params: {total_n:,}  ({total_b:.2f} B)")

    results = []
    for n_frames in FRAME_COUNTS:
        print(f"  ── Frames: {n_frames} ──", end=" ", flush=True)
        try:
            m = benchmark_one(model, tokenizer, image_processor, n_frames)
            results.append(m)
            print(
                f"1st={m['first_token_s']:.3f}s  "
                f"sub_avg={m['subsequent_avg_s']:.3f}s  "
                f"total={m['total_s']:.3f}s  "
                f"mem={m['peak_mem_gib']:.2f}GiB(res={m['mem_reserved_gib']:.2f})  "
                f"seq_len={m['input_seq_len']}"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"OOM — stopping this config.")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"ERROR: {e}")
            torch.cuda.empty_cache()
        finally:
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Frame counts: {FRAME_COUNTS[0]}–{FRAME_COUNTS[-1]} (step {FRAME_COUNTS[1] - FRAME_COUNTS[0]})")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    print(f"Prompt: {PROMPT}")
    print(f"Model: {MODEL_PATH}")
    print(f"Compress loops: {COMPRESS_LOOPS}")

    all_results: Dict[str, List[Dict]] = {}

    for loop in COMPRESS_LOOPS:
        key = f"mm_max_compress_loop={loop}"
        try:
            all_results[key] = run_benchmark_for_loop(loop)
        except Exception as e:
            print(f"  FATAL for loop={loop}: {e}")
            import traceback
            traceback.print_exc()
            all_results[key] = []

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "gpu": torch.cuda.get_device_name(0),
            "model_path": MODEL_PATH,
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "frame_counts_range": [
                FRAME_COUNTS[0],
                FRAME_COUNTS[-1],
                FRAME_COUNTS[1] - FRAME_COUNTS[0],
            ],
            "compress_loops": COMPRESS_LOOPS,
        },
        "results": all_results,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_JSON}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'Config':<30} {'MaxFrames':>10} {'PeakMem':>8} {'MemRes':>8} {'1stTok':>8} {'SubAvg':>8} {'Total':>8} {'SeqLen':>8}")
    print("-" * 100)
    for key, entries in all_results.items():
        if entries:
            last = entries[-1]
            seq_len = last.get('input_seq_len', 'N/A')
            print(
                f"{key:<30} {last['num_frames']:>10} "
                f"{last['peak_mem_gib']:>7.2f}GiB {last['mem_reserved_gib']:>7.2f}GiB "
                f"{last['first_token_s']:>7.3f}s {last['subsequent_avg_s']:>7.3f}s "
                f"{last['total_s']:>7.3f}s {str(seq_len):>8}"
            )
        else:
            print(f"{key:<30} {'N/A':>10}")
    print("=" * 100)


if __name__ == "__main__":
    main()

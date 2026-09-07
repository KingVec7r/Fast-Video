#!/usr/bin/env python3
"""
Fast Video Model Delay & Memory Benchmark
==========================================

Benchmark: total params, peak GPU memory, first-token latency,
and subsequent-token latency for the Fast Video model across 8–1024 frames,
with mm_max_compress_loop swept over [1, 2, 4, 8, 16].

Settings:
  - Frames: 8, 16, 24, ..., 1024 (step 8) from model_test/video_frames
  - Output limited to 101 tokens (greedy decoding)
  - flash_attention_2
  - Peak GPU memory INCLUDES model weights, visual tensors, KV cache,
    and activation buffers — representing the minimum GPU required to
    avoid OOM under transformers + flash_attention_2 (production budget).
  - Timing starts AFTER frame tensors are on CUDA (i.e. model.generate
    call is the timed section).
  - Results saved to model_test/fast_video_test_results.json
  - CUDA 4 (set via CUDA_VISIBLE_DEVICES=4).
"""

import json
import os
import sys
import time
import warnings
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ──────────────────────────────────────────────────
# Monkey-patches for transformers 5.x compatibility
# ──────────────────────────────────────────────────
import transformers
import transformers.modeling_utils as _tf_mu

_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])
if _TRANSFORMERS_MAJOR >= 5:
    _ORIG_FINALIZE = _tf_mu.PreTrainedModel.__dict__["_finalize_model_loading"]

    def _patched_finalize(model, load_config, loading_info):
        if not hasattr(model, "all_tied_weights_keys") or model.all_tied_weights_keys is None:
            if hasattr(model, "_tied_weights_keys") and model._tied_weights_keys is not None:
                old = model._tied_weights_keys
                model.all_tied_weights_keys = old if isinstance(old, dict) else {k: [] for k in old}
            else:
                model.all_tied_weights_keys = {}
        return _ORIG_FINALIZE(model, load_config, loading_info)

    _tf_mu.PreTrainedModel._finalize_model_loading = staticmethod(_patched_finalize)

    _ORIG_GET_TOTAL = _tf_mu.get_total_byte_count

    def _patched_get_total_byte_count(model, *args, **kwargs):
        if not hasattr(model, "all_tied_weights_keys"):
            old = getattr(model, "_tied_weights_keys", None)
            model.all_tied_weights_keys = old if isinstance(old, dict) else {}
        return _ORIG_GET_TOTAL(model, *args, **kwargs)

    _tf_mu.get_total_byte_count = _patched_get_total_byte_count

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FRAMES_DIR = BASE_DIR / "video_frames"
OUTPUT_JSON = BASE_DIR / "fast_video_test_results.json"
CHECKPOINT_DIR = BASE_DIR.parent / "checkpoints" / "checkpoints_video_f32"

FRAME_COUNTS = list(range(8, 1025, 8))  # 8, 16, 24, ..., 1024
MAX_NEW_TOKENS = 101
PROMPT = "Describe this video in detail."
LOOP_VALUES = [1, 2, 4, 8, 16]

IMAGE_SIZE = 512  # standard preprocessing size used by the pipeline


# ══════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════

def sample_frames(n: int) -> list[Image.Image]:
    """Sample n frames uniformly. If n > available, cycle through with offset."""
    all_frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if not all_frames:
        raise FileNotFoundError(f"No frames in {FRAMES_DIR}")
    avail = len(all_frames)
    idx = np.linspace(0, avail - 1, min(n, avail), dtype=int)
    frames = [Image.open(all_frames[i]).convert("RGB") for i in idx]
    # If more frames requested than available, repeat with stride
    while len(frames) < n:
        needed = n - len(frames)
        extra_idx = np.linspace(0, avail - 1, min(needed, avail), dtype=int)
        frames += [Image.open(all_frames[i]).convert("RGB") for i in extra_idx]
    return frames[:n]


def count_params(model) -> tuple[int, float]:
    """Return (total_params, billions)."""
    n = sum(p.numel() for p in model.parameters())
    return n, n / 1e9


# ══════════════════════════════════════════════════
# Fast Video prompt builder
# ══════════════════════════════════════════════════

def build_fast_video_prompt(question: str, num_frames: int) -> str:
    """Build the Qwen3-style chat prompt with video placeholders.

    Uses ``<|video_pad|>`` tokens — one per frame because the compressor
    maps each frame to exactly 1 compressed token.
    """
    video_pads = "<|video_pad|>" * num_frames
    vision_block = f"<|vision_start|>{video_pads}<|vision_end|>"
    return (
        f"<|im_start|>user\n{vision_block}\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ══════════════════════════════════════════════════
# Benchmark helper
# ══════════════════════════════════════════════════

def benchmark_one_config(
    model,
    tokenizer,
    image_processor,
    mm_max_compress_loop: int,
) -> list[dict]:
    """Benchmark all frame counts for a given mm_max_compress_loop setting.

    The model's mm_max_compress_loop is set BEFORE this function is called.
    Timing starts when the preprocessed pixel_values tensor is already on CUDA,
    just before calling model.generate().
    """
    print(f"\n{'─' * 55}")
    print(f"  mm_max_compress_loop = {mm_max_compress_loop}")
    print(f"{'─' * 55}")

    results = []
    for n_frames in FRAME_COUNTS:
        print(f"    Frames: {n_frames:>4} ...", end=" ", flush=True)
        try:
            # ── Sample & preprocess frames (on CPU, not timed) ─────
            frames = sample_frames(n_frames)
            prompt = build_fast_video_prompt(PROMPT, n_frames)

            input_ids = tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors='pt'
            )

            frames_pil = [Image.fromarray(np.array(f)) for f in frames]
            pixel_values = image_processor(
                frames_pil,
                size=IMAGE_SIZE,
                crop_size=IMAGE_SIZE,
                return_tensors='pt'
            )['pixel_values']  # [T, C, H, W]

            # Move to CUDA — this is the "frames on CUDA" starting point
            pixel_values = pixel_values.to(
                dtype=model.dtype, device=model.device
            )
            input_ids = input_ids.to(model.device)
            # Add batch dim: [1, T, C, H, W]
            pixel_values = pixel_values.unsqueeze(0)

            gen_kwargs = {
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            # ── Warmup (1 token, not measured) ────────────────────
            _ = model.generate(
                input_ids=input_ids,
                images=pixel_values,
                max_new_tokens=1,
                **gen_kwargs,
            )
            torch.cuda.synchronize()

            # ── Peak memory + total latency (full generation) ────
            # Reset peak *after* warmup → baseline = model + inputs on GPU
            torch.cuda.reset_peak_memory_stats()
            start_total = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)
            start_total.record()
            with torch.inference_mode():
                model.generate(
                    input_ids=input_ids,
                    images=pixel_values,
                    max_new_tokens=MAX_NEW_TOKENS,
                    **gen_kwargs,
                )
            end_total.record()
            torch.cuda.synchronize()
            total_s = start_total.elapsed_time(end_total) / 1000.0
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

            # ── First-token latency (lightweight, separate 1-token call) ──
            start_first = torch.cuda.Event(enable_timing=True)
            end_first = torch.cuda.Event(enable_timing=True)
            start_first.record()
            with torch.inference_mode():
                model.generate(
                    input_ids=input_ids,
                    images=pixel_values,
                    max_new_tokens=1,
                    **gen_kwargs,
                )
            end_first.record()
            torch.cuda.synchronize()
            first_s = start_first.elapsed_time(end_first) / 1000.0

            subsequent_avg = (
                (total_s - first_s) / (MAX_NEW_TOKENS - 1)
                if MAX_NEW_TOKENS > 1 else 0.0
            )

            result = {
                "num_frames": n_frames,
                "mm_max_compress_loop": mm_max_compress_loop,
                "first_token_s": round(first_s, 4),
                "total_s": round(total_s, 4),
                "subsequent_avg_s": round(subsequent_avg, 4),
                "peak_mem_gib": round(peak_mem, 4),
                "mem_reserved_gib": round(mem_reserved, 4),
            }
            results.append(result)

            print(f"1st={first_s:.3f}s  sub_avg={subsequent_avg:.3f}s  "
                  f"total={total_s:.3f}s  mem={peak_mem:.2f}GiB "
                  f"(res={mem_reserved:.2f})")

        except torch.cuda.OutOfMemoryError:
            print(f"OOM — stopping at {n_frames} frames for loop={mm_max_compress_loop}")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"ERROR: {e}")
            torch.cuda.empty_cache()
        finally:
            # Reset allocator state between iterations
            torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════

def main():
    # ── Force CUDA 0 ─────────────────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.  Is GPU 4 present?")
        sys.exit(1)

    device = "cuda:0"  # CUDA_VISIBLE_DEVICES=4 → this GPU becomes cuda:0
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Frame counts: {FRAME_COUNTS[0]}–{FRAME_COUNTS[-1]} "
          f"(step {FRAME_COUNTS[1] - FRAME_COUNTS[0]})")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    print(f"mm_max_compress_loop sweep: {LOOP_VALUES}")
    print(f"Prompt: {PROMPT}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")

    # ── Load model once ───────────────────────────────────────────────
    print("\n── Loading Fast Video Model ──")
    from transformers import AutoTokenizer
    from fast_onevision.model import FastOneVisionForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        str(CHECKPOINT_DIR), trust_remote_code=True, use_fast=False,
    )

    model = FastOneVisionForCausalLM.from_pretrained(
        str(CHECKPOINT_DIR),
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
        trust_remote_code=True,
    ).eval()

    total_n, total_b = count_params(model)
    print(f"  Params: {total_n:,}  ({total_b:.2f} B)")
    print(f"  dtype: {model.dtype}")

    image_processor = model.get_vision_tower().image_processor
    print(f"  Vision tower: {type(model.get_vision_tower()).__name__}")
    print(f"  Image processor: {type(image_processor).__name__}")

    # ── Sweep mm_max_compress_loop ────────────────────────────────────
    all_results = {}

    for loop_val in LOOP_VALUES:
        print(f"\n{'=' * 60}")
        print(f"  Sweep: mm_max_compress_loop = {loop_val}")
        print(f"{'=' * 60}")

        # Apply loop setting to both config and compressor
        model.config.mm_max_compress_loop = loop_val
        compressor = model.get_compressor()
        if compressor is not None:
            compressor.max_encode_loop = loop_val

        try:
            results = benchmark_one_config(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                mm_max_compress_loop=loop_val,
            )
            all_results[str(loop_val)] = results
        except Exception as e:
            print(f"  FATAL at loop={loop_val}: {e}")
            import traceback
            traceback.print_exc()
            all_results[str(loop_val)] = []

    # ── Save to JSON ──────────────────────────────────────────────────
    output = {
        "metadata": {
            "gpu": gpu_name,
            "checkpoint": str(CHECKPOINT_DIR),
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "image_size": IMAGE_SIZE,
            "frame_counts_range": [
                FRAME_COUNTS[0],
                FRAME_COUNTS[-1],
                FRAME_COUNTS[1] - FRAME_COUNTS[0],
            ],
            "loop_values": LOOP_VALUES,
            "total_params": total_n,
            "total_params_billions": round(total_b, 2),
            "dtype": str(model.dtype),
        },
        "results": all_results,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {OUTPUT_JSON}")

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "=" * 110)
    header = (f"{'Loop':>6} {'MaxFrames':>10} {'PeakMem':>8} {'MemRes':>8} "
              f"{'1stTok':>8} {'SubAvg':>8} {'Total':>8}")
    print(header)
    print("-" * 110)
    for loop_str, entries in all_results.items():
        if entries:
            last = entries[-1]
            print(f"{loop_str:>6} {last['num_frames']:>10} "
                  f"{last['peak_mem_gib']:>7.2f}GiB {last['mem_reserved_gib']:>7.2f}GiB "
                  f"{last['first_token_s']:>7.3f}s {last['subsequent_avg_s']:>7.3f}s "
                  f"{last['total_s']:>7.3f}s")
        else:
            print(f"{loop_str:>6} {'N/A':>10}")
    print("=" * 110)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
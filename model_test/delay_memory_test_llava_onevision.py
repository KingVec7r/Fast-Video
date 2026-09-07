#!/usr/bin/env python3
"""
LLaVA OneVision (0.5B) Delay & Memory Benchmark
================================================

Benchmark: total params, peak GPU memory, first-token latency,
and subsequent-token latency for LLaVA OneVision across 8–1024 frames.

Settings:
  - Frames: 8, 16, 24, ..., 1024 (step 8) from model_test/video_frames
  - Output limited to 101 tokens (greedy decoding)
  - flash_attention_2
  - Peak GPU memory INCLUDES model weights, visual tensors, KV cache,
    and activation buffers.
  - Timing starts AFTER frame tensors are on CUDA.
  - Results saved to model_test/llava_onevision_test_results.json
  - CUDA 0

CRITICAL for video:
  - Use image_processor.preprocess() directly (NOT process_images — that
    triggers AnyRes tiling meant for single-image).
  - Use modalities=["video"] so the model uses spatial pooling (196 tok/frame)
    instead of AnyRes multi-patch (729+ tok/image).
  - Only ONE <image> token in the prompt for video input.
"""

import json
import os
import sys
import warnings
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ═══════════════════════════════════════════════════════════
# Monkey-patches for transformers 5.x + LLaVA compatibility
# ═══════════════════════════════════════════════════════════
import transformers
import transformers.modeling_utils as _tf_mu

_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])

# ── Patch 1: transformers 5.x _finalize_model_loading ──
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

# ── Patch 2: functions removed from transformers.modeling_utils in v5.x ──
def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for t in input_tensors:
            if t.shape[chunk_dim] != tensor_shape:
                raise ValueError(f"shape mismatch along chunk_dim {chunk_dim}")
        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(f"chunk_size {chunk_size} not divisible by {tensor_shape}")
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
        input_chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
        output_chunks = tuple(forward_fn(*chunk) for chunk in zip(*input_chunks))
        return torch.cat(output_chunks, dim=chunk_dim)
    return forward_fn(*input_tensors)

def _find_pruneable_heads_and_indices(heads, num_heads, head_size, already_pruned_heads):
    heads_to_prune = set(heads) - set(already_pruned_heads)
    heads_to_keep = sorted(set(range(num_heads)) - heads_to_prune)
    index = torch.arange(num_heads * head_size).view(num_heads, head_size)
    index = index[heads_to_keep].flatten()
    return heads_to_prune, index

def _prune_linear_layer(layer, index, dim=0):
    import torch.nn as nn
    if dim == 0:
        layer.weight = nn.Parameter(layer.weight.index_select(0, index).clone().detach())
        if layer.bias is not None:
            layer.bias = nn.Parameter(layer.bias.index_select(0, index).clone().detach())
    elif dim == 1:
        layer.weight = nn.Parameter(layer.weight.index_select(1, index).clone().detach())
    return layer

_tf_mu.apply_chunking_to_forward = _apply_chunking_to_forward
_tf_mu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
_tf_mu.prune_linear_layer = _prune_linear_layer

# Now safe to import LLaVA
sys.path.insert(0, str(Path(__file__).resolve().parent / "LLaVA-NeXT"))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# Paths & constants
# ═══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
FRAMES_DIR = BASE_DIR / "video_frames"
OUTPUT_JSON = BASE_DIR / "llava_onevision_test_results.json"
CHECKPOINT_DIR = "/path/to/llava-onevision-qwen2-0.5b-ov"

FRAME_COUNTS = list(range(8, 129, 8))  # 8, 16, 24, ..., 128
MAX_NEW_TOKENS = 101
PROMPT = "Describe this video in detail."
CONV_TEMPLATE = "qwen_1_5"
MODEL_NAME = "llava_qwen"
VIS_TOKENS_PER_FRAME = 196  # 27x27 → 2D bilinear pool stride=2 → 14x14


def sample_frames(n: int) -> list[Image.Image]:
    all_frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if not all_frames:
        raise FileNotFoundError(f"No frames in {FRAMES_DIR}")
    avail = len(all_frames)
    idx = np.linspace(0, avail - 1, min(n, avail), dtype=int)
    frames = [Image.open(all_frames[i]).convert("RGB") for i in idx]
    while len(frames) < n:
        needed = n - len(frames)
        extra_idx = np.linspace(0, avail - 1, min(needed, avail), dtype=int)
        frames += [Image.open(all_frames[i]).convert("RGB") for i in extra_idx]
    return frames[:n]


def count_params(model) -> tuple[int, float]:
    n = sum(p.numel() for p in model.parameters())
    return n, n / 1e9


def build_llava_video_prompt(question: str) -> str:
    """ONE <image> token for video (LLaVA OneVision format)."""
    conv = conv_templates[CONV_TEMPLATE].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    device = "cuda:0"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Frame counts: {FRAME_COUNTS[0]}–{FRAME_COUNTS[-1]} "
          f"(step {FRAME_COUNTS[1] - FRAME_COUNTS[0]})")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    print(f"Prompt: {PROMPT}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")

    # ── Load model ──
    print("\n── Loading LLaVA OneVision 0.5B ──")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=CHECKPOINT_DIR,
        model_base=None,
        model_name=MODEL_NAME,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    model.eval()

    total_n, total_b = count_params(model)
    print(f"  Params: {total_n:,}  ({total_b:.2f} B)")
    print(f"  dtype: {model.dtype}")
    print(f"  Vision tower: {type(model.get_vision_tower()).__name__}")

    # ── Benchmark ──
    results = []
    for n_frames in FRAME_COUNTS:
        print(f"  Frames: {n_frames:>4} ...", end=" ", flush=True)
        try:
            frames = sample_frames(n_frames)

            # Video preprocessing: use image_processor.preprocess directly
            # (NOT process_images) to avoid AnyRes tiling
            frames_np = np.stack([np.array(f) for f in frames])
            pixel_values = image_processor.preprocess(
                frames_np, return_tensors="pt"
            )["pixel_values"]
            pixel_values = pixel_values.to(dtype=model.dtype, device=device)

            prompt = build_llava_video_prompt(PROMPT)
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            image_sizes = [f.size for f in frames]
            vis_tokens_est = n_frames * VIS_TOKENS_PER_FRAME

            gen_kwargs = {
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            # Warmup (1 token)
            _ = model.generate(
                input_ids,
                images=[pixel_values],
                image_sizes=image_sizes,
                modalities=["video"],
                max_new_tokens=1,
                **gen_kwargs,
            )
            torch.cuda.synchronize()

            # Peak memory + total latency (full generation)
            torch.cuda.reset_peak_memory_stats()
            start_total = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)
            start_total.record()
            with torch.inference_mode():
                model.generate(
                    input_ids,
                    images=[pixel_values],
                    image_sizes=image_sizes,
                    modalities=["video"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    **gen_kwargs,
                )
            end_total.record()
            torch.cuda.synchronize()
            total_s = start_total.elapsed_time(end_total) / 1000.0
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

            # First-token latency (separate 1-token call)
            start_first = torch.cuda.Event(enable_timing=True)
            end_first = torch.cuda.Event(enable_timing=True)
            start_first.record()
            with torch.inference_mode():
                model.generate(
                    input_ids,
                    images=[pixel_values],
                    image_sizes=image_sizes,
                    modalities=["video"],
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
                "first_token_s": round(first_s, 4),
                "total_s": round(total_s, 4),
                "subsequent_avg_s": round(subsequent_avg, 4),
                "peak_mem_gib": round(peak_mem, 4),
                "mem_reserved_gib": round(mem_reserved, 4),
                "vis_tokens_est": vis_tokens_est,
            }
            results.append(result)

            print(f"1st={first_s:.3f}s  sub_avg={subsequent_avg:.3f}s  "
                  f"total={total_s:.3f}s  mem={peak_mem:.2f}GiB "
                  f"(res={mem_reserved:.2f})  vis_tok={vis_tokens_est}")

        except torch.cuda.OutOfMemoryError:
            print(f"OOM — stopping at {n_frames} frames")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"ERROR: {e}")
            torch.cuda.empty_cache()
        finally:
            torch.cuda.empty_cache()

    # ── Save ──
    output = {
        "metadata": {
            "gpu": gpu_name,
            "model": "LLaVA-OneVision-0.5B",
            "checkpoint": str(CHECKPOINT_DIR),
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "frame_counts_range": [
                FRAME_COUNTS[0], FRAME_COUNTS[-1],
                FRAME_COUNTS[1] - FRAME_COUNTS[0],
            ],
            "total_params": total_n,
            "total_params_billions": round(total_b, 2),
            "dtype": str(model.dtype),
            "vis_tokens_per_frame": VIS_TOKENS_PER_FRAME,
            "vision_tower": type(model.get_vision_tower()).__name__,
        },
        "results": results,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {OUTPUT_JSON}")

    if results:
        last = results[-1]
        print(f"LLaVA OneVision | max_frames={last['num_frames']} | "
              f"1st={last['first_token_s']:.3f}s | "
              f"sub_avg={last['subsequent_avg_s']:.3f}s | "
              f"peak_mem={last['peak_mem_gib']:.2f}GiB")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()

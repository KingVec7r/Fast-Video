#!/usr/bin/env python3
"""
FLOP Test: Measure floating-point operations for VLM models
across 8–max_frames with gap=8.

Models:
  1. Qwen3.5-0.8B        (max 1024 frames) — calflops (already done, skip)
  2. InternVL3.5-1B       (max 128 frames)  — calflops (transformers 4.57.6)
  3. Fast-OneVision-0.5B  (max 128 frames)  — calflops (transformers 4.57.6)
  4. Fast-Video-0.8B      (max 1024 frames, swept over mm_max_compress_loop)
  5. SmolVLM2-500M        (max 64 frames)   — calflops (already done, skip)

Approach:
  - transformers 4.57.6 for InternVL/Fast-OneVision/Fast-Video compatibility
  - Keep existing Qwen/SmolVLM results from previous run
  - Use SDPA attention for calflops compatibility
  - Results saved incrementally to model_test/flop_test_results.json
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

# ──────────────────────────────────────────────────
# Monkey-patches for transformers compatibility
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

# Always apply these patches (needed by LLaVA code in transformers 4.x and 5.x)
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

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FRAMES_DIR = BASE_DIR / "video_frames"
OUTPUT_JSON = BASE_DIR / "flop_test_results.json"
PROMPT = "Describe this video in detail."
DEFAULT_ATTN = "sdpa"

MODEL_CONFIGS = {
    "Qwen3.5-0.8B": {
        "path": "/path/to/Qwen3.5-0.8B",
        "max_frames": 1024,
        "attn": DEFAULT_ATTN,
    },
    "InternVL3.5-1B": {
        "path": "/path/to/InternVL3_5-1B",
        "max_frames": 128,
        "attn": "native",
    },
    "Fast-OneVision-0.5B": {
        "path": "/path/to/llava-onevision-qwen2-0.5b-ov",
        "max_frames": 128,
        "attn": DEFAULT_ATTN,
    },
    "Fast-Video-0.8B": {
        "path": str(BASE_DIR.parent / "checkpoints" / "checkpoints_video_f32"),
        "max_frames": 1024,
        "attn": DEFAULT_ATTN,
        "loop_values": [1, 2, 4, 8, 16],
    },
    "SmolVLM2-500M": {
        "path": "/path/to/SmolVLM2-500M-Video-Instruct",
        "max_frames": 64,
        "attn": DEFAULT_ATTN,
    },
}


# ══════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════

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


def count_params(model) -> tuple:
    n = sum(p.numel() for p in model.parameters())
    return n, n / 1e9


def measure_flops_calflops(model, kwargs, forward_mode="forward"):
    """Measure FLOPs using calflops. Returns (flops, macs, params) or Nones."""
    try:
        from calflops import calculate_flops
        flops, macs, params = calculate_flops(
            model=model, kwargs=kwargs, forward_mode=forward_mode,
            print_results=False, output_as_string=False, output_precision=4,
        )
        return (float(flops) if not isinstance(flops, str) else 0.0,
                float(macs) if not isinstance(macs, str) else 0.0,
                float(params) if not isinstance(params, str) else 0.0)
    except Exception:
        return None, None, None


def estimate_flops_analytical(model, num_tokens: int) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    matmul_flops = 2.0 * n_params * num_tokens
    try:
        config = model.config
        if hasattr(config, "text_config"):
            cfg = config.text_config
        else:
            cfg = config
        n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", 24)
        n_heads = getattr(cfg, "num_attention_heads", 16)
        head_dim = getattr(cfg, "hidden_size", 2048) // n_heads
        attn_flops = 2.0 * n_layers * n_heads * head_dim * num_tokens * num_tokens
    except Exception:
        attn_flops = 0.0
    return matmul_flops + attn_flops


def save_results(results_dict, metadata):
    output = {"metadata": metadata, "results": results_dict}
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════
# Model-specific benchmarks
# ══════════════════════════════════════════════════

# ─── InternVL3.5-1B ───

def bench_internvl() -> list[dict]:
    """Benchmark InternVL3.5-1B using analytical FLOP estimation.
    The model's custom chat() interface is incompatible with calflops tracing."""
    import json as _json

    cfg = MODEL_CONFIGS["InternVL3.5-1B"]
    mp, max_frames = cfg["path"], cfg["max_frames"]

    # Read config for analytical estimation
    config_path = Path(mp) / "config.json"
    with open(config_path) as f:
        model_config = _json.load(f)

    llm_config = model_config.get("llm_config", {})
    hidden_size = llm_config.get("hidden_size", 2048)
    num_layers = llm_config.get("num_hidden_layers", 24)
    num_heads = llm_config.get("num_attention_heads", 16)
    head_dim = hidden_size // num_heads

    # InternVL: 1.06B params, 256 visual tokens per frame
    total_n = 1_060_897_792
    vis_tokens_per_frame = 256
    text_tokens_base = 50

    print(f"\n── InternVL3.5-1B (analytical from config) ──")
    print(f"  Params: {total_n:,}  ({total_n/1e9:.2f} B)  max_frames: {max_frames}")
    print(f"  hidden_size={hidden_size}  layers={num_layers}  heads={num_heads}")
    print(f"  Note: chat() interface incompatible with calflops; using analytical")

    frame_list = [f for f in range(8, max_frames + 1, 8)]
    results = []

    for n_frames in frame_list:
        print(f"  Frames: {n_frames:>4} ...", end=" ", flush=True)
        try:
            num_tokens = text_tokens_base + n_frames * vis_tokens_per_frame
            matmul_flops = 2.0 * total_n * num_tokens
            attn_flops = 2.0 * num_layers * num_heads * head_dim * num_tokens * num_tokens
            flops = matmul_flops + attn_flops
            macs = flops / 2.0

            result = {
                "num_frames": n_frames, "num_tokens": num_tokens,
                "flops": round(flops, 2), "macs": round(macs, 2),
                "params": total_n, "method": "analytical_config",
            }
            results.append(result)
            print(f"tokens={num_tokens}  FLOPs={flops:.4g}  [analytical]")
        except Exception as e:
            import traceback
            print(f"ERR: {type(e).__name__}: {e}")
            traceback.print_exc()

    return results


# ─── Fast-OneVision-0.5B (LLaVA OneVision) ───

def bench_fast_onevision() -> list[dict]:
    cfg = MODEL_CONFIGS["Fast-OneVision-0.5B"]
    mp, max_frames = cfg["path"], cfg["max_frames"]

    sys.path.insert(0, str(BASE_DIR / "LLaVA-NeXT"))
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    CONV_TEMPLATE = "qwen_1_5"
    MODEL_NAME = "llava_qwen"
    VIS_TOKENS_PER_FRAME = 196

    def _prompt(q):
        conv = conv_templates[CONV_TEMPLATE].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{q}")
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    print("\n── Loading Fast-OneVision-0.5B (sdpa) ──")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=mp, model_base=None, model_name=MODEL_NAME,
        torch_dtype="bfloat16", attn_implementation="sdpa", device_map="cuda:0",
    )
    model.eval()
    total_n, total_b = count_params(model)
    print(f"  Params: {total_n:,}  ({total_b:.2f} B)  max_frames: {max_frames}")

    gen_kwargs = {
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    frame_list = [f for f in range(8, max_frames + 1, 8)]
    results = []

    for n_frames in frame_list:
        print(f"  Frames: {n_frames:>4} ...", end=" ", flush=True)
        try:
            frames = sample_frames(n_frames)
            frames_np = np.stack([np.array(f) for f in frames])
            pixel_values = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(dtype=model.dtype, device="cuda:0")
            prompt = _prompt(PROMPT)
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to("cuda:0")
            image_sizes = [f.size for f in frames]
            num_tokens = input_ids.shape[1] - 1 + n_frames * VIS_TOKENS_PER_FRAME

            # Warmup
            _ = model.generate(
                input_ids, images=[pixel_values], image_sizes=image_sizes,
                modalities=["video"], max_new_tokens=1, **gen_kwargs,
            )
            torch.cuda.synchronize()

            # FLOPs via calflops generate mode
            gk = {
                "input_ids": input_ids, "images": [pixel_values],
                "image_sizes": image_sizes, "modalities": ["video"],
                "max_new_tokens": 1, **gen_kwargs,
            }
            flops, macs, params_measured = measure_flops_calflops(
                model, gk, forward_mode="generate")
            method = "calflops"
            if flops is None:
                flops = estimate_flops_analytical(model, num_tokens)
                macs = flops / 2.0
                params_measured = total_n
                method = "analytical"

            result = {
                "num_frames": n_frames, "num_tokens": num_tokens,
                "flops": round(flops, 2), "macs": round(macs, 2),
                "params": round(params_measured, 2) if params_measured else total_n,
                "method": method,
            }
            results.append(result)
            print(f"tokens={num_tokens}  FLOPs={flops:.4g}  [{method}]")
        except torch.cuda.OutOfMemoryError:
            print(f"OOM — stopping at {n_frames} frames")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            import traceback
            print(f"ERR: {type(e).__name__}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
        finally:
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ─── Fast-Video-0.8B ───

def bench_fast_video() -> dict:
    from transformers import AutoTokenizer
    from fast_onevision.model import FastOneVisionForCausalLM

    cfg = MODEL_CONFIGS["Fast-Video-0.8B"]
    mp, max_frames, loop_values = cfg["path"], cfg["max_frames"], cfg["loop_values"]
    IMAGE_SIZE = 512

    def _prompt(q, n):
        vp = "<|video_pad|>" * n
        return (f"<|im_start|>user\n<|vision_start|>{vp}<|vision_end|>\n"
                f"{q}<|im_end|>\n<|im_start|>assistant\n")

    attn = cfg.get("attn", DEFAULT_ATTN)
    print(f"\n── Loading Fast-Video-0.8B ({attn}) ──")
    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, use_fast=False)
    model = FastOneVisionForCausalLM.from_pretrained(
        mp, dtype=torch.bfloat16, attn_implementation=attn,
        device_map="cuda:0", trust_remote_code=True,
    ).eval()
    total_n, total_b = count_params(model)
    print(f"  Params: {total_n:,}  ({total_b:.2f} B)  max_frames: {max_frames}")
    print(f"  Vision tower: {type(model.get_vision_tower()).__name__}")

    image_processor = model.get_vision_tower().image_processor
    gen_kwargs = {
        "do_sample": False, "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    frame_list = [f for f in range(8, max_frames + 1, 8)]
    all_results = {}

    for loop_val in loop_values:
        print(f"\n{'─' * 55}\n  mm_max_compress_loop = {loop_val}\n{'─' * 55}")
        model.config.mm_max_compress_loop = loop_val
        comp = model.get_compressor()
        if comp is not None:
            comp.max_encode_loop = loop_val

        loop_results = []
        for n_frames in frame_list:
            print(f"    Frames: {n_frames:>4} ...", end=" ", flush=True)
            try:
                frames = sample_frames(n_frames)
                prompt = _prompt(PROMPT, n_frames)
                input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
                frames_pil = [Image.fromarray(np.array(f)) for f in frames]
                pixel_values = image_processor(
                    frames_pil, size=IMAGE_SIZE, crop_size=IMAGE_SIZE, return_tensors='pt'
                )['pixel_values']
                pixel_values = pixel_values.to(dtype=model.dtype, device=model.device).unsqueeze(0)
                input_ids = input_ids.to(model.device)
                nt = input_ids.shape[1] + n_frames

                # Warmup
                _ = model.generate(input_ids=input_ids, images=pixel_values,
                                   max_new_tokens=1, **gen_kwargs)
                torch.cuda.synchronize()

                # FLOPs via calflops generate mode
                gk = {"input_ids": input_ids, "images": pixel_values,
                      "max_new_tokens": 1, **gen_kwargs}
                flops, macs, params_measured = measure_flops_calflops(
                    model, gk, forward_mode="generate")
                method = "calflops"
                if flops is None:
                    flops = estimate_flops_analytical(model, nt)
                    macs = flops / 2.0
                    params_measured = total_n
                    method = "analytical"

                result = {
                    "num_frames": n_frames, "mm_max_compress_loop": loop_val,
                    "num_tokens": nt, "flops": round(flops, 2),
                    "macs": round(macs, 2),
                    "params": round(params_measured, 2) if params_measured else total_n,
                    "method": method,
                }
                loop_results.append(result)
                print(f"tokens={nt}  FLOPs={flops:.4g}  [{method}]")
            except torch.cuda.OutOfMemoryError:
                print(f"OOM — stopping at {n_frames} frames")
                torch.cuda.empty_cache()
                break
            except Exception as e:
                import traceback
                print(f"ERR: {type(e).__name__}: {e}")
                traceback.print_exc()
                torch.cuda.empty_cache()
            finally:
                torch.cuda.empty_cache()
        all_results[str(loop_val)] = loop_results

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return all_results


# ══════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Output: {OUTPUT_JSON}")
    for name, cfg in MODEL_CONFIGS.items():
        loops = cfg.get("loop_values", [])
        loop_str = f"  loops={loops}" if loops else ""
        print(f"  {name:<25s}  max_frames={cfg['max_frames']:>4}{loop_str}")

    # Load existing results
    all_results = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON) as f:
                existing = json.load(f)
            all_results = existing.get("results", {})
            print(f"  Loaded existing: {list(all_results.keys())}")
        except Exception:
            pass

    metadata = {"gpu": gpu_name, "prompt": PROMPT, "transformers": transformers.__version__}

    # Only run models that need calflops measurement:
    # Qwen, SmolVLM already have calflops data → skip
    # InternVL uses analytical (chat() incompatible with calflops) → skip
    # Fast-OneVision, Fast-Video → run now with calflops
    benchmarks = [
        ("Fast-OneVision-0.5B", bench_fast_onevision),
        ("Fast-Video-0.8B", bench_fast_video),
    ]

    for name, bench_fn in benchmarks:
        existing_data = all_results.get(name)
        if existing_data:
            if isinstance(existing_data, dict):
                # Fast-Video: dict of lists
                all_calflops = all(
                    (v[0].get("method") == "calflops" if v else False)
                    for v in existing_data.values() if v
                )
                if all_calflops:
                    total = sum(len(v) for v in existing_data.values())
                    print(f"\n  Skip {name} (already {total} calflops entries)")
                    continue
            elif isinstance(existing_data, list) and len(existing_data) > 0:
                if existing_data[0].get("method") == "calflops":
                    print(f"\n  Skip {name} (already {len(existing_data)} calflops entries)")
                    continue

        print(f"\n{'='*60}\n  Benchmarking {name}\n{'='*60}")
        try:
            all_results[name] = bench_fn()
            save_results(all_results, metadata)
            print(f"  [Saved to {OUTPUT_JSON}]")
        except Exception as e:
            print(f"FATAL: {e}")
            import traceback
            traceback.print_exc()
            save_results(all_results, metadata)

    print(f"\n✅ All results saved to {OUTPUT_JSON}")

    # Summary
    print("\n" + "=" * 100)
    print(f"{'Model':<25s} {'Loop':>6s} {'MaxFrames':>10s} {'MaxTokens':>10s} "
          f"{'MaxFLOPs':>14s} {'Method':>12s}")
    print("-" * 100)
    for mname, mres in all_results.items():
        if isinstance(mres, dict) and any(isinstance(v, list) for v in mres.values()):
            for lstr, entries in mres.items():
                if entries:
                    e = entries[-1]
                    fs = f"{e['flops']:.4g}" if e.get('flops') else "N/A"
                    print(f"{mname:<25s} {lstr:>6s} {e['num_frames']:>10} "
                          f"{str(e.get('num_tokens','N/A')):>10s} {fs:>14s} "
                          f"{e.get('method','N/A'):>12s}")
        elif isinstance(mres, list) and mres:
            e = mres[-1]
            fs = f"{e['flops']:.4g}" if e.get('flops') else "N/A"
            print(f"{mname:<25s} {'-':>6s} {e['num_frames']:>10} "
                  f"{str(e.get('num_tokens','N/A')):>10s} {fs:>14s} "
                  f"{e.get('method','N/A'):>12s}")
        else:
            print(f"{mname:<25s} {'-':>6s} {'N/A':>10} {'N/A':>10} {'N/A':>14s} {'N/A':>12s}")
    print("=" * 100)


if __name__ == "__main__":
    main()

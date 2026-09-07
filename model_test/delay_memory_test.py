#!/usr/bin/env python3
"""
Benchmark: total params, peak GPU memory, first-token latency,
and subsequent-token latency for three VLM models across 8–1024 frames.

Models:
  1. InternVL3_5-1B
  2. Qwen3.5-0.8B
  3. SmolVLM2-500M-Video-Instruct

Settings:
  - Frames: 8, 16, 24, ..., 1024 (step 8) from model_test/video_frames
  - Output limited to 101 tokens (greedy decoding)
  - transformers + flash_attention_2
  - Peak GPU memory INCLUDES model weights, visual tensors, KV cache,
    and activation buffers — representing the minimum GPU required to
    avoid OOM under transformers + flash_attention_2 (production budget).
  - Results saved to model_test/vlm_test_results.json
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

# ──────────────────────────────────────────────────
# Monkey-patches for InternVL + transformers 5.x
# (only applied when transformers >= 5.0)
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
OUTPUT_JSON = BASE_DIR / "vlm_test_results.json"
FRAME_COUNTS = list(range(8, 1025, 8))  # 8, 16, 24, ..., 1024
MAX_NEW_TOKENS = 101
PROMPT = "Describe this video in detail."

MODEL_PATHS = {
    "internvl": "/path/to/InternVL3_5-1B",
    "qwen":     "/path/to/Qwen3.5-0.8B",
    "smolvlm":  "/path/to/SmolVLM2-500M-Video-Instruct",
}

# Per-model max frame caps
MAX_FRAMES = {
    "internvl": 128,
    "qwen":     1024,
    "smolvlm":  64,
}

# InternVL preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_INPUT_SIZE = 448


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
# InternVL preprocessing
# ══════════════════════════════════════════════════
def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def dynamic_preprocess(image: Image.Image, image_size: int = 448, max_num: int = 1):
    w, h = image.size
    ar = w / h
    if ar > 1:
        nc, nr = min(max_num, max(1, round(ar))), 1
    else:
        nr, nc = min(max_num, max(1, round(1.0 / ar))), 1
    scale = min(nc * image_size / w, nr * image_size / h)
    nw, nh = (int(w * scale), int(h * scale)) if scale < 1 else (w, h)
    resized = image.resize((nw, nh), Image.BICUBIC)
    tiles = []
    for r in range(nr):
        for c in range(nc):
            tiles.append(resized.crop((c * image_size, r * image_size,
                                       (c + 1) * image_size, (r + 1) * image_size)))
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size), Image.BICUBIC))
    return tiles


def internvl_preprocess(frames: list[Image.Image]) -> tuple[torch.Tensor, list[int]]:
    transform = build_transform(INTERNVL_INPUT_SIZE)
    pv_list, np_list = [], []
    for f in frames:
        tiles = dynamic_preprocess(f, INTERNVL_INPUT_SIZE, max_num=1)
        pv_list.append(torch.stack([transform(t) for t in tiles]))
        np_list.append(len(tiles))
    return torch.cat(pv_list, dim=0), np_list


# ══════════════════════════════════════════════════
# Benchmark helpers
# ══════════════════════════════════════════════════
def measure_generate(model, gen_kwargs: dict) -> dict:
    """
    Measure first-token latency, total latency, and **true peak GPU memory**
    during generation.  Peak includes model weights, visual tensors, KV cache,
    and activation buffers — matching the minimum GPU required to avoid OOM
    under transformers + flash_attention_2 with no extra optimisations.

    Strategy:
      1. Warmup with 1 token (primes CUDA context, not measured).
      2. reset_peak_memory_stats() — baseline = model + inputs already on GPU.
      3. Generate MAX_NEW_TOKENS tokens — captures KV-cache / activation peak.
      4. First-token latency measured afterwards (lightweight, separate call).

    No torch.cuda.empty_cache() is called so that PyTorch's caching allocator
    behaves as it would in production.
    """
    # ── Warmup (1 token, not measured) ──
    with torch.inference_mode():
        model.generate(**{**gen_kwargs, "max_new_tokens": 1})
    torch.cuda.synchronize()

    # ── Peak memory + total latency (full generation) ──
    # Reset peak *after* warmup → peak starts from model + inputs baseline
    torch.cuda.reset_peak_memory_stats()
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    with torch.inference_mode():
        model.generate(**{**gen_kwargs, "max_new_tokens": MAX_NEW_TOKENS})
    end2.record()
    torch.cuda.synchronize()
    total_s = start2.elapsed_time(end2) / 1000.0
    # max_memory_allocated includes model weights + inputs (baseline) +
    # KV cache + activations allocated during generation
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

    # ── First-token latency (lightweight, separate 1-token call) ──
    s1 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    s1.record()
    with torch.inference_mode():
        model.generate(**{**gen_kwargs, "max_new_tokens": 1})
    e1.record()
    torch.cuda.synchronize()
    first_s = s1.elapsed_time(e1) / 1000.0

    subsequent_avg = (total_s - first_s) / (MAX_NEW_TOKENS - 1) if MAX_NEW_TOKENS > 1 else 0

    return {
        "first_token_s": first_s,
        "total_s": total_s,
        "subsequent_avg_s": subsequent_avg,
        "peak_mem_gib": peak_mem,
        "mem_reserved_gib": mem_reserved,
    }


# ══════════════════════════════════════════════════
# Model 1: InternVL3_5-1B
# ══════════════════════════════════════════════════
def bench_internvl() -> list[dict]:
    """Load InternVL once, benchmark all frame counts, return list of results."""
    from transformers import AutoModel, AutoTokenizer

    mp = MODEL_PATHS["internvl"]
    print("\n── Loading InternVL3_5-1B ──")
    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        mp, trust_remote_code=True, dtype=torch.bfloat16,
        use_flash_attn=True, device_map="cuda:0",
    ).eval()

    total_n, total_b = count_params(model)
    print(f"  Params: {total_n:,}  ({total_b:.2f} B)  max_frames: {MAX_FRAMES['internvl']}")

    frame_list = [f for f in FRAME_COUNTS if f <= MAX_FRAMES["internvl"]]
    results = []
    for n_frames in frame_list:
        print(f"  ── Frames: {n_frames} ──")
        try:
            frames = sample_frames(n_frames)
            pv, np_list = internvl_preprocess(frames)
            pv = pv.to(dtype=torch.bfloat16, device=model.device)
            video_prefix = "".join([f"Frame-{i+1}: <image>\n" for i in range(len(np_list))])
            question = video_prefix + PROMPT
            n_vis_tokens = sum(np_list) * 256  # InternVL: 256 visual tokens per tile

            # Warmup (1 token – primes CUDA context, not measured)
            _ = model.chat(tokenizer, pv, question, dict(max_new_tokens=1, do_sample=False),
                           num_patches_list=np_list, history=None, return_history=True)
            torch.cuda.synchronize()

            # ── Peak memory + total latency (full generation) ──
            # Reset peak *after* warmup → baseline includes model weights + visual tensors
            torch.cuda.reset_peak_memory_stats()
            s2 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
            s2.record()
            with torch.inference_mode():
                model.chat(tokenizer, pv, question, dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False),
                           num_patches_list=np_list, history=None, return_history=True)
            e2.record(); torch.cuda.synchronize()
            total_s = s2.elapsed_time(e2) / 1000.0
            # max_memory_allocated includes model weights + visual tensors (baseline)
            # + KV cache + activations allocated during generation
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

            # ── First-token latency (lightweight, separate 1-token call) ──
            s1 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            s1.record()
            with torch.inference_mode():
                model.chat(tokenizer, pv, question, dict(max_new_tokens=1, do_sample=False),
                           num_patches_list=np_list, history=None, return_history=True)
            e1.record(); torch.cuda.synchronize()
            first_s = s1.elapsed_time(e1) / 1000.0

            sub_avg = (total_s - first_s) / (MAX_NEW_TOKENS - 1) if MAX_NEW_TOKENS > 1 else 0

            results.append({
                "num_frames": n_frames,
                "first_token_s": round(first_s, 4),
                "total_s": round(total_s, 4),
                "subsequent_avg_s": round(sub_avg, 4),
                "peak_mem_gib": round(peak_mem, 4),
                "mem_reserved_gib": round(mem_reserved, 4),
                "vis_tiles": len(np_list),
                "vis_tokens_est": n_vis_tokens,
            })
            print(f"    1st={first_s:.3f}s  sub_avg={sub_avg:.3f}s  total={total_s:.3f}s  "
                  f"mem={peak_mem:.2f}GiB(res={mem_reserved:.2f})  vis_tok={n_vis_tokens}")
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at {n_frames} frames, stopping this model.")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"    ERROR at {n_frames} frames: {e}")
            torch.cuda.empty_cache()
        finally:
            # Reset allocator state between iterations so each frame count
            # is measured as a fresh single-inference scenario.
            torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache()
    return results


# ══════════════════════════════════════════════════
# Model 2: Qwen3.5-0.8B
# ══════════════════════════════════════════════════
def bench_qwen() -> list[dict]:
    """Load Qwen once, benchmark all frame counts, return list of results."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    mp = MODEL_PATHS["qwen"]
    print("\n── Loading Qwen3.5-0.8B ──")
    processor = AutoProcessor.from_pretrained(mp)
    model = AutoModelForImageTextToText.from_pretrained(
        mp, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map="cuda:0",
    ).eval()

    total_n, total_b = count_params(model)
    print(f"  Params: {total_n:,}  ({total_b:.2f} B)  max_frames: {MAX_FRAMES['qwen']}")

    frame_list = [f for f in FRAME_COUNTS if f <= MAX_FRAMES["qwen"]]
    results = []
    for n_frames in frame_list:
        print(f"  ── Frames: {n_frames} ──")
        try:
            frames = sample_frames(n_frames)

            msgs = [{"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": PROMPT},
            ]}]
            text = processor.apply_chat_template(msgs, add_generation_prompt=True)
            raw = processor(text=text, videos=[frames], return_tensors="pt",
                            num_frames=n_frames, fps=None)

            gen_kwargs = {}
            model_inputs = {}
            for k, v in raw.items():
                if k in ("pixel_values_videos", "video_grid_thw", "pixel_values", "image_grid_thw"):
                    gen_kwargs[k] = v.to(model.device)
                elif isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to(model.device)

            base_kwargs = {
                **model_inputs, **gen_kwargs,
                "do_sample": False,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            }

            # Diagnostics: input sequence length & visual token estimate
            input_seq_len = model_inputs.get("input_ids", torch.tensor([])).shape[-1]
            vg_thw = gen_kwargs.get("video_grid_thw")
            if vg_thw is not None and vg_thw.dim() == 2:
                vis_tokens_est = int((vg_thw.prod(dim=1)).sum().item())
            else:
                vis_tokens_est = 0

            # Warmup
            _ = model.generate(**base_kwargs, max_new_tokens=1)
            torch.cuda.synchronize()

            m = measure_generate(model, base_kwargs)
            m["num_frames"] = n_frames
            m["input_seq_len"] = input_seq_len
            m["vis_tokens_est"] = vis_tokens_est
            # Round for cleaner JSON
            for k in ("first_token_s", "total_s", "subsequent_avg_s", "peak_mem_gib", "mem_reserved_gib"):
                m[k] = round(m[k], 4)
            results.append(m)
            print(f"    1st={m['first_token_s']:.3f}s  sub_avg={m['subsequent_avg_s']:.3f}s  "
                  f"total={m['total_s']:.3f}s  mem={m['peak_mem_gib']:.2f}GiB(res={m['mem_reserved_gib']:.2f})  "
                  f"seq_len={input_seq_len}  vis_tok={vis_tokens_est}")
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at {n_frames} frames, stopping this model.")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"    ERROR at {n_frames} frames: {e}")
            torch.cuda.empty_cache()
        finally:
            torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache()
    return results


# ══════════════════════════════════════════════════
# Model 3: SmolVLM2-500M-Video-Instruct
# ══════════════════════════════════════════════════
def bench_smolvlm() -> list[dict]:
    """Load SmolVLM once, benchmark all frame counts, return list of results."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    mp = MODEL_PATHS["smolvlm"]
    print("\n── Loading SmolVLM2-500M-Video-Instruct ──")
    processor = AutoProcessor.from_pretrained(mp)
    processor.image_processor.do_image_splitting = False  # video branch

    model = AutoModelForImageTextToText.from_pretrained(
        mp, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map="cuda:0",
    ).eval()

    total_n, total_b = count_params(model)
    max_context = model.config.text_config.max_position_embeddings
    print(f"  Params: {total_n:,}  ({total_b:.2f} B)  max_context: {max_context}  max_frames: {MAX_FRAMES['smolvlm']}")

    frame_list = [f for f in FRAME_COUNTS if f <= MAX_FRAMES["smolvlm"]]
    results = []
    for n_frames in frame_list:
        print(f"  ── Frames: {n_frames} ──")
        try:
            frames = sample_frames(n_frames)

            msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]
                     + [{"type": "image"} for _ in range(n_frames)]}]
            prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
            # Use processor_kwargs to suppress transformers 5.x warning
            inputs = processor(text=prompt, images=frames, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Check context overflow
            input_seq_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
            overflow = input_seq_len > max_context
            if overflow:
                print(f"    ⚠ CONTEXT OVERFLOW: input_seq_len={input_seq_len} > max={max_context} (model will truncate!)")

            gen_kwargs = {**inputs, "do_sample": False}

            # Warmup
            _ = model.generate(**gen_kwargs, max_new_tokens=1)
            torch.cuda.synchronize()

            m = measure_generate(model, gen_kwargs)
            m["num_frames"] = n_frames
            m["input_seq_len"] = input_seq_len
            m["context_overflow"] = overflow
            # Per-frame visual tokens in SmolVLM2/Idefics3: image_seq_len learned queries
            # Default is typically 64 (from idefics3 config), check config for exact value
            img_seq_len = getattr(model.config, "image_seq_len", None) or \
                          getattr(model.config, "perceiver_config", {}).get("resampler_n_latents", 64) if hasattr(model.config, "perceiver_config") else 64
            m["vis_tokens_est"] = n_frames * img_seq_len
            for k in ("first_token_s", "total_s", "subsequent_avg_s", "peak_mem_gib", "mem_reserved_gib"):
                m[k] = round(m[k], 4)
            results.append(m)
            print(f"    1st={m['first_token_s']:.3f}s  sub_avg={m['subsequent_avg_s']:.3f}s  "
                  f"total={m['total_s']:.3f}s  mem={m['peak_mem_gib']:.2f}GiB(res={m['mem_reserved_gib']:.2f})  "
                  f"seq_len={input_seq_len}  overflow={overflow}")
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at {n_frames} frames, stopping this model.")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"    ERROR at {n_frames} frames: {e}")
            torch.cuda.empty_cache()
        finally:
            torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache()
    return results


# ══════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════
def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Frame counts: {FRAME_COUNTS[0]}–{FRAME_COUNTS[-1]} (step {FRAME_COUNTS[1]-FRAME_COUNTS[0]})")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    print(f"Prompt: {PROMPT}")

    all_results = {}

    for name, bench_fn in [("InternVL3_5-1B", bench_internvl),
                            ("Qwen3.5-0.8B", bench_qwen),
                            ("SmolVLM2-500M-Video-Instruct", bench_smolvlm)]:
        print(f"\n{'='*60}\n  Benchmarking {name}\n{'='*60}")
        try:
            all_results[name] = bench_fn()
        except Exception as e:
            print(f"  FATAL: {e}")
            import traceback; traceback.print_exc()
            all_results[name] = []

    # ── Save to JSON ──
    output = {
        "metadata": {
            "gpu": torch.cuda.get_device_name(0),
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "frame_counts_range": [FRAME_COUNTS[0], FRAME_COUNTS[-1], FRAME_COUNTS[1] - FRAME_COUNTS[0]],
        },
        "results": all_results,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_JSON}")

    # ── Print summary table (last data point per model) ──
    print("\n" + "=" * 110)
    print(f"{'Model':<30} {'MaxFrames':>10} {'PeakMem':>8} {'MemRes':>8} {'1stTok':>8} {'SubAvg':>8} {'Total':>8} {'SeqLen':>8}")
    print("-" * 110)
    for name, entries in all_results.items():
        if entries:
            last = entries[-1]
            seq_len = last.get('input_seq_len', 'N/A')
            mem_res = last.get('mem_reserved_gib', 0)
            print(f"{name:<30} {last['num_frames']:>10} {last['peak_mem_gib']:>7.2f}GiB {mem_res:>7.2f}GiB "
                  f"{last['first_token_s']:>7.3f}s {last['subsequent_avg_s']:>7.3f}s {last['total_s']:>7.3f}s {str(seq_len):>8}")
        else:
            print(f"{name:<30} {'N/A':>10}")
    print("=" * 110)
    print(f"Full per-frame-count data saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
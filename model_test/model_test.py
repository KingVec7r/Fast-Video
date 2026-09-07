#!/usr/bin/env python3
"""
Video Model Batch Evaluation Script using vLLM.

Evaluates 4 models across 6 video benchmarks using vLLM offline batch inference.

Usage:
    python model_test.py                          # all models × all benchmarks
    python model_test.py --models llava_onevision # single model
    python model_test.py --benchmarks videomme    # single benchmark
    python model_test.py --num-samples 20          # 20 random samples per dataset

Configuration (top of file):
    MODELS             – which models to test 
    BENCHMARKS         – which benchmarks to test 
    GPU_IDS            – CUDA devices to use 
    GPU_MEM_UTIL       – GPU memory utilization fraction 
    NUM_FRAMES         – frames sampled per video (default: 8)
    NUM_SAMPLES        – random samples per dataset 
    OUTPUT_DIR         – directory for result JSON files 
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Fix SmolVLM get_number_of_image_patches bug (must be at module level) ──
# SmolVLMImageProcessorPil.get_number_of_image_patches() returns 0 when
# size.longest_edge ≤ max_image_size.longest_edge, breaking vLLM profiling.
# This patch must be at module level because vLLM EngineCore runs in a
# spawn()-ed subprocess that re-imports everything from scratch.
try:
    from transformers.models.smolvlm.image_processing_pil_smolvlm import (
        SmolVLMImageProcessorPil,
    )
    _smolvlm_orig_get_num_patches = SmolVLMImageProcessorPil.get_number_of_image_patches

    def _smolvlm_patched_get_num_patches(self, height, width, images_kwargs=None):
        if images_kwargs is None:
            images_kwargs = {}
        num_patches, num_rows, num_cols = _smolvlm_orig_get_num_patches(
            self, height, width, images_kwargs
        )
        if num_patches <= 0:
            # Bug: when image fits in a single tile and longest_edge ≤ 512,
            # the function returns (0,0,0) instead of (1,0,0).
            # Return (1,0,0) to signal "1 patch, no grid splitting"
            # — NOT (1,1,1) which would trigger tile+global doubling.
            num_patches, num_rows, num_cols = 1, 0, 0
        return num_patches, num_rows, num_cols

    SmolVLMImageProcessorPil.get_number_of_image_patches = _smolvlm_patched_get_num_patches
    _smolvlm_patched = True
except ImportError:
    _smolvlm_patched = False

#  Configuration

# Models to evaluate (keys from MODEL_REGISTRY in model_test_datasets.py)
MODELS: List[str] = [
    "llava_onevision",  # Fixed: updated preprocessor_config.json, tokenizer, config.json
                        # (image_aspect_ratio="pad" disables anyres for video benchmarks),
                        # and converted weight checkpoint to vLLM-compatible format.
    "qwen35",
    "internvl",
    "smolvlm",
]

# Benchmarks to evaluate
BENCHMARKS: List[str] = [
    "videomme",
    "mlvu",
    "longvideobench",
    "lvbench",
    "egoschema",
    "mvbench",
]

# GPU configuration
GPU_IDS: str = "0,1,2,3,4,5,6,7"      # CUDA_VISIBLE_DEVICES (supports 1 / 2 / 4 / 8 GPUs)
GPU_MEM_UTIL: float = 0.95           # gpu_memory_utilization for vLLM

# Sampling
NUM_FRAMES: int = 512                    # frames per video
NUM_SAMPLES: int = 5                 # random samples per dataset
RANDOM_SEED: int = 42

# Output
OUTPUT_DIR: str = "video_results"

# vLLM sampling parameters
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 256
TOP_P: float = 1.0

# Chunked inference — process dataset in fixed-size chunks via a data-loader
# pipeline: NUM_DATA_WORKERS threads build chunks in the background while
# vLLM infers the current one, with up to NUM_DATA_WORKERS × PREFETCH_FACTOR
# chunks buffered ahead.
CHUNK_SIZE: int = 16               # samples per chunk
NUM_DATA_WORKERS: int = 4           # background threads preparing chunks
PREFETCH_FACTOR: int = 2            # max prefetched chunks per worker

# vLLM model configuration
LIMIT_MM_PER_PROMPT: int = 1024          # max images/video frames per prompt

# vLLM pre-allocates the encoder cache as max(max_num_batched_tokens,
# max_tokens_per_mm_item).  On modern GPUs max_num_batched_tokens
# defaults to 16384, which is far smaller than the ~200K visual tokens
# produced by 1024 video frames.  We compute it per-model in evaluate_model()
# as max(max_model_len, MIN_BATCHED_TOKENS) to ensure the encoder cache
# is sized to accommodate all frames.
MIN_BATCHED_TOKENS: int = 2048  # floor; actual value computed per model

#  Each model's max_model_len is set to its native max_position_embeddings.
#  Setting it higher (e.g. 264192 for all) causes RoPE positions to exceed
#  the trained range, producing NaN attention scores and CUBLAS crashes.
#  Models with insufficient native context for the configured NUM_FRAMES
#  will be rejected by vLLM with a clear prompt-length error rather than
#  a cryptic CUDA device-side assert.
#
#  Native max_position_embeddings:
#    llava_onevision  → 32768   (Qwen2-0.5B base)
#    qwen35           → 262144  (Qwen3.5-0.8B, native 256K)
#    internvl         → 40960   (InternVL3.5-1B → Qwen3 llm_config)
#    smolvlm          → 8192    (SmolVLM2-500M, native 8K)

MODEL_VLLM_CONFIG: Dict[str, Dict[str, Any]] = {
    "llava_onevision": {
        "weight_path": "/path/to/llava-onevision-qwen2-0.5b-ov",
        "display_name": "LLaVA-OneVision (Qwen2-0.5B-OV)",
        "tensor_parallel_size": 1,  # 14 attn heads → TP ∈ {1,2,7,14}; 1 fastest for 0.5B
        "max_model_len": 32768,      # native max_position_embeddings (Qwen2-0.5B)
        "mm_processor_kwargs": {"max_pixels": 2844352},
        "mm_data_key": "video",      # use <video> placeholder for multi-frame
    },
    "qwen35": {
        "weight_path": "/path/to/Qwen3.5-0.8B",
        "display_name": "Qwen3.5-0.8B",
        "tensor_parallel_size": 1,  # 12 attn heads → TP ∈ {1,2,3,4,6,12}; 1 fastest for 0.8B
        "max_model_len": 262144,     # native max_position_embeddings (Qwen3.5-0.8B)
        "mm_processor_kwargs": {"max_pixels": 23075472},
        "mm_data_key": "video",      # uses <|video_pad|> placeholder
    },
    "internvl": {
        "weight_path": "/path/to/InternVL3_5-1B",
        "display_name": "InternVL3.5-1B",
        "tensor_parallel_size": 1,  # 16 attn heads → TP ∈ {1,2,4,8,16}; 1 fastest for 1B
        "max_model_len": 40960,     # llm_config.max_position_embeddings (Qwen3)
        "mm_data_key": "video",     # InternVL video path: 448² resize → 1024 patches → pixel_shuffle → 256 tok/frame
    },
    "smolvlm": {
        "weight_path": "/path/to/SmolVLM2-500M-Video-Instruct",
        "display_name": "SmolVLM2-500M-Video-Instruct",
        "tensor_parallel_size": 1,  # 15 attn heads + 5 KV heads → TP ∈ {1, 5}
        "max_model_len": 8192,       # native max_position_embeddings (SmolVLM2-500M)
        "mm_processor_kwargs": {"size": {"longest_edge": 512}},
        # SmolVLM2 config.json has pad_token_id=128002 (Llama3 legacy) but
        # vocab_size is only 49280.  Override to 2 (<|im_end|>) which matches
        # text_config.pad_token_id and special_tokens_map.json.
        "hf_overrides": {"pad_token_id": 2},
    },
}


#  Helper utilities
def ensure_output_dir(path: str) -> str:
    """Create output directory if it doesn't exist and return absolute path."""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def timestamp() -> str:
    """Return a compact ISO-like timestamp for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_batch_for_indices(
    chunk_indices: np.ndarray,
    dataset,
    mm_data_key: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build (batch_prompts, sample_ids) for a slice of indices."""
    from PIL import Image

    batch_prompts: List[Dict[str, Any]] = []
    sample_ids: List[str] = []

    for idx in chunk_indices:
        try:
            item = dataset[idx]
        except Exception as e:
            print(f"    ⚠  Skipping index {idx}: {e}")
            continue

        frames_pil = [Image.fromarray(f) for f in item["frames"]]

        if mm_data_key == "video":
            num_frames_item = len(frames_pil)
            frames_np = np.stack([np.array(f) for f in frames_pil])
            video_metadata: Dict[str, Any] = {
                "fps": 2.0,
                "total_num_frames": num_frames_item,
                "frames_indices": list(range(num_frames_item)),
                "do_sample_frames": False,
            }
            mm_data: Dict[str, Any] = {mm_data_key: (frames_np, video_metadata)}
        else:
            mm_data = {mm_data_key: frames_pil}

        batch_prompts.append({
            "prompt": item["prompt"],
            "multi_modal_data": mm_data,
        })
        sample_ids.append(item["id"])

    return batch_prompts, sample_ids


def run_vllm_inference(
    llm: Any,
    sampling_params: Any,
    dataset,
    num_samples: int = 10,
    mm_data_key: str = "image",
    chunk_size: int = 16,
    skip_ids: Optional[set] = None,
) -> List[Dict[str, str]]:
    """
    Run vLLM batch inference on the dataset.

    Uses a data-loader pipeline: ``NUM_DATA_WORKERS`` background threads
    build chunks while vLLM infers the current one, with up to
    ``NUM_DATA_WORKERS × PREFETCH_FACTOR`` chunks buffered ahead.
    This overlaps video decoding / frame preprocessing with GPU inference.

    Args:
        llm: vLLM LLM instance (already loaded).
        sampling_params: vLLM SamplingParams.
        dataset: model-specific Dataset instance.
        num_samples: number of random samples to evaluate.
        chunk_size: samples per chunk (default: 16).
        skip_ids: set of sample IDs to skip (already completed).

    Returns:
        List of dicts: [{"id": ..., "output": ...}, ...]
    """
    import gc
    import threading
    from queue import Queue

    # ── Pick and shuffle indices ─────────────────────────────────────────
    shuffle_seed = RANDOM_SEED + (os.getpid() % 10007)
    rng = np.random.RandomState(shuffle_seed)
    if num_samples <= 0 or num_samples >= len(dataset):
        indices = rng.permutation(len(dataset))
    else:
        indices = rng.choice(
            len(dataset), size=num_samples, replace=False
        )

    # ── Filter out already-completed IDs (resume support) ───────────────
    if skip_ids:
        kept_indices = []
        for idx in indices:
            item_id = str(dataset.data[idx].get("id", ""))
            if item_id and item_id not in skip_ids:
                kept_indices.append(idx)
        skipped = len(indices) - len(kept_indices)
        if skipped > 0:
            print(f"    ⏭  Skipping {skipped} already-completed samples")
        indices = np.array(kept_indices) if kept_indices else np.array([], dtype=int)

    total_samples = len(indices)
    if total_samples == 0:
        print("    All samples already completed — nothing to infer.")
        return []

    if chunk_size <= 0 or chunk_size > total_samples:
        chunk_size = total_samples

    # ── Divide into chunks ───────────────────────────────────────────────
    chunk_boundaries: List[Tuple[int, int]] = []
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        chunk_boundaries.append((start, end))
    total_chunks = len(chunk_boundaries)

    if total_chunks == 0:
        return []

    # ── Prefetch queue (bounded, back-pressures workers when full) ───────
    prefetch_q: "Queue[Tuple[int, List, List]]" = Queue(
        maxsize=NUM_DATA_WORKERS * PREFETCH_FACTOR
    )

    # Atomic dispatch counter across workers
    next_cid: List[int] = [0]
    dispatch_lock = threading.Lock()

    def data_worker() -> None:
        """Background thread: grab next chunk, decode frames, put in queue."""
        while True:
            with dispatch_lock:
                cid = next_cid[0]
                if cid >= total_chunks:
                    break
                next_cid[0] += 1
                start, end = chunk_boundaries[cid]

            chunk_idx_arr = indices[start:end]
            prompts, ids = _build_batch_for_indices(
                chunk_idx_arr, dataset, mm_data_key,
            )
            # Always enqueue — even if empty — to keep the counting
            # consistent with the main loop.
            prefetch_q.put((cid, prompts, ids))

    # ── Launch data workers ──────────────────────────────────────────────
    workers: List[threading.Thread] = []
    for i in range(NUM_DATA_WORKERS):
        t = threading.Thread(
            target=data_worker, name=f"data-wkr-{i}", daemon=True,
        )
        t.start()
        workers.append(t)

    # ── Main inference loop ──────────────────────────────────────────────
    all_results: List[Dict[str, str]] = []
    results_map: Dict[int, List[Dict[str, str]]] = {}

    for chunk_idx in range(total_chunks):
        cid, batch_prompts, sample_ids = prefetch_q.get()

        if not batch_prompts:
            if total_chunks > 1:
                print(f"    Chunk {cid + 1}/{total_chunks}: empty, skipping.")
            continue

        if total_chunks > 1:
            print(f"    Chunk {cid + 1}/{total_chunks}: "
                  f"inferring {len(batch_prompts)} samples ...")
        else:
            print(f"    Running inference on {len(batch_prompts)} samples ...")

        outputs = llm.generate(
            prompts=batch_prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        chunk_results: List[Dict[str, str]] = []
        for sid, out in zip(sample_ids, outputs):
            text = out.outputs[0].text.strip() if out.outputs else ""
            chunk_results.append({"id": sid, "output": text})
        results_map[cid] = chunk_results

        # Aggressive CUDA memory cleanup between chunks.
        # For large num_frames (e.g. 1024), each chunk allocates substantial
        # GPU memory for the encoder cache.  Without explicit cleanup the
        # EngineCore can crash with an out-of-memory error after several chunks.
        del batch_prompts, sample_ids, outputs
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    # ── Wait for workers to drain ────────────────────────────────────────
    for t in workers:
        t.join()

    # ── Assemble results in original chunk order ─────────────────────────
    for cid in range(total_chunks):
        if cid in results_map:
            all_results.extend(results_map[cid])

    return all_results

#  Main evaluation logic
def evaluate_model(
    model_key: str,
    benchmarks: List[str],
    num_frames: int = 8,
    num_samples: int = 10,
    output_dir: str = "video_results",
    gpu_ids: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single model across all specified benchmarks.

    Args:
        gpu_ids: GPU IDs to use for this model. If None, uses the global GPU_IDS.
                 In multiprocessing mode, each process sets its own subset.

    Returns a summary dict.
    """
    from model_test_datasets import create_dataset, MODEL_REGISTRY

    vllm_cfg = MODEL_VLLM_CONFIG[model_key]
    weight_path = vllm_cfg["weight_path"]
    display_name = vllm_cfg["display_name"]

    if not os.path.isdir(weight_path):
        print(f"  ⚠  Weight path not found: {weight_path}  —  SKIPPING {display_name}")
        return {"model": model_key, "status": "skipped", "reason": "weight_path missing"}

    print(f"\n{'='*70}")
    print(f"  Model: {display_name}")
    print(f"  Weights: {weight_path}")
    print(f"{'='*70}")

    # ── Load vLLM model ────────────────────────────────────────────────────
    print("  Loading model with vLLM ...")
    try:
        from vllm import LLM, SamplingParams

        effective_gpu_ids = gpu_ids if gpu_ids else GPU_IDS
        num_gpus = len(effective_gpu_ids.split(",")) if effective_gpu_ids else 1
        tp_size = _resolve_tp_size(model_key, num_gpus)

        # Use model-specific max_model_len matching native max_position_embeddings.
        # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 is set globally in main() for models
        # where max_model_len slightly exceeds the derived limit (e.g. internvl).
        # Models with native context too small for the configured NUM_FRAMES
        # will get a clear prompt-length error from vLLM.
        model_max_len = vllm_cfg["max_model_len"]

        # Build LLM kwargs, optionally including hf_overrides
        #
        # limit_mm_per_prompt must include both "image" and "video" keys
        # because models like Qwen2-VL/Qwen3-VL use the "video" modality
        # for multi-frame inputs.  Without the "video" key, vLLM may
        # under-allocate the encoder cache.
        mm_limit: Dict[str, int] = {"image": LIMIT_MM_PER_PROMPT}
        mm_data_key = vllm_cfg.get("mm_data_key", "image")
        if mm_data_key == "video":
            mm_limit["video"] = LIMIT_MM_PER_PROMPT

        # Encoder cache must be ≥ max possible visual tokens per prompt.
        # Use model_max_len as the upper bound since total tokens (visual +
        # text) cannot exceed it.  On modern GPUs this overrides the default
        # 16384 which is far too small for multi-frame video.
        per_model_batched_tokens = max(model_max_len, MIN_BATCHED_TOKENS)

        llm_kwargs: Dict[str, Any] = {
            "model": weight_path,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": GPU_MEM_UTIL,
            "max_model_len": model_max_len,
            "trust_remote_code": True,
            "limit_mm_per_prompt": mm_limit,
            "max_num_batched_tokens": per_model_batched_tokens,
            "seed": RANDOM_SEED,
            "max_num_seqs": 64,  # Conservative: avoid Mamba cache block exhaustion
        }
        if vllm_cfg.get("hf_overrides"):
            llm_kwargs["hf_overrides"] = vllm_cfg["hf_overrides"]
        if vllm_cfg.get("mm_processor_kwargs"):
            llm_kwargs["mm_processor_kwargs"] = vllm_cfg["mm_processor_kwargs"]

        llm = LLM(**llm_kwargs)

        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            stop=["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_utterance>"],
        )
        print("  ✓ Model loaded successfully.")

    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        traceback.print_exc()
        return {"model": model_key, "status": "error", "error": str(e)}

    # ── Evaluate each benchmark ────────────────────────────────────────────
    all_results: Dict[str, Any] = {
        "model_key": model_key,
        "model_name": display_name,
        "weight_path": weight_path,
        "num_frames": num_frames,
        "num_samples_per_benchmark": num_samples,
        "timestamp": timestamp(),
        "benchmarks": {},
    }

    for bench in benchmarks:
        print(f"\n  --- Benchmark: {bench} ---")
        try:
            # Create dataset
            ds = create_dataset(model_key, bench, num_frames=num_frames)
            total = len(ds)
            print(f"    Dataset size: {total}")

            if total == 0:
                print(f"    ⚠  Empty dataset, skipping.")
                all_results["benchmarks"][bench] = {"status": "empty", "count": 0}
                continue

            # ── Load existing results for resume ─────────────────────────
            bench_dir = os.path.join(output_dir, model_key)
            os.makedirs(bench_dir, exist_ok=True)
            json_path = os.path.join(bench_dir, f"{bench}.json")

            existing_results: List[Dict[str, str]] = []
            skip_ids: set = set()
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        prev = json.load(f)
                    prev_results = prev.get("results", [])
                    # Only reuse results whose IDs are in the current dataset
                    valid_ids = {str(ds.data[i].get("id", "")) for i in range(len(ds))}
                    for r in prev_results:
                        rid = r.get("id", "")
                        if rid and rid in valid_ids:
                            existing_results.append(r)
                            skip_ids.add(rid)
                    if existing_results:
                        print(f"    📋 Loaded {len(existing_results)} existing results "
                              f"→ will skip these samples")
                except (json.JSONDecodeError, OSError) as e:
                    print(f"    ⚠  Could not read existing results ({e}) — starting fresh")

            # ── Run inference on remaining samples ───────────────────────
            new_results = run_vllm_inference(
                llm=llm,
                sampling_params=sampling_params,
                dataset=ds,
                num_samples=num_samples,
                mm_data_key=vllm_cfg.get("mm_data_key", "image"),
                chunk_size=CHUNK_SIZE,
                skip_ids=skip_ids,
            )

            # ── Merge & save ─────────────────────────────────────────────
            all_results_list = existing_results + new_results
            completed = len(all_results_list)
            completion_pct = (completed / total * 100) if total > 0 else 100.0

            bench_result = {
                "model": model_key,
                "model_name": display_name,
                "benchmark": bench,
                "num_frames": num_frames,
                "num_samples": completed,
                "dataset_total": total,
                "completion_rate": round(completion_pct, 1),
                "timestamp": timestamp(),
                "results": all_results_list,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(bench_result, f, ensure_ascii=False, indent=2)

            print(f"    ✓ Saved {completed} results ({completion_pct:.1f}% of {total}) → {json_path}")
            if new_results:
                print(f"      (existing: {len(existing_results)}, new: {len(new_results)})")

            # Print a few examples from new results
            for r in new_results[:3]:
                out_preview = r["output"].replace("\n", "\\n")
                if len(out_preview) > 120:
                    out_preview = out_preview[:120] + "..."
                print(f"      [{r['id']}] → {out_preview}")

            all_results["benchmarks"][bench] = {
                "status": "ok",
                "count": completed,
                "dataset_total": total,
                "completion_rate": round(completion_pct, 1),
                "json_path": json_path,
            }

        except Exception as e:
            print(f"    ✗ Error on {bench}: {e}")
            traceback.print_exc()
            all_results["benchmarks"][bench] = {
                "status": "error",
                "error": str(e),
            }

    # Clean up vLLM to free GPU memory
    try:
        del llm
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return all_results

#  Tensor-parallel size resolution
#  ─────────────────────────────────────────

def _resolve_tp_size(model_key: str, num_gpus: int) -> int:
    """Resolve tensor_parallel_size based on available GPUs.

    Clamps the configured TP to the number of available GPUs.
    Warns if the configured TP exceeds available GPUs so the user
    can adjust ``MODEL_VLLM_CONFIG`` or ``--gpu-ids`` accordingly.
    """
    cfg = MODEL_VLLM_CONFIG[model_key]
    configured_tp = cfg.get("tensor_parallel_size", 1)
    if configured_tp > num_gpus:
        print(
            f"  ⚠  {model_key}: configured TP={configured_tp} "
            f"> available GPUs={num_gpus}, falling back to TP={num_gpus}"
        )
        return num_gpus
    return configured_tp


#  CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch video model evaluation with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models", nargs="*", default=None,
        help="Model keys to evaluate (default: all 4). "
             "Choices: llava_onevision, qwen35, internvl, smolvlm",
    )
    p.add_argument(
        "--benchmarks", nargs="*", default=None,
        help="Benchmark keys to evaluate (default: all 6). "
             "Choices: videomme, mlvu, longvideobench, lvbench, egoschema, mvbench",
    )
    p.add_argument(
        "--gpu-ids", type=str, default=GPU_IDS,
        help=f"Comma-separated CUDA device IDs (default: {GPU_IDS})",
    )
    p.add_argument(
        "--gpu-mem-util", type=float, default=GPU_MEM_UTIL,
        help=f"GPU memory utilization fraction (default: {GPU_MEM_UTIL})",
    )
    p.add_argument(
        "--num-frames", type=int, default=NUM_FRAMES,
        help=f"Frames per video (default: {NUM_FRAMES})",
    )
    p.add_argument(
        "--num-samples", type=int, default=NUM_SAMPLES,
        help=f"Random samples per dataset (default: {NUM_SAMPLES}). "
             "Set to 0 to evaluate ALL samples.",
    )
    p.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    p.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})",
    )
    p.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS,
        help=f"Max generated tokens (default: {MAX_TOKENS})",
    )
    p.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Samples per chunk (default: {CHUNK_SIZE}). "
             "Data is prepared by {NUM_DATA_WORKERS} background workers "
             f"with prefetch factor {PREFETCH_FACTOR}.",
    )
    p.add_argument(
        "--min-batched-tokens", type=int, default=MIN_BATCHED_TOKENS,
        help=f"Floor for max_num_batched_tokens / encoder cache size per model "
             f"(default: {MIN_BATCHED_TOKENS}). Each model uses max(max_model_len, this).",
    )
    p.add_argument(
        "--serial", action="store_true", default=False,
        help="(Deprecated) Models always run serially using the leading GPUs.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Apply CLI overrides
    models = args.models if args.models else MODELS
    benchmarks = args.benchmarks if args.benchmarks else BENCHMARKS
    global GPU_IDS, GPU_MEM_UTIL, NUM_FRAMES, NUM_SAMPLES, OUTPUT_DIR
    global TEMPERATURE, MAX_TOKENS, MIN_BATCHED_TOKENS, CHUNK_SIZE

    GPU_IDS = args.gpu_ids
    GPU_MEM_UTIL = args.gpu_mem_util
    NUM_FRAMES = args.num_frames
    NUM_SAMPLES = args.num_samples
    OUTPUT_DIR = args.output_dir
    TEMPERATURE = args.temperature
    MAX_TOKENS = args.max_tokens
    MIN_BATCHED_TOKENS = args.min_batched_tokens
    CHUNK_SIZE = args.chunk_size

    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    # Allow max_model_len to exceed native max_position_embeddings
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    print(f"CUDA_VISIBLE_DEVICES = {GPU_IDS}")
    print(f"GPU memory utilization = {GPU_MEM_UTIL}")
    print(f"Frames per video = {NUM_FRAMES}")
    print(f"Samples per dataset = {NUM_SAMPLES}")
    print(f"Chunk size = {CHUNK_SIZE} {'(auto)' if CHUNK_SIZE == 0 else ''}")
    print(f"Min batched tokens (floor) = {MIN_BATCHED_TOKENS}")
    print(f"Output directory = {os.path.abspath(OUTPUT_DIR)}")

    # Build output directory with frame-count suffix
    frame_output_dir = os.path.join(OUTPUT_DIR, f"{NUM_FRAMES}frames")
    print(f"Actual output path = {os.path.abspath(frame_output_dir)}")

    # Validate
    from model_test_datasets import MODEL_REGISTRY
    valid_models = set(MODEL_REGISTRY.keys())
    valid_benchmarks = {
        "videomme", "mlvu", "longvideobench", "lvbench", "egoschema", "mvbench"
    }

    for m in models:
        if m not in valid_models:
            print(f"ERROR: unknown model '{m}'. Valid: {sorted(valid_models)}")
            sys.exit(1)
    for b in benchmarks:
        if b not in valid_benchmarks:
            print(f"ERROR: unknown benchmark '{b}'. Valid: {sorted(valid_benchmarks)}")
            sys.exit(1)

    print(f"\nModels to evaluate: {models}")
    print(f"Benchmarks to evaluate: {benchmarks}")

    # Ensure output directory exists
    ensure_output_dir(frame_output_dir)

    # ── Run evaluation ── 
    summary: Dict[str, Any] = {
        "config": {
            "models": models,
            "benchmarks": benchmarks,
            "gpu_ids": GPU_IDS,
            "gpu_memory_utilization": GPU_MEM_UTIL,
            "num_frames": NUM_FRAMES,
            "num_samples": NUM_SAMPLES,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "min_batched_tokens": MIN_BATCHED_TOKENS,
        },
        "models": {},
    }

    # ── Run evaluation (models always run serially, each using all GPUs) ── 
    for model_key in models:
        result = evaluate_model(
            model_key=model_key,
            benchmarks=benchmarks,
            num_frames=NUM_FRAMES,
            num_samples=NUM_SAMPLES,
            output_dir=frame_output_dir,
        )
        summary["models"][model_key] = result

    # ── Write global summary ─ 
    summary_path = os.path.join(frame_output_dir, f"summary_{timestamp()}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  Evaluation complete! Summary → {summary_path}")
    print(f"{'='*70}")

    # Quick summary table
    print(f"\n  {'Model':<25} {'Benchmark':<20} {'Status':<10} {'Completed':<12} {'Rate':<8}")
    print(f"  {'-'*75}")
    has_failures = False
    for mk, mr in summary["models"].items():
        if "benchmarks" in mr:
            for bk, br in mr["benchmarks"].items():
                status = br.get("status", "?")
                count = br.get("count", 0)
                total = br.get("dataset_total", 0)
                rate = br.get("completion_rate", None)
                rate_str = f"{rate:.1f}%" if rate is not None else "-"
                count_str = f"{count}/{total}" if total else str(count)
                print(f"  {mk:<25} {bk:<20} {status:<10} {count_str:<12} {rate_str:<8}")
                if status == "error":
                    has_failures = True
                elif NUM_SAMPLES == 0 and status == "ok":
                    if rate is not None and rate < 100.0:
                        print(f"  ⚠  WARNING: {mk}/{bk} completion={rate:.1f}% < 100% — incomplete!")
                        has_failures = True
                    elif rate is None:
                        # Fallback: no rate available, use heuristic minimums
                        _expected_mins = {
                            "videomme": 300, "mlvu": 10, "longvideobench": 3,
                            "lvbench": 30, "egoschema": 30, "mvbench": 30,
                        }
                        min_expected = _expected_mins.get(bk, 50)
                        if count < min_expected:
                            print(f"  ⚠  WARNING: {mk}/{bk} count={count} < expected min={min_expected} — likely partial results")
                            has_failures = True

    if has_failures:
        print(f"\n  ✗ Some benchmarks failed or produced partial results — exiting with code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
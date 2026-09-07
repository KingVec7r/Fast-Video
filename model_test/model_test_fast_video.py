#!/usr/bin/env python3
"""
Fast Video Model Batch Evaluation Script (Native HF, no vLLM).

Evaluates the custom Fast Video model across 6 video benchmarks using
HuggingFace native generate with **batched inference** (configurable batch size).

Usage:
    python model_test_fast_video.py                          # default config (batch=16)
    python model_test_fast_video.py --num-frames 64          # custom frame count
    python model_test_fast_video.py --mm-max-compress-loop 1024 # custom compress loop
    python model_test_fast_video.py --num-samples 20         # 20 random samples
    python model_test_fast_video.py --benchmarks videomme    # single benchmark
    python model_test_fast_video.py --benchmarks mvbench     # MVBench only
    python model_test_fast_video.py --batch-size 8           # smaller batch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import gc
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
from transformers import AutoTokenizer

# ── Configuration ────────────────────────────────────────────────────────────

# Model checkpoint (resolved relative to this script)
# MODEL_PATH: str = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "..", "checkpoints", "checkpoints_video_part4", "checkpoint-4380"
# )
MODEL_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "checkpoints", "checkpoints_video_f32"
)

# Benchmarks to evaluate (6 video benchmarks including MVBench)
BENCHMARKS: List[str] = [
    "videomme",
    "mlvu",
    "longvideobench",
    "lvbench",
    "egoschema",
    "mvbench",
]

# GPU configuration (single-GPU per process; multi-GPU orchestrated externally)
GPU_IDS: str = "0"

# Sampling
NUM_FRAMES: int = 8
NUM_SAMPLES: int = 2              # test mode — 2 random samples per benchmark
RANDOM_SEED: int = 42

# Compressor loop
MM_MAX_COMPRESS_LOOP: int = 1024

# Output
OUTPUT_DIR: str = "fast_video_results"

# Generation parameters
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 256
TOP_P: float = 1.0

# Batch inference
BATCH_SIZE: int = 16

# DataLoader workers (async data preparation, same as training pipeline)
DATALOADER_NUM_WORKERS: int = 4
DATALOADER_PREFETCH_FACTOR: int = 2

# Image preprocessing size
IMAGE_SIZE: int = 512


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset wrappers — reuse abstract_test_class base classes, but override
#  __getitem__ for native HF inference (no vLLM)
# ═══════════════════════════════════════════════════════════════════════════════

from abstract_test_class import (
    BaseTestDataset,
    VideoMMEDataset,
    MLVUDataset,
    LongVideoBenchDataset,
    LVBenchDataset,
    EgoSchemaDataset,
    MVBenchDataset,
    _format_mc_question,
)


class FastVideoPromptMixin:
    """Mixin providing Fast Video model prompt formatting for native HF inference.

    Uses Qwen3-style chat format with ``<|video_pad|>`` tokens.
    The number of ``<|video_pad|>`` tokens equals ``num_frames`` because
    the compressor maps each frame to exactly 1 compressed token.
    """

    MODEL_TYPE = "fast_video"

    def _format_prompt(self, question: str, num_frames: int) -> str:
        """Build the complete prompt string with video placeholders."""
        video_pads = "<|video_pad|>" * num_frames
        vision_block = f"<|vision_start|>{video_pads}<|vision_end|>"
        return (
            f"<|im_start|>user\n{vision_block}\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def __getitem__(self, idx: int) -> Dict:
        """Return a dict ready for batch collation & native HF inference.

        Image preprocessing (frame sampling + CLIP processor) happens here
        so that DataLoader workers can parallelise it — same pattern as the
        training ``LazySupervisedDataset.__getitem__``.

        Keys:
            id          – str, sample identifier
            input_ids   – torch.LongTensor (1, L)
            image_pixel – torch.FloatTensor (T, C, H, W)  *already preprocessed*
            prompt      – str (for debugging)
        """
        item = self.data[idx]
        question = item["question"]
        video_path = item["_video_path"]

        frames = self.sample_video_frames(video_path, self._num_frames)
        question += "\nAnswer with the option letter from the given choices directly."

        prompt = self._format_prompt(question, len(frames))
        input_ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors='pt'
        )

        # Preprocess frames through the vision tower's image processor.
        # Done inside __getitem__ so DataLoader workers parallelise the
        # PIL conversion + resize + normalize work.
        frames_pil = [Image.fromarray(f) for f in frames]
        image_pixel = self.image_processor(
            frames_pil,
            size=self._image_size,
            crop_size=self._image_size,
            return_tensors='pt'
        )['pixel_values']  # [T, C, H, W]

        return {
            "id": item["id"],
            "input_ids": input_ids,
            "image_pixel": image_pixel,
            "prompt": prompt,
        }


# ── Create model-specific dataset classes ─────────────────────────────────────

def _make_fast_video_dataset_class(base_cls: type, class_name: str) -> type:
    return type(class_name, (FastVideoPromptMixin, base_cls), {})


FastVideo_VideoMME = _make_fast_video_dataset_class(VideoMMEDataset, "FastVideo_VideoMME")
FastVideo_MLVU = _make_fast_video_dataset_class(MLVUDataset, "FastVideo_MLVU")
FastVideo_LongVideoBench = _make_fast_video_dataset_class(LongVideoBenchDataset, "FastVideo_LongVideoBench")
FastVideo_LVBench = _make_fast_video_dataset_class(LVBenchDataset, "FastVideo_LVBench")
FastVideo_EgoSchema = _make_fast_video_dataset_class(EgoSchemaDataset, "FastVideo_EgoSchema")
FastVideo_MVBench = _make_fast_video_dataset_class(MVBenchDataset, "FastVideo_MVBench")

_BENCHMARK_CLASSES: Dict[str, type] = {
    "videomme": FastVideo_VideoMME,
    "mlvu": FastVideo_MLVU,
    "longvideobench": FastVideo_LongVideoBench,
    "lvbench": FastVideo_LVBench,
    "egoschema": FastVideo_EgoSchema,
    "mvbench": FastVideo_MVBench,
}


def create_fast_video_dataset(
    benchmark: str,
    tokenizer: AutoTokenizer,
    image_processor,
    num_frames: int = 8,
    image_size: int = 512,
) -> BaseTestDataset:
    """Instantiate a Fast Video dataset for the given benchmark.

    Attaches ``tokenizer``, ``image_processor``, and ``_image_size`` to the
    dataset instance so that ``__getitem__`` (which may run inside DataLoader
    worker processes) has everything it needs to produce a fully-preprocessed
    sample.
    """
    if benchmark not in _BENCHMARK_CLASSES:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Choose from: {list(_BENCHMARK_CLASSES.keys())}"
        )
    cls = _BENCHMARK_CLASSES[benchmark]
    ds = cls(num_frames=num_frames)
    ds.tokenizer = tokenizer
    ds.image_processor = image_processor
    ds._image_size = image_size
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_output_dir(path: str) -> str:
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def cleanup_gpu_memory():
    """Force CUDA memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════════════════
#  Data collator (mirrors DataCollatorForSupervisedDataset in train.py)
# ═══════════════════════════════════════════════════════════════════════════════

class DataCollatorForEval:
    """Collate ``__getitem__`` outputs into left-padded eval sub-batches.

    Samples are **grouped by frame count** (``image_pixel.shape[0]``) so that
    every sub-batch has uniform ``T``, avoiding the dimension mismatch that
    occurs when videos with different frame counts are concatenated.

    * ``input_ids`` are *left-padded* (causal-LM generation standard).
    * ``image_pixel`` tensors are stacked → ``[B, T, C, H, W]``.
    * ``attention_mask`` marks real (non-padding) tokens.

    Returns a **list** of sub-batch dicts (one per distinct frame count).
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, instances: List[Dict]) -> List[Dict[str, Any]]:
        # ── Group instances by frame count (T dimension of image_pixel) ─
        groups: Dict[int, List[Dict]] = {}
        for inst in instances:
            t = inst["image_pixel"].shape[0] if inst["image_pixel"] is not None else 0
            groups.setdefault(t, []).append(inst)

        sub_batches: List[Dict[str, Any]] = []
        for t, group in groups.items():
            sub_batches.append(self._collate_group(group))
        return sub_batches

    def _collate_group(self, instances: List[Dict]) -> Dict[str, Any]:
        """Collate a group of instances that share the same frame count."""
        ids = [inst["id"] for inst in instances]
        prompt_lens = [inst["input_ids"].shape[1] for inst in instances]

        # ── Left-pad input_ids ──────────────────────────────────────────
        max_len = max(prompt_lens)
        padded_ids_list = []
        for inst in instances:
            ids_t = inst["input_ids"].squeeze(0)  # (L,)
            pad_len = max_len - ids_t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len,), self.pad_token_id, dtype=ids_t.dtype
                )
                padded = torch.cat([padding, ids_t], dim=0)
            else:
                padded = ids_t
            padded_ids_list.append(padded)

        input_ids = torch.stack(padded_ids_list, dim=0)  # [B, L_max]

        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, plen in enumerate(prompt_lens):
            attention_mask[i, max_len - plen:] = True

        # ── Stack image_pixel tensors (all have same T by construction) ─
        images_list = [inst["image_pixel"] for inst in instances]
        if images_list[0] is not None:
            images = torch.stack(images_list, dim=0)  # [B, T, C, H, W]
        else:
            images = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "prompt_lens": prompt_lens,
            "ids": ids,
        }


def run_batch_inference(
    model,
    tokenizer: AutoTokenizer,
    batch: Dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 256,
    top_p: float = 1.0,
) -> List[str]:
    """Run batched generation.

    Args:
        model: FastOneVisionForCausalLM
        tokenizer: HF tokenizer
        batch: dict from ``DataCollatorForEval`` (or equivalently-shaped dict
               with keys input_ids, attention_mask, images, prompt_lens, ids).

    Returns:
        List of decoded responses (one per sample), stop tokens stripped.
    """
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    images = batch["images"].to(dtype=model.dtype, device=model.device)
    prompt_lens = batch["prompt_lens"]

    gen_kwargs = {
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "attention_mask": attention_mask,
    }

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            images=images,
            **gen_kwargs
        )

    # Strip prompt portion and decode each sample
    max_len_batch = input_ids.shape[1]  # L_max (padded)
    responses: List[str] = []
    for i, plen in enumerate(prompt_lens):
        # prompt occupies the rightmost `plen` positions in left-padded input
        prompt_start = max_len_batch - plen
        generated_ids = outputs[i][prompt_start + plen:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Remove any remaining stop tokens
        for stop_str in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            if stop_str in response:
                response = response.split(stop_str)[0].strip()

        responses.append(response)

    return responses


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_benchmark(
    model,
    tokenizer: AutoTokenizer,
    image_processor,
    benchmark: str,
    num_frames: int,
    num_samples: int,
    output_dir: str,
    image_size: int = 512,
    temperature: float = 0.0,
    max_tokens: int = 256,
    top_p: float = 1.0,
    mm_max_compress_loop: int = 16,
    batch_size: int = 16,
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> Dict[str, Any]:
    """Evaluate model on a single benchmark using DataLoader-based batching."""
    print(f"\n  --- Benchmark: {benchmark} ---")
    t_start = time.monotonic()

    # ── Temporarily override mm_max_compress_loop ─────────────────────────
    original_loop = model.config.mm_max_compress_loop
    model.config.mm_max_compress_loop = mm_max_compress_loop
    compressor = model.get_compressor()
    if compressor is not None:
        original_comp_loop = compressor.max_encode_loop
        compressor.max_encode_loop = mm_max_compress_loop

    try:
        ds = create_fast_video_dataset(
            benchmark, tokenizer, image_processor,
            num_frames=num_frames, image_size=image_size,
        )
        total = len(ds)
        print(f"    Dataset size: {total}")

        if total == 0:
            print(f"    ⚠  Empty dataset, skipping.")
            return {"status": "empty", "count": 0}

        # ── Pick random samples ──────────────────────────────────────────
        rng = np.random.RandomState(RANDOM_SEED + hash(benchmark) % 10007)
        if num_samples <= 0 or num_samples >= total:
            indices = rng.permutation(total)
        else:
            indices = rng.choice(total, size=num_samples, replace=False)

        # ── Resume: load existing results ────────────────────────────────
        bench_dir = os.path.join(output_dir, "fast_video")
        os.makedirs(bench_dir, exist_ok=True)
        json_path = os.path.join(bench_dir, f"{benchmark}.json")

        existing_results: Dict[str, str] = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                for r in prev.get("results", []):
                    rid = r.get("id", "")
                    if rid:
                        existing_results[rid] = r.get("output", "")
                if existing_results:
                    print(f"    📋 Loaded {len(existing_results)} existing results → will skip")
            except (json.JSONDecodeError, OSError) as e:
                print(f"    ⚠  Could not read existing results ({e}) — starting fresh")

        # ── Build pending index list (exclude already-completed samples) ─
        # Use ds.data[idx]["id"] (metadata-only, no __getitem__ call) to
        # avoid paying the cost of frame sampling just to check the ID.
        skipped = 0
        pending_indices: List[int] = []
        for idx in indices:
            sid = str(ds.data[idx]["id"])
            if sid in existing_results:
                skipped += 1
                continue
            pending_indices.append(idx)

        total_pending = len(pending_indices)
        print(f"    Pending samples: {total_pending}  (skipped {skipped} already done)")

        if total_pending == 0:
            print(f"    ✓ All samples already completed.")
            return {
                "status": "ok",
                "count": len(existing_results),
                "dataset_total": total,
                "completion_rate": round(len(existing_results) / total * 100, 1),
                "json_path": json_path,
                "errors": 0,
            }

        # ── DataLoader: async multi-worker data preparation ──────────────
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        collator = DataCollatorForEval(pad_token_id=pad_id)

        eval_subset = Subset(ds, pending_indices)
        dataloader = DataLoader(
            eval_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            collate_fn=collator,
        )

        # ── Batch inference loop ─────────────────────────────────────────
        new_results: List[Dict[str, str]] = []
        errors = 0
        num_batches = len(dataloader)

        for batch_idx, sub_batches in enumerate(dataloader):
            # sub_batches is a list of dicts, one per distinct frame count
            for sub_idx, sub_batch in enumerate(sub_batches):
                sub_size = len(sub_batch["ids"])
                sub_frames = sub_batch["images"].shape[1] if sub_batch["images"] is not None else 0
                desc = f"Batch [{batch_idx+1}/{num_batches}]"
                if len(sub_batches) > 1:
                    desc += f".{sub_idx+1}/{len(sub_batches)}"
                print(f"    {desc} (size={sub_size}, frames={sub_frames}) ...", flush=True)

                t0 = time.monotonic()
                try:
                    outputs = run_batch_inference(
                        model=model,
                        tokenizer=tokenizer,
                        batch=sub_batch,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )

                    elapsed = time.monotonic() - t0
                    avg_per_sample = elapsed / sub_size

                    for j, output in enumerate(outputs):
                        sid = sub_batch["ids"][j]
                        preview = output.replace("\n", "\\n")
                        if len(preview) > 100:
                            preview = preview[:100] + "..."
                        print(f"      [{j+1}/{sub_size}] id={sid} → {preview}")
                        new_results.append({"id": sid, "output": output})

                    print(f"    ✓ sub-batch done ({elapsed:.1f}s total, "
                          f"{avg_per_sample:.2f}s/sample)")

                except Exception as e:
                    errors += sub_size
                    print(f"    ✗ SUB-BATCH ERROR: {e}")
                    traceback.print_exc()
                    for sid in sub_batch["ids"]:
                        new_results.append({"id": sid, "output": f"[ERROR] {e}"})

            if (batch_idx + 1) % 5 == 0:
                cleanup_gpu_memory()

        # ── Merge & save ─────────────────────────────────────────────────
        all_results_list = [
            {"id": rid, "output": out}
            for rid, out in existing_results.items()
        ]
        all_results_list.extend(new_results)

        completed = len(all_results_list)
        completion_pct = (completed / total * 100) if total > 0 else 100.0

        bench_result = {
            "model": "fast_video",
            "model_name": "Fast Video (Qwen3-0.6B base)",
            "benchmark": benchmark,
            "num_frames": num_frames,
            "mm_max_compress_loop": mm_max_compress_loop,
            "num_samples": completed,
            "dataset_total": total,
            "completion_rate": round(completion_pct, 1),
            "timestamp": timestamp(),
            "results": all_results_list,
            "errors": errors,
            "skipped": skipped,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(bench_result, f, ensure_ascii=False, indent=2)

        elapsed_total = time.monotonic() - t_start
        print(f"    ✓ Saved {completed} results ({completion_pct:.1f}% of {total}) "
              f"→ {json_path}  ({elapsed_total:.1f}s)")
        if new_results:
            print(f"      (existing: {len(existing_results)}, new: {len(new_results)}, "
                  f"skipped: {skipped}, errors: {errors})")

        return {
            "status": "ok",
            "count": completed,
            "dataset_total": total,
            "completion_rate": round(completion_pct, 1),
            "json_path": json_path,
            "errors": errors,
        }

    except Exception as e:
        print(f"    ✗ Error on {benchmark}: {e}")
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

    finally:
        # Restore original config
        model.config.mm_max_compress_loop = original_loop
        if compressor is not None:
            compressor.max_encode_loop = original_comp_loop


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(
    model_path: str,
    gpu_id: int = 0,
    mm_max_compress_loop: int = 16,
) -> Tuple[Any, AutoTokenizer, Any]:
    """Load the Fast Video model, tokenizer, and image processor.

    Note: ``gpu_id`` is only used for logging.  The actual device is always
    ``cuda:0`` because the orchestrator isolates GPUs via
    ``CUDA_VISIBLE_DEVICES`` — when it assigns GPU *N* to this process, that
    GPU becomes ``cuda:0`` inside the subprocess.
    """
    # Always use the first (and only) visible CUDA device.  The orchestrator
    # sets CUDA_VISIBLE_DEVICES to a single GPU, so ``cuda:0`` is correct.
    device = "cuda:0"

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    print(f"Loading model from {model_path} ...")
    # Import here to avoid failures when torch isn't available yet
    from fast_onevision.model import FastOneVisionForCausalLM

    model = FastOneVisionForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
        trust_remote_code=True,
    ).eval()

    # Override mm_max_compress_loop
    if mm_max_compress_loop is not None:
        model.config.mm_max_compress_loop = mm_max_compress_loop
        compressor = model.get_compressor()
        if compressor is not None:
            compressor.max_encode_loop = mm_max_compress_loop

    image_processor = model.get_vision_tower().image_processor

    print(f"  Model loaded on {device}.  dtype={model.dtype}")
    print(f"  Vision tower:    {type(model.get_vision_tower()).__name__}")
    print(f"  Image processor: {type(image_processor).__name__}")
    print(f"  mm_max_compress_loop: {model.config.mm_max_compress_loop}")

    return model, tokenizer, image_processor


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast Video model batch evaluation (native HF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-path", type=str, default=MODEL_PATH,
                   help=f"Model checkpoint path")
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Benchmark keys (default: all 6). "
                        "Choices: videomme, mlvu, longvideobench, lvbench, egoschema, mvbench")
    p.add_argument("--gpu-ids", type=str, default=GPU_IDS,
                   help=f"CUDA device ID for this process (default: {GPU_IDS})")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES,
                   help=f"Frames per video (default: {NUM_FRAMES})")
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                   help=f"Random samples per dataset (default: {NUM_SAMPLES}). "
                        "Set to 0 for ALL samples.")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                   help=f"Output directory (default: {OUTPUT_DIR})")
    p.add_argument("--temperature", type=float, default=TEMPERATURE,
                   help=f"Sampling temperature (default: {TEMPERATURE})")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                   help=f"Max generated tokens (default: {MAX_TOKENS})")
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                   help=f"Image preprocessing size (default: {IMAGE_SIZE})")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help=f"Batch size for inference (default: {BATCH_SIZE})")
    p.add_argument("--num-workers", type=int, default=DATALOADER_NUM_WORKERS,
                   help=f"DataLoader workers (default: {DATALOADER_NUM_WORKERS})")
    p.add_argument("--prefetch-factor", type=int, default=DATALOADER_PREFETCH_FACTOR,
                   help=f"DataLoader prefetch factor (default: {DATALOADER_PREFETCH_FACTOR})")
    p.add_argument("--mm-max-compress-loop", type=int, default=MM_MAX_COMPRESS_LOOP,
                   help=f"Compressor max loop iterations (default: {MM_MAX_COMPRESS_LOOP})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(model_path):
        print(f"ERROR: Model path not found: {model_path}")
        sys.exit(1)

    benchmarks = args.benchmarks if args.benchmarks else BENCHMARKS
    gpu_id = int(args.gpu_ids.split(",")[0])

    # Validate
    valid_benchmarks = set(_BENCHMARK_CLASSES.keys())
    for b in benchmarks:
        if b not in valid_benchmarks:
            print(f"ERROR: unknown benchmark '{b}'. Valid: {sorted(valid_benchmarks)}")
            sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # Build output directory with frame+loop suffix
    frame_output_dir = os.path.join(
        args.output_dir,
        f"f{args.num_frames}_l{args.mm_max_compress_loop}"
    )
    ensure_output_dir(frame_output_dir)

    print("=" * 70)
    print("  Fast Video Model — Batch Evaluation")
    print("=" * 70)
    print(f"  Model path:      {model_path}")
    print(f"  GPU:             {args.gpu_ids}")
    print(f"  Frames:          {args.num_frames}")
    print(f"  Compress loop:   {args.mm_max_compress_loop}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  DataLoader:      {args.num_workers} workers × prefetch {args.prefetch_factor}")
    print(f"  Samples/bench:   {args.num_samples} {'(FULL)' if args.num_samples == 0 else ''}")
    print(f"  Benchmarks:      {benchmarks}")
    print(f"  Temperature:     {args.temperature}")
    print(f"  Max tokens:      {args.max_tokens}")
    print(f"  Output:          {os.path.abspath(frame_output_dir)}")
    print("=" * 70)

    # ── Load model ──────────────────────────────────────────────────────────
    print("\nLoading model ...")
    model, tokenizer, image_processor = load_model(
        model_path=model_path,
        gpu_id=gpu_id,
        mm_max_compress_loop=args.mm_max_compress_loop,
    )

    # ── Evaluate each benchmark ──────────────────────────────────────────────
    all_results: Dict[str, Any] = {
        "model_key": "fast_video",
        "model_name": "Fast Video (Qwen3-0.6B base)",
        "model_path": model_path,
        "num_frames": args.num_frames,
        "mm_max_compress_loop": args.mm_max_compress_loop,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "num_samples_per_benchmark": args.num_samples,
        "timestamp": timestamp(),
        "benchmarks": {},
    }

    for bench in benchmarks:
        br = evaluate_benchmark(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            benchmark=bench,
            num_frames=args.num_frames,
            num_samples=args.num_samples,
            output_dir=frame_output_dir,
            image_size=args.image_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=TOP_P,
            mm_max_compress_loop=args.mm_max_compress_loop,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        all_results["benchmarks"][bench] = br

    # ── Save summary ─────────────────────────────────────────────────────────
    summary_path = os.path.join(frame_output_dir, f"summary_{timestamp()}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  Evaluation complete!  Summary → {summary_path}")
    print(f"{'='*70}")

    for bench, br in all_results["benchmarks"].items():
        status = br.get("status", "?")
        count = br.get("count", 0)
        total = br.get("dataset_total", 0)
        rate = br.get("completion_rate", 0)
        errs = br.get("errors", 0)
        print(f"  {bench:20s}  {status:6s}  {count:4d}/{total:<6d}  "
              f"({rate:.0f}%)  errors={errs}")

    del model
    cleanup_gpu_memory()


if __name__ == "__main__":
    main()

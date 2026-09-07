#!/usr/bin/env python3
"""
Parallel Fast Video Model Evaluation Orchestrator (per-benchmark tasks)
=======================================================================

Runs (num_frames × mm_max_compress_loop × benchmark) tasks across 4 GPUs
in parallel.  Tasks are sorted by num_frames descending, then by
mm_max_compress_loop descending (largest / heaviest first).
When a GPU becomes free, the next task is immediately assigned to it.

Each task evaluates the Fast Video model on a **single** benchmark with the
given (num_frames, compress_loop) configuration, using 4 DataLoader workers
and batch size 16.

Usage:
    # Test mode (2 samples per benchmark):
    python run_parallel_fast_video.py

    # Full evaluation (all samples):
    python run_parallel_fast_video.py --full

    # Custom:
    python run_parallel_fast_video.py --num-samples 10 --gpus 0,1,2,3
""" 

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from queue import Queue
from typing import Dict, List, NamedTuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  Task definitions  — per-benchmark tasks
# ═══════════════════════════════════════════════════════════════════════════════

class Task(NamedTuple):
    num_frames: int
    mm_max_compress_loop: int
    benchmark: str


# Frame counts and compress loop values
FRAME_COUNTS = [8, 16, 32, 64, 128, 256, 512, 1024]
LOOP_VALUES = [16, 1024]

# Benchmarks (including MVBench)
BENCHMARKS: List[str] = [
    "videomme",
    "mlvu",
    "longvideobench",
    "lvbench",
    "egoschema",
    "mvbench",
]


def _build_all_tasks() -> List[Task]:
    """
    Build every (num_frames, compress_loop, benchmark) combination.

    Sorted by num_frames descending, then mm_max_compress_loop descending.
    Total: 8 × 3 × 6 = 144 tasks.
    """
    tasks: List[Task] = []
    for frames in FRAME_COUNTS:
        for loop in LOOP_VALUES:
            for bench in BENCHMARKS:
                tasks.append(Task(frames, loop, bench))
    # Sort: largest frames first, largest loop first
    tasks.sort(key=lambda t: (t.num_frames, t.mm_max_compress_loop), reverse=True)
    return tasks


RAW_TASKS: List[Task] = _build_all_tasks()

# ── GPU config ────────────────────────────────────────────────────────────────

DEFAULT_GPU_IDS: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TEST_SCRIPT = os.path.join(SCRIPT_DIR, "model_test_fast_video.py")


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def tstamp() -> str:
    """Compact timestamp for log lines."""
    return datetime.now().strftime("%m-%d %H:%M:%S")


def task_label(num_frames: int, compress_loop: int, benchmark: str) -> str:
    """Human-readable task label."""
    return f"f{num_frames}_l{compress_loop} [{benchmark}]"


def _gpu_memory_summary() -> str:
    """One-line GPU memory snapshot via nvidia-smi (best-effort)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        parts = []
        for line in out.stdout.strip().split("\n"):
            idx, used, total = line.split(", ")
            parts.append(f"GPU{idx}:{used}/{total}MiB")
        return " | ".join(parts)
    except Exception:
        return "nvidia-smi unavailable"


def _check_summary_json(num_frames: int, compress_loop: int, benchmark: str, num_samples: int = 0) -> int:
    """
    Validate the most recent summary JSON containing the specific benchmark.

    When ``num_samples > 0`` (test/sample mode), only checks that the
    expected number of samples were processed rather than requiring 100%
    completion of the full dataset.

    Returns 0 if the benchmark appears genuinely successful.
    Returns a magic number (≥ 99) if no valid summary was found.
    """
    output_dir = os.path.join(
        SCRIPT_DIR, "fast_video_results",
        f"f{num_frames}_l{compress_loop}"
    )
    if not os.path.isdir(output_dir):
        return 999  # no output directory at all → major failure

    # Find the most recent summary file
    pattern = os.path.join(output_dir, "summary_*.json")
    summary_files = sorted(glob.glob(pattern), reverse=True)  # newest first
    if not summary_files:
        return 998  # no summary file at all → major failure

    # Search summaries from newest to oldest for one that contains this benchmark
    summary = None
    for sp in summary_files:
        try:
            with open(sp, "r", encoding="utf-8") as f:
                candidate = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if benchmark in candidate.get("benchmarks", {}):
            summary = candidate
            break

    if summary is None:
        return 997  # benchmark not found in any summary → task produced no output

    br = summary["benchmarks"][benchmark]

    status = br.get("status", "?")
    count = br.get("count", 0)
    total = br.get("dataset_total", 0)
    errors = br.get("errors", 0)
    rate = br.get("completion_rate", None)

    if status == "error":
        return 1

    if errors > 0:
        return 1

    # ── Determine expected sample count ─────────────────────────────
    if num_samples > 0:
        # Test/sample mode: expect min(num_samples, total) samples
        expected = min(num_samples, total)
    else:
        # Full mode: expect all samples
        expected = total

    if count < expected:
        return 1

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Task runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_task(
    num_frames: int,
    compress_loop: int,
    benchmark: str,
    gpu_id: int,
    num_samples: int,
    *,
    extra_args: Optional[List[str]] = None,
) -> int:
    """
    Run ``model_test_fast_video.py`` for one (num_frames, compress_loop,
    benchmark) combination on the specified GPU.

    Returns exit code (0 = success, non-zero = failure).
    Also validates the summary JSON for false-success detection.
    """
    cmd: List[str] = [
        sys.executable, MODEL_TEST_SCRIPT,
        "--num-frames", str(num_frames),
        "--mm-max-compress-loop", str(compress_loop),
        "--benchmarks", benchmark,
        "--gpu-ids", str(gpu_id),
        "--num-samples", str(num_samples),
        "--batch-size", "2",
        "--num-workers", "4",
        "--prefetch-factor", "1",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    label = task_label(num_frames, compress_loop, benchmark)
    print(f"[{tstamp()}]  GPU {gpu_id} ▶ START  {label}")

    t0 = time.monotonic()

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
        )
        elapsed = time.monotonic() - t0
        elapsed_str = f"{elapsed/60:.1f}m" if elapsed >= 60 else f"{elapsed:.1f}s"

        # Validate via summary JSON
        json_failures = _check_summary_json(num_frames, compress_loop, benchmark, num_samples)

        if proc.returncode != 0 or json_failures:
            if proc.returncode != 0:
                print(f"[{tstamp()}]  GPU {gpu_id} ✗ FAIL   {label}  "
                      f"(exit={proc.returncode}, {elapsed_str})")
            else:
                print(f"[{tstamp()}]  GPU {gpu_id} ✗ FAIL   {label}  "
                      f"(exit=0 but benchmark still incomplete, {elapsed_str})")
            # Print last 30 lines for diagnostics
            combined = (proc.stdout + proc.stderr).strip().split("\n")
            for line in combined[-30:]:
                print(f"    │ {line}")
            return 1
        else:
            print(f"[{tstamp()}]  GPU {gpu_id} ✓ DONE   {label}  ({elapsed_str})")
            return 0
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"[{tstamp()}]  GPU {gpu_id} ✗ ERROR  {label}  "
              f"({elapsed:.1f}s): {exc}")
        return 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel Fast Video model evaluation across 4 GPUs (per-benchmark tasks)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=2,
        help="Samples per benchmark (default: 2 for test mode; use 0 for full)",
    )
    parser.add_argument(
        "--full", action="store_true", default=False,
        help="Full evaluation: equivalent to --num-samples 0",
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Override model checkpoint path",
    )
    args = parser.parse_args()

    num_samples = 0 if args.full else args.num_samples

    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        gpu_ids = list(DEFAULT_GPU_IDS)

    # Build extra args (no --benchmarks; each task specifies its own)
    extra_args: List[str] = []
    if args.model_path:
        extra_args.extend(["--model-path", args.model_path])

    # ── Print plan ────────────────────────────────────────────────────────
    num_gpus = len(gpu_ids)
    print("=" * 70)
    print("  Parallel Fast Video Model Evaluation Orchestrator")
    print("=" * 70)
    print(f"  Tasks:           {len(RAW_TASKS)}  "
          f"(frames × loops × benchmarks: "
          f"{len(FRAME_COUNTS)}×{len(LOOP_VALUES)}×{len(BENCHMARKS)})")
    print(f"  GPUs:            {gpu_ids}  ({num_gpus} workers)")
    print(f"  Samples/bench:   {num_samples} {'(FULL)' if num_samples == 0 else '(test mode)'}")
    if extra_args:
        print(f"  Extra args:      {' '.join(extra_args)}")
    print(f"  GPU memory:      {_gpu_memory_summary()}")
    print(f"\n  Task queue (largest frames first, largest loop first):")
    for i, t in enumerate(RAW_TASKS):
        print(f"    {i+1:3d}. {task_label(t.num_frames, t.mm_max_compress_loop, t.benchmark)}")
    print("=" * 70)
    print()

    # ── Queues ────────────────────────────────────────────────────────────
    task_queue: "Queue[Optional[Task]]" = Queue()
    gpu_queue: "Queue[int]" = Queue()

    for t in RAW_TASKS:
        task_queue.put(t)
    for _ in range(num_gpus):
        task_queue.put(None)  # sentinel

    for g in gpu_ids:
        gpu_queue.put(g)

    # ── Progress tracking ─────────────────────────────────────────────────
    progress_lock = threading.Lock()
    completed = 0
    failed = 0
    total = len(RAW_TASKS)
    start_ts = time.monotonic()

    def worker(worker_idx: int):
        nonlocal completed, failed
        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                break

            gpu_id = gpu_queue.get()

            rc = run_task(
                num_frames=task.num_frames,
                compress_loop=task.mm_max_compress_loop,
                benchmark=task.benchmark,
                gpu_id=gpu_id,
                num_samples=num_samples,
                extra_args=extra_args,
            )

            with progress_lock:
                if rc == 0:
                    completed += 1
                else:
                    failed += 1
                done = completed + failed
                remaining = total - done
                elapsed = time.monotonic() - start_ts
                eta = (elapsed / done * remaining) if done > 0 else 0
                eta_str = f"{eta/60:.0f}m" if eta >= 60 else f"{eta:.0f}s"
                print(f"  📊 Progress: {done}/{total}  "
                      f"(✓{completed} ✗{failed}  remaining:{remaining}  "
                      f"ETA:{eta_str})  [{_gpu_memory_summary()}]")

            gpu_queue.put(gpu_id)
            task_queue.task_done()

    # ── Launch workers ────────────────────────────────────────────────────
    threads = []
    for i in range(num_gpus):
        t = threading.Thread(target=worker, args=(i,), name=f"worker-{i}", daemon=True)
        t.start()
        threads.append(t)

    # ── Wait for completion ───────────────────────────────────────────────
    for t in threads:
        t.join()

    total_elapsed = time.monotonic() - start_ts
    elapsed_str = (f"{total_elapsed/60:.1f}m" if total_elapsed >= 60
                   else f"{total_elapsed:.1f}s")

    print(f"\n{'='*70}")
    print(f"  All tasks complete!  ({elapsed_str})")
    print(f"    ✓ Success:  {completed}/{total}")
    print(f"    ✗ Failed:   {failed}/{total}")
    print(f"{'='*70}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

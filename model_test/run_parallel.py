#!/usr/bin/env python3
"""
Parallel Video Model Evaluation Orchestrator (per-benchmark retry)
==================================================================

Runs model × num_frames × benchmark tasks across 8 GPUs in parallel.
Tasks are sorted by num_frames descending (largest first).
When a GPU becomes free, the next task from the queue is
immediately assigned to it.

This version targets only specific (model, num_frames, benchmark)
combinations that previously failed or were incomplete.

Usage:
    # Retry failed tasks (test mode, 2 samples):
    python run_parallel_v2.py

    # Retry failed tasks (full evaluation):
    python run_parallel_v2.py --full

    # Custom:
    python run_parallel_v2.py --num-samples 10 --gpus 0,1,2,3
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
#  Task definitions  — per-benchmark retry tasks
# ═══════════════════════════════════════════════════════════════════════════════

class Task(NamedTuple):
    model: str
    num_frames: int
    benchmark: str


# Retry tasks (previously failed/incomplete) + MVBench for all models.
# Sorted below in main() by num_frames descending.
#
# Model constraints:
#   smolvlm         ≤ 64 frames
#   llava_onevision ≤ 128 frames
#   internvl        ≤ 128 frames
#   qwen35          no limit
RAW_TASKS: List[Task] = [
    # ── 1024 frames: qwen35 (retry 4 + mvbench) ──────────────────
    Task("qwen35", 1024, "mlvu"),
    Task("qwen35", 1024, "longvideobench"),
    Task("qwen35", 1024, "lvbench"),
    Task("qwen35", 1024, "egoschema"),
    Task("qwen35", 1024, "mvbench"),
    # ── 512 frames: qwen35 (mvbench) ──────────────────────────────
    Task("qwen35", 512, "mvbench"),
    # ── 256 frames: qwen35 (mvbench) ──────────────────────────────
    Task("qwen35", 256, "mvbench"),
    # ── 128 frames: qwen35, internvl, llava_onevision (mvbench) ───
    Task("qwen35", 128, "mvbench"),
    Task("internvl", 128, "mvbench"),
    Task("llava_onevision", 128, "mvbench"),
    # ── 64 frames: qwen35, internvl, llava_onevision, smolvlm ─────
    Task("qwen35", 64, "mvbench"),
    Task("internvl", 64, "mvbench"),
    Task("llava_onevision", 64, "mvbench"),
    Task("smolvlm", 64, "mvbench"),
    # ── 32 frames: qwen35, internvl, llava_onevision, smolvlm ─────
    Task("qwen35", 32, "mvbench"),
    Task("internvl", 32, "mvbench"),
    Task("llava_onevision", 32, "mvbench"),
    Task("smolvlm", 32, "mvbench"),
    # ── 16 frames: qwen35, internvl, llava_onevision, smolvlm ─────
    Task("qwen35", 16, "mvbench"),
    Task("internvl", 16, "mvbench"),
    Task("llava_onevision", 16, "mvbench"),
    Task("smolvlm", 16, "mvbench"),
    # ── 8 frames: qwen35, internvl, llava_onevision, smolvlm ──────
    Task("qwen35", 8, "mvbench"),
    Task("internvl", 8, "mvbench"),
    Task("llava_onevision", 8, "mvbench"),
    Task("smolvlm", 8, "mvbench"),
]

# ── Model display names ───────────────────────────────────────────────────────

MODEL_DISPLAY: Dict[str, str] = {
    "qwen35":          "Qwen3.5-0.8B",
    "internvl":        "InternVL3.5-1B",
    "llava_onevision": "LLaVA-OneVision",
    "smolvlm":         "SmolVLM2-500M",
}

# ── GPU config ────────────────────────────────────────────────────────────────

DEFAULT_GPU_IDS: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TEST_SCRIPT = os.path.join(SCRIPT_DIR, "model_test_v2.py")


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def tstamp() -> str:
    """Compact timestamp for log lines."""
    return datetime.now().strftime("%m-%d %H:%M:%S")


def task_label(model: str, num_frames: int, benchmark: str) -> str:
    """Human-readable task label."""
    name = MODEL_DISPLAY.get(model, model)
    return f"{name} @ {num_frames}f [{benchmark}]"


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


def _check_summary_json(model: str, num_frames: int, benchmark: str) -> int:
    """
    Validate the most recent summary JSON containing *model* for the given task.

    Checks whether the specific *benchmark* completed successfully.

    Returns 0 if the benchmark appears genuinely successful.
    Returns a magic number (≥ 99) if no valid summary was found.
    """
    output_dir = os.path.join(SCRIPT_DIR, "video_results", f"{num_frames}frames")
    if not os.path.isdir(output_dir):
        return 999  # no output directory at all → major failure

    # Find the most recent summary file that contains this model.
    pattern = os.path.join(output_dir, "summary_*.json")
    summary_files = sorted(glob.glob(pattern), reverse=True)  # newest first
    if not summary_files:
        return 998  # no summary file at all → major failure

    # Search summaries from newest to oldest for one that includes *model* and *benchmark*
    summary = None
    for sp in summary_files:
        try:
            with open(sp, "r", encoding="utf-8") as f:
                candidate = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if model in candidate.get("models", {}):
            # Also verify the benchmark exists in this summary
            mr = candidate["models"][model]
            if benchmark in mr.get("benchmarks", {}):
                summary = candidate
                break

    if summary is None:
        return 997  # model/benchmark not found in any summary → task produced no output

    mr = summary["models"][model]
    br = mr["benchmarks"][benchmark]

    status = br.get("status", "?")
    count = br.get("count", 0)
    total = br.get("dataset_total", 0)
    rate = br.get("completion_rate", None)

    if status == "error":
        return 1

    # Trust completion_rate when available (handles resume-from-full-eval case)
    if rate is not None:
        if rate < 100.0:
            return 1
        return 0

    # Fallback: check count vs total
    if total > 0 and count < total:
        return 1

    # Last resort: expected minimum sample counts per benchmark
    _expected_mins: Dict[str, int] = {
        "videomme": 300, "mlvu": 10, "longvideobench": 3,
        "lvbench": 30, "egoschema": 30, "mvbench": 30,
    }
    min_expected = _expected_mins.get(benchmark, 50)
    if count < min_expected:
        return 1

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Task runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_task(
    model: str,
    num_frames: int,
    benchmark: str,
    gpu_id: int,
    num_samples: int,
    *,
    extra_args: Optional[List[str]] = None,
) -> int:
    """
    Run ``model_test.py`` for *model* × *benchmark* with *num_frames* on *gpu_id*.

    Returns the subprocess exit code (0 = success, non-zero = failure).
    Also validates the summary JSON for false-success detection.
    """
    cmd: List[str] = [
        sys.executable, MODEL_TEST_SCRIPT,
        "--models", model,
        "--benchmarks", benchmark,
        "--num-frames", str(num_frames),
        "--gpu-ids", str(gpu_id),
        "--num-samples", str(num_samples),
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("DECORD_EOF_RETRY_MAX", "204800")

    label = task_label(model, num_frames, benchmark)
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

        # ── Validate results via summary JSON (defense-in-depth) ──────
        json_failures = _check_summary_json(model, num_frames, benchmark)

        if proc.returncode != 0 or json_failures:
            if proc.returncode != 0:
                print(f"[{tstamp()}]  GPU {gpu_id} ✗ FAIL   {label}  "
                      f"(exit={proc.returncode}, {elapsed_str})")
            else:
                print(f"[{tstamp()}]  GPU {gpu_id} ✗ FAIL   {label}  "
                      f"(exit=0 but benchmark still incomplete, {elapsed_str})")
            # Print last 30 lines of combined output for diagnostics
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
        description="Parallel video model evaluation across 8 GPUs",
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
        help="Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)",
    )
    parser.add_argument(
        "--benchmarks", type=str, default=None,
        help="Comma-separated benchmark names (passed to model_test.py)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=16,
        help="Samples per chunk (default: 16). "
             "Passed through to model_test.py --chunk-size.",
    )
    parser.add_argument(
        "--gpu-mem-util", type=float, default=None,
        help="Override GPU memory utilization (passed to model_test.py)",
    )
    args = parser.parse_args()

    # ── Resolve config ────────────────────────────────────────────────────
    num_samples = 0 if args.full else args.num_samples

    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        gpu_ids = list(DEFAULT_GPU_IDS)

    # ── Build sorted task queue (largest num_frames first) ────────────────
    tasks = sorted(RAW_TASKS, key=lambda t: t.num_frames, reverse=True)

    # ── Build extra args (no --benchmarks; each task specifies its own) ───
    extra_args: List[str] = []
    if args.chunk_size > 0:
        extra_args.extend(["--chunk-size", str(args.chunk_size)])
    if args.gpu_mem_util is not None:
        extra_args.extend(["--gpu-mem-util", str(args.gpu_mem_util)])

    # ── Print plan ────────────────────────────────────────────────────────
    num_gpus = len(gpu_ids)
    print("=" * 70)
    print("  Parallel Video Model Evaluation Orchestrator")
    print("=" * 70)
    print(f"  Tasks:           {len(tasks)}")
    print(f"  GPUs:            {gpu_ids}  ({num_gpus} workers)")
    print(f"  Samples/bench:   {num_samples} {'(FULL)' if num_samples == 0 else '(test mode)'}")
    if extra_args:
        print(f"  Extra args:      {' '.join(extra_args)}")
    print(f"  GPU memory:      {_gpu_memory_summary()}")
    print(f"\n  Task queue (largest frames first):")
    for i, t in enumerate(tasks):
        print(f"    {i+1:2d}. {task_label(t.model, t.num_frames, t.benchmark)}")
    print("=" * 70)
    print()

    # ── Queues ────────────────────────────────────────────────────────────
    task_queue: "Queue[Optional[Task]]" = Queue()
    gpu_queue: "Queue[int]" = Queue()

    for t in tasks:
        task_queue.put(t)
    # Sentinel values: one None per worker to signal shutdown
    for _ in range(num_gpus):
        task_queue.put(None)

    for g in gpu_ids:
        gpu_queue.put(g)

    # ── Progress tracking ─────────────────────────────────────────────────
    progress_lock = threading.Lock()
    completed = 0
    failed = 0
    total = len(tasks)
    start_ts = time.monotonic()

    def worker(worker_idx: int):
        nonlocal completed, failed
        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                break  # shutdown signal

            gpu_id = gpu_queue.get()  # wait for available GPU

            rc = run_task(
                model=task.model,
                num_frames=task.num_frames,
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

            gpu_queue.put(gpu_id)      # return GPU to pool
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

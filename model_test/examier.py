#!/usr/bin/env python3
"""
Video Benchmark Scoring Tool
=============================

Extracts ground-truth answers for 6 video benchmarks, parses model outputs
to extract option letters, and computes per-task accuracy.

Output: ``video_results/scores.json``

Usage:
    python examier.py                           # default: video_results
    python examier.py /path/to/video_results    # custom root
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional


# ==============================================================================
#  Ground-truth answer extraction per benchmark
# ==============================================================================

FAST_VIDEO_TEST_BASE = "/path/to/fast_video_test"

_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _load_videomme_answers() -> Dict[str, str]:
    """Video-MME: answer is already a letter (A/B/C/D)."""
    import pyarrow.parquet as pq

    path = os.path.join(FAST_VIDEO_TEST_BASE,
                        "Video-MME/videomme/test-00000-of-00001.parquet")
    table = pq.read_table(path)
    df = table.to_pandas()
    return {
        str(row["question_id"]): str(row["answer"]).strip().upper()
        for _, row in df.iterrows()
    }


def _load_mlvu_answers() -> Dict[str, str]:
    """MLVU: answer is full text; map to A/B/C/D/E by matching candidates."""
    json_dir = os.path.join(FAST_VIDEO_TEST_BASE, "MLVU/MLVU/json")
    _SKIP_MLVU = {"9_summary.json", "8_sub_scene.json"}

    answers: Dict[str, str] = {}
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json") or fname in _SKIP_MLVU:
            continue
        fpath = os.path.join(json_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            entries = json.load(f)
        sub_dir = os.path.splitext(fname)[0]
        for i, entry in enumerate(entries):
            uid = f"{entry.get('question_type', sub_dir)}_{i}"
            correct_text = str(entry.get("answer", "")).strip()
            candidates = entry.get("candidates", [])
            letter = _match_text_to_option(correct_text, candidates)
            answers[uid] = letter
    return answers


def _load_longvideobench_answers() -> Dict[str, str]:
    """LongVideoBench: correct_choice is 0-indexed integer -> map to A-E."""
    import pyarrow.parquet as pq

    path = os.path.join(FAST_VIDEO_TEST_BASE,
                        "LongVideoBench/validation-00000-of-00001.parquet")
    table = pq.read_table(path)
    df = table.to_pandas()
    answers: Dict[str, str] = {}
    for _, row in df.iterrows():
        idx = int(row["correct_choice"])
        letter = _LABELS[idx] if 0 <= idx < len(_LABELS) else str(idx)
        answers[str(row["id"])] = letter
    return answers


def _load_lvbench_answers() -> Dict[str, str]:
    """LVBench: answer is already a letter (A/B/C/D)."""
    jsonl_path = os.path.join(FAST_VIDEO_TEST_BASE,
                              "LVBench/video_info.meta.jsonl")
    _MISSING = {"28CIeC8cZks", "idZkam9zqAs", "gXnhqF0TqqI", "QgWRyDV9Ozs"}

    answers: Dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            video_entry = json.loads(line)
            if video_entry["key"] in _MISSING:
                continue
            for qa in video_entry.get("qa", []):
                uid = str(qa.get("uid", ""))
                letter = str(qa.get("answer", "")).strip().upper()
                answers[uid] = letter
    return answers


def _load_egoschema_answers() -> Dict[str, str]:
    """EgoSchema: answer is 0-indexed string integer -> map to A-E."""
    import pyarrow.parquet as pq

    path = os.path.join(FAST_VIDEO_TEST_BASE,
                        "EgoSchema/Subset/test-00000-of-00001.parquet")
    table = pq.read_table(path)
    df = table.to_pandas()
    answers: Dict[str, str] = {}
    for _, row in df.iterrows():
        idx = int(row["answer"])
        letter = _LABELS[idx] if 0 <= idx < len(_LABELS) else str(idx)
        answers[str(row["question_idx"])] = letter
    return answers


def _load_mvbench_answers() -> Dict[str, str]:
    """MVBench: answer is full text; map to A/B/C/D/E by matching candidates."""
    json_dir = os.path.join(FAST_VIDEO_TEST_BASE, "MVBench/json")

    answers: Dict[str, str] = {}
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(json_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            entries = json.load(f)
        category = os.path.splitext(fname)[0]
        for i, entry in enumerate(entries):
            uid = f"{category}_{i}"
            correct_text = str(entry.get("answer", "")).strip()
            candidates = entry.get("candidates", [])
            letter = _match_text_to_option(correct_text, candidates)
            answers[uid] = letter
    return answers


# -- Registry ------------------------------------------------------------------

_ANSWER_LOADERS: Dict[str, Callable[[], Dict[str, str]]] = {
    "videomme":        _load_videomme_answers,
    "mlvu":            _load_mlvu_answers,
    "longvideobench":  _load_longvideobench_answers,
    "lvbench":         _load_lvbench_answers,
    "egoschema":       _load_egoschema_answers,
    "mvbench":         _load_mvbench_answers,
}

_answer_cache: Dict[str, Dict[str, str]] = {}


def get_answers(benchmark: str) -> Dict[str, str]:
    """Return {sample_id: correct_option_letter} for a benchmark (cached)."""
    if benchmark not in _answer_cache:
        loader = _ANSWER_LOADERS.get(benchmark)
        if loader is None:
            print(f"  !!  Unknown benchmark: {benchmark}", file=sys.stderr)
            _answer_cache[benchmark] = {}
        else:
            _answer_cache[benchmark] = loader()
    return _answer_cache[benchmark]


# ==============================================================================
#  Answer text -> option letter matcher
# ==============================================================================

_STRIP_LABEL_RE = re.compile(
    r'^\s*(?:\(?[A-Z]\)?|[A-Z])\s*[\.\)\:]\s*', re.IGNORECASE
)


def _strip_option_label(opt_text: str) -> str:
    return _STRIP_LABEL_RE.sub("", opt_text, count=1).strip()


def _match_text_to_option(correct_text: str, candidates: List[str]) -> str:
    """Given a text answer and list of candidate options, return A/B/C/D/E."""
    correct = correct_text.strip().lower()
    if not correct:
        return "?"

    # 1) Exact match
    for i, cand in enumerate(candidates):
        if str(cand).strip().lower() == correct:
            return _LABELS[i] if i < len(_LABELS) else str(i)

    # 2) Match after stripping option labels
    for i, cand in enumerate(candidates):
        stripped = _strip_option_label(str(cand)).strip().lower()
        if stripped == correct:
            return _LABELS[i] if i < len(_LABELS) else str(i)

    # 3) Substring match
    for i, cand in enumerate(candidates):
        c = str(cand).strip().lower()
        if correct in c or c in correct:
            return _LABELS[i] if i < len(_LABELS) else str(i)

    # 4) Alphanumeric-only match
    correct_norm = re.sub(r'[^a-z0-9]', '', correct)
    for i, cand in enumerate(candidates):
        cand_norm = re.sub(r'[^a-z0-9]', '', str(cand).strip().lower())
        if cand_norm == correct_norm and correct_norm:
            return _LABELS[i] if i < len(_LABELS) else str(i)

    return "?"


# ==============================================================================
#  Robust option-letter extractor from model outputs
# ==============================================================================

def _extract_paren_letter(text: str) -> Optional[str]:
    """(A), (B), etc. -- highest precision signal."""
    m = re.search(r'(?<!\w)\(([A-E])\)', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _extract_letter_dot_start(text: str) -> Optional[str]:
    """Output that starts with 'A.' or 'B)' etc."""
    m = re.match(r'^\s*([A-E])\s*[\.\)]', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _extract_single_letter_line(text: str) -> Optional[str]:
    """A line containing ONLY a single letter A-E."""
    for line in text.strip().split('\n'):
        line = line.strip()
        m = re.match(r'^([A-E])$', line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _extract_last_standalone_letter(text: str) -> Optional[str]:
    """Last standalone A-E letter bounded by word boundaries."""
    matches = list(re.finditer(r'\b([A-E])\b', text))
    if matches:
        return matches[-1].group(1).upper()
    return None


def _extract_think_tag_letter(text: str) -> Optional[str]:
    """Handle <think>...</think> prefix, then extract first letter."""
    cleaned = re.sub(
        r'<think>.*?</think>', '', text,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()
    if not cleaned:
        return None
    for ch in ['A', 'B', 'C', 'D', 'E']:
        if cleaned.startswith(ch) and (
            len(cleaned) == 1 or cleaned[1] in ' .:)\n'
        ):
            return ch
    return None


_EXTRACTION_PATTERNS: List[Callable[[str], Optional[str]]] = [
    _extract_paren_letter,
    _extract_letter_dot_start,
    _extract_single_letter_line,
    _extract_think_tag_letter,
    _extract_last_standalone_letter,
]


def extract_option_letter(output: str) -> Optional[str]:
    """Extract the predicted option letter (A-E) from a model output string.

    Returns None if no letter can be confidently extracted.
    """
    if not output or not isinstance(output, str):
        return None

    text = output.strip()
    if not text:
        return None

    for pattern in _EXTRACTION_PATTERNS:
        result = pattern(text)
        if result is not None:
            return result

    return None


# ==============================================================================
#  Result file discovery & scoring
# ==============================================================================

def find_result_files(root_dir: str) -> List[str]:
    """Recursively find all non-summary ``*.json`` files under *root_dir*."""
    result_files: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".json") and not fn.startswith("summary") \
               and fn != "scores.json":
                result_files.append(os.path.join(dirpath, fn))
    return sorted(result_files)


def score_result_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Score a single result JSON file.

    Returns a dict with keys: num_frames, loop_count, benchmark, model,
    score, extraction_rate, or None if the file cannot be scored.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  !!  Could not read {filepath}: {e}", file=sys.stderr)
        return None

    benchmark = data.get("benchmark", "")
    model = data.get("model", data.get("model_key", "unknown"))
    num_frames = data.get("num_frames", 0)
    loop_count = data.get("mm_max_compress_loop", 0)
    results = data.get("results", [])

    if not benchmark or not results:
        return None

    gt_answers = get_answers(benchmark)
    if not gt_answers:
        print(f"  !!  No ground-truth for benchmark '{benchmark}'",
              file=sys.stderr)
        return None

    correct = 0
    matched = 0          # samples where GT exists AND letter was extracted
    extraction_failures = 0
    missing_gt = 0

    for r in results:
        sample_id = str(r.get("id", ""))
        output = r.get("output", "")

        gt_letter = gt_answers.get(sample_id)
        if gt_letter is None:
            missing_gt += 1
            continue

        pred_letter = extract_option_letter(output)
        if pred_letter is None:
            extraction_failures += 1
            continue

        matched += 1
        if pred_letter == gt_letter:
            correct += 1

    if matched == 0:
        return None

    score = correct / matched * 100.0
    extraction_rate = matched / (matched + extraction_failures) * 100.0

    return {
        "num_frames": num_frames,
        "loop_count": loop_count,
        "benchmark": benchmark,
        "model": model,
        "score": round(score, 2),
        "extraction_rate": round(extraction_rate, 2),
    }


# ==============================================================================
#  Main
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score video benchmark results -> scores.json",
    )
    parser.add_argument(
        "root_dir", nargs="?", default="fast_video_results_adaptive_f32",
        help="Root directory with benchmark result JSONs (default: fast_video_results)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON path (default: <root_dir>/scores.json)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-file scoring details",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root_dir)
    if not os.path.isdir(root):
        print(f"Error: directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or os.path.join(root, "scores.json")

    # Pre-load all ground truth answers
    print("Loading ground-truth answers ...")
    for bench_name in _ANSWER_LOADERS:
        ans = get_answers(bench_name)
        print(f"  {bench_name:<18s} {len(ans):>6d} answers")

    print(f"\nScanning: {root}")
    result_files = find_result_files(root)
    print(f"Found {len(result_files)} result file(s)\n")

    # Structure: {benchmark: {loop_count: {num_frames: {model: {score, extraction_rate}}}}}
    all_scores: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    scored = 0
    skipped = 0

    for fp in result_files:
        info = score_result_file(fp)
        if info is None:
            skipped += 1
            continue
        scored += 1

        bench = info["benchmark"]
        loop = str(info["loop_count"])
        nf = str(info["num_frames"])
        model = info["model"]

        all_scores[bench][loop][nf][model] = {
            "score": info["score"],
            "extraction_rate": info["extraction_rate"],
        }

        if args.verbose:
            print(
                f"  {bench:<18s}  loop={info['loop_count']:>3d}  "
                f"frames={info['num_frames']:>4d}  {model:<20s}  "
                f"score={info['score']:6.2f}%  "
                f"extraction={info['extraction_rate']:6.2f}%"
            )

    # Build nested output: {benchmark: {loop_count: {num_frames: {model: {score, extraction_rate}}}}}
    output_data: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
    for bench in sorted(all_scores):
        output_data[bench] = {}
        for loop in sorted(all_scores[bench], key=lambda x: int(x)):
            output_data[bench][loop] = {}
            for nf in sorted(all_scores[bench][loop], key=lambda x: int(x)):
                output_data[bench][loop][nf] = {}
                for model in sorted(all_scores[bench][loop][nf]):
                    output_data[bench][loop][nf][model] = all_scores[bench][loop][nf][model]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Scored:  {scored} result files")
    print(f"  Skipped: {skipped}")
    print(f"  Output:  {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

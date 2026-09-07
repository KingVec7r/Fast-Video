#!/usr/bin/env python3
"""
Abstract test dataset base classes for 6 video benchmarks.

Each class inherits from torch.utils.data.Dataset and provides:
- __init__: loads raw data, normalizes to unified item format {id, source_file, question}
- __len__: returns dataset size
- __getitem__: abstract — to be implemented by model-specific subclasses
- Frame sampling: decord-based utility mounted on video benchmarks

Benchmarks:
  Long Term Video (4): Video-MME, MLVU (7/9 subsets), LongVideoBench, LVBench
  Video (2): EgoSchema, MVBench

Usage:
  from abstract_test_class import VideoMMEDataset
  ds = VideoMMEDataset()          # uses default absolute paths
  print(len(ds), ds[0])           # __getitem__ must be implemented first
"""

import os
import json
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import numpy as np
from torch.utils.data import Dataset

# Base directory configuration
_BASE_DIR_DEFAULT = "/path/to/fast_video_test"

if os.path.isdir(_BASE_DIR_DEFAULT):
    FAST_VIDEO_TEST_BASE = _BASE_DIR_DEFAULT
else:
    raise RuntimeError(
        f"Base directory not found: {_BASE_DIR_DEFAULT}\n"
        "Please set FAST_VIDEO_TEST_BASE manually or ensure the data directory exists."
    )

# Decord-based frame sampling utility
os.environ.setdefault("DECORD_EFM_LOG_LEVEL", "quiet")
# Some videos have corrupt/malformed EOF markers that cause decord to retry
# endlessly.  Raise the limit so decord can work through them.
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "204800")
def _extract_frames_pyav(vid_path: str, num_frames: int) -> Optional[np.ndarray]:
    """Fallback frame extractor using PyAV when decord fails."""
    try:
        import av
        container = av.open(vid_path)
        stream = container.streams.video[0]
        total_frames = stream.frames
        if total_frames == 0:
            container.close()
            return None

        n = min(num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, n, dtype=int)
        indices_set = set(indices.tolist())

        frames_list = []
        for i, frame in enumerate(container.decode(stream)):
            if i in indices_set:
                img = frame.to_ndarray(format='rgb24')
                frames_list.append(img)
                if len(frames_list) == n:
                    break

        container.close()
        if len(frames_list) == n:
            return np.stack(frames_list).astype(np.uint8)
        return None
    except Exception:
        return None


def _repair_video(vid_path: str, overwrite: bool = False) -> Optional[str]:
    """Attempt to repair a corrupted video via ffmpeg re-encode."""
    base_path = os.path.splitext(vid_path)[0]
    final_path = f"{base_path}_repaired.mp4"

    if not overwrite and os.path.exists(final_path):
        return final_path

    # Use /tmp for temp files to avoid issues with non-existent or non-writable
    # directories (e.g. when video_path points to a wrong location)
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".mp4", prefix=".repair_", dir="/tmp"
    )
    os.close(tmp_fd)

    # Ensure the directory for the final repaired file exists
    final_dir = os.path.dirname(final_path)
    if final_dir:
        os.makedirs(final_dir, exist_ok=True)

    try:
        subprocess.run([
            'ffmpeg', '-y', '-err_detect', 'ignore_err',
            '-i', vid_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'superfast',
            '-x264-params', 'keyint=1',
            '-vsync', 'cfr',
            '-an',
            '-movflags', '+faststart',
            tmp_path
        ], check=True, capture_output=True, timeout=120)
        os.replace(tmp_path, final_path)
        return final_path
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None

def sample_frames_from_directory(
    dir_path: str,
    num_frames: int,
    return_img: bool = False,
) -> np.ndarray:
    """
    Sample uniformly-spaced frames from a directory of image files.

    The directory is expected to contain sequentially-named image files
    (e.g. ``00001.jpg``, ``00002.jpg``, ...).  Files are sorted by name
    and sampled uniformly.

    Args:
        dir_path:   Absolute path to a directory of frame images.
        num_frames: Number of frames to extract (uniformly spaced).
        return_img: If True, return list of PIL.Image instead of ndarray.

    Returns:
        np.ndarray of shape (num_frames, H, W, C), dtype uint8,
        or list[PIL.Image] if return_img=True.
    """
    from PIL import Image as PILImage

    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Frame directory not found: {dir_path}")

    # Collect image files sorted by name
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    frame_files = sorted(
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in img_exts
    )
    if not frame_files:
        raise RuntimeError(f"No image files found in directory: {dir_path}")

    total = len(frame_files)
    n = min(num_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int)

    frames_list = []
    for idx in indices:
        img_path = os.path.join(dir_path, frame_files[idx])
        img = PILImage.open(img_path).convert('RGB')
        frames_list.append(np.array(img))

    frames = np.stack(frames_list).astype(np.uint8)
    if return_img:
        return [PILImage.fromarray(f) for f in frames]
    return frames


def sample_frames_from_video(
    vid_path: str,
    num_frames: int,
    return_img: bool = False,
    reencode: bool = True,
) -> np.ndarray:
    """
    Sample uniformly-spaced frames from a video using decord.

    If ``vid_path`` is a directory, delegates to
    :func:`sample_frames_from_directory` for frame-image directories.

    Args:
        vid_path:   Absolute path to the video file, or a directory of frames.
        num_frames: Number of frames to extract (uniformly spaced).
        return_img: If True, return list of PIL.Image instead of ndarray.
        reencode:   If True, attempt ffmpeg repair on corrupted videos.

    Returns:
        np.ndarray of shape (num_frames, H, W, C), dtype uint8,
        or list[PIL.Image] if return_img=True.
    """
    # Dispatch to directory-based frame reader when vid_path is a directory
    if os.path.isdir(vid_path):
        return sample_frames_from_directory(vid_path, num_frames, return_img)

    from decord import VideoReader, cpu

    if not os.path.exists(vid_path):
        raise FileNotFoundError(f"Video file not found: {vid_path}")

    try:
        vr = VideoReader(vid_path, ctx=cpu(0))
    except Exception:
        vr = None

    if vr is None:
        # Try pyav fallback before resorting to ffmpeg re-encode
        if reencode:
            pyav_frames = _extract_frames_pyav(vid_path, num_frames)
            if pyav_frames is not None:
                if return_img:
                    from PIL import Image as PILImage
                    return [PILImage.fromarray(f) for f in pyav_frames]
                return pyav_frames.astype(np.uint8)

            repaired = _repair_video(vid_path)
            if repaired is None:
                raise RuntimeError(f"Cannot read video (decord+pyav+ffmpeg all failed): {vid_path}")
            return sample_frames_from_video(repaired, num_frames, return_img, reencode=False)
        raise RuntimeError(f"Cannot read video: {vid_path}")

    total = len(vr)
    if total == 0:
        raise RuntimeError(f"Zero frames in video: {vid_path}")

    n = min(num_frames, total)
    idx = np.linspace(0, total - 1, n, dtype=int)

    try:
        frames = vr.get_batch(idx).asnumpy()
    except Exception as e:
        if not reencode:
            raise RuntimeError(f"Frame extraction failed: {e}")
        # Try pyav first, then ffmpeg re-encode
        pyav_frames = _extract_frames_pyav(vid_path, num_frames)
        if pyav_frames is not None:
            if return_img:
                from PIL import Image as PILImage
                return [PILImage.fromarray(f) for f in pyav_frames]
            return pyav_frames.astype(np.uint8)

        repaired = _repair_video(vid_path)
        if repaired is None:
            raise RuntimeError(f"Frame extraction failed after all recovery attempts: {e}")
        return sample_frames_from_video(repaired, num_frames, return_img, reencode=False)

    if return_img:
        from PIL import Image as PILImage
        return [PILImage.fromarray(f) for f in frames]
    return frames.astype(np.uint8)


# Helper: format multiple-choice options

_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

import re as _re

# Regex to detect leading option labels like "A.", "A)", "(A)", "A )", etc.
_STRIP_LABEL_RE = _re.compile(
    r'^\s*(?:\(?[A-Z]\)?|[A-Z])\s*[\.\)\:]\s*', _re.IGNORECASE
)

def _strip_option_label(opt_text: str) -> str:
    """Remove a leading option label (e.g. 'A.', 'B)', '(C) ') if present."""
    return _re.sub(_STRIP_LABEL_RE, '', opt_text, count=1).strip()

def _has_option_label(opt_text: str) -> bool:
    """Check whether the first option already carries an A./B./C. prefix."""
    return bool(_re.match(_STRIP_LABEL_RE, opt_text.strip()))

def _format_mc_question(question: str, options: List[str]) -> str:
    """Compose a question string with labeled options.

    If options already have labels (like 'A. Apples'), the existing labels
    are stripped first to avoid 'A. A. Apples' doubling.
    """
    if not options:
        return question.strip()

    parts = [question.strip()]
    # Detect whether existing options already have consistent A-Z labels
    any_has_label = any(_has_option_label(o) for o in options)

    for i, opt in enumerate(options):
        label = _LABELS[i] if i < len(_LABELS) else f"({i + 1})"
        opt_text = str(opt).strip()
        if any_has_label:
            opt_text = _strip_option_label(opt_text)
        parts.append(f"{label}. {opt_text}")
    return "\n".join(parts)

# MVBench helper: locate video across nested subdirectories
def _build_mvbench_video_index(video_dir: str) -> Dict[str, str]:
    """Walk MVBench/video/ and map bare filename & relative paths → full path.

    Stores multiple keys per video so that JSON entries with partial relative
    paths (e.g. ``left/4504_frame52.mp4``) can still find files located in
    nested subdirectories (e.g. ``vlnqa/left/4504_frame52.mp4``).

    Also indexes frame-image directories (directories containing .jpg/.png/etc.
    but no actual video container files).  The bare directory name is used as
    the key, pointing to the directory itself.
    """
    index: Dict[str, str] = {}
    if not os.path.isdir(video_dir):
        return index

    _img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

    for root, _dirs, files in os.walk(video_dir):
        for f in files:
            if f.endswith(('.mp4', '.webm', '.mkv', '.avi', '.mov')):
                full_path = os.path.join(root, f)
                # Bare filename (first one wins)
                if f not in index:
                    index[f] = full_path
                # Full relative path from video_dir
                rel_path = os.path.relpath(full_path, video_dir)
                if rel_path not in index:
                    index[rel_path] = full_path
                # All suffix paths (e.g. for vlnqa/left/f.mp4 also store left/f.mp4)
                rel_parts = rel_path.split(os.sep)
                for i in range(1, len(rel_parts)):
                    suffix = os.sep.join(rel_parts[i:])
                    if suffix not in index:
                        index[suffix] = full_path

    # ── Index frame-image directories ───────────────────────────────────
    # Walk again looking for directories that hold frame images (no video
    # files).  The directory name itself is the clip identifier (no extension).
    for root, dirs, files in os.walk(video_dir):
        for d in dirs:
            d_full = os.path.join(root, d)
            try:
                children = os.listdir(d_full)
            except PermissionError:
                continue
            # Check if this directory contains image files (and no video files)
            has_images = any(
                os.path.splitext(c)[1].lower() in _img_exts
                for c in children
            )
            has_videos = any(
                c.endswith(('.mp4', '.webm', '.mkv', '.avi', '.mov'))
                for c in children
            )
            if has_images and not has_videos:
                # Bare directory name as key → directory path as value
                if d not in index:
                    index[d] = d_full
                # Relative path from video_dir
                rel_path = os.path.relpath(d_full, video_dir)
                if rel_path not in index:
                    index[rel_path] = d_full
    return index

# Abstract base class
class BaseTestDataset(Dataset, ABC):
    """Abstract base for all benchmark test datasets.

    Subclasses MUST override __getitem__ for model-specific inference.
    Subclasses SHOULD call super().__init__() and populate self.data.
    """

    def __init__(self, num_frames: int = 16):
        super().__init__()
        self.data: List[Dict] = []
        self._num_frames = num_frames

    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        """Return a single sample.  Model-specific subclasses implement this."""
        ...

    def sample_video_frames(
        self, vid_path: str, num_frames: Optional[int] = None
    ) -> np.ndarray:
        """Sample frames from a video, always decoding from the original or repaired video."""
        n = num_frames if num_frames is not None else self._num_frames
        return sample_frames_from_video(vid_path, n)

    def random_samples(self, k: int = 10, seed: int = 42) -> List[Dict]:
        """Return k random items for quick inspection."""
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(self.data), size=min(k, len(self.data)), replace=False)
        return [self.data[i] for i in indices]

# 1. Video-MME  (Long Term Video, parquet, multiple-choice)
class VideoMMEDataset(BaseTestDataset):
    """Video-MME benchmark.

    Parquet fields: video_id, duration, domain, sub_category, url, videoID,
                    question_id, task_type, question, options, answer
    """

    PARQUET_PATH = os.path.join(FAST_VIDEO_TEST_BASE, "Video-MME/videomme/test-00000-of-00001.parquet")
    VIDEO_DIR    = os.path.join(FAST_VIDEO_TEST_BASE, "Video-MME/data")

    def __init__(
        self,
        parquet_path: Optional[str] = None,
        video_dir: Optional[str] = None,
        num_frames: int = 16,
    ):
        super().__init__(num_frames=num_frames)
        parquet_path = parquet_path or self.PARQUET_PATH
        self.video_dir = video_dir or self.VIDEO_DIR

        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        for _, row in df.iterrows():
            options = row['options']
            if isinstance(options, np.ndarray):
                options = options.tolist()
            elif not isinstance(options, list):
                options = [options]

            question = _format_mc_question(row['question'], options)
            video_path = os.path.join(self.video_dir, f"{row['videoID']}.mp4")

            self.data.append({
                "id": str(row['question_id']),
                "source_file": video_path,
                "question": question,
                "_video_path": video_path,
            })

    def __getitem__(self, idx):
        raise NotImplementedError(
            "VideoMMEDataset.__getitem__ must be implemented by a model-specific subclass"
        )
    
# 2. MLVU  (Long Term Video, JSON, multiple-choice)
class MLVUDataset(BaseTestDataset):
    """MLVU benchmark — 9 JSON files, videos organised by question_type subdir.

    JSON fields: video, duration, question, candidates (list[4]), answer, question_type
    """

    JSON_DIR   = os.path.join(FAST_VIDEO_TEST_BASE, "MLVU/MLVU/json")
    VIDEO_DIR  = os.path.join(FAST_VIDEO_TEST_BASE, "MLVU/MLVU/video")

    def __init__(
        self,
        json_dir: Optional[str] = None,
        video_dir: Optional[str] = None,
        num_frames: int = 16,
    ):
        super().__init__(num_frames=num_frames)
        json_dir = json_dir or self.JSON_DIR
        self.video_dir = video_dir or self.VIDEO_DIR

        # Skip summary-generation and sub-scene-description subsets
        _SKIP_MLVU = {'9_summary.json', '8_sub_scene.json'}

        for fname in sorted(os.listdir(json_dir)):
            if not fname.endswith('.json'):
                continue
            if fname in _SKIP_MLVU:
                continue
            fpath = os.path.join(json_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            sub_dir = os.path.splitext(fname)[0]  # e.g. "1_plotQA"
            video_subdir = os.path.join(self.video_dir, sub_dir)

            for i, entry in enumerate(entries):
                question = _format_mc_question(
                    entry['question'], entry.get('candidates', [])
                )
                video_filename = entry['video']
                video_path = os.path.join(video_subdir, video_filename)
                uid = f"{entry.get('question_type', sub_dir)}_{i}"

                self.data.append({
                    "id": uid,
                    "source_file": video_path,
                    "question": question,
                    "_video_path": video_path,
                })

    def __getitem__(self, idx):
        raise NotImplementedError(
            "MLVUDataset.__getitem__ must be implemented by a model-specific subclass"
        )

# 3. LongVideoBench  (Long Term Video, parquet, 5-option MC)
class LongVideoBenchDataset(BaseTestDataset):
    """LongVideoBench benchmark — validation split.

    Parquet fields: video_id, id, video_path, correct_choice, question,
                    option0-option4, duration, ...
    """

    PARQUET_PATH = os.path.join(FAST_VIDEO_TEST_BASE, "LongVideoBench/validation-00000-of-00001.parquet")
    VIDEO_DIR    = os.path.join(FAST_VIDEO_TEST_BASE, "LongVideoBench/videos")

    def __init__(
        self,
        parquet_path: Optional[str] = None,
        video_dir: Optional[str] = None,
        num_frames: int = 16,
    ):
        super().__init__(num_frames=num_frames)
        parquet_path = parquet_path or self.PARQUET_PATH
        self.video_dir = video_dir or self.VIDEO_DIR

        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        for _, row in df.iterrows():
            options = [str(row.get(f'option{i}', '')) for i in range(5)]
            options = [o for o in options if o and o != 'N/A']

            question = _format_mc_question(row['question'], options)
            video_path = os.path.join(self.video_dir, str(row['video_path']))

            self.data.append({
                "id": str(row['id']),
                "source_file": video_path,
                "question": question,
                "_video_path": video_path,
            })

    def __getitem__(self, idx):
        raise NotImplementedError(
            "LongVideoBenchDataset.__getitem__ must be implemented by a model-specific subclass"
        )

# 4. LVBench  (Long Term Video, JSONL, options embedded in question)
class LVBenchDataset(BaseTestDataset):
    """LVBench benchmark — JSONL with per-video QA arrays.

    JSONL fields: key, type, qa[], video_info
    Each QA: uid, question (with (A)...(B)...(C)...(D) already embedded), answer

    Note: questions already contain (A)...(B)...(C)...(D) formatted options,
    so we do NOT re-format them.
    """

    JSONL_PATH = os.path.join(FAST_VIDEO_TEST_BASE, "LVBench/video_info.meta.jsonl")
    VIDEO_DIR  = os.path.join(FAST_VIDEO_TEST_BASE, "LVBench/videos")

    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        video_dir: Optional[str] = None,
        num_frames: int = 16,
    ):
        super().__init__(num_frames=num_frames)
        jsonl_path = jsonl_path or self.JSONL_PATH
        self.video_dir = video_dir or self.VIDEO_DIR

        # Videos known to be missing from the dataset — skip all their QAs
        _MISSING_VIDEO_KEYS = {"28CIeC8cZks", "idZkam9zqAs", "gXnhqF0TqqI", "QgWRyDV9Ozs"}

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                video_entry = json.loads(line)
                video_key = video_entry['key']
                if video_key in _MISSING_VIDEO_KEYS:
                    continue
                video_path = os.path.join(self.video_dir, f"{video_key}.mp4")

                for qa in video_entry.get('qa', []):
                    question = qa['question'].strip()
                    uid = str(qa.get('uid', ''))

                    self.data.append({
                        "id": uid,
                        "source_file": video_path,
                        "question": question,
                        "_video_path": video_path,
                    })

    def __getitem__(self, idx):
        raise NotImplementedError(
            "LVBenchDataset.__getitem__ must be implemented by a model-specific subclass"
        )

# 5. EgoSchema  (Video, parquet, 5-option MC, Subset split)
class EgoSchemaDataset(BaseTestDataset):
    """EgoSchema benchmark — egocentric video, Subset split.

    Parquet fields: question_idx, question, video_idx, option, answer
    option is a list of 5 strings like "A. ...", "B. ..."
    """

    PARQUET_PATH = os.path.join(FAST_VIDEO_TEST_BASE, "EgoSchema/Subset/test-00000-of-00001.parquet")
    VIDEO_DIR    = os.path.join(FAST_VIDEO_TEST_BASE, "EgoSchema/videos")

    def __init__(
        self,
        parquet_path: Optional[str] = None,
        video_dir: Optional[str] = None,
        num_frames: int = 16,
    ):
        super().__init__(num_frames=num_frames)
        parquet_path = parquet_path or self.PARQUET_PATH
        self.video_dir = video_dir or self.VIDEO_DIR

        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        for _, row in df.iterrows():
            options = row['option']
            if isinstance(options, np.ndarray):
                options = options.tolist()
            elif isinstance(options, str):
                import ast
                try:
                    options = ast.literal_eval(options)
                except Exception:
                    options = [options]
            elif not isinstance(options, list):
                options = [options]

            question = _format_mc_question(row['question'], options)
            video_path = os.path.join(self.video_dir, f"{row['video_idx']}.mp4")

            self.data.append({
                "id": str(row['question_idx']),
                "source_file": video_path,
                "question": question,
                "_video_path": video_path,
            })

    def __getitem__(self, idx):
        raise NotImplementedError(
            "EgoSchemaDataset.__getitem__ must be implemented by a model-specific subclass"
        )

# 6. MVBench  (Video, JSON, multiple-choice, videos in subdirs)
class MVBenchDataset(BaseTestDataset):
    """MVBench benchmark — 20 JSON task files, videos across subdirectories.

    JSON fields: video (filename), question, candidates (list), answer

    Some action_antonym videos (186/200) have corrupt VP9 bitstreams that
    cannot be decoded by decord, PyAV, or ffmpeg.  These are permanently
    unprocessable and are excluded from the dataset so that completion
    metrics are accurate.
    """

    JSON_DIR  = os.path.join(FAST_VIDEO_TEST_BASE, "MVBench/json")
    VIDEO_DIR = os.path.join(FAST_VIDEO_TEST_BASE, "MVBench/video")

    # Path to a newline-separated list of video paths (or bare filenames)
    # that are known to be permanently corrupt.  One video per line.
    # If the file doesn't exist, all videos are included (backward compatible).
    _CORRUPT_BLACKLIST_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "mvbench_corrupt_videos.txt",
    )

    @classmethod
    def _load_corrupt_blacklist(cls) -> set:
        """Load the set of video paths or filenames known to be corrupt."""
        blacklist = set()
        if os.path.exists(cls._CORRUPT_BLACKLIST_PATH):
            with open(cls._CORRUPT_BLACKLIST_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        blacklist.add(line)
                        # Also add just the basename for flexible matching
                        blacklist.add(os.path.basename(line))
        return blacklist

    def __init__(
        self,
        json_dir: Optional[str] = None,
        video_dir: Optional[str] = None,
        num_frames: int = 16,
    ):
        super().__init__(num_frames=num_frames)
        json_dir = json_dir or self.JSON_DIR
        self.video_dir = video_dir or self.VIDEO_DIR

        # Build video filename → full path index (videos are in nested subdirs)
        self._video_index = _build_mvbench_video_index(self.video_dir)

        # Load corrupt-video blacklist (permanently un-decodable videos)
        _corrupt_blacklist = self._load_corrupt_blacklist()
        _excluded_count = 0

        for fname in sorted(os.listdir(json_dir)):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(json_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                entries = json.load(f)

            category = os.path.splitext(fname)[0]  # e.g. "action_antonym"

            for i, entry in enumerate(entries):
                question = _format_mc_question(
                    entry['question'], entry.get('candidates', [])
                )
                video_filename = entry['video']
                video_path = self._video_index.get(
                    video_filename,
                    os.path.join(self.video_dir, video_filename),  # fallback
                )

                # ── Skip permanently-corrupt videos ─────────────────────
                if video_path in _corrupt_blacklist or video_filename in _corrupt_blacklist:
                    _excluded_count += 1
                    continue

                uid = f"{category}_{i}"

                self.data.append({
                    "id": uid,
                    "source_file": video_path,
                    "question": question,
                    "_video_path": video_path,
                })

        if _excluded_count > 0:
            print(f"    ⚠  MVBench: excluded {_excluded_count} permanently-corrupt "
                  f"video(s) (see {self._CORRUPT_BLACKLIST_PATH})")

    def __getitem__(self, idx):
        raise NotImplementedError(
            "MVBenchDataset.__getitem__ must be implemented by a model-specific subclass"
        )

# Registry
DATASET_REGISTRY = {
    "videomme":        VideoMMEDataset,
    "mlvu":            MLVUDataset,
    "longvideobench":  LongVideoBenchDataset,
    "lvbench":         LVBenchDataset,
    "egoschema":       EgoSchemaDataset,
    "mvbench":         MVBenchDataset,
}

# Quick test on module load
def _test_all(output_json: Optional[str] = None):
    """Instantiate each dataset and print 10 random entries for inspection.

    Args:
        output_json: If provided, also write full (untruncated) samples to this JSON file.
    """
    if output_json is None:
        output_json = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_samples.json"
        )

    all_results: Dict[str, Dict] = {}

    for name, cls in DATASET_REGISTRY.items():
        print(f"\n{'='*70}")
        print(f"  Testing: {name}  ({cls.__name__})")
        print(f"{'='*70}")
        try:
            ds = cls()
            print(f"  Total items: {len(ds)}")
            samples = ds.random_samples(k=10, seed=2026)

            json_samples = []
            for i, s in enumerate(samples):
                q_preview = s['question'].replace('\n', '\\n')
                if len(q_preview) > 120:
                    q_preview = q_preview[:120] + '...'

                # Build JSON-safe copy
                js = {"id": s["id"], "question": s["question"]}
                src_preview = s['source_file'][:100] + ('...' if len(s['source_file']) > 100 else '')
                js["source_file"] = s['source_file']

                json_samples.append(js)

                print(f"  [{i}] id={s['id']}")
                print(f"       source_file={src_preview}")
                print(f"       question={q_preview}")

            all_results[name] = {
                "class": cls.__name__,
                "total_items": len(ds),
                "samples": json_samples,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"class": cls.__name__, "error": str(e)}

    # Write JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*70}")
    print(f"  Full (untruncated) samples written to: {output_json}")

if __name__ == "__main__":
    _test_all()

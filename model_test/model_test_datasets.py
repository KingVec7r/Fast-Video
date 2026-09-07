#!/usr/bin/env python3
"""
Model-specific test dataset implementations for vLLM batch inference.

Each model has its own subclass of every benchmark dataset that implements
__getitem__ to return data in the format expected by vLLM's multimodal offline
inference (``llm.generate`` or ``llm.chat``).

Supported models (4):
  - llava_onevision : LLaVA-OneVision (Qwen2-0.5B-OV)
  - qwen35          : Qwen3.5-0.8B
  - internvl        : InternVL3.5-1B
  - smolvlm         : SmolVLM2-500M-Video-Instruct

Supported benchmarks (6):
  - videomme, mlvu, longvideobench, lvbench, egoschema, mvbench
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from abstract_test_class import (
    BaseTestDataset,
    VideoMMEDataset,
    MLVUDataset,
    LongVideoBenchDataset,
    LVBenchDataset,
    EgoSchemaDataset,
    MVBenchDataset,
    _format_mc_question,
    sample_frames_from_video,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Model prompt templates  (mixins)
# ═══════════════════════════════════════════════════════════════════════════════

class ModelPromptMixin:
    """Mixin providing model-specific prompt formatting for vLLM.

    Each subclass defines:
      - IMAGE_TOKEN : the placeholder string vLLM replaces with image embeddings
      - _format_prompt(question, num_frames) : builds a complete prompt
    """

    MODEL_TYPE: str = "base"
    IMAGE_TOKEN: str = "<image>"  # vLLM-compatible image placeholder

    def _format_prompt(self, question: str, num_frames: int) -> str:
        """Return a prompt string with *num_frames* image placeholders."""
        image_block = self.IMAGE_TOKEN * num_frames
        return f"{image_block}{question}"

    def _build_vllm_input(self, idx: int) -> Dict:
        """
        Build the complete per-sample dict for vLLM offline inference.

        Returns a dict with keys:
          - id          : str   – unique sample identifier
          - prompt      : str   – model-specific formatted prompt (vLLM-compatible)
          - video_path  : str   – absolute path to the video file
          - frames      : np.ndarray – (T, H, W, C) uint8 sampled frames
        """
        item = self.data[idx]
        question = item["question"]
        video_path = item["_video_path"]

        frames = self.sample_video_frames(video_path, self._num_frames)
        question += "\nAnswer with the option letter from the given choices directly."
        prompt = self._format_prompt(question, len(frames))

        return {
            "id": item["id"],
            "prompt": prompt,
            "video_path": video_path,
            "frames": frames,
        }

    def __getitem__(self, idx: int) -> Dict:
        return self._build_vllm_input(idx)


# ── LLaVA-OneVision (Qwen2-0.5B-OV) ────────────────────────────────────────────

class LLaVAOneVisionMixin(ModelPromptMixin):
    """
    LLaVA-OneVision prompt format.

    Uses ``<image>`` token placeholder per frame (vLLM-compatible).
    ChatML format: <|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n
    Reference: https://github.com/LLaVA-VL/LLaVA-NeXT
    """

    MODEL_TYPE = "llava_onevision"
    IMAGE_TOKEN = "<video>"

    def _format_prompt(self, question: str, num_frames: int) -> str:
        image_block = self.IMAGE_TOKEN * num_frames
        return (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{image_block}{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


# ── Qwen3.5-0.8B ──────────────────────────────────────────────────────────────

class Qwen35Mixin(ModelPromptMixin):
    """
    Qwen3.5 (Qwen2-VL style) prompt format.

    Uses ``<|vision_start|><|video_pad|><|vision_end|>`` as a single video
    placeholder block.  vLLM expands ``<|video_pad|>`` to the correct number
    of visual tokens based on grid_thw / spatial_merge_size.
    ChatML format with system + user + assistant roles.
    Reference: https://github.com/QwenLM/Qwen3.6

    Thinking mode: the official chat template pre-fills ``<think>\\n\\n</think>\\n\\n``
    after ``<|im_start|>assistant\\n`` when ``enable_thinking=False``, which
    signals the model to skip chain-of-thought reasoning and answer directly.
    We replicate this so that ``max_tokens`` is spent on the actual answer
    instead of being consumed by verbose reasoning.
    """

    MODEL_TYPE = "qwen35"
    IMAGE_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"

    def _format_prompt(self, question: str, num_frames: int) -> str:
        # Single vision block for the whole video — vLLM expands <|video_pad|>
        # to the correct number of visual tokens based on grid_thw / merge_size.
        # Pre-filled empty <think> block disables chain-of-thought reasoning,
        # matching the official chat template with enable_thinking=False.
        return (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{self.IMAGE_TOKEN}\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )


# ── InternVL3.5-1B ────────────────────────────────────────────────────────────

class InternVLMixin(ModelPromptMixin):
    """
    InternVL3.5 prompt format (video modality).

    Uses a single ``<|video_pad|>`` placeholder because InternVL's video
    processor handles all frames as one multimodal item (448² resize →
    1024 patches → pixel_shuffle → 256 tok/frame, with temporal modeling).
    The video placeholder token is determined by the text model (Qwen3).
    """

    MODEL_TYPE = "internvl"
    IMAGE_TOKEN = "<video>"  # InternVL vLLM processor hardcodes target="<video>"

    def _format_prompt(self, question: str, num_frames: int) -> str:
        # Single video placeholder for all frames (not repeated)
        return (
            f"<|im_start|>user\n{self.IMAGE_TOKEN}\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


# ── SmolVLM2-500M-Video-Instruct ──────────────────────────────────────────────

class SmolVLMMixin(ModelPromptMixin):
    """
    SmolVLM2 prompt format.

    Uses ``<image>`` token placeholder per frame.
    Format: <|im_start|>User:{images}{question}<end_of_utterance>\\nAssistant:
    Reference: https://github.com/huggingface/smollm
    """

    MODEL_TYPE = "smolvlm"
    IMAGE_TOKEN = "<image>"

    def _format_prompt(self, question: str, num_frames: int) -> str:
        image_block = self.IMAGE_TOKEN * num_frames
        return (
            f"<|im_start|>User:{image_block}{question}<end_of_utterance>\n"
            f"Assistant:"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Model-specific dataset factories  (one class per model × benchmark)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Helper: dynamically create a model-specific dataset class ──────────────────

def _make_model_dataset_class(
    base_cls: type,
    mixin_cls: type,
    class_name: str,
) -> type:
    """
    Create a new class that inherits from (mixin_cls, base_cls) so that
    mixin_cls.__getitem__ takes precedence over base_cls's abstract stub.
    """
    return type(class_name, (mixin_cls, base_cls), {})


# ── All model mixins ───────────────────────────────────────────────────────────

_MODEL_MIXINS: Dict[str, type] = {
    "llava_onevision": LLaVAOneVisionMixin,
    "qwen35": Qwen35Mixin,
    "internvl": InternVLMixin,
    "smolvlm": SmolVLMMixin,
}

# ── All benchmark base classes ─────────────────────────────────────────────────

_BENCHMARK_BASES: Dict[str, type] = {
    "videomme": VideoMMEDataset,
    "mlvu": MLVUDataset,
    "longvideobench": LongVideoBenchDataset,
    "lvbench": LVBenchDataset,
    "egoschema": EgoSchemaDataset,
    "mvbench": MVBenchDataset,
}


def get_dataset_class(
    model_type: str,
    benchmark: str,
) -> type:
    """
    Return the model-specific dataset class for *model_type* × *benchmark*.

    Args:
        model_type: one of ``llava_onevision``, ``qwen35``, ``internvl``, ``smolvlm``
        benchmark:  one of ``videomme``, ``mlvu``, ``longvideobench``, ``lvbench``,
                    ``egoschema``, ``mvbench``

    Returns:
        A Dataset **class** (not instance).  Instantiate with
        ``cls(num_frames=8)`` to get a ready-to-use vLLM-compatible dataset.
    """
    if model_type not in _MODEL_MIXINS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {list(_MODEL_MIXINS.keys())}"
        )
    if benchmark not in _BENCHMARK_BASES:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Choose from: {list(_BENCHMARK_BASES.keys())}"
        )

    mixin = _MODEL_MIXINS[model_type]
    base = _BENCHMARK_BASES[benchmark]
    class_name = f"{mixin.MODEL_TYPE.upper()}_{benchmark.upper()}"

    return _make_model_dataset_class(base, mixin, class_name)


def create_dataset(
    model_type: str,
    benchmark: str,
    num_frames: int = 8,
) -> BaseTestDataset:
    """
    Instantiate a model-specific dataset for *model_type* × *benchmark*.

    Args:
        model_type: one of ``llava_onevision``, ``qwen35``, ``internvl``, ``smolvlm``
        benchmark:  one of the 6 benchmark keys
        num_frames: number of frames to sample per video (default: 8)

    Returns:
        A ready-to-use Dataset instance whose ``__getitem__`` returns
        vLLM-compatible dicts.
    """
    cls = get_dataset_class(model_type, benchmark)
    return cls(num_frames=num_frames)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model registry  –  weights & hub paths for the 4 models
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "llava_onevision": {
        "name": "LLaVA-OneVision (Qwen2-0.5B-OV)",
        "weight_path": "/path/to/llava-onevision-qwen2-0.5b-ov",
        "hub_path": "/path/to/LLaVA-NeXT",
    },
    "qwen35": {
        "name": "Qwen3.5-0.8B",
        "weight_path": "/path/to/Qwen3.5-0.8B",
        "hub_path": "/path/to/Qwen3.6",
    },
    "internvl": {
        "name": "InternVL3.5-1B",
        "weight_path": "/path/to/InternVL3_5-1B",
        "hub_path": "/path/to/InternVL",
    },
    "smolvlm": {
        "name": "SmolVLM2-500M-Video-Instruct",
        "weight_path": "/path/to/SmolVLM2-500M-Video-Instruct",
        "hub_path": "/path/to/smollm",
    },
}

# Alias for backward compatibility
MODEL_CONFIGS = MODEL_REGISTRY

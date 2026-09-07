import os
import sys
import subprocess
import atexit
import numpy as np
import json
import random
import tempfile
from typing import Optional
from decord import VideoReader, cpu
from PIL import Image
from safetensors import safe_open
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from multiprocessing import Pool
from typing import Optional
import pickle
from fast_onevision.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, ASSISTANT_PREFIX_IDS, STOP_TOKEN_IDS, IGNORE_INDEX, IMG_PAD_ID, VID_PAD_ID
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output (including C++ logs)."""
    # Save original stderr
    original_stderr = sys.stderr
    original_stderr_fd = os.dup(2)  # Duplicate file descriptor 2 (stderr)
    
    # Open /dev/null
    devnull = os.open(os.devnull, os.O_WRONLY)
    
    try:
        # Redirect stderr to /dev/null
        os.dup2(devnull, 2)
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        # Restore
        os.dup2(original_stderr_fd, 2)
        sys.stderr = original_stderr
        os.close(devnull)
        os.close(original_stderr_fd)

logger = logging.getLogger(__name__)

def gen_qwen3_messages(
        raw_messages,
        image_token_num=None,  # int
        video_token_num=None,  # int
        plain=False
        ):

    total_img = sum(m.get("value", "").count(DEFAULT_IMAGE_TOKEN) for m in raw_messages)
    total_vid = sum(m.get("value", "").count(DEFAULT_VIDEO_TOKEN) for m in raw_messages)

    qwen3_messages = []
    
    def get_img_tag():
        n = image_token_num or 1
        return f"<|vision_start|>{'<|image_pad|>' * n}<|vision_end|>"
    
    def get_vid_tag():
        n = video_token_num or 1
        return f"<|vision_start|>{'<|video_pad|>' * n}<|vision_end|>"

    for msg in raw_messages:
        if msg["from"] not in ["human", "gpt"]:
            if msg["from"] == "Answer":
                if not qwen3_messages:
                    raise ValueError("Answer message appears before any user/assistant message")
                qwen3_messages[-1]["content"] += f"\nAnswer: {msg['value']}"
                continue
            else:
                raise ValueError(f"Unexpected 'from' value: {msg['from']}")
        assert isinstance(msg["value"], str), f"Message 'value' must be a string, got {type(msg['value'])}"

        role = "user" if msg["from"] == "human" else "assistant"
        content = msg["value"]
        
        if role == "user":
            if plain:
                if total_img > 0: content = get_img_tag()
                elif total_vid > 0: content = get_vid_tag()
            else:
                stripped = content.strip()
                is_pure_img = stripped == DEFAULT_IMAGE_TOKEN
                is_pure_vid = stripped == DEFAULT_VIDEO_TOKEN
                
                content = content.replace(DEFAULT_IMAGE_TOKEN, get_img_tag())
                content = content.replace(DEFAULT_VIDEO_TOKEN, get_vid_tag())
                
                # add prompts for pure image/video inputs
                if is_pure_img: content += " Please generate detailed descriptions of the given image."
                if is_pure_vid: content += " Please generate detailed descriptions of the given image."
        
        qwen3_messages.append({"role": role, "content": content})

    # no placeholder tokens but config specifies image/video tokens, prepend a tag to the first user message
    if total_img == 0 and total_vid == 0:
        replacement = ""
        if image_token_num is not None:
            replacement = get_img_tag()
        elif video_token_num is not None:
            replacement = get_vid_tag()
            
        if replacement:
            for msg in qwen3_messages:
                if msg["role"] == "user":
                    msg["content"] = replacement + msg["content"]
                    break

    return qwen3_messages
    
_MAX_TASKS_PER_PROCESS = float("inf")

_VIDEO_LOADER_WORKER_SCRIPT = r"""
import os as _os, sys as _sys, pickle as _pickle
_os.environ.setdefault("DECORD_EFM_LOG_LEVEL", "quiet")
from decord import VideoReader, cpu as _cpu
import numpy as _np, subprocess as _subprocess

def _repair(vid_path, overwrite=False):

    base_path = _os.path.splitext(vid_path)[0]
    final_path = f"{base_path}_repaired.mp4"

    if not overwrite and _os.path.exists(final_path):
        return final_path

    # Write to a temp file first, then atomically rename to final_path.
    # This prevents a half-written file from being left behind if ffmpeg
    # is killed mid-encoding (e.g. by the 15 s subprocess restart timeout).
    import tempfile as _tempfile
    tmp_fd, tmp_path = _tempfile.mkstemp(
        suffix=".mp4", prefix=".repair_", dir=_os.path.dirname(vid_path) or "."
    )
    _os.close(tmp_fd)

    try:
        _subprocess.run([
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
        ], check=True, capture_output=True)

        _os.replace(tmp_path, final_path)
        return final_path
    except:
        # Clean up temp file on failure; final_path is never touched.
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        return None

def _load_one(vid_path, sample_fps, min_sample, max_sample,
              return_img, reencode_flag=False):
    try:
        vr = VideoReader(vid_path, ctx=_cpu(0))
    except Exception:
        vr = None
        
    if vr is None:
        if reencode_flag:
            raise RuntimeError(f'Corrupted after re-encode: {vid_path}')
        
        repaired_path = _repair(vid_path)
        if repaired_path is None:
            raise RuntimeError(f'Re-encode failed: {vid_path}')
        
        return _load_one(repaired_path, sample_fps, min_sample, max_sample,
                         return_img, reencode_flag=True)

    total = len(vr)
    if total == 0:
        raise RuntimeError(f'Zero frames: {vid_path}')
    try:
        fps = float(vr.get_avg_fps())
        if not (0 < fps < 1e6): fps = 25.0
    except Exception:
        fps = 25.0
        
    n = max(min(int(total / fps * sample_fps), max_sample), min_sample)
    n = min(n, total)
    idx = _np.linspace(0, total - 1, n, dtype=int)
    
    try:
        frames = vr.get_batch(idx).asnumpy()
    except Exception as e:
        if reencode_flag:
            raise RuntimeError(f'Frame extraction failed after re-encode: {e}')
        
        repaired_path = _repair(vid_path)
        if repaired_path is None:
            raise RuntimeError(f'Re-encode failed: {vid_path}')
            
        return _load_one(repaired_path, sample_fps, min_sample, max_sample,
                         return_img, reencode_flag=True)

    if return_img:
        from PIL import Image as _PILImage
        return [_PILImage.fromarray(f) for f in frames]
    return frames.astype(_np.uint8)

while True:
    try:
        req = _pickle.load(_sys.stdin.buffer)
    except EOFError:
        break
    if req is None:
        break
    vid_path, kwargs = req

    try:
        result = _load_one(vid_path, **kwargs)
        _pickle.dump((True, result), _sys.stdout.buffer)
    except Exception as e:
        _pickle.dump((False, str(e)), _sys.stdout.buffer)
    _sys.stdout.buffer.flush()
"""

class _VideoLoaderSubprocess:
    """Persistent subprocess (via ``subprocess.Popen``) that loads videos.

    Restarted every *max_tasks* jobs so that any leaked FFmpeg allocations
    are reclaimed by the OS.  Communication is via pickle over stdin/stdout,
    protected by ``threading.Lock`` for thread safety.
    """

    def __init__(self, max_tasks: int = _MAX_TASKS_PER_PROCESS):
        self._max_tasks = max_tasks
        self._task_count = 0
        self._lock = __import__("threading").Lock()
        self._start()

    def _start(self):
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _VIDEO_LOADER_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._task_count = 0

    def load(self, vid_path, **kwargs):
        with self._lock:
            if self._task_count >= self._max_tasks:
                self._restart_locked()
            pickle.dump((vid_path, kwargs), self._proc.stdin)
            self._proc.stdin.flush()
            ok, result = pickle.load(self._proc.stdout)
            self._task_count += 1
        if not ok:
            raise RuntimeError(f"Subprocess video load failed: {result}")
        return result

    def _restart(self):
        with self._lock:
            self._restart_locked()

    def _restart_locked(self):
        """Must be called while holding ``self._lock``."""
        try:
            import pickle as _pickle
            _pickle.dump(None, self._proc.stdin)  # sentinel
            self._proc.stdin.flush()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)
        self._start()

    def shutdown(self):
        try:
            self._restart_locked() if self._lock.locked() else self._restart()
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

# Global singleton, lazily initialised per process
_video_loader_instance: Optional["_VideoLoaderSubprocess"] = None

def _get_video_loader() -> _VideoLoaderSubprocess:
    global _video_loader_instance
    if _video_loader_instance is None:
        _video_loader_instance = _VideoLoaderSubprocess()
    return _video_loader_instance

def load_video(
    vid_path,
    sample_fps=1,
    min_sample=1,
    max_sample=1,
    return_img=False,
):
    """Load video frames — delegates to a persistent subprocess that is
    periodically restarted to eliminate decord / FFmpeg memory leaks."""
    loader = _get_video_loader()
    return loader.load(
        vid_path,
        sample_fps=sample_fps,
        min_sample=min_sample,
        max_sample=max_sample,
        return_img=return_img,
    )

def _shutdown_video_loader():
    global _video_loader_instance
    if _video_loader_instance is not None:
        _video_loader_instance.shutdown()
        _video_loader_instance = None

atexit.register(_shutdown_video_loader)

def init_layers_from_pretrain(target_layers, pretrained_path):
    """Initialize decoder layers in-place from a pretrained LLM checkpoint.

    Loads the first ``len(target_layers)`` layers (``model.layers.0.*``,
    ``model.layers.1.*``, ...) from the pretrained weights and copies them
    into ``target_layers`` via ``load_state_dict(strict=False)``.  No value
    is returned — the weights are written directly into the model.

    Args:
        target_layers: List of ``nn.Module`` (e.g. ``nn.ModuleList``)
            representing the decoder layers to be initialized.
        pretrained_path: Path to a single ``.safetensors`` file, or a
            directory containing ``.safetensors`` checkpoint(s).

    Note:
        When ``pretrained_path`` is a directory with **sharded** checkpoints
        (e.g. ``model-00001-of-00003.safetensors``), only the first file
        returned by ``os.listdir()`` is read.  Because directory listing
        order is filesystem-dependent, the code may pick a shard that does
        **not** contain the required early-layer weights, causing all layers
        to log ``"No matching weights found"``.  For robust shard handling,
        all shards should be loaded and merged, or the shards should be
        sorted explicitly by index.
    """
    if os.path.isdir(pretrained_path):
        # Support SafeTensors
        checkpoint_files = [f for f in os.listdir(pretrained_path) 
                        if f.endswith('.safetensors')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint found in {pretrained_path}")
        checkpoint_path = os.path.join(pretrained_path, checkpoint_files[0])
    else:
        checkpoint_path = pretrained_path
    
    # Load weights
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    
    # Map and load the weights onto the query generation layer
    for layer_idx, layer in enumerate(target_layers):
        target_idx = layer_idx  # Start loading from the 0th layer.
        
        # Build the weight mapping
        layer_state_dict = {}
        prefix = f"model.layers.{target_idx}."
        
        for key, value in state_dict.items():
            if key.startswith(prefix):
                # Convert the key to the key of the current layer
                new_key = key.replace(prefix, "", 1)
                layer_state_dict[new_key] = value
        
        if layer_state_dict:
            # Load onto the current layer
            missing, unexpected = layer.load_state_dict(layer_state_dict, strict=False)
            if missing:
                logger.warning(f"Layer {layer_idx}: Missing keys: {missing}")
            if unexpected:
                logger.warning(f"Layer {layer_idx}: Unexpected keys: {unexpected}")
        else:
            logger.error(f"No matching weights found for layer {layer_idx}")

def patch_qwen_template_no_think(tokenizer):
    """
    remove <think> block
    """
    new_template = (
        "{%- if tools %}\n"
        "    {{- '<|im_start|>system\\n' }}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {{- messages[0].content + '\\n\\n' }}\n"
        "    {%- endif %}\n"
        "    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n"
        "    {%- for tool in tools %}\n"
        "        {{- \"\\n\" }}\n"
        "        {{- tool | tojson }}\n"
        "    {%- endfor %}\n"
        "    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n"
        "{%- else %}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}\n"
        "{%- endif %}\n"
        "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n"
        "{%- for message in messages[::-1] %}\n"
        "    {%- set index = (messages|length - 1) - loop.index0 %}\n"
        "    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n"
        "        {%- set ns.multi_step_tool = false %}\n"
        "        {%- set ns.last_query_index = index %}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
        "{%- for message in messages %}\n"
        "    {%- if message.content is string %}\n"
        "        {%- set content = message.content %}\n"
        "    {%- else %}\n"
        "        {%- set content = '' %}\n"
        "    {%- endif %}\n"
        "    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n"
        "    {%- elif message.role == \"assistant\" %}\n"
        "        {# Simplification: Directly output `content`, no longer processing `reasoning_content` and `think` tags. #}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
        "        {%- if message.tool_calls %}\n"
        "            {%- for tool_call in message.tool_calls %}\n"
        "                {%- if (loop.first and content) or (not loop.first) %}\n"
        "                    {{- '\\n' }}\n"
        "                {%- endif %}\n"
        "                {%- if tool_call.function %}\n"
        "                    {%- set tool_call = tool_call.function %}\n"
        "                {%- endif %}\n"
        "                {{- '<tool_call>\\n{\"name\": \"' }}\n"
        "                {{- tool_call.name }}\n"
        "                {{- '\", \"arguments\": ' }}\n"
        "                {%- if tool_call.arguments is string %}\n"
        "                    {{- tool_call.arguments }}\n"
        "                {%- else %}\n"
        "                    {{- tool_call.arguments | tojson }}\n"
        "                {%- endif %}\n"
        "                {{- '}\\n</tool_call>' }}\n"
        "            {%- endfor %}\n"
        "        {%- endif %}\n"
        "        {{- '<|im_end|>\\n' }}\n"
        "    {%- elif message.role == \"tool\" %}\n"
        "        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n"
        "            {{- '<|im_start|>user' }}\n"
        "        {%- endif %}\n"
        "        {{- '\\n<tool_response>\\n' }}\n"
        "        {{- content }}\n"
        "        {{- '\\n</tool_response>' }}\n"
        "        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n"
        "            {{- '<|im_end|>\\n' }}\n"
        "        {%- endif %}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n"
        "    {{- '<|im_start|>assistant\\n' }}\n"
        "{%- endif %}"
    )
    tokenizer.chat_template = new_template
    return tokenizer

def gen_labels_vectorized(
    input_ids: torch.Tensor, 
    assistant_prefix_ids: list = ASSISTANT_PREFIX_IDS, 
    stop_token_ids: list = STOP_TOKEN_IDS, 
    ignore_index: int = IGNORE_INDEX
):
    """
    Python loop -> unfold and cumsum
    """
    device = input_ids.device
    seq_len = input_ids.size(0)
    
    prefix_tensor = torch.tensor(assistant_prefix_ids, device=device)
    stop_tensor = torch.tensor(stop_token_ids, device=device)
    p_len = len(assistant_prefix_ids)
    s_len = len(stop_token_ids)

    # 1. Find all Prefix and Stop starting positions
    prefix_matches = (input_ids.unfold(0, p_len, 1) == prefix_tensor).all(dim=-1)
    stop_matches = (input_ids.unfold(0, s_len, 1) == stop_tensor).all(dim=-1)

    # 1. Get all start and end positions
    start_indices = torch.where(prefix_matches)[0] + p_len
    all_stop_indices = torch.where(stop_matches)[0] + s_len

    if len(start_indices) == 0:
        return torch.full_like(input_ids, ignore_index)

    # 2. Core: find the nearest Stop after each Assistant Start
    # side='left' with > start_indices ensures finding the first one "after"
    idx = torch.searchsorted(all_stop_indices, start_indices, side='left')
    
    valid_mask = idx < len(all_stop_indices)
    valid_start_indices = start_indices[valid_mask]
    valid_end_indices = all_stop_indices[idx[valid_mask]]

    # 3. Build Event Diff
    change_events = torch.zeros(seq_len + 1, device=device, dtype=torch.long)
    
    # Note: if multiple Assistant Starts map to the same Stop (anomalous data)
    # or Start order overlaps, using index_put_ is more controlled than direct +=
    change_events[valid_start_indices] = 1
    change_events[valid_end_indices] = -1 
    
    # 4. Handle truncation: if the last Assistant segment is incomplete (no corresponding Stop)
    # Logic: no Stop after the last Start
    if len(start_indices) > 0 and (len(all_stop_indices) == 0 or start_indices[-1] > all_stop_indices[-1]):
        # Set end of sequence as the stop point
        change_events[start_indices[-1]] = 1
        change_events[seq_len] = -1

    # 5. Generate mask (using cumsum)
    mask_state = torch.cumsum(change_events, dim=0)[:seq_len]
    is_assistant_content = mask_state > 0

    targets = torch.full_like(input_ids, ignore_index)
    targets[is_assistant_content] = input_ids[is_assistant_content]

    return targets

def init_projector_weights(module, checkpoint_path: Optional[str] = None):
    """
    Initialize multimodal projector weights.

    When ``checkpoint_path`` is provided (a directory or a .safetensors file),
    loads the saved projector weights from the projector pre-training stage.
    Otherwise falls back to best-practice random initialization:
      - Hidden layers: Xavier uniform (good gradient flow for GELU/ReLU nets)
      - Final layer: near-zero init, so the model starts behaving like the original LLM
    """
    logger.info(f"Initializing projector weights...")
    # ── branch 2: random initialization (no checkpoint provided) ────────
    linear_layers = [
        m for m in module.modules() 
        if isinstance(m, nn.Linear)
    ]
    n = len(linear_layers)
    
    for i, linear in enumerate(linear_layers):
        if i == n - 1:
            # Last layer: zero-init so projector output ~ 0 at start,
            # model initially behaves like pure LLM, stabilizes early training
            torch.nn.init.zeros_(linear.weight)
            if linear.bias is not None:
                torch.nn.init.zeros_(linear.bias)
        else:
            # Hidden layers: Xavier uniform
            torch.nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                torch.nn.init.zeros_(linear.bias)

def load_adapter_weights(model, checkpoint_path: Optional[str] = None):
    if checkpoint_path is None:
        return

    # ── 1. Resolve checkpoint paths ───────────────────────────────────
    if os.path.isdir(checkpoint_path):
        shard_files = sorted([f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')])
        if not shard_files:
            raise FileNotFoundError(f"No .safetensors found in: {checkpoint_path}")
        paths = [os.path.join(checkpoint_path, f) for f in shard_files]
    else:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        paths = [checkpoint_path]

    # ── 2. Filter and Load adapter weights ───────────────────────────
    ADAPTER_PREFIXES = ("model.projector.", "model.compressor.")
    adapter_state = {}
    for p in paths:
        with safe_open(p, framework="pt") as f:
            for key in f.keys():
                if key.startswith(ADAPTER_PREFIXES):
                    adapter_state[key] = f.get_tensor(key)

    if not adapter_state:
        logger.warning(f"No valid adapter weights found in {checkpoint_path}")
        return

    # ── 3. Determine components and Log loading ──────────────────────
    found_types = []
    if any(k.startswith("model.projector.") for k in adapter_state):
        found_types.append("projector")
    if any(k.startswith("model.compressor.") for k in adapter_state):
        found_types.append("compressor")
    
    type_str = " and ".join(found_types)
    logger.info(f"Loading {type_str} weights from {checkpoint_path}")

    # ── 4. Apply weights and check missing trainable parameters ──────
    all_missing_params = []
    configs = [
        ("model.projector.", model.get_projector),
        ("model.compressor.", model.get_compressor),
    ]

    for prefix, getter_fn in configs:
        sub_module = getter_fn()
        if sub_module is None:
            continue

        # Extract weights for the current module and strip the prefix
        sub_state = {k[len(prefix):]: v for k, v in adapter_state.items() if k.startswith(prefix)}
        if not sub_state:
            continue

        # Load weights
        sub_module.load_state_dict(sub_state, strict=False)

        # Check for missing trainable parameters (excluding buffers)
        missing_trainable = [
            name for name, param in sub_module.named_parameters()
            if param.requires_grad and name not in sub_state
        ]
        if missing_trainable:
            # Add prefix for clearer identification in logs
            all_missing_params.extend([f"{prefix}{n}" for n in missing_trainable])

    # ── 5. Final Warning for missing parameters ──────────────────────
    if all_missing_params:
        logger.warning(
            f"Missing parameters: {all_missing_params} when loading {type_str} weights from {checkpoint_path}"
        )

def data_debugger(tokenizer, data_item, raw_item=None):
    """
    Debug a data item: print input_ids, labels, their decoded text, and force exit the process.
    """
    input_ids = data_item.get("input_ids")
    labels = data_item.get("labels")
    
    # If Tensor, convert to list first for processing
    if isinstance(input_ids, torch.Tensor):
        input_ids_list = input_ids.detach().cpu().tolist()
    else:
        input_ids_list = input_ids

    if isinstance(labels, torch.Tensor):
        labels_list = labels.detach().cpu().tolist()
    else:
        labels_list = labels

    # 1. Decode input_ids back to text
    # skip_special_tokens=False lets us see [SEP], [CLS], <|endoftext|> etc., useful for debugging
    decoded_text = tokenizer.decode(input_ids_list, skip_special_tokens=False)

    # 2. Decode non -100 parts of labels back to text (optional, to see where loss is actually computed)
    # Typically -100 in labels represents ignored positions
    label_tokens = [t if t != -100 else tokenizer.pad_token_id for t in labels_list]
    decoded_labels = tokenizer.decode(label_tokens, skip_special_tokens=False)

    # 3. Print debug info
    logger.debug("="*5 + " DATA DEBUGGER START " + "="*5)
    logger.debug(f"Input IDs: {input_ids_list}")
    logger.debug(f"Labels (Raw): {labels_list}")
    logger.debug("-" * 7)
    logger.debug(f"Decoded Input Text:\n{decoded_text}")
    logger.debug("-" * 7)
    logger.debug(f"Decoded Labels Text (where not -100):\n{decoded_labels}")
    if raw_item is not None:
        logger.debug("-" * 7)
        logger.debug(f"Raw Data Item:\n{raw_item['conversations']}")
    
    if "image_pixel" in data_item:
        pixel_values = data_item["image_pixel"]
        shape = pixel_values.shape if hasattr(pixel_values, 'shape') else "N/A"
        logger.debug(f"Image Pixel Shape: {shape}")
        
    logger.debug("="*5 + " DATA DEBUGGER END " + "="*5)

    # 4. Force terminate the process
    # os._exit(0) ensures immediate stop in multi-process environments
    logger.info("Debugger triggered: Force exiting process...")
    sys.stdout.flush()
    os._exit(0)

def get_max_len_suggestions(dataset, output_dir, num_samples=1000, seed=42, detailed=False):
    """
    Randomly sample data from dataset and compute max-length suggestions at
    various percentiles. Only writes results on rank 0.
    """
    dataset_size = len(dataset)
    sample_size = min(num_samples, dataset_size)

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(dataset_size), sample_size)

    lengths = []
    broken_in_sample = 0
    for idx in sampled_indices:
        try:
            item = dataset[idx]
            lengths.append(len(item["input_ids"]))
        except RuntimeError as e:
            logger.warning(
                f"[get_max_len_suggestions] Skipping idx={idx} due to load error "
                f"(likely a broken video that escaped filtering): {e}"
            )
            broken_in_sample += 1
            continue

    if broken_in_sample:
        logger.warning(
            f"[get_max_len_suggestions] {broken_in_sample}/{sample_size} sampled items "
            f"failed to load. Consider re-running test_video_loads to update the broken list."
        )

    lengths = np.array(lengths)
    max_val = int(lengths.max())

    suggestions = {
        "num_samples": sample_size,
        "max_length": max_val,
        "min_length": int(lengths.min()),
        "mean_length": round(float(lengths.mean()), 2),
        "median_length": round(float(np.median(lengths)), 2),
        "percentiles": {},
    }

    # Coverage: 90%, 91%, ..., 99%
    for i in range(90, 100):
        pct = np.percentile(lengths, i)
        suggestions["percentiles"][f"{i}%"] = int(np.ceil(pct))

    # Coverage: 99.1%, 99.2%, ..., 99.9%
    for j in range(1, 10):
        pct = np.percentile(lengths, 99 + j / 10)
        suggestions["percentiles"][f"99.{j}%"] = int(np.ceil(pct))

    # detailed support
    if detailed:
        detailed_coverage = {}
        sorted_lengths = np.sort(lengths)
        for threshold in range(5, max_val + 5, 5):
            count = np.sum(sorted_lengths <= threshold)
            coverage = count / sample_size
            detailed_coverage[str(threshold)] = round(float(coverage), 4)
        
        suggestions["detailed_coverage"] = detailed_coverage

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "max_len_suggestions.json")
    with open(output_path, "w") as f:
        json.dump(suggestions, f, indent=2, ensure_ascii=False)

    logger.info(f"Max length suggestions saved to {output_path}")
    logger.info(
        f"  Max: {suggestions['max_length']}, "
        f"90%: {suggestions['percentiles']['90%']}, "
        f"95%: {suggestions['percentiles']['95%']}, "
        f"99%: {suggestions['percentiles']['99%']}"
    )
    return suggestions

def save_param_info(model, output_dir):
    parameters_info = []
    for name, param in model.named_parameters():
        parameters_info.append((name, param.size(), param.requires_grad))

    output_file = f"{output_dir}/model_parameters_info.txt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        for name, size, requires_grad in parameters_info:
            f.write(f"{name}: Size={size}, {'Trainable' if requires_grad else 'Frozen'}\n")

def verify_data(dataset, rank=0, world_size=1):

    total = len(dataset)

    local_indices = [i for i in range(total) if i % world_size == rank]
    local_total = len(local_indices)

    logger.info(
        f"[Rank {rank}/{world_size}] Verifying {local_total} / {total} samples "
        f"(shard idx % {world_size} == {rank}) ..."
    )

    pbar = tqdm(
        local_indices,
        position=rank,
        leave=True,
        desc=f"Rank {rank}",
        disable=False,
    )

    errors = []
    mismatches = []
    for i in pbar:
        try:
            data_item = dataset.__getitem__(i, test=True)
        except Exception as e:
            errors.append(f"idx={i} | {e}")
            continue

        image_token_num = data_item.get("image_token_num")
        if image_token_num is None:
            continue

        input_ids = data_item["input_ids"]
        img_pad_count = (input_ids == IMG_PAD_ID).sum().item()

        expected = image_token_num
        if img_pad_count != expected:
            mismatches.append(
                f"idx={i} | expected={expected} found={img_pad_count} input_ids_len={len(input_ids)}"
            )

    pbar.close()

    # ── Per-rank report (self-contained, no cross-rank ops) ──
    if len(errors) == 0 and len(mismatches) == 0:
        logger.info(
            f"[Rank {rank}/{world_size}] Verification OK: {local_total} samples, "
            f"0 mismatches, 0 errors."
        )
    else:
        logger.warning(
            f"[Rank {rank}/{world_size}] Verification done: {local_total} samples, "
            f"{len(mismatches)} IMG_PAD_ID mismatches, {len(errors)} errors."
        )
        if errors:
            logger.warning(f"[Rank {rank}] Errors ({len(errors)}):")
            for err_str in errors:
                logger.warning(f"[Rank {rank}]   {err_str}")
        if mismatches:
            logger.warning(f"[Rank {rank}] IMG_PAD_ID mismatches ({len(mismatches)}):")
            for detail_str in mismatches:
                logger.warning(f"[Rank {rank}]   {detail_str}")

def repair_video(vid_path, overwrite=False):

    base_path = os.path.splitext(vid_path)[0]
    final_path = f"{base_path}_repaired.mp4"

    if not overwrite and os.path.exists(final_path):
        return final_path

    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".mp4", prefix=".repair_", dir=os.path.dirname(vid_path) or "."
    )
    os.close(tmp_fd)

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
        ], check=True, capture_output=True)

        os.replace(tmp_path, final_path)
        return final_path
    except:
        # Clean up temp file on failure; final_path is never touched.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None

def load_video_fast(
        vid_path = None, 
        sample_fps = 1, 
        min_sample = 1, 
        max_sample = 1,
        return_img = False, 
        reencode_flag=False):
    try:
        vr = VideoReader(vid_path, ctx=cpu(0))
    except Exception:
        vr = None
        
    if vr is None:
        if reencode_flag:
            raise RuntimeError(f'Corrupted after re-encode: {vid_path}')
        
        repaired_path = repair_video(vid_path)
        if repaired_path is None:
            raise RuntimeError(f'Re-encode failed: {vid_path}')
        
        return load_video_fast(repaired_path, sample_fps, min_sample, max_sample,
                         return_img, reencode_flag=True)

    total = len(vr)
    if total == 0:
        raise RuntimeError(f'Zero frames: {vid_path}')
    try:
        fps = float(vr.get_avg_fps())
        if not (0 < fps < 1e6): fps = 25.0
    except Exception:
        fps = 25.0
        
    n = max(min(int(total / fps * sample_fps), max_sample), min_sample)
    n = min(n, total)
    idx = np.linspace(0, total - 1, n, dtype=int)
    
    try:
        with suppress_stderr():
            frames = vr.get_batch(idx).asnumpy()
    except Exception as e:
        if reencode_flag:
            raise RuntimeError(f'Frame extraction failed after re-encode: {e}')
        
        repaired_path = repair_video(vid_path)
        if repaired_path is None:
            raise RuntimeError(f'Re-encode failed: {vid_path}')
            
        return load_video_fast(repaired_path, sample_fps, min_sample, max_sample,
                         return_img, reencode_flag=True)

    if return_img:
        from PIL import Image as _PILImage
        return [_PILImage.fromarray(f) for f in frames]
    return frames.astype(np.uint8)
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import logging
import sys
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List, Union
from transformers import BaseImageProcessor

import math
import torch
import random
import yaml
import transformers

from fast_onevision.constants import IGNORE_INDEX, SEQ_PAD_ID, IMAGE_PATCH_NUM, IMG_PAD_ID, DEFAULT_IMAGE_TOKEN

from torch.utils.data import Dataset
from fast_onevision.train.llava_trainer import LLaVATrainer

from fast_onevision.model import FastOneVisionForCausalLM

from PIL import Image

import numpy as np
from PIL import ImageFile
from tqdm import tqdm

from fast_onevision.utils import gen_qwen3_messages, patch_qwen_template_no_think, gen_labels_vectorized, init_projector_weights, load_adapter_weights, data_debugger, get_max_len_suggestions, save_param_info, verify_data

from fast_onevision.utils import load_video_fast as load_video

ImageFile.LOAD_TRUNCATED_IMAGES = True

local_rank = None

BROKEN_VIDEO = ["y6ReUXtm_VE", "xq61UM-xAyU", "WBxWdD9YKdY", "QIkBXj_F8hk", "GqeRnxSuLFI", "bRwdpNx6bdM", "93RkWNK3BZc"]

# Use a parent logger so that all submodules (train, model.*, etc.)
# can propagate their log records to the same handlers.
LOGGER_NAME = "fast_onevision"
logger = logging.getLogger(LOGGER_NAME)

def init_training_logger(output_dir: str):
    """Initialize training logger: console + per-rank file, on ALL ranks.

    Each rank writes to its own log file (training_rank{N}.log) so there
    is no file contention.  All logging calls throughout the codebase
    already include a ``[Rank {rank}]`` prefix, so there is no output
    pollution even when all ranks log to console.
    """
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        # same format as DeepSpeed logs
        fmt="[%(asctime)s,%(msecs)03d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (all ranks)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (per-rank log file to avoid contention)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_rank{local_rank}.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"[Rank {local_rank}] Training log initialized. Log file: {log_file}")

@dataclass
class ModelArguments:

    mm_version: Optional[str] = field(default=None)
    mm_vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=None) # field(default=-2)
    mm_vision_select_feature: Optional[str] = field(default=None) # field(default="patch")
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')

    mm_max_compress_loop: Optional[int] = field(default=None)

    compress_intent_modeler_layer_num: Optional[int] = field(default=2)
    compress_intent_token_num: Optional[int] = field(default=4)
    compress_guidance_gen_layer_num: Optional[int] = field(default=2)
    compress_layer_num: Optional[int] = field(default=2)

@dataclass
class DataArguments:
    data_path: Optional[str] = field(metadata={"help": "Paths to the training data."})
    conversation_format: Optional[str] = field(default="qwen3")
    image_processor: Optional[Union[str, BaseImageProcessor]] = field(default=None)
    image_size: Optional[int] = field(default=512)
    fixed_image_token_num: Optional[int] = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-0.6B")
    adapter_checkpoint: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=512)
    group_strategy: str = field(default=None)
    output_dir: str = field(default=None)
    gradient_checkpointing: bool = field(default=True)
    lr_vision_tower: Optional[float] = None
    lr_projector: Optional[float] = None
    lr_llm: Optional[float] = None
    lr_llm_decay: Optional[float] = None
    lr_query: Optional[float] = None
    lr_compressor: Optional[float] = None
    # DataLoader prefetch factor per worker; None = auto (num_workers * 2)
    dataloader_prefetch_factor: Optional[int] = field(default=None)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self,
                 data_args: DataArguments,
                 tokenizer: transformers.PreTrainedTokenizer,
                 ):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = []

        self.image_size = data_args.image_size
        self.image_processor = data_args.image_processor
        self.fixed_image_token_num = data_args.fixed_image_token_num
        self.mm_version = data_args.mm_version
        self.mm_max_compress_loop = data_args.mm_max_compress_loop

        self.plain = False
        if data_args.conversation_format == "plain":
            self.plain = True

        self.conversation_formatter = gen_qwen3_messages
        self.video_lader = load_video
        self.labels_generator = gen_labels_vectorized

        if data_args.data_path.endswith(".yaml"):
            with open(data_args.data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets")
            for dataset in datasets:
                json_path = dataset.get("json_path")
                sampling_strategy = dataset.get("sampling_strategy", "all")
                sampling_number = None

                logger.info(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as json_file:
                        cur_data_dict = json.load(json_file)
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                # Apply the sampling strategy
                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    state = random.getstate()  # Save random state
                    random.seed(1)
                    random.shuffle(cur_data_dict)
                    random.setstate(state)  # Restore to random stat
                    cur_data_dict = cur_data_dict[:sampling_number]

                logger.info(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                self.list_data_dict.extend(cur_data_dict)

        else:
            with open(data_args.data_path, "r") as f:
                data_dict = json.load(f)
            self.list_data_dict.extend(data_dict)  # Append data from each file to the list

        # Determine fixed_image_token_num and assign per-item image_token_num
        if self.fixed_image_token_num is None:
            if self.mm_version == "projector":
                self.fixed_image_token_num = IMAGE_PATCH_NUM
            else:
                new_list_data_dict = []
                for item in self.list_data_dict:
                    video_path = item.get("video")
                    if video_path:
                        item["image_token_num"] = 32
                        # item["image_token_num"] = random.choice(list(range(8, 33))) # f32
                        # item["image_token_num"] = random.choice([8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 48, 64]) # f64
                        # item["image_token_num"] = random.choice([8, 10, 12, 14] + [i * 16 for i in range(1, 13)]) # f192
                        # item["image_token_num"] = random.choice([256])
                    elif item.get("image"):
                        # item["image_token_num"] = random.randint(1, self.mm_max_compress_loop)
                        item["image_token_num"] = random.randint(1, 16)
                    else:
                        item["image_token_num"] = None
                    new_list_data_dict.append(item)
                self.list_data_dict = new_list_data_dict
        else:
            if self.mm_version == "projector":
                self.fixed_image_token_num = IMAGE_PATCH_NUM
            else:
                new_list_data_dict = []
                for item in self.list_data_dict:
                    video_path = item.get("video")
                    if video_path:
                        assert False, "Should not have video data here"
                    elif item.get("image"):
                        item["image_token_num"] = self.fixed_image_token_num
                    else:
                        item["image_token_num"] = None
                    new_list_data_dict.append(item)
                self.list_data_dict = new_list_data_dict
        # logger.debug(f"Random 3 samples: {self.list_data_dict[66]['conversations']}, {self.list_data_dict[668]['conversations']}, {self.list_data_dict[1668]['conversations']}")
        logger.info(f"DATA SIZE: {len(self.list_data_dict)}")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if sample.get('image') or sample.get('video') or sample.get('feature'):
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    @property
    def image_token_num(self):
        result = []
        for item in self.list_data_dict:
            if item.get("image"):
                val = item.get("image_token_num")
                result.append(val if val is not None else 0)
            elif item.get("video"):
                val = item.get("image_token_num")
                result.append(-val if val is not None else 0)
            else:
                result.append(0)
        return result
            
    def __getitem__(self, i, test=False) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i] 
        images = None
        image_pixel = None

        if not test:
            if item.get("image"): # image-text data
                image_file = item['image']
                images = []
                if isinstance(image_file, str): # single image
                    image = Image.open(image_file).convert('RGB')
                    images.append(image)
                else: # image_file is list
                    for path in image_file:
                        image = Image.open(path).convert('RGB')
                        images.append(image)

            elif item.get("video"):
                image_file = item['video']
                sample_num = item["image_token_num"]
                try:
                    images = self.video_lader(image_file, min_sample= sample_num, max_sample=sample_num)
                except RuntimeError as e:
                    logger.error(
                        f"[__getitem__ idx={i}] Failed to load video '{image_file}' "
                        f"(even after re-encode): {e}.  Returning a single black frame "
                        f"as fallback to avoid crashing the whole training run."
                    )
                    # Return a single black frame as fallback to keep training alive.
                    images = [Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))]
                # image: tensor[image_num, C, H, W]

        if images is not None:
            image_pixel = self.image_processor(
                images, 
                size=self.image_size,
                crop_size=self.image_size,
                return_tensors='pt'
                )['pixel_values']
            if item.get("video") and image_pixel.shape[0] != item["image_token_num"]:
                n = image_pixel.shape[0]
                target = item["image_token_num"]
                # logger.warning(f"video {item['video']}: video frame count anomaly, need {target} frames, got {n} frames")
                indices = torch.linspace(0, n - 1, target).long().to(image_pixel.device)
                image_pixel = image_pixel[indices]

            # logger.debug(f"image_pixel.shape: {image_pixel.shape}")
            # torch.Size([1, 3, 512, 512])

        # Determine img_num and token_num_per_video for conversation formatter.
        # In test mode we derive these from raw metadata without loading media.
        if test:
            if item.get("image"):
                img_file = item["image"]
                vid_frame_num = None
            elif item.get("video"):
                # In test mode we cannot know the frame count; use image_token_num
                # as a rough estimate so the formatter can still produce vision tokens.
                vid_frame_num = item.get("image_token_num", 1)
            else:
                vid_frame_num = None
        else:
            vid_frame_num = len(images) if images is not None else None
        # raw_messages,
        # image_token_num=None,  # int
        # video_token_num=None,  # int
        # plain=False
        try:
            conversations = self.conversation_formatter(
                raw_messages = item["conversations"],
                image_token_num = self.fixed_image_token_num if self.fixed_image_token_num is not None else item.get("image_token_num"),
                video_token_num = vid_frame_num,
                plain = self.plain)
        except ValueError as e:
            logger.error(f"catch sth wrong: {e}")
            logger.info(f"item.conversations: {item["conversations"]}")
            assert False

        input_ids = self.tokenizer.apply_chat_template(
            conversations, 
            tokenize=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )[0]

        labels =  self.labels_generator(input_ids)

        data_item = {
            "input_ids": input_ids,
            "labels": labels,
            "image_pixel": image_pixel,
            "image_token_num": item.get("image_token_num"),
        }
        # data_debugger(self.tokenizer, data_item, raw_item = item)
        return data_item

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, pad_token_id=SEQ_PAD_ID):
        self.pad_token_id = pad_token_id

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:  
        input_ids, labels, images = tuple([instance[key] for instance in instances]
                                      for key in ("input_ids", "labels", "image_pixel"))
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id)
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        if images[0] is not None:
            # torch.stack adds a batch dimension: [1,C,H,W] → [B,1,C,H,W] for images
            # or [T,C,H,W] → [B,T,C,H,W] for videos, matching encode_images's 5D expectation.
            if all(img.shape == images[0].shape for img in images):
                images = torch.stack(images, dim=0)  # -> [B, 1 or T, C, H, W]
            else:
                images = torch.cat(images, dim=0)    # fallback: 4D for mismatched shapes
        else:
            images = None

        # Safely extract image_token_num from the first instance, defaulting to None
        img_compressed_len = None
        if instances and "image_token_num" in instances[0]:
            img_compressed_len = instances[0].get("image_token_num")

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id),
            images=images,
            img_compressed_len=img_compressed_len,
        )

        return batch

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    init_training_logger(training_args.output_dir)

    # ── CUDA / cuDNN / Flash Attention optimizations ──
    # These help avoid int32 indexing overflows and improve throughput when
    # processing many video frames in parallel through the vision encoder.
    if torch.cuda.is_available():
        # Let cuDNN auto-tune the best convolution algorithm (e.g. Winograd
        # instead of im2col) for the current input sizes.  This is critical
        # for large-batch depthwise conv7 in ConvFFN / ReparamLargeKernelConv.
        torch.backends.cudnn.benchmark = True
        # Allow TF32 on Ampere+ GPUs for faster matmul/conv without noticeable
        # precision loss.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Prefer Flash Attention / memory-efficient attention in SDPA backend.
        # The vision encoder's MHSA also uses flash_attn_func directly (see
        # mci.py) which avoids materialising the N×N attention matrix entirely.
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        try:
            import flash_attn  # noqa: F401
            _flash_attn_ok = True
        except ImportError:
            _flash_attn_ok = False

        logger.info(
            f"[Rank {local_rank}] CUDA optimizations enabled: "
            f"cudnn.benchmark=True, tf32=True, flash_sdp=True, "
            f"flash_attn2={_flash_attn_ok}"
        )

    model = FastOneVisionForCausalLM.from_pretrained(
        training_args.model_name_or_path,
        attn_implementation=attn_implementation,
        dtype=(torch.bfloat16 if training_args.bf16 else torch.float16)
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Config training model:

    if getattr(model.config, 'mm_vision_tower', None) is None:
        # init from original qwen3
        # Projector Pre-training or Initial Full Training
        # Copy all ModelArguments fields to model.config for vision module initialization.
        for field_name, value in vars(model_args).items():
            if value is not None:
                setattr(model.config, field_name, value)
        model.model.multimodal_init(model.config)
        model.config.mm_vision_tower = model.get_vision_tower().config
        if model_args.mm_version == "projector":
            # Projector Pre-training
            init_projector_weights(model.get_projector())
        if model_args.mm_version=="compressor": 
            # Compressor Pre-training
            load_adapter_weights(model, training_args.adapter_checkpoint)
            model.get_compressor().init_weights(training_args.model_name_or_path)
        else:
            # Initial Full Training
            load_adapter_weights(model, training_args.adapter_checkpoint)

        model.get_vision_tower().to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16)
        model.get_compressor().to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16)
        model.get_projector().to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16)

        tokenizer = patch_qwen_template_no_think(tokenizer)
    
    if model_args.mm_max_compress_loop is not None:
        model.config.mm_max_compress_loop = model_args.mm_max_compress_loop
        model.get_compressor().max_encode_loop = model_args.mm_max_compress_loop

    if model.config.mm_version != "projector" and model.config.mm_max_compress_loop is None:
        logger.error("mm_max_compress_loop is not set! Please set it to a positive integer.")

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        # Qwen3Model.forward() is decorated with @check_model_inputs, which
        # falls back to config.use_cache (default True) when use_cache=None
        # is passed. With gradient checkpointing enabled, this triggers a
        # warning and auto-sets it to False. Set it explicitly to avoid the warning.
        model.config.use_cache = False

    if model_args.mm_version=="projector": # train projector
        model.requires_grad_(False)
        model.get_projector().requires_grad_(True)   
    elif model_args.mm_version=="compressor": # train compressor
        model.requires_grad_(False)
        model.get_projector().requires_grad_(True)
        model.get_compressor().requires_grad_(True)
    elif model_args.mm_version=="vlm": # Full-scale fine-tuning
        model.requires_grad_(True)
    elif model_args.mm_version=="video": # Frozen Vision Tower
        model.requires_grad_(True)
        model.get_vision_tower().requires_grad_(False)
        # model.get_projector().requires_grad_(False)
        # model.get_compressor().compressor.requires_grad_(False)
    else:
        raise ValueError(f"unknown training recipe {model_args.mm_version}")

    if local_rank == 0:
        save_param_info(model, training_args.output_dir)

    # Set up training data:

    if data_args.fixed_image_token_num is not None and model.config.mm_max_compress_loop is not None and data_args.fixed_image_token_num > model.config.mm_max_compress_loop:
        logger.warning("fixed_image_token_num is larger than mm_max_compress_loop! it would be truncated to mm_max_compress_loop.")
        data_args.fixed_image_token_num = model.config.mm_max_compress_loop

    data_args.image_processor = model.get_vision_tower().image_processor
    data_args.mm_version = model.config.mm_version
    data_args.mm_max_compress_loop = model.config.mm_max_compress_loop

    random.seed(training_args.seed)

    train_dataset = LazySupervisedDataset(
        tokenizer = tokenizer,
        data_args = data_args)

    if local_rank == 0:
        pass
        # get_max_len_suggestions(train_dataset, training_args.output_dir, num_samples=2000, seed=6, detailed=True)
        # assert False
    # verify_data(train_dataset, rank=local_rank, world_size=training_args.world_size)

    data_collator = DataCollatorForSupervisedDataset(pad_token_id = tokenizer.pad_token_id)

    training_args.mm_version = model_args.mm_version
    trainer = LLaVATrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    logger.info("Training completed.")

if __name__ == "__main__":
    train()
import argparse
import json
import os
from typing import List, Dict, Optional
from copy import deepcopy
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image

from fast_onevision.constants import DEFAULT_IMAGE_TOKEN
from fast_onevision.conversation import conv_templates
from fast_onevision.model import FastOneVisionForCausalLM
from dataclasses import dataclass, field

from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm
import sys


@dataclass
class InferenceArguments:
    """Inference argument configuration."""
    model_path: str = field(default="")
    tokenizer_path: str = field(default="")
    device: str = field(default="cuda:0")
    image_size: int = field(default=512)
    query_path: str = field(default="/path/to/charades")
    video_path: str = field(default="/path/to/charades/videos")
    save_path: str = field(default="/path/to/charades/custom_features")
    batch_size: int = field(default=4)
    num_workers: int = field(default=8)


def get_seq_frames(total_num_frames, desired_num_frames):
    total_num_frames = int(total_num_frames)
    desired_num_frames = int(desired_num_frames)
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq


def load_video(video_path, min_frm_num=10, max_frm_num=20000):
    """Load video and return a list of PIL images."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = vr.get_avg_fps()
        # if fps <= 0: fps = 24

        # Sampling logic: sample about 0.5 frames per second (consistent with original script)
        sample_num = min(total_frame_num, total_frame_num // int(fps) if fps > 1 else total_frame_num, max_frm_num)
        sample_num = max(sample_num, min_frm_num)
        
        frame_idx = get_seq_frames(total_frame_num, sample_num)
        img_array = vr.get_batch(frame_idx).asnumpy()
        clip_imgs = [Image.fromarray(img_array[j]) for j in range(len(frame_idx))]
        return clip_imgs
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None


class VideoTextDataset(Dataset):
    def __init__(self, query_path: str, video_path: str, save_path: str, 
                 tokenizer=None, image_processor=None, image_size: Optional[int] = None):
        self.video_dir = video_path
        self.save_dir = save_path
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self.image_size = image_size

        self.samples = []
        # Read only jsonl files directly under query_path, do not recurse
        for filename in os.listdir(query_path):
            if not filename.endswith('.jsonl'):
                continue
            filepath = os.path.join(query_path, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    qid = item['qid']
                    vid = item['vid']
                    
                    video_full_path = os.path.join(self.video_dir, f"{vid}.mp4")
                    # Target feature file name format: {vid}_{qid}.pt
                    target_feature_file = os.path.join(self.save_dir, f"{vid}_{qid}.pt")
                    
                    # Skip if video file does not exist
                    if not os.path.exists(video_full_path):
                        continue

                    self.samples.append({
                        'qid': qid,
                        'vid': vid,
                        'query': item['query'],
                        'video_path': video_full_path,
                        'target_file': target_feature_file
                    })
        random.seed(42)
        random.shuffle(self.samples)
        # Keep only even-indexed samples (qid filtering as in original logic)
        self.samples = [self.samples[i] for i in range(len(self.samples)) if not i % 2]
        # Skip already processed samples
        self.samples = [data for data in self.samples if not os.path.exists(data["target_file"])]
        
        print(f"Total valid samples to process: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _get_conversation(self, query: str) -> str:
        messages = [{"from": "human", "value": f"DEFAULT_IMAGE_TOKEN\n{query}"}]
        for message in messages:
            if DEFAULT_IMAGE_TOKEN not in message['value']:
                message['value'] = f"{DEFAULT_IMAGE_TOKEN}\n{message['value']}"
            message['value'] = message['value'].replace(
                DEFAULT_IMAGE_TOKEN, '<|vision_start|><|vision_pad|><|vision_end|>'
            ).strip()

        conv = conv_templates["qwen3"].copy()
        conv.messages = messages
        return conv.get_prompt(generate=True)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video frames
        pil_images = load_video(sample['video_path'])
        if pil_images is None: return None

        # Process images
        if self._image_processor is not None:
            image_processor = deepcopy(self._image_processor)
            pixel_values = image_processor(
                pil_images,
                size=self.image_size,
                crop_size=self.image_size,
                return_tensors='pt'
            )['pixel_values']
        else:
            pixel_values = None

        # Process text
        if self._tokenizer is not None:
            tokenizer = deepcopy(self._tokenizer)
            conversation = self._get_conversation(sample['query'])
            tokenized = tokenizer(conversation, padding=False, truncation=True, return_tensors='pt')
            input_ids = tokenized['input_ids'].squeeze(0)
            attention_mask = tokenized['attention_mask'].squeeze(0)
        else:
            input_ids, attention_mask = None, None

        return {
            'qid': sample['qid'],
            'vid': sample['vid'],
            'target_file': sample['target_file'],
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def collate_fn(batch: List[Dict]) -> Dict:
    batch = [item for item in batch if item is not None]
    if not batch: return {}

    qids = [item['qid'] for item in batch]
    vids = [item['vid'] for item in batch]
    target_files = [item['target_file'] for item in batch]
    video_frames = [item['pixel_values'] for item in batch]
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    
    max_seq_len = max(ids.shape[0] for ids in input_ids_list)
    padded_input_ids = torch.full((len(batch), max_seq_len), 0, dtype=torch.long)
    padded_attention_mask = torch.zeros((len(batch), max_seq_len), dtype=torch.long)
    
    for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        seq_len = ids.shape[0]
        padded_input_ids[i, :seq_len] = ids
        padded_attention_mask[i, :seq_len] = mask

    return {
        'qids': qids,
        'vids': vids,
        'target_files': target_files,
        'video_frames': video_frames,
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask
    }


@torch.inference_mode()
def run_batch_inference(model, batch, device):
    input_ids = batch['input_ids'].to(device)
    video_frames_list = batch['video_frames']
    
    # Visual encoding
    image_embeddings = []
    for frames in video_frames_list:
        frames = frames.to(dtype=model.dtype, device=device)
        emb = model.encode_images([frames])
        image_embeddings.append(emb[0])

    # Text embeddings (for compressor)
    text_embeddings = model.get_input_embeddings()(input_ids)
    pad_embedding = model.get_input_embeddings().weight[model.config.pad_token_id].to(device=device, dtype=text_embeddings.dtype)

    # Generate compressed features (1 token per frame)
    compress_queries = model.get_compressor().query_generator(text_embeddings, input_ids, pad_embedding)
    compressed_embeddings_list = model.get_compressor().image_compressor(compress_queries, image_embeddings)
    
    return compressed_embeddings_list


def save_results(compressed_embeddings_list, batch):
    for i in range(len(compressed_embeddings_list)):
        target_file = batch['target_files'][i]
        # compressed_embeddings_list[i] shape: [num_frames, 1, hidden_size]
        # Squeeze to [num_frames, hidden_size]
        video_feat = compressed_embeddings_list[i].squeeze(1).detach().cpu().to(torch.float16)
        
        # Save as .pt file
        torch.save(video_feat, target_file)


# Configuration parameters (use relative placeholder paths)
QWEN_3 = "Qwen/Qwen3-0.6B"
CHECK_POINT = "checkpoint_ov"
IMAGE_SIZE = 512
DEVICE = "cuda:0"  # Adjust according to available GPU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=CHECK_POINT)
    parser.add_argument("--tokenizer_path", type=str, default=QWEN_3)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--query_path", type=str, default="/path/to/charades")
    parser.add_argument("--video_path", type=str, default="/path/to/charades/Charades_v1")
    parser.add_argument("--save_path", type=str, default="/path/to/charades/custom_features")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False, trust_remote_code=True)
    model = FastOneVisionForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", trust_remote_code=True
    ).eval().to(args.device)
    
    image_processor = model.get_vision_tower().image_processor
    
    dataset = VideoTextDataset(
        query_path=args.query_path,
        video_path=args.video_path,
        save_path=args.save_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_size=IMAGE_SIZE
    )
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    for batch in tqdm(dataloader, desc="Processing Charades"):
        if not batch: continue
        try:
            feats = run_batch_inference(model, batch, args.device)
            save_results(feats, batch)
        except Exception as e:
            print(f"Error in batch {batch.get('qids', 'unknown')}: {e}")
            continue

    print("Task finished!")


if __name__ == "__main__":
    main()
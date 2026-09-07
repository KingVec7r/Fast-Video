# Generate text and video features for moment_detr training (batch inference version, parallel data loading)

import argparse
import json
import os
from typing import List, Dict, Optional
from copy import deepcopy

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
    model_path: str = field(default="Qwen/Qwen3-0.6B")
    tokenizer_path: str = field(default="Qwen/Qwen3-0.6B")
    device: str = field(default="cuda:0")
    image_size: int = field(default=None)
    query_path: str = field(default=None)
    video_path: str = field(default=None)
    save_path: str = field(default=None)
    batch_size: int = field(default=4)
    num_workers: int = field(default=4)


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
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()

    sample_num = min(total_frame_num, total_frame_num // fps // 2, max_frm_num)
    sample_num = max(sample_num, min_frm_num)
    sample_num = int(sample_num)

    frame_idx = get_seq_frames(total_frame_num, sample_num)
    img_array = vr.get_batch(frame_idx).asnumpy()
    clip_imgs = [Image.fromarray(img_array[j]) for j in range(sample_num)]
    return clip_imgs


class VideoTextDataset(Dataset):
    """
    Video-Text Dataset - processes video frames in __getitem__ in parallel.
    Returns: {
        'qid': str,
        'vid': str,
        'query': str,
        'video_path': str,
        'text_feature_file': str,
        'video_feature_file': str,
        'pixel_values': torch.Tensor,  # [num_frames, C, H, W]
        'input_ids': torch.Tensor,     # [seq_len]
        'attention_mask': torch.Tensor # [seq_len]
    }
    """
    def __init__(self, query_path: str, video_path: str, save_path: str, 
                 text_feature_path: str, video_feature_path: str,
                 tokenizer=None, image_processor=None, image_size: Optional[int] = None):
        self.video_path = video_path
        self.text_feature_path = text_feature_path
        self.video_feature_path = video_feature_path
        
        # Save configuration but not processor/tokenizer instances directly (multi-process safe).
        # In __getitem__, independent copies are created via deepcopy.
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self.image_size = image_size

        # Collect all samples to process
        self.samples = []
        for filename in os.listdir(query_path):
            if not filename.endswith('.jsonl'):
                continue
            filepath = os.path.join(query_path, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    qid = item['qid']
                   
                    try:
                        if isinstance(qid, str):
                            if int(qid[-1]) % 2:
                                continue  # Keep only even qids
                        else:
                            if qid % 2:
                                continue  # Keep only even qids
                    except Exception as e:
                        print(e)
                        print(item)
                        print(filepath)
                        sys.exit(1)

                    vid = item['vid']

                    text_feature_file = os.path.join(text_feature_path, f"qid{qid}.npz")
                    video_feature_file = os.path.join(video_feature_path, f"{vid}_{qid}.npz")
                    video_full_path = os.path.join(video_path, f"{vid}.mp4")

                    # Skip already existing features
                    if os.path.exists(text_feature_file) and os.path.exists(video_feature_file):
                        continue
                    # Skip missing videos
                    if not os.path.exists(video_full_path):
                        continue

                    self.samples.append({
                        'qid': qid,
                        'vid': vid,
                        'query': item['query'],
                        'video_path': video_full_path,
                        'text_feature_file': text_feature_file,
                        'video_feature_file': video_feature_file
                    })

        print(f"Total samples to process: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _get_conversation(self, query: str) -> str:
        """Construct the conversation text."""
        messages = [
            {
                "from": "human",
                "value": f"DEFAULT_IMAGE_TOKEN\n{query}"
            }
        ]
        # Preprocess multimodal messages
        for message in messages:
            if DEFAULT_IMAGE_TOKEN not in message['value']:
                message['value'] = f"{DEFAULT_IMAGE_TOKEN}\n{message['value']}"
            message['value'] = message['value'].replace(
                DEFAULT_IMAGE_TOKEN,
                '<|vision_start|><|vision_pad|><|vision_end|>'
            ).strip()

        conv = conv_templates["qwen3"].copy()
        conv.messages = messages
        conversation = conv.get_prompt(generate=True)
        return conversation

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # In multi-process environment, create independent processor and tokenizer copies for each worker.
        # This is crucial to avoid state contamination from sharing across processes.
        if self._image_processor is not None:
            image_processor = deepcopy(self._image_processor)
        else:
            image_processor = None
            
        if self._tokenizer is not None:
            tokenizer = deepcopy(self._tokenizer)
        else:
            tokenizer = None

        # 1. Load video frames
        pil_images = load_video(sample['video_path'])
        
        # 2. Process video frames -> Tensor
        if image_processor is not None:
            if self.image_size is not None:
                pixel_values = image_processor(
                    pil_images,
                    size=self.image_size,
                    crop_size=self.image_size,
                    return_tensors='pt'
                )['pixel_values']  # [num_frames, C, H, W]
            else:
                pixel_values = image_processor(
                    pil_images,
                    return_tensors='pt'
                )['pixel_values']  # [num_frames, C, H, W]
        else:
            # If no processor, return raw PIL images (handled in collate_fn)
            pixel_values = pil_images

        # 3. Process text -> Tokenize
        if tokenizer is not None:
            conversation = self._get_conversation(sample['query'])
            tokenized = tokenizer(
                conversation,
                padding=False,  # Do not pad here; done uniformly in collate_fn
                truncation=True,
                return_tensors='pt'
            )
            input_ids = tokenized['input_ids'].squeeze(0)  # [seq_len]
            attention_mask = tokenized['attention_mask'].squeeze(0)  # [seq_len]
        else:
            input_ids = None
            attention_mask = None

        return {
            'qid': sample['qid'],
            'vid': sample['vid'],
            'query': sample['query'],
            'video_path': sample['video_path'],
            'text_feature_file': sample['text_feature_file'],
            'video_feature_file': sample['video_feature_file'],
            'pixel_values': pixel_values,  # [num_frames, C, H, W]
            'input_ids': input_ids,        # [seq_len]
            'attention_mask': attention_mask # [seq_len]
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function - only handles batch assembly and padding.
    Assumes pixel_values in each item are already tensors.
    """
    qids = [item['qid'] for item in batch]
    vids = [item['vid'] for item in batch]
    queries = [item['query'] for item in batch]
    video_paths = [item['video_path'] for item in batch]
    text_feature_files = [item['text_feature_file'] for item in batch]
    video_feature_files = [item['video_feature_file'] for item in batch]

    # 1. Video frames are already tensors, just collect them
    video_frames = [item['pixel_values'] for item in batch]  # List[Tensor], each [num_frames, C, H, W]

    # 2. Handle text padding
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    
    # Manual padding to the maximum sequence length in the batch
    max_seq_len = max(ids.shape[0] for ids in input_ids_list)
    batch_size = len(batch)
    
    # Obtain pad_token_id (from the tokenizer of the first input_ids, or use a default)
    # Here we default to 0; the actual value will be handled in run_batch_inference or main.
    pad_token_id = 0  # default, adjusted in run_batch_inference
    
    # Create padded tensors
    padded_input_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    padded_attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    
    for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        seq_len = ids.shape[0]
        padded_input_ids[i, :seq_len] = ids
        padded_attention_mask[i, :seq_len] = mask

    return {
        'qids': qids,
        'vids': vids,
        'queries': queries,
        'video_frames': video_frames,  # List[Tensor], each [num_frames, C, H, W]
        'input_ids': padded_input_ids,          # [batch_size, max_seq_len]
        'attention_mask': padded_attention_mask, # [batch_size, max_seq_len]
        'text_feature_files': text_feature_files,
        'video_feature_files': video_feature_files
    }


@torch.inference_mode()
def run_batch_inference(
    model: FastOneVisionForCausalLM,
    tokenizer: AutoTokenizer,
    batch: Dict,
    inference_args: InferenceArguments
):
    """
    Execute batch inference.
    """
    device = inference_args.device

    # Move input_ids to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # Process video frames (each video may have different number of frames, encode separately)
    video_frames_list = batch['video_frames']  # List[Tensor]
    batch_size = len(video_frames_list)

    # Encode each video individually for visual features
    image_embeddings = []
    for frames in video_frames_list:
        frames = frames.to(
            dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16,
            device=device
        )
        # encode_images expects List[Tensor] input
        emb = model.encode_images([frames])  # returns List[Tensor], take first
        image_embeddings.append(emb[0])

    # Obtain text embeddings
    text_embeddings = model.get_input_embeddings()(input_ids)  # [batch_size, seq_len, hidden_size]

    # Obtain pad embedding
    pad_embedding = model.get_input_embeddings().weight[model.config.pad_token_id]
    pad_embedding = pad_embedding.to(device=device, dtype=text_embeddings.dtype)

    # Ensure all image embeddings are on the same device
    for i, emb in enumerate(image_embeddings):
        image_embeddings[i] = emb.to(device=device, dtype=text_embeddings.dtype)

    assert model.config.mm_version != "projector"

    # Generate compress queries in batch
    compress_queries = model.get_compressor().query_generator(
        text_embeddings,
        input_ids,
        pad_embedding
    )  # [batch_size, query_num, hidden_size]

    # Perform image compression in batch
    compressed_embeddings_list = model.get_compressor().image_compressor(
        compress_queries, 
        image_embeddings 
    )
    
    return compressed_embeddings_list, input_ids, text_embeddings


def save_batch_results(
    compressed_embeddings_list: List[torch.Tensor],
    input_ids: torch.Tensor,
    text_embeddings: torch.Tensor,
    batch: Dict
):
    """
    Save batch inference results.
    """
    batch_size = len(compressed_embeddings_list)
    
    for i in range(batch_size):
        qid = batch['qids'][i]
        text_feature_file = batch['text_feature_files'][i]
        video_feature_file = batch['video_feature_files'][i]
        
        # 1. Process video features
        # compressed_embeddings_list[i] shape: [num_frames, 1, hidden_size] -> [num_frames, hidden_size]
        img_feat = compressed_embeddings_list[i].squeeze(1)
        img_feat_np = img_feat.detach().cpu().to(torch.float32).numpy().astype(np.float16)
        
        # 2. Process text features
        # text_embeddings[i] shape: [seq_len, hidden_size]
        last_hidden_state = text_embeddings[i]  # [seq_len, hidden_size]
        
        # Find actual length (based on attention_mask)
        seq_len = batch['attention_mask'][i].sum().item()
        last_hidden_state = last_hidden_state[:seq_len]  # truncate to actual length
        
        # Remove the first token and last 5 tokens (following original logic)
        if last_hidden_state.shape[0] > 6:
            last_hidden_state = last_hidden_state[1:-5]
        
        last_hidden_state_np = last_hidden_state.detach().cpu().to(torch.float32).numpy().astype(np.float16)
        
        # pooler_output: average pooling
        pooler_output = last_hidden_state.mean(dim=0)
        pooler_output_np = pooler_output.detach().cpu().to(torch.float32).numpy().astype(np.float16)
        
        # 3. Save features
        np.savez(video_feature_file, features=img_feat_np)
        np.savez(text_feature_file, 
                last_hidden_state=last_hidden_state_np, 
                pooler_output=pooler_output_np)


QWEN_3 = "Qwen/Qwen3-0.6B"
CHECK_POINT = "checkpoint_ov"
IMAGE_SIZE = 512
DEVICE = "cuda:0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=CHECK_POINT)
    parser.add_argument("--tokenizer_path", type=str, default=QWEN_3)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--query_path", type=str, default="/path/to/moment_detr/data")
    parser.add_argument("--video_path", type=str, default="/path/to/qvhl/videos")
    parser.add_argument("--save_path", type=str, default="/path/to/features")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of data loading threads")
    
    args = parser.parse_args()
    
    inference_args = InferenceArguments(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        image_size=args.image_size,
        query_path=args.query_path,
        video_path=args.video_path,
        save_path=args.save_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create save directories
    text_feature_path = os.path.join(args.save_path, "clip_text_features")
    video_feature_path = os.path.join(args.save_path, "clip_features")
    os.makedirs(text_feature_path, exist_ok=True)
    os.makedirs(video_feature_path, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model from {inference_args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        inference_args.tokenizer_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side='right'  # Ensure padding on the right
    )
    
    model = FastOneVisionForCausalLM.from_pretrained(
        inference_args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).eval()
    model.to(inference_args.device)
    
    image_processor = model.get_vision_tower().image_processor
    
    # Create Dataset - pass processor and tokenizer
    # These will be deep-copied in each worker process
    dataset = VideoTextDataset(
        query_path=args.query_path,
        video_path=args.video_path,
        save_path=args.save_path,
        text_feature_path=text_feature_path,
        video_feature_path=video_feature_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_size=inference_args.image_size
    )
    
    # Create DataLoader - use multi-process data loading
    # collate_fn now requires no extra arguments because processing is already done in Dataset
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        # Optionally set multiprocessing start method
        multiprocessing_context='spawn' if args.num_workers > 0 else None
    )
    
    # Batch inference
    print(f"Starting batch inference with batch_size={args.batch_size}, num_workers={args.num_workers}...")
    for batch in tqdm(dataloader, desc="Processing batches"):
        try:
            # Execute batch inference
            compressed_embeddings_list, input_ids, text_embeddings = run_batch_inference(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                inference_args=inference_args
            )
            
            # Save results
            save_batch_results(
                compressed_embeddings_list=compressed_embeddings_list,
                input_ids=input_ids,
                text_embeddings=text_embeddings,
                batch=batch
            )
            
        except Exception as e:
            print(f"Error processing batch with qids {batch['qids']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("Batch inference completed!")


if __name__ == "__main__":
    main()
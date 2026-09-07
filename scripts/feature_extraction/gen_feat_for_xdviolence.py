import os
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoTokenizer
import random
import time

from fast_onevision.constants import DEFAULT_IMAGE_TOKEN
from fast_onevision.conversation import conv_templates
from fast_onevision.model import FastOneVisionForCausalLM

# ================= Configuration =================
ANNO_PATH = "/path/to/XD-Violence/annos.json"
VIDEO_ROOT = "/path/to/XD-Violence/videos/train"
QWEN_3 = "Qwen/Qwen3-0.6B"
CHECK_POINT = "checkpoint_ov"
DEVICE = "cuda:0" 
IMAGE_SIZE = 512
VISION_CHUNK_SIZE = 256  # Critical: number of frames per chunk to prevent index overflow and memory crash

SAVE_DIRS = {
    "common": "/path/to/XD-Violence/FAST_features_common/features",
    "violence": "/path/to/XD-Violence/FAST_features_violence/features",
    "specific": "/path/to/XD-Violence/FAST_features_specific/features"
}
# ==================================================

def apply_crop(frames: np.ndarray, crop_type: int) -> np.ndarray:
    """Apply crop and horizontal flip based on crop_type."""
    resized = [cv2.resize(f, (777, 585)) for f in frames]
    img = np.array(resized)
    if crop_type % 5 == 0: img = img[:, 36:548, 132:644, :]
    elif crop_type % 5 == 1: img = img[:, :512, :512, :]
    elif crop_type % 5 == 2: img = img[:, :512, -512:, :]
    elif crop_type % 5 == 3: img = img[:, -512:, :512, :]
    elif crop_type % 5 == 4: img = img[:, -512:, -512:, :]
    if crop_type >= 5: img = img[:, :, ::-1, :]
    return img

class FeatureExtractor:
    def __init__(self):
        print(f"Loading model from {CHECK_POINT}...")
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_3, trust_remote_code=True)
        self.model = FastOneVisionForCausalLM.from_pretrained(
            CHECK_POINT,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).eval().to(DEVICE)
        self.image_processor = self.model.get_vision_tower().image_processor

    def get_prompt(self, query: str) -> str:
        """Build conversation prompt."""
        content = f"{DEFAULT_IMAGE_TOKEN}\n{query}"
        content = content.replace(DEFAULT_IMAGE_TOKEN, '<|vision_start|><|vision_pad|><|vision_end|>')
        conv = conv_templates["qwen3"].copy()
        conv.messages = [{"from": "human", "value": content}]
        return conv.get_prompt(generate=True)

    @torch.inference_mode()
    def extract(self, pixel_values: torch.Tensor, query_text: str):
        """Extract compressed video features for a given query."""
        # 1. Process text
        prompt = self.get_prompt(query_text)
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
        text_embeddings = self.model.get_input_embeddings()(input_ids)
        pad_embedding = self.model.get_input_embeddings().weight[self.model.config.pad_token_id].to(text_embeddings.dtype)

        # 2. Chunked visual encoding (prevents 32-bit index error)
        all_image_features = []
        num_frames = pixel_values.shape[0]
        
        for i in range(0, num_frames, VISION_CHUNK_SIZE):
            chunk = pixel_values[i : i + VISION_CHUNK_SIZE].to(DEVICE, dtype=torch.bfloat16)
            # encode_images returns List[Tensor], take features of the first video
            chunk_features = self.model.encode_images([chunk])[0]
            all_image_features.append(chunk_features)
            del chunk  # free memory immediately
        
        # Concatenate visual features of all frames [N, L, C]
        image_embeddings_full = torch.cat(all_image_features, dim=0)
        
        # 3. Compression (Query-Aware Compression)
        # Compressor expects input as List[Tensor]
        compress_queries = self.model.get_compressor().query_generator(
            text_embeddings, input_ids, pad_embedding
        )
        
        compressed_embeddings = self.model.get_compressor().image_compressor(
            compress_queries, [image_embeddings_full] 
        )
        
        feat = compressed_embeddings[0].squeeze(1).cpu().to(torch.float32).numpy().astype(np.float16)
        return feat

def main(global_rank=10, local_rank=0, task_id=None):
    global DEVICE
    if local_rank % 2 == 0:
        DEVICE = "cuda:2"
    else:
        DEVICE = "cuda:3"
    # DEVICE = f"cuda:{local_rank % 4}"
    
    # Ensure output directories exist
    for path in SAVE_DIRS.values():
        os.makedirs(path, exist_ok=True)
    
    # Initialize model after DEVICE is assigned
    extractor = FeatureExtractor()

    with open(ANNO_PATH, 'r') as f:
        annos = json.load(f)
    
    # Data sharding
    todo_file = f"fast_onevision/eval/single_image/task_{task_id}.json"
    if local_rank == 0:
        todo_id = []
        for i in range(len(annos)):
            filename = annos[i]["filename"]
            all_exist = True
            for q_key in SAVE_DIRS.keys():
                for ct in range(10):
                    if not os.path.exists(os.path.join(SAVE_DIRS[q_key], f"{filename}__{ct}.npy")):
                        all_exist = False
                        break
                if not all_exist:
                    break
            if not all_exist:
                todo_id.append(i)
        tmp_file = todo_file + ".tmp"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(todo_id, f, ensure_ascii=False, indent=2)
        os.rename(tmp_file, todo_file)
        print(f"[Rank {local_rank}] TODO file created.")
    else:
        print(f"[Rank {local_rank}] Waiting for {todo_file}...")
        while not os.path.exists(todo_file):
            time.sleep(0.5)  # check every 0.5 seconds
        print(f"[Rank {local_rank}] File detected, loading...")
        with open(todo_file, 'r', encoding='utf-8') as f:
            todo_id = json.load(f)
    annos = [annos[i] for i in todo_id]
    random.seed(55)
    random.shuffle(annos)  # optional: shuffle to balance work across ranks

    annos = [annos[i] for i in range(len(annos)) if i % global_rank == local_rank]

    for item in tqdm(annos, desc="Processing Videos"):
        filename = item["filename"]
        video_path = os.path.join(VIDEO_ROOT, f"{filename}.mp4")

        if not os.path.exists(video_path):
            continue

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            frame_indices = list(range(0, len(vr), 16))
            if not frame_indices:
                continue
            raw_frames = vr.get_batch(frame_indices).asnumpy()
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        for crop_type in range(10):
            # Crop frames
            cropped_imgs = apply_crop(raw_frames, crop_type)
            # Convert to Tensor, keep on CPU to save GPU memory
            pil_imgs = [Image.fromarray(f) for f in cropped_imgs]
            pixel_values = extractor.image_processor(
                pil_imgs, size=IMAGE_SIZE, crop_size=IMAGE_SIZE, return_tensors='pt'
            )['pixel_values']  # [N, 3, 512, 512]

            queries = {
                "common": item["common_query"],
                "violence": item["violence_query"],
                "specific": item["specific_query"]
            }

            for q_key, q_text in queries.items():
                save_path = os.path.join(SAVE_DIRS[q_key], f"{filename}__{crop_type}.npy")
                if os.path.exists(save_path):
                    continue
                
                try:
                    feat = extractor.extract(pixel_values, q_text)
                    np.save(save_path, feat)
                except Exception as e:
                    print(f"\nError extracting {filename} | Crop {crop_type} | Query {q_key}: {e}")
            
            # Clear GPU cache after each crop type
            torch.cuda.empty_cache()
            
        # Clear memory for this video
        del raw_frames
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_rank", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--task_id", type=str, default="")
    args = parser.parse_args()
    
    main(global_rank=args.global_rank,
         local_rank=args.local_rank,
         task_id=args.task_id)
# Support prompt.json and single image or plain text inference
# use: python infer.py --model_name_or_path /path/to/model --prompt_json prompts.json --image_folder /path/to/images

import argparse
import json
import os
from typing import List, Dict, Optional, Tuple

# ── Suppress FFmpeg warnings from decord ──
os.environ.setdefault("DECORD_EFM_LOG_LEVEL", "error")

import torch
from PIL import Image
from transformers import AutoTokenizer

from fast_onevision.constants import DEFAULT_IMAGE_TOKEN
from fast_onevision.conversation import conv_templates
from fast_onevision.model import FastOneVisionForCausalLM
from fast_onevision.mm_utils import tokenizer_image_token
from dataclasses import dataclass, field

from decord import VideoReader, cpu
import numpy as np

@dataclass
class InferenceArguments:
    """Inference parameter configuration"""
    model_path: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "Model path"}
    )
    prompt_json: str = field(
        default="prompts.json",
        metadata={"help": "Prompt file path"}
    )
    image_folder: str = field(
        default="fast_onevision/eval/images",
        metadata={"help": "Image folder path"}
    )
    video_folder: str = field(
        default="fast_onevision/eval/videos",
        metadata={"help": "Video folder path"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Generate temperature parameters"}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p sampling parameter"}
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of generated tokens"}
    )
    device: str = field(
        default="cuda:0",
        metadata={"help": "Reasoning device"}
    )
    image_size: int = field(
        default=512,
        metadata={"help": "Image processing size"}
    )

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

def load_video(video_path,
               min_frm_num=10, 
               max_frm_num=20000):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps=vr.get_avg_fps()

    sample_num = min(total_frame_num, total_frame_num//fps, max_frm_num)
    sample_num = max(sample_num, min_frm_num)
    sample_num = int(sample_num)
    
    frame_idx = get_seq_frames(total_frame_num, sample_num)

    img_array = vr.get_batch(frame_idx).asnumpy()

    clip_imgs = [Image.fromarray(img_array[j]) for j in range(sample_num)]

    return clip_imgs

def extract_visual_input(image_path):

    if isinstance(image_path, list):
        return [Image.open(path).convert('RGB') for path in image_path]
    elif isinstance(image_path, str):
        suffix = image_path.split('.')[-1].lower()
        video_suffixes = {'mp4', 'avi', 'webm'}
        image_suffixes = {'jpg', 'png', 'jpeg'}
        
        if suffix in video_suffixes:
            return load_video(image_path)
        elif suffix in image_suffixes:
            return [Image.open(image_path).convert('RGB')]
    else:
        return None

def load_prompts(prompt_json: str) -> List[Dict]:
    """
    Load the "prompt.json" file
    Format: [{"prompt": "Describe this picture", "image": "demo.jpg"}]
    The "image" field can be null or omitted, indicating text-based reasoning.
    """
    with open(prompt_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts

def preprocess_multimodal_for_infer(
    messages: List[Dict],
    has_image: Optional[List] = None
) -> List[Dict]:
    """
    Preprocessing of multimodal input (simplified version)
    """
    for message in messages:
        if has_image and DEFAULT_IMAGE_TOKEN not in message['value']:
            message['value'] = f"{DEFAULT_IMAGE_TOKEN}\n{message['value']}"
        
        if DEFAULT_IMAGE_TOKEN in message['value']:
            message['value'] = message['value'].replace(
                DEFAULT_IMAGE_TOKEN,
                '<|vision_start|><|vision_pad|><|vision_end|>'
            ).strip()
    
    return messages

def build_input_prompt(prompt: str, image_path: Optional[str]) -> Tuple[List[Dict], bool]:
    """
    Construct the input prompt
    Return: (messages, has_image)
    """
    visual_input = extract_visual_input(image_path)
    
    messages = [
        {
            "from": "human",
            "value": prompt if not visual_input else f"<image>\n{prompt}"
        }
    ]
    
    return messages, visual_input


@torch.inference_mode()
def run_inference(
    model: FastOneVisionForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    image_path: Optional[str],
    image_processor,
    inference_args: InferenceArguments
) -> str:
    """
    Carry out single inference
    """
    messages, visual_input = build_input_prompt(prompt, image_path)
    
    messages = preprocess_multimodal_for_infer(messages, visual_input)
    
    conv = conv_templates["qwen3"].copy()
    conv.messages = messages
    conversation = conv.get_prompt(generate=True)
    
    if visual_input:

        input_ids = tokenizer_image_token(
            conversation,
            tokenizer,
            return_tensors='pt'
        ).unsqueeze(0).to(model.device)
        
        if inference_args.image_size is not None:
            pixel_values = image_processor(
                visual_input,
                size=inference_args.image_size,
                crop_size=inference_args.image_size,
                return_tensors='pt'
            )['pixel_values'].to(
                dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16,
                device=model.device
            )
        else:
            pixel_values = image_processor(
                visual_input,
                return_tensors='pt'
            )['pixel_values'].to(
                dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16,
                device=model.device
            )

        gen_kwargs = {
            "do_sample": True,
            "temperature": inference_args.temperature,
            "top_p": inference_args.top_p,
            "max_new_tokens": inference_args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        outputs = model.generate(
            input_ids=input_ids,
            images=[pixel_values],
            **gen_kwargs
        ).to('cpu')
        
    else:
        # Pure text reasoning

        input_ids = tokenizer(
            conversation,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids.to(model.device)
        
        gen_kwargs = {
            "do_sample": True,
            "temperature": inference_args.temperature,
            "top_p": inference_args.top_p,
            "max_new_tokens": inference_args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        outputs = model.generate(
            input_ids=input_ids,
            **gen_kwargs
        ).to('cpu')

    generated_ids = outputs[0]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    response = response.split("<|im_end|>")[0].strip()
    
    return response

# CHECK_POINT="./checkpoint_ov"
CHECK_POINT="./checkpoint_si"

PROMPTS="./fast_onevision/eval/prompt.json"

OUTPUT_FILE="./fast_onevision/eval/inference_results.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=CHECK_POINT)
    parser.add_argument("--prompt_json", type=str, default=PROMPTS)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image_size", type=int, default=512)
    
    args = parser.parse_args()

    inference_args = InferenceArguments(
        model_path=args.model_path,
        prompt_json=args.prompt_json,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        image_size=args.image_size
    )

    print(f"Loading model from {inference_args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        inference_args.model_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    model = FastOneVisionForCausalLM.from_pretrained(
        inference_args.model_path,
        aux_init_path = inference_args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=inference_args.device,
        trust_remote_code=True
    ).eval()

    image_processor = model.get_vision_tower().image_processor
    
    print(f"Loading prompts from {inference_args.prompt_json}...")
    prompts = load_prompts(inference_args.prompt_json)
    
    print(f"\nRunning inference on {len(prompts)} samples...\n")
    results = []
    
    for idx, item in enumerate(prompts):
        prompt = item.get("text", "")
        image_file = item.get("image", None)
        
        print(f"--- Sample {idx + 1}/{len(prompts)} ---")
        print(f"Prompt: {prompt}")
        
        image_file = item.get("image", None)
        video_file = item.get("video", None)

        if image_file and video_file:
            raise ValueError("Each item should contain only one of 'image' or 'video'.")

        if image_file:
            media_path = os.path.join(inference_args.image_folder, image_file)
            print(f"Image: {image_file} (from {media_path})")
        elif video_file:
            media_path = os.path.join(inference_args.video_folder, video_file)
            print(f"Video: {video_file} (from {media_path})")
        else:
            media_path = None
            print("Media: None (text-only)")
        
        response = run_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            image_path=media_path,
            image_processor=image_processor,
            inference_args=inference_args
        )
        print(f"Response: {response}\n")
        
        results.append({
            "prompt": prompt,
            "image": image_file,
            "response": response
        })
            
    output_file = OUTPUT_FILE
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
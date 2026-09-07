import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from safetensors.torch import load_file
from fast_onevision.model import *
from fast_onevision.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# Finally returns tokenizer, model, image_processor, and context_len (max context length)
# return tokenizer, model, image_processor, context_len
pass
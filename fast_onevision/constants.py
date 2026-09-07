IGNORE_INDEX = -100
MAX_MM_ENCODE_LOOP = 16
IMAGE_PATCH_NUM = 64

# Model Constants 
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# stop_token = "<|im_end|>\n"
# assistant_prefix = "<|im_start|>assistant\n"
STOP_TOKEN_IDS = [151645, 198]         
ASSISTANT_PREFIX_IDS = [151644, 77091, 198]

# Qwen/Qwen3-0.6B
V_START_ID = 151652 # <|vision_start|>
V_END_ID = 151653 # <|vision_end|>
IMG_PAD_ID = 151655 # <|image_pad|>
VID_PAD_ID = 151656 # <|video_pad|>
SEQ_PAD_ID = 151643 # <|endoftext|>
ASSISTANT_ID = 77091 # assistant
USER_ID = 827 # user
SPLIT_ID = 198 # \n
START_ID = 151644 # <|im_start|>
END_ID = 151645 # <|im_end|>

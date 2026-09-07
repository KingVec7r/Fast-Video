#    Copyright 2023 Haotian Liu
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

from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_compress import MultimodalCompressLayer

import logging
import torch.distributed as dist
import os
import sys

from fast_onevision.constants import SEQ_PAD_ID

logger = logging.getLogger(__name__)

def safe_exit_all(msg, exit_code=1):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if dist.is_initialized() and dist.get_world_size() > 1:
        has_err = 1.0 if msg else 0.0
        err_flag = torch.tensor(has_err, device=f"cuda:{local_rank}")
        dist.all_reduce(err_flag, op=dist.ReduceOp.SUM)
        if err_flag.item() > 0:
            if dist.get_rank() == 0:
                log_msg = msg if msg else "At least one rank encountered dummy/None vision data. Aborting training."
                logger.error(log_msg)
            
            try:
                dist.barrier()
            except Exception:
                pass
            finally:
                dist.destroy_process_group()
            sys.exit(exit_code)
    else:
        if msg:
            logger.error(msg)
        sys.exit(exit_code)

class FastOneVisionMetaModel:

    def __init__(self, config):
        super(FastOneVisionMetaModel, self).__init__(config)

        self.register_buffer("pad_id_tensor", torch.tensor([SEQ_PAD_ID], dtype=torch.long), persistent=False)

        if getattr(config, "mm_vision_tower", None):
            # build_vision_tower：When the parameter is a string, need manually call the `load_model` method later.
            self.vision_tower = build_vision_tower(config)
            self.projector = build_vision_projector(config)
            self.compressor = MultimodalCompressLayer(config)

    def multimodal_init(self, config):
        self.vision_tower = build_vision_tower(config)
        # config.mm_hidden_size = self.vision_tower.hidden_size
        config.mm_vision_tower = self.get_vision_tower().config

        self.projector = build_vision_projector(config)
        self.compressor = MultimodalCompressLayer(config)
                
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower
    
    def get_projector(self):
        projector = getattr(self, 'projector', None)
        return projector
    
    def get_compressor(self):
        compressor = getattr(self, 'compressor', None)
        return compressor
    
    def get_pad_id_tensor(self):
        pad_id_tensor = getattr(self, 'pad_id_tensor', None)
        return pad_id_tensor

class FastOneVisionMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    def get_projector(self):
        return self.get_model().get_projector()
    def get_compressor(self):
        return self.get_model().get_compressor()
    def get_pad_id_tensor(self):
        return self.get_model().get_pad_id_tensor()

    def encode_images(self, images, enable_chunked_encode: bool = False):

        # ctx = nullcontext()
        ctx = torch.no_grad()

        # for image data or video data with same sample frame num
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:  # image: [batchsize, C, H, W]
                images = images.unsqueeze(1)  # -> [batchsize, 1, C, H, W]
            assert images.dim() == 5
            batch_size, num_frames, C, H, W = images.shape
            images_4d = images.reshape(batch_size * num_frames, C, H, W)

        
            if enable_chunked_encode:
                # Fixed chunk size to bound peak memory during vision encoding.
                MAX_FRAMES_PER_CHUNK = 256
                total_frames = images_4d.shape[0]

                if total_frames > MAX_FRAMES_PER_CHUNK:
                    chunk_features = []
                    for chunk in torch.split(images_4d, MAX_FRAMES_PER_CHUNK, dim=0):
                        chunk = chunk.to(memory_format=torch.channels_last)
                        with ctx:
                            feat = self.get_vision_tower()(chunk)
                        feat = self.get_projector()(feat)
                        chunk_features.append(feat)
                    features_4d_processed = torch.cat(chunk_features, dim=0)
                else:
                    images_4d = images_4d.to(memory_format=torch.channels_last)
                    with ctx:
                        features_4d = self.get_vision_tower()(images_4d)
                    features_4d_processed = self.get_projector()(features_4d)
            else:
                features_4d = self.get_vision_tower()(images_4d)
                features_4d_processed = self.get_projector()(features_4d)

            feature_shape = features_4d_processed.shape[1:]
            image_embedding = features_4d_processed.reshape(batch_size, num_frames, *feature_shape) # [batchsize, frames_num, patches_num, hidden_size]

            return image_embedding

        # image_embedding = []
        if isinstance(images, list):
            raise NotImplementedError("Image instances within list objects are currently not supported.")
        
            local_has_none = any(img is None for img in images)
            err_msg = "Using dummy/None vision data in batch. Aborting training." if local_has_none else ""
            safe_exit_all(err_msg)

            image_embedding = []
            for image in images:
                embedding = self.get_vision_tower()(image) # [frames_num, patches_num, hidden_size]
                processed_embedding = self.get_projector()(embedding)
                image_embedding.append(processed_embedding)

            return image_embedding
        
        raise TypeError(f"Unsupported images type: {type(images)}")

    def prepare_embeddings_for_multimodal(
        self, 
        input_ids,
        images, 
        img_features=None,
        img_compressed_len=None,
        enable_chunked_encode: bool = False,
    ):
        if img_features is None:
            vision_tower = self.get_vision_tower()
            if vision_tower is None or images is None or input_ids.shape[1] == 1:
                return input_ids, None
            image_embedding = self.encode_images(images, enable_chunked_encode=enable_chunked_encode)
        else:
            # image_embedding = img_features
            image_embedding = self.get_projector()(img_features)

        text_embedding = self.get_input_embeddings()(input_ids)
        pad_embedding = None
        pad_id_tensor = self.get_pad_id_tensor()
        if pad_id_tensor:
            # [1, hidden_dim]
            pad_embedding = self.get_input_embeddings()(pad_id_tensor)

        if self.config.mm_version == "projector":
            # When preparing data, ensure that the length of the image patch sequence matches that of the image pad sequence.
            simple_replace = True
        else:
            simple_replace = False
            
        embeddings = self.get_compressor()(
            text_embeddings = text_embedding,
            input_ids = input_ids,
            image_embeddings = image_embedding,
            pad_embedding = pad_embedding,
            img_compressed_len = img_compressed_len,
            simple_replace = simple_replace
            )

        return None, embeddings

    # NOTE: enable_input_require_grads and gradient_checkpointing_enable have
    # been moved to FastOneVisionForCausalLM in
    # fast_onevision/model/language_model/fast_onevision.py because
    # PreTrainedModel defines both methods and appears earlier in the MRO,
    # which would shadow this mixin's definitions.
    # - enable_input_require_grads: vision tower hook would not be registered,
    #   breaking gradient flow through visual features.
    # - gradient_checkpointing_enable: GuidanceGenerator's KV cache would be
    #   silently disabled, breaking the chunked autoregressive loop.
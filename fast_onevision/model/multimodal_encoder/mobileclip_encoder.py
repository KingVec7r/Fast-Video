#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, PretrainedConfig
from fast_onevision.model.fast_vit_arch import FastViTImageConfig, FastViTImageEncoder

import logging
logger = logging.getLogger(__name__)

class MobileCLIPVisionTower(nn.Module):
    """
    Vision tower that can be initialized from:
      - A pretrained hub name (str): loads config + weights from HuggingFace Hub.
        Used when starting training from scratch.
      - A config dict: loads only the config structure. The actual weights are
        expected to be loaded later via ``load_state_dict`` from a full checkpoint.
        Used when resuming from a fine-tuned checkpoint that includes vision tower weights.

    Supports delayed loading: when ``delay_load=True``, only the lightweight config
    (``cfg_only``) is loaded. Call ``load_model()`` later to instantiate the encoder weights.
    """

    def __init__(self, vision_tower):
        """
        Args:
            vision_tower: Hub name (str) or config dict.
            args: Model config / argparse.Namespace with optional ``unfreeze_mm_vision_tower``.
            delay_load: If True, defer weight loading (only load cfg_only).
        """
        super().__init__()

        if isinstance(vision_tower, str):
            # Hub path: load config + image processor from pretrained
            self.config = FastViTImageConfig.from_pretrained(vision_tower)
            self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
            self.vision_tower = FastViTImageEncoder.from_pretrained(vision_tower)
            logger.info(f"Loaded vision tower from pretrained hub: {vision_tower}")
        elif isinstance(vision_tower, PretrainedConfig):
            # Config dict: construct config in-memory (weights come from checkpoint)
            self.config = vision_tower
            img_size = getattr(vision_tower, 'image_size', 512)
            self.image_processor = CLIPImageProcessor(size=img_size, crop_size=img_size)
            self.vision_tower = FastViTImageEncoder(vision_tower)
            logger.info(f"Initialized vision tower from PretrainedConfig")
        else:
            raise TypeError(
                f"vision_tower must be str (hub name) or PretrainedConfig, "
                f"got {type(vision_tower)}"
            )
        
    def feature_select(self, image_forward_outs):
        """Select features from encoder output (identity by default)."""
        return image_forward_outs

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype)
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def hidden_size(self):
        return self.config.embed_dim

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

"""
HF-compatible wrapper that turns the FastViT backbone into a pure *image encoder*.
Output: a single (B, embed_dim) vector obtained with the built-in GlobalPool2D head.
"""
import torch
from transformers import PreTrainedModel, PretrainedConfig
from fast_onevision.model.multimodal_encoder.mobileclip.mci import fastvithd, GlobalPool2D     # imports your backbone factory


# -----------------------  Config  -----------------------
class FastViTImageConfig(PretrainedConfig):
    """Minimal config so HF knows the image size & embed dim."""
    model_type = "fastvit_image_encoder"

    def __init__(
        self,
        image_size: int = 1024,
        embed_dim:  int = 3072,   # channels after conv_exp
        patch_size: int = 16,
        **kwargs
    ):
        self.image_size = image_size
        self.embed_dim  = embed_dim
        self.patch_size = patch_size
        super().__init__(**kwargs)

    @property
    def hidden_size(self):
        return self.embed_dim


# -----------------------  Model  ------------------------
class FastViTImageEncoder(PreTrainedModel):
    """
    Wraps FastViT-HD and exposes an `.embeddings` output;
    no text tower, no CLIP logits, only a pooled image embedding.
    """
    config_class    = FastViTImageConfig
    main_input_name = "pixel_values"

    def __init__(self, config: FastViTImageConfig):
        super().__init__(config)

        # We **keep** GlobalPool2D by asking for `num_classes = embed_dim`
        # (FastViT replaces the classifier with GlobalPool2D in that case).
        self.backbone = fastvithd(num_classes=0)
        self.backbone.head = GlobalPool2D(
            in_dim  = 3072,
            out_dim = 768
        )

        # HF helper that registers weights for bf16/half-precision etc.
        self.post_init()

    # ------------------------------------------
    # def forward(self, pixel_values, return_dict=True, **unused):
    #     """
    #     Args:
    #         pixel_values: (B, 3, H, W) tensor (already resized/normalized).
    #     Returns:
    #         Dict with a single key `"embeddings"` of shape (B, embed_dim).
    #     """
    #     # FastViT forward returns the pooled tensor directly because
    #     # `num_classes == embed_dim` and head == GlobalPool2D.
    #     embeddings = self.backbone(pixel_values)     # (B, embed_dim)

    #     if not return_dict:
    #         return (embeddings,)

    #     return {"embeddings": embeddings}
    
    def forward(self, images):
        return self.forward_images(images)

    def feature_select(self, image_forward_outs):
        # Features from penultimate layer
        image_features = image_forward_outs["image_embeddings"]

        # Reshape 4D tensor to 3D
        B, C, H, W = image_features.shape
        image_features = image_features.reshape(B, C, H*W)
        image_features = image_features.transpose(1, 2)
        return image_features

    def forward_images(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.backbone(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), return_image_embeddings=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.backbone(images.to(device=self.device, dtype=self.dtype), return_image_embeddings=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features
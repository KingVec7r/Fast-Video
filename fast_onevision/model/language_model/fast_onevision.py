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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import logging

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen3Model, Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..fast_onevision_arch import FastOneVisionMetaModel, FastOneVisionMetaForCausalLM

logger = logging.getLogger(__name__)

class FastOneVisionConfig(Qwen3Config):
    model_type = "fast_onevision"
    
    def __init__(self, **kwargs):
        
        self.mm_version = kwargs.pop("mm_version", None)
        # vision tower config — normalize dict→FastViTImageConfig on checkpoint load
        mm_vision_tower = kwargs.pop("mm_vision_tower", None)
        if isinstance(mm_vision_tower, dict):
            from fast_onevision.model.fast_vit_arch import FastViTImageConfig
            mm_vision_tower = FastViTImageConfig(**mm_vision_tower)
        self.mm_vision_tower = mm_vision_tower
        self.mm_vision_select_layer = kwargs.pop("mm_vision_select_layer", None)
        self.mm_vision_select_feature = kwargs.pop("mm_vision_select_feature", None)
        self.mm_projector_type = kwargs.pop("mm_projector_type", None)
        self.mm_max_compress_loop = kwargs.pop("mm_max_compress_loop", None)
        # compress config
        self.compress_intent_modeler_layer_num = kwargs.pop("compress_intent_modeler_layer_num", None)
        self.compress_intent_token_num = kwargs.pop("compress_intent_token_num", None)
        self.compress_guidance_gen_layer_num = kwargs.pop("compress_guidance_gen_layer_num", None)
        self.compress_layer_num = kwargs.pop("compress_layer_num", None)

        super().__init__(**kwargs)
        self.architectures = ["FastOneVisionForCausalLM"]

class FastOneVisionModel(FastOneVisionMetaModel, Qwen3Model):
    def __init__(self, config: Qwen3Config):
        super(FastOneVisionModel, self).__init__(config)

class FastOneVisionForCausalLM(Qwen3ForCausalLM, FastOneVisionMetaForCausalLM):
    config_class = FastOneVisionConfig
    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = FastOneVisionModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    # def enable_input_require_grads(self):
    #     """
    #     Overrides the parent implementation to also register a hook on the
    #     vision tower's output, ensuring the visual feature path has gradient
    #     tracking independent of whether the projector has trainable parameters.
    #     """

    #     def make_inputs_require_grads(module, input, output):
    #         output.requires_grad_(True)

    #     # Hook on the text embedding layer (original behavior)
    #     self._require_grads_hook = self.get_input_embeddings().register_forward_hook(
    #         make_inputs_require_grads
    #     )

    #     # Hook on the vision tower output to ensure visual features carry gradients
    #     vision_tower = self.get_vision_tower()
    #     if vision_tower is not None:
    #         self._vision_require_grads_hook = vision_tower.register_forward_hook(
    #             make_inputs_require_grads
    #         )

    # def disable_input_require_grads(self):
    #     """
    #     Removes both the text embedding hook and the vision tower hook.
    #     """
    #     self._require_grads_hook.remove()
    #     vision_hook = getattr(self, '_vision_require_grads_hook', None)
    #     if vision_hook is not None:
    #         vision_hook.remove()
    #         self._vision_require_grads_hook = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Enable checkpointing on all supported modules (LLM + compressor layers)
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)

        # past_key_values (KV cache) which is fundamentally incompatible with
        # gradient checkpointing (recomputed forward would lose the accumulated
        # cache state).  The Compressor uses full attention without a KV cache,
        # so it can safely keep gradient checkpointing enabled.
        if self.get_compressor():
            for layer in self.get_compressor().guidance_generator.generate_layers:
                layer.gradient_checkpointing = False
            logger.info("Disabled gradient checkpointing on guidance generator.")

            for layer in self.get_compressor().compressor.generate_layers:
                layer.gradient_checkpointing = False
            logger.info("Disabled gradient checkpointing on compressor.")
        # compressor = self.get_compressor()
        # if compressor is not None:
        #     # for submodule_name in ('guidance_generator', 'compressor', 'intent_modeler'):
        #     for submodule_name in ('guidance_generator'):
        #         submodule = getattr(compressor, submodule_name, None)
        #         if submodule is not None:
        #             for layer in submodule.generate_layers:
        #                 layer.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        img_features: Optional[torch.FloatTensor] = None,
        img_compressed_len: Optional[int] = None,
        enable_chunked_encode: bool = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                inputs_embeds
            ) = self.prepare_embeddings_for_multimodal(
                input_ids = input_ids,
                images = images,
                img_features = img_features,
                img_compressed_len = img_compressed_len,
                enable_chunked_encode = enable_chunked_encode,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

AutoConfig.register("fast_onevision", FastOneVisionConfig)
AutoModelForCausalLM.register(FastOneVisionConfig, FastOneVisionForCausalLM)
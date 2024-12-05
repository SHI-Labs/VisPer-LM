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
import torch.nn.functional as F
import numpy as np

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM


from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from dataclasses import dataclass
from ..ola_arch import OlaLlavaMetaModel, OlaLlavaMetaForCausalLM

from ola_vlm.model.aux_heads import GenHead, DepthHead, DAv2_Head
from ola_vlm.model.aux_heads.depth_anything_v2.dpt import DepthAnythingV2
from ola_vlm.model.aux_heads.oneformer_head import OneFormerHead, OneFormerSegHead

from transformers import OneFormerProcessor

from diffusers import (
    DPMSolverMultistepScheduler,
    StableUnCLIPImg2ImgPipeline,
)

import torch.distributed as dist
try:
    import wandb
except:
    pass
import os
import matplotlib

from ola_vlm.model.language_model.base_probe_vlm import BaseProbe_VLM

@dataclass
class ProbeDSGCausalLLMOutputWithPast(CausalLMOutputWithPast):
    image_embs: Optional[Tuple[torch.FloatTensor]] = None
    seg_embs: Optional[Tuple[torch.FloatTensor]] = None
    depth_embs: Optional[Tuple[torch.FloatTensor]] = None
    depth_preds: Optional[Tuple[torch.FloatTensor]] = None


class ProbeDSGLlavaLlamaConfig(LlamaConfig):
    model_type = "probe_dsg_llava_llama"


class ProbeDSGLlavaLlamaModel(OlaLlavaMetaModel, LlamaModel):
    config_class = ProbeDSGLlavaLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(ProbeDSGLlavaLlamaModel, self).__init__(config)


class ProbeDSGLlavaLlamaForCausalLM(LlamaForCausalLM, OlaLlavaMetaForCausalLM, BaseProbe_VLM):
    config_class = ProbeDSGLlavaLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        config.rope_scaling = None
        self.model = ProbeDSGLlavaLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pil_images = None,
        
    ) -> Union[Tuple, ProbeDSGCausalLLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        with torch.no_grad():
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

        hidden_states = outputs[0]

        layer_states = outputs[-1][1:]

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        log_dict = {}
        
        seg_embs, seg_loss, seg_log_dict = self.seg_emb_forward(pil_images, hidden_states, layer_states)
        if self.mode == "seg":
            if pil_images is not None:
                if dist.get_rank() == 0:
                    log_dict = {
                        **log_dict,
                        **seg_log_dict
                    }
        
        depth_preds, depth_embs, depth_loss, depth_log_dict = self.depth_emb_forward(pil_images, layer_states)
        if self.mode == "depth" and hidden_states.shape[1] > 1:
            if dist.get_rank() == 0:
                log_dict = {
                    **log_dict,
                    **depth_log_dict
                }

        img_embs, gen_loss, log_dict = self.gen_emb_forward(pil_images, hidden_states, layer_states)
        if self.mode == "gen" and hidden_states.shape[1] > 1:
            if dist.get_rank() == 0:
                log_dict = {
                    **log_dict,
                    **depth_log_dict
                }
        
        loss = seg_loss + depth_loss + gen_loss

        try:
            if dist.get_rank() == 0:
                log_dict = {
                    **log_dict,
                    "depth_loss": depth_loss,
                    "gen_loss": gen_loss,
                    "seg_loss": seg_loss,
                }
                filtered_log_dict = {key: value for key, value in log_dict.items() if value > 0}
                wandb.log(filtered_log_dict)
                self.steps += 1
        except:
            pass
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ProbeDSGCausalLLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_embs=img_embs,
            seg_embs=seg_embs,
            depth_embs=depth_embs,
            depth_preds=depth_preds,
        )    
    
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
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        pil_images: Optional[List[object]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return self._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pil_images=pil_images
        )

AutoConfig.register("probe_dsg_llava_llama", ProbeDSGLlavaLlamaConfig)
AutoModelForCausalLM.register(ProbeDSGLlavaLlamaConfig, ProbeDSGLlavaLlamaForCausalLM)
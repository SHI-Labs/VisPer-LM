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

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaForCausalLM, LlamaModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass

from ..ola_arch import OlaLlavaMetaModel, OlaLlavaMetaForCausalLM
import torch.distributed as dist
try:
    import wandb
except:
    pass
from torch.nn import CrossEntropyLoss
from .base_lm import BaseCausalLM
from .base_ola_vlm import BaseOLA_VLM



@dataclass
class OlaCausalLLMOutputWithPast(CausalLMOutputWithPast):
    image_embs: Optional[Tuple[torch.FloatTensor]] = None
    seg_embs: Optional[Tuple[torch.FloatTensor]] = None
    depth_embs: Optional[Tuple[torch.FloatTensor]] = None
    depth_preds: Optional[Tuple[torch.FloatTensor]] = None


class OlaLlavaLlamaConfig(LlamaConfig):
    model_type = "ola_llama"


class OlaLlavaLlamaModel(OlaLlavaMetaModel, LlamaModel):
    config_class = OlaLlavaLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(OlaLlavaLlamaModel, self).__init__(config)


class OlaLlavaLlamaForCausalLM(LlamaForCausalLM, OlaLlavaMetaForCausalLM, BaseOLA_VLM):
    config_class = OlaLlavaLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = OlaLlavaLlamaModel(config)
        self.vocab_size = config.vocab_size
        if self.vocab_size < 128000:
            self.NUM_SYS_TOKENS = 26 # vicuna-7b
        else:
            self.NUM_SYS_TOKENS = 38 # llama3-8b
        print(f"Number of System Tokens: {self.NUM_SYS_TOKENS}")
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _forward(
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
        return_dict: Optional[bool] = None,
        pil_images = None,
        gen_mask: Optional[torch.FloatTensor] = None,
        seg_mask: Optional[torch.FloatTensor] = None,
        depth_mask: Optional[torch.FloatTensor] = None,
        
    ) -> Union[Tuple, OlaCausalLLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        text_loss = None
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            text_loss = loss_fct(shift_logits, shift_labels)

        
        depth_preds, depth_embs, depth_loss, depth_l1_loss, depth_cont_loss = self.depth_emb_forward(pil_images, layer_states, depth_mask)
        seg_embs, seg_loss, seg_l1_loss, seg_contrastive_loss = self.seg_emb_forward(pil_images, hidden_states, layer_states, seg_mask)
        img_embs, gen_loss, gen_mse_loss, gen_con_loss = self.gen_emb_forward(pil_images, hidden_states, layer_states, gen_mask)
            
        if text_loss is not None:
            loss = text_loss + seg_loss + depth_loss + gen_loss
        
        try:
            if dist.get_rank() == 0:
                if loss > text_loss:
                    log_dict = {
                        "depth_loss": depth_loss,
                        "gen_loss": gen_loss,
                        "depth_l1_loss": depth_l1_loss,
                        "depth_contrastive_loss": depth_cont_loss,
                        "gen_l1_loss": gen_mse_loss,
                        "gen_contrastive_loss": gen_con_loss,
                        "seg_loss": seg_loss,
                        "seg_l1_loss": seg_l1_loss,
                        "seg_contrastive_loss": seg_contrastive_loss,
                        "text_loss": text_loss,
                        "loss": loss,
                    }
                    filtered_log_dict = {key: value for key, value in log_dict.items() if value > 0}
                    wandb.log(filtered_log_dict)
                else:
                    wandb.log({
                        "text_loss": text_loss,
                        "loss": loss,
                    })

                self.steps += 1
        except:
            pass
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return OlaCausalLLMOutputWithPast(
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
        gen_mask: Optional[torch.FloatTensor] = None,
        seg_mask: Optional[torch.FloatTensor] = None,
        depth_mask: Optional[torch.FloatTensor] = None,
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
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pil_images=pil_images,
            gen_mask=gen_mask,
            seg_mask=seg_mask,
            depth_mask=depth_mask,
        )

AutoConfig.register("ola_llama", OlaLlavaLlamaConfig)
AutoModelForCausalLM.register(OlaLlavaLlamaConfig, OlaLlavaLlamaForCausalLM)
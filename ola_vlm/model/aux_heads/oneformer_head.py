import torch
from typing import Optional
from torch import Tensor, nn
from ola_vlm.model.multimodal_projector.resampler import Resampler, TaskTokenResampler
import math
from torch.nn import functional as F
from transformers import OneFormerModel
from transformers.models.oneformer.modeling_oneformer import OneFormerForUniversalSegmentationOutput, OneFormerModelOutput, OneFormerPixelLevelModule, OneFormerPixelLevelModuleOutput


class AuxOneFormerPixelLevelModule(OneFormerPixelLevelModule):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False, last_backbone_feats: Tensor = None, all_backbone_features: Tensor = None, return_features: bool = False, return_all_features: bool = False):
        if all_backbone_features is None:
            features = self.encoder(pixel_values).feature_maps
            if return_all_features:
                return features
        else:
            features = all_backbone_features
        if last_backbone_feats is not None:
            features = list(features)
            last_backbone_feats = F.interpolate(last_backbone_feats, size=features[-1].shape[-2:], mode='bilinear', align_corners=False)
            features[-1] = last_backbone_feats
            for i in range(3):
                features[i] = F.interpolate(features[i], size=features[-1].shape[-2:], mode='bilinear', align_corners=False)
            features = tuple(features)
        elif return_features:
            return F.interpolate(features[-1], size=(24, 24), mode='bilinear', align_corners=False)
        decoder_output = self.decoder(features, output_hidden_states=output_hidden_states)
        return OneFormerPixelLevelModuleOutput(
            encoder_features=tuple(features),
            decoder_features=decoder_output.multi_scale_features,
            decoder_last_feature=decoder_output.mask_features,
        )

class OneFormerHead(OneFormerModel):
    def __init__(self, config):
        super().__init__(config)
        self.pixel_level_module = AuxOneFormerPixelLevelModule(config)
    
    def forward_features(
        self,
        pixel_values: Tensor,
        task_inputs: Tensor,
        text_inputs: Tensor = None,
        pixel_mask: Tensor = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)

        backbone_last_feature = self.pixel_level_module(pixel_values, output_hidden_states, return_features=True)

        return backbone_last_feature
    
    def get_backbone_feats(
        self,
        pixel_values: Tensor,
        task_inputs: Tensor,
        text_inputs: Tensor = None,
        pixel_mask: Tensor = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)

        backbone_last_feature = self.pixel_level_module(pixel_values, output_hidden_states, return_all_features=True)

        return backbone_last_feature
    
    def get_masks(
        self,
        pixel_values: Tensor,
        task_inputs: Tensor,
        text_inputs: Tensor = None,
        pixel_mask: Tensor = None,
        backbone_last_feature: Tensor = None,
        all_backbone_features: Tensor = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)

        pixel_level_module_output = self.pixel_level_module(pixel_values, output_hidden_states, backbone_last_feature, all_backbone_features)

        multi_scale_features = pixel_level_module_output.decoder_features
        mask_features = pixel_level_module_output.decoder_last_feature

        task_token = self.task_encoder(task_inputs.to(self.dtype))

        if self.is_training:
            text_queries = self.text_mapper(text_inputs)
        else:
            text_queries = None

        transformer_module_output = self.transformer_module(
            multi_scale_features=multi_scale_features,
            mask_features=mask_features,
            task_token=task_token,
            output_attentions=output_attentions,
        )

        queries = transformer_module_output.object_queries

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_features
            pixel_decoder_hidden_states = (pixel_level_module_output.decoder_last_feature,)
            for f in pixel_level_module_output.decoder_features:
                pixel_decoder_hidden_states += (f,)
            transformer_decoder_hidden_states = transformer_module_output.auxiliary_predictions

        outputs = OneFormerModelOutput(
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            transformer_decoder_object_queries=queries,
            transformer_decoder_contrastive_queries=transformer_module_output.contrastive_logits,
            transformer_decoder_mask_predictions=transformer_module_output.prediction_masks,
            transformer_decoder_class_predictions=transformer_module_output.prediction_class,
            transformer_decoder_auxiliary_predictions=transformer_module_output.auxiliary_predictions,
            text_queries=text_queries,
            task_token=task_token,
            attentions=transformer_module_output.attentions,
        )

        class_queries_logits = outputs.transformer_decoder_class_predictions
        masks_queries_logits = outputs.transformer_decoder_mask_predictions
        contrastive_queries_logits = outputs.transformer_decoder_contrastive_queries
        auxiliary_predictions = outputs.transformer_decoder_auxiliary_predictions
        text_queries = outputs.text_queries

        output = OneFormerForUniversalSegmentationOutput(
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits,
            auxiliary_predictions=auxiliary_predictions,
            loss=None,
            **outputs,
        )

        return output
    
class OneFormerSegHead(nn.Module):

    def __init__(
        self,
        proj_config: dict = None,
        llm_hidden_size: int = 4096,
    ) -> None:
        super().__init__()

        self.projector = Resampler(
            dim=proj_config["output_dim"],
            depth=proj_config["depth"],
            dim_head=proj_config["dim_head"],
            heads=proj_config["num_heads"],
            num_queries=proj_config["num_tokens"],
            embedding_dim=llm_hidden_size,
            output_dim=proj_config["output_dim"],
            ff_mult=proj_config["ff_mult"],
        )
        
    
    def forward(
        self,
        llm_feats: torch.Tensor,
    ):
        visual_feats = self.projector(llm_feats)
        b, n, c = visual_feats.shape
        b = int(b)
        c = int(c)
        h = w = int(math.sqrt(int(n)))
        visual_feats = visual_feats.permute(0, 2, 1)
        image_embeddings = visual_feats.reshape(b, c, h, w)
        
        return image_embeddings


class OneFormerTaskTokenSegHead(nn.Module):

    def __init__(
        self,
        proj_config: dict = None,
        llm_hidden_size: int = 4096,
    ) -> None:
        super().__init__()

        self.projector = TaskTokenResampler(
            dim=proj_config["output_dim"],
            depth=proj_config["depth"],
            dim_head=proj_config["dim_head"],
            heads=proj_config["num_heads"],
            num_queries=proj_config["num_tokens"],
            embedding_dim=llm_hidden_size,
            output_dim=proj_config["output_dim"],
            ff_mult=proj_config["ff_mult"],
        )
        
    
    def forward(
        self,
        llm_feats: torch.Tensor,
        latents: torch.Tensor,
    ):
        visual_feats = self.projector(llm_feats, latents)
        b, n, c = visual_feats.shape
        b = int(b)
        c = int(c)
        h = w = int(math.sqrt(int(n)))
        visual_feats = visual_feats.permute(0, 2, 1)
        image_embeddings = visual_feats.reshape(b, c, h, w)
        
        return image_embeddings

def build_mlp(in_hidden_size, hidden_size):
    modules = [nn.Linear(in_hidden_size, hidden_size)]
    modules.append(nn.GELU())
    modules.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*modules)
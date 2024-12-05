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

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from ola_vlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from ola_vlm.mm_utils import get_anyres_image_grid_shape
import numpy as np

from ola_vlm.model.aux_heads.sam_utils.build_sam import sam_model_registry
from ola_vlm.model.aux_heads.sam_utils.automatic_mask_generator import SamAutomaticMaskGenerator
from ola_vlm.model.aux_heads.depth_anything_v2.dpt import DepthAnythingV2
from diffusers import StableUnCLIPImg2ImgPipeline
import torch.nn.functional as F
import copy

from ola_vlm.model.aux_heads.oneformer_head import OneFormerHead, OneFormerSegHead, OneFormerTaskTokenSegHead
from transformers import OneFormerProcessor, OneFormerConfig

# import torch
from torchvision import transforms
from PIL import Image



def build_mlp(in_hidden_size, hidden_size):
    modules = [nn.Linear(in_hidden_size, hidden_size)]
    modules.append(nn.GELU())
    modules.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*modules)

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class MultiEncLlavaMetaModel:

    def __init__(self, config):
        super(MultiEncLlavaMetaModel, self).__init__(config)
        self.attn_mask_type = 'causal'

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

            self.aggr = getattr(config, 'aggregation', "features")
            if  self.aggr == "tokens":
                depth_config = copy.deepcopy(config)
                depth_config.mm_hidden_size = config.depth_dim
                self.depth_projector = build_vision_projector(depth_config)

                gen_config = copy.deepcopy(config)
                gen_config.mm_hidden_size = config.gen_dim
                self.gen_projector = build_vision_projector(gen_config)

                seg_config = copy.deepcopy(config)
                seg_config.mm_hidden_size = config.seg_dim
                self.seg_projector = build_vision_projector(seg_config)
            
            self.init_encoders(config)

            self.set_attn_mask_type(config)

    def init_encoders(self, config):
        encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
        self.dav2_model = DepthAnythingV2(**model_configs[encoder])
        self.dav2_model.load_state_dict(torch.load(config.depth_estimator, map_location='cpu'))
        self.dav2_model.eval()
        
        self.aggr = getattr(config, 'aggregation', "features")

        try:
            self.dav2_model = self.dav2_model.cuda()
        except:
            pass

        self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(config.image_generator, torch_dtype=torch.float16, variant="fp16")

        self.seg_teacher = getattr(config, "seg_teacher", "oneformer")
        if self.seg_teacher == "sam":
            self.sam = sam_model_registry["vit_l"](checkpoint=self.config.image_segmentor)
            try:
                self.sam = self.sam.to("cuda")
            except:
                pass
            for p in self.sam.parameters():
                p.requires_grad = False
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        
        elif self.seg_teacher == "oneformer":
            self.oneformer_processor = OneFormerProcessor.from_pretrained(config.image_segmentor)
            self.oneformer = OneFormerHead.from_pretrained(config.image_segmentor)
            for p in self.oneformer.parameters():
                p.requires_grad = False
            try:
                self.oneformer = self.oneformer.to("cuda")
            except:
                pass
            self.mask_generator = None

    def set_attn_mask_type(self, config):
        if hasattr(config, 'attn_mask_type'):
            self.attn_mask_type = config.attn_mask_type
        else:
            self.attn_mask_type = 'causal'
        print(f"Setting attn_mask_type to {self.attn_mask_type}")

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            if getattr(model_args, 'aggregation', "features") == "features":
                self.config.mm_hidden_size = self.config.mm_hidden_size + model_args.depth_dim + model_args.seg_dim + model_args.gen_dim 
            self.mm_projector = build_vision_projector(self.config)

            if getattr(model_args, 'aggregation', "features") == "tokens":
                depth_config = copy.deepcopy(self.config)
                depth_config.mm_hidden_size = model_args.depth_dim
                self.depth_projector = build_vision_projector(depth_config)

                gen_config = copy.deepcopy(self.config)
                gen_config.mm_hidden_size = model_args.gen_dim
                self.gen_projector = build_vision_projector(gen_config)

                seg_config = copy.deepcopy(self.config)
                seg_config.mm_hidden_size = model_args.seg_dim
                self.seg_projector = build_vision_projector(seg_config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor

def unpad_prep_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        mode = "height"
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]
        mode = "width"

    return unpadded_tensor, mode, padding


def reverse_convnext_preprocess(preprocessed_tensor):
    unnormalize = transforms.Normalize(mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5], std=[1/0.5, 1/0.5, 1/0.5])
    image_tensor = torch.clamp(unnormalize(preprocessed_tensor), 0, 1)
    return transforms.ToPILImage()(image_tensor)

class MultiEncLlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    @property
    def attn_mask_type(self):
        return self.get_model().attn_mask_type

    def get_seg_targets(self, pil_images, preds):
        def _get_feats(img, mask_generator):
            if self.get_model().seg_teacher == "oneformer":
                img = img.resize((768, 768))
                inputs = self.get_model().oneformer_processor(img, ["panoptic"], return_tensors="pt")
                self.get_model().oneformer = self.get_model().oneformer.to(preds.device, preds.dtype)
                inputs["pixel_values"] = inputs["pixel_values"].to(preds.device, preds.dtype)
                with torch.no_grad():
                    feats = self.get_model().oneformer.forward_features(**inputs)
            else:
                img = np.array(img)
                mask_generator.predictor.set_image(img, dtype=preds.dtype)
                feats = mask_generator.predictor.features
                mask_generator.predictor.reset_image()
                feats = F.interpolate(feats, (24, 24), mode="bicubic", align_corners=False)
            feats = feats.permute(0, 2, 3, 1)
            feats = feats.reshape(1, -1, feats.shape[-1])
            return feats

        seg_targets = []
        for img in pil_images:
            feat = _get_feats(img, self.get_model().mask_generator)
            seg_targets.append(feat)

        seg_targets = torch.stack(seg_targets, dim=0).squeeze(1)
        return seg_targets

    def get_dav2_feats(self, pil_images, device):
        self.get_model().dav2_model = self.get_model().dav2_model.to(device)
        dav2_feats = []
        for img in pil_images:
            img = img.resize((336, 336))
            img = np.array(img)
            feat = self.get_model().dav2_model.infer_image(img, is_dsg=True)
            feat = (feat[0][0] + feat[1][0] + feat[2][0] + feat[3][0]) / 4
            dav2_feats.append(feat.to(device))

        dav2_feats = torch.stack(dav2_feats, dim=0).squeeze(1)
        return dav2_feats

    def get_gen_feats(self, pil_images, device):
        gen_feats = []
        self.get_model().pipe.image_encoder = self.get_model().pipe.image_encoder.to(device)
        for img in pil_images:
            clip_ims = self.get_model().pipe.feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)
            feat = self.get_model().pipe.image_encoder(clip_ims).image_embeds
            gen_feats.append(feat)

        gen_feats = torch.stack(gen_feats, dim=0)
        return gen_feats
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images).to(images.dtype).to(images.device)
        
        if self.get_model().aggr == "tokens":
            image_features = self.get_model().mm_projector(image_features)

        pil_images = [reverse_convnext_preprocess(images[i].float()) for i in range(images.shape[0])]
        
        depth_feats = self.get_dav2_feats(pil_images, image_features.device).to(image_features.dtype)
        
        if self.get_model().aggr == "tokens":
            depth_feats = depth_feats.permute(0, 2, 1)
            depth_feats = F.avg_pool1d(depth_feats, kernel_size=72)
            depth_feats = depth_feats.permute(0, 2, 1)
            depth_feats = self.get_model().depth_projector(depth_feats)

        gen_feats = self.get_gen_feats(pil_images, image_features.device).to(image_features.dtype)
        
        if self.get_model().aggr == "tokens":
            gen_feats = gen_feats.repeat(1, 8, 1)
            gen_feats = self.get_model().gen_projector(gen_feats)
        else:
            gen_feats = gen_feats.repeat(1, image_features.shape[1], 1)

        seg_feats = self.get_seg_targets(pil_images, image_features).to(image_features.dtype)
        
        if self.get_model().aggr == "tokens":
            seg_feats = seg_feats.permute(0, 2, 1)
            seg_feats = F.avg_pool1d(seg_feats, kernel_size=72)
            seg_feats = seg_feats.permute(0, 2, 1)
            seg_feats = self.get_model().seg_projector(seg_feats)
        
        if self.get_model().aggr == "tokens":
            # image_features = torch.cat([image_features, depth_feats, seg_feats, gen_feats], dim=1)
            image_features = torch.cat([image_features, gen_feats, depth_feats, seg_feats], dim=1)
        else:
            # image_features = torch.cat([image_features, depth_feats, seg_feats, gen_feats], dim=2)
            image_features = torch.cat([image_features, gen_feats, depth_feats, seg_feats], dim=2)
            image_features = self.get_model().mm_projector(image_features)

        return image_features
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)
        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            do_sample = False
        else:
            do_sample = True

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)] 
        
        new_input_embeds = []
        new_labels = []
        block_indices = [None] * len(input_ids)
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            num_tokens = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    num_tokens += cur_input_embeds_no_im[i].shape[0]
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    num_tokens += cur_image_features.shape[0]
                        
            if self.attn_mask_type == "block-causal":
                indices = ["block-causal", image_token_indices[1], num_tokens]
                block_indices[batch_idx] = indices

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, block_indices

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
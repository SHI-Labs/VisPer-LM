from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.generation.utils import GenerateOutput

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
from .base_lm import BaseCausalLM
from tqdm import tqdm

from ola_vlm.ola_utils import *

class BaseProbe_VLM(BaseCausalLM):

    def __init__(self, config):
        super(BaseCausalLM, self).__init__(config)
        self.steps = 0
        self.config = config
        self.num_layers = config.num_hidden_layers

        # Initialize weights and apply final processing
        self.post_init()
        self.is_trained = False
        if hasattr(config, "probe_mode"):
            self.is_trained = True
            self.init_heads(config)

        try:
            if dist.get_rank() == 0:
                wandb.init(project=os.environ['WANDB_PROJECT'], name=f"{os.environ['WANDB_NAME']}")
        except:
            pass

    def get_model(self):
        return self.model

    def init_heads(self, config):
        self.mode = config.probe_mode
        
        if self.mode == "gen":            
            self.image_gen_heads = nn.ModuleList([
                GenHead(config.image_gen, llm_hidden_size=config.hidden_size)
                for _ in range(self.num_layers)
            ])
            
            if not self.is_trained:
                self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(config.image_generator, torch_dtype=torch.float16, variant="fp16")
                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
                self.gen_encoder = self.pipe.image_encoder
                self.feature_extractor = self.pipe.feature_extractor
                for p in self.gen_encoder.parameters():
                    p.requires_grad = False

        elif self.mode == "seg":
            if not self.is_trained:
                self.oneformer_processor = OneFormerProcessor.from_pretrained(config.image_segmentor)
                self.oneformer = OneFormerHead.from_pretrained(config.image_segmentor)
                for p in self.oneformer.parameters():
                    p.requires_grad = False
                try:
                    self.oneformer = self.oneformer.to("cuda")
                except:
                    pass
            self.image_seg_heads = nn.ModuleList([
                OneFormerSegHead(config.image_seg, llm_hidden_size=config.hidden_size)
                for _ in range(self.num_layers)
            ])
            

        if self.mode == "depth":
            self.image_depth_heads = nn.ModuleList([
                DepthHead(proj_config=config.image_depth, llm_hidden_size=config.hidden_size, use_intermediate_depth=False)
                for _ in range(self.num_layers)
            ])

            dav2_cfg = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
            self.dav2_backbone = DepthAnythingV2(**dav2_cfg)
            self.dav2_backbone.load_state_dict(torch.load(config.depth_estimator, map_location='cpu'))
            for p in self.dav2_backbone.parameters():
                p.requires_grad = False

            self.da_v2_head = DAv2_Head()
            self.da_v2_head.load_state_dict(torch.load(config.depth_estimator), strict=False)
            for p in self.da_v2_head.parameters():
                p.requires_grad = False
    
    def _get_layer_loss_weight(self, config, prefix):
        layer_indices = config[f"{prefix}_layer_indices"]
        layer_indices = layer_indices.split("-")
        layer_indices = [int(i) - 1 for i in layer_indices]
        loss_weight = config[f"{prefix}_loss_weight"]
        return layer_indices, loss_weight
    
    def log_gen(self, img_embeds, pil_images, layer_idx, is_train=False):
        device = "cuda" if torch.cuda.is_available() else "hip"
        pipe = self.pipe.to(device)

        images = []

        if len(pil_images) > 2:
            pil_images = pil_images[:2]
            img_embeds = img_embeds[:2]

        for img_embed in img_embeds:
            image = pipe(image_embeds=img_embed.float().detach(),
                    num_inference_steps=25,
                    # guidance_scale=1,,
                ).images[0]
            images.append(image)
        
        if not is_train:
            return images

        n = len(images)
        c = min(n, 16)
        r = n // c
        images = images[:c*r]
        image_grid = make_grid(images, pil_images)
        
        wandb.log({
            f"val_gen_images/step_{self.steps}": wandb.Image(image_grid, caption=f"Layer-{layer_idx}")
        })
    
    def log_depth(self, depth_preds, layer_idx, depth_targets=None, is_train=False):
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        depth_preds = depth_preds.float().detach()
        def _visualize_depth(depth):
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
            return Image.fromarray(colored_depth)

        pred_depths, gt_depths = [], []

        if depth_targets is None:
            depth_targets = [None] * len(depth_preds)

        from tqdm import tqdm
        for pred, target in tqdm(zip(depth_preds, depth_targets), desc="Visualizing Depth..."):
            if target is not None:
                gt = _visualize_depth(target.float())
                gt_depths.append(gt)

            pred = _visualize_depth(pred)
            pred_depths.append(pred)
        
        if not is_train:
            return pred_depths

        n = len(pred_depths)
        c = min(n, 16)
        r = n // c
        pred_depths = pred_depths[:c*r]
        gt_depths = gt_depths[:c*r]
        masks_grid = make_grid(pred_depths, gt_depths)
        
        wandb.log({
            f"val_depth_images/step_{self.steps}": wandb.Image(masks_grid, caption=f"Layer-{layer_idx}")
        })
    
    def log_seg(self, seg_embeds, pil_images, layer_idx, seg_targets=None, is_train=False):
        def _oneformer_prepare_panoptic_instance_prediction(
            segmentation: torch.Tensor, segments_info: dict
        ):
            masks = []
            classes = []

            for segment in segments_info:
                id = segment["id"]
                label_id = segment["label_id"]
                label = self.oneformer.config.id2label[label_id]
                mask = segmentation == id
                masks.append(mask.float())
                classes.append(label)

            return masks, classes
        
        pred_masks, gt_masks = [], []

        seg_embeds = seg_embeds.detach()

        if seg_targets is None:
            seg_targets = [None] * len(seg_embeds)

        if len(pil_images) > 2:
            pil_images = pil_images[:2]
            seg_embeds = seg_embeds[:2]
            seg_targets = seg_targets[:2]

        from tqdm import tqdm
        for emb, target, img in tqdm(zip(seg_embeds, seg_targets, pil_images), desc=f"Predicting Segmentation Map..."):
            with torch.no_grad():
                inputs = self.oneformer_processor(img, ["panoptic"], return_tensors="pt")
                inputs["pixel_values"] = inputs["pixel_values"].to(emb.device, emb.dtype)
                inputs["task_inputs"] = inputs["task_inputs"].to(emb.device, emb.dtype)
                gt = self.oneformer.get_masks(**inputs, backbone_last_feature=target.unsqueeze(0))
                gt = self.oneformer_processor.post_process_panoptic_segmentation(
                                        gt, target_sizes=[img.size[::-1]]
                                    )[0]
                gt_msk, gt_cls = _oneformer_prepare_panoptic_instance_prediction(**gt)
                gt = visualize_oneformer_masks_on_image(img, gt_msk, gt_cls)

                pred = self.oneformer.get_masks(**inputs, backbone_last_feature=emb.unsqueeze(0))
                pred = self.oneformer_processor.post_process_panoptic_segmentation(
                                        pred, target_sizes=[img.size[::-1]]
                                    )[0]
                pred_msk, pred_cls = _oneformer_prepare_panoptic_instance_prediction(**pred)
                pred = visualize_oneformer_masks_on_image(img, pred_msk, pred_cls)

            gt_masks.append(gt)
            pred_masks.append(pred)

        if not is_train:
            return pred_masks

        n = len(pred_masks)
        c = min(n, 16)
        r = n // c
        pred_masks = pred_masks[:c*r]
        gt_masks = gt_masks[:c*r]
        masks_grid = make_grid(pred_masks, gt_masks)
        
        wandb.log({
            f"val_seg_images/step_{self.steps}": wandb.Image(masks_grid, caption=f"Layer-{layer_idx}")
        })
    

    def _emb_loss(self, emb_preds, emb_targets):
        emb_targets = emb_targets.to(emb_preds.dtype).to(emb_preds.device)

        if emb_targets.shape[0] != emb_preds.shape[0]:
            repeat_factor = emb_preds.shape[0] // emb_targets.shape[0]
            emb_targets = emb_targets.repeat(repeat_factor, 1, 1)

            if emb_targets.shape[0] != emb_preds.shape[0]:
                emb_targets = emb_targets[:emb_preds.shape[0]]
                emb_mask = emb_mask[:emb_preds.shape[0]]

        emb_loss = F.smooth_l1_loss(
            emb_preds.float(), emb_targets.float(), reduction="none"
        ).mean()

        return emb_loss


    def _get_gen_feats(self, pil_images, device):
        gen_feats = []
        for img in pil_images:
            with torch.no_grad():
                clip_ims = self.pipe.feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)
                feat = self.pipe.image_encoder(clip_ims).image_embeds
                gen_feats.append(feat)

        gen_feats = torch.stack(gen_feats, dim=0)
        return gen_feats
    
    def _forward_gen(self, gen_preds, layer_index, pil_images, gen_targets):        
        gen_loss = self._emb_loss(gen_preds, gen_targets)

        if dist.get_rank() == 0:
            if self.steps % 500 == 0:
                try:
                    self.log_gen(gen_preds.detach(), pil_images, layer_index, is_train=True)
                except:
                    pass
    
        return gen_loss
    

    def _get_dav2_feats(self, pil_images, device):
        dav2_gts = []
        depth_targets = [[]]
        for img in pil_images:
            img = img.resize((336, 336))
            img = np.array(img)
            with torch.no_grad():
                feat = self.dav2_backbone.infer_image(img, is_dsg=True)
                depth_gts = self.da_v2_head([feat[-1]] * 4)
                depth_targets[0].append(feat[-1][0])
            min_val = depth_gts.amin(dim=(1, 2), keepdim=True)
            max_val = depth_gts.amax(dim=(1, 2), keepdim=True)
            depth_gts = (depth_gts - min_val) / (max_val - min_val)
            dav2_gts.append(depth_gts.to(device))
        dav2_gts = torch.stack(dav2_gts, dim=0).squeeze(1)
        for i in range(len(depth_targets)):
            depth_targets[i] = (torch.stack(depth_targets[i], dim=0).squeeze(1), None)
        return depth_targets, dav2_gts
    
    def _forward_depth(self, all_depth_feats, layer_index, all_depth_targets, depth_pred_maps, depth_gts):                

        depth_feats, depth_targets = all_depth_feats[0][0], all_depth_targets[0][0]
        depth_loss = self._emb_loss(depth_feats, depth_targets)

        if dist.get_rank() == 0:
            if self.steps % 200 == 0:
                try:
                    self.log_depth(depth_pred_maps.detach(), layer_index, depth_gts, is_train=True)
                except:
                    pass

        return depth_loss


    def _get_seg_targets(self, pil_images, seg_preds):
        def _get_feats(img):
            img = img.resize((768, 768))
            inputs = self.oneformer_processor(img, ["panoptic"], return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(seg_preds.device, seg_preds.dtype)
            with torch.no_grad():
                feats = self.oneformer.forward_features(**inputs)
            return feats

        seg_targets = []
        for img in pil_images:
            feat = _get_feats(img)
            seg_targets.append(feat)

        seg_targets = torch.stack(seg_targets, dim=0).squeeze(1)
        return seg_targets

    def _forward_seg(self, seg_preds, layer_index, pil_images, seg_targets):
        
        seg_loss = self._emb_loss(seg_preds, seg_targets)
        
        if dist.get_rank() == 0:
            if self.steps % 200 == 0:
                try:
                    self.log_seg(seg_preds.detach(), pil_images, layer_index, seg_targets, is_train=True)
                except:
                    pass
    
        return seg_loss


    def forward_emb_predictor(self, layer_states, idx, i, heads):
        inp_tokens = layer_states[idx]
        task_emb = heads[i](inp_tokens)
        return task_emb

    def depth_emb_forward(self, pil_images, layer_states):
        depth_preds = []
        depth_embs = []
        depth_loss = 0
        log_dict = {}
        if self.mode == "depth": 
            if pil_images is not None:
                depth_targets, depth_gts = self._get_dav2_feats(pil_images, layer_states[0].device)
            else:
                depth_targets, depth_gts = None, None
            
            for i, idx in enumerate(range(self.num_layers)):
                
                depth_feats = self.forward_emb_predictor(layer_states, idx, i, self.image_depth_heads)
                depth_embs.append(depth_feats)

                with torch.no_grad():
                    depth_pred = self.da_v2_head([depth_feats[0]] * 4)
                    min_val = depth_pred.amin(dim=(1, 2), keepdim=True)
                    max_val = depth_pred.amax(dim=(1, 2), keepdim=True)
                    depth_pred = (depth_pred - min_val) / (max_val - min_val)
                    depth_preds.append(depth_pred)

                if depth_targets is not None:
                    layer_depth_loss = self._forward_depth(depth_feats, idx+1, depth_targets, depth_pred, depth_gts)
                    depth_loss += layer_depth_loss
                    if dist.get_rank() == 0:
                        log_dict = {
                            **log_dict,
                            f"{idx}_depth_loss": layer_depth_loss.item(),
                        }
    
    
        return depth_preds, depth_embs, depth_loss, log_dict
    
    def seg_emb_forward(self, pil_images, hidden_states, layer_states):
        seg_embs = []
        seg_loss = 0
        log_dict = {}
        if "seg" in self.mode:
            if pil_images is not None:
                seg_targets = self._get_seg_targets(pil_images, hidden_states)
            else:
                seg_targets = None
            for i, idx in enumerate(range(self.num_layers)):

                seg_emb = self.forward_emb_predictor(layer_states, idx, i, self.image_seg_heads)
                seg_embs.append(seg_emb)

                if seg_targets is not None:
                    layer_seg_loss = self._forward_seg(seg_emb, idx+1, pil_images, seg_targets)
                    seg_loss += layer_seg_loss
                    if dist.get_rank() == 0:
                        log_dict = {
                            **log_dict,
                            f"{idx}_seg_loss": layer_seg_loss.item(),
                        }
        
        
        return seg_embs, seg_loss, log_dict
    
    def gen_emb_forward(self, pil_images, hidden_states, layer_states):
        img_embs = []
        gen_loss = 0
        log_dict = {}
        if "gen" in self.mode:
            if pil_images is not None:
                gen_targets = self._get_gen_feats(pil_images, hidden_states.device)
            else:
                gen_targets = None
            
            for i, idx in enumerate(range(self.num_layers)):
                
                img_emb = self.forward_emb_predictor(layer_states, idx, i, self.image_gen_heads)
                img_embs.append(img_emb)

                if gen_targets is not None:
                    layer_gen_loss = self._forward_gen(img_emb, idx+1, pil_images, gen_targets)
                    gen_loss += layer_gen_loss
                    if dist.get_rank() == 0:
                        log_dict = {
                            **log_dict,
                            f"{idx}_gen_loss": layer_gen_loss.item(),
                        }

        return img_embs, gen_loss, log_dict

    @torch.no_grad()
    def get_visual_interpretations(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if True:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )

        
        return self.forward(
            input_ids=inputs,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        pil_images = kwargs.pop("pil_images", None)
        
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if pil_images is not None:
            inputs['pil_images'] = pil_images
        return inputs
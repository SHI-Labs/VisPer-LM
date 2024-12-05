from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.generation.utils import GenerateOutput

from ola_vlm.model.aux_heads import GenHead, DepthHead, DAv2_Head, TaskTokenGenHead, TaskTokenDepthHead
from ola_vlm.model.aux_heads.depth_anything_v2.dpt import DepthAnythingV2
from ola_vlm.model.aux_heads.oneformer_head import OneFormerHead, OneFormerSegHead, OneFormerTaskTokenSegHead

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


class BaseOLA_VLM(BaseCausalLM):

    def __init__(self, config):
        super(BaseCausalLM, self).__init__(config)
        self.steps = 0
        self.config = config

        if hasattr(config, "image_gen"):
            self.init_heads(config)

        try:
            if dist.get_rank() == 0:
                wandb.init(project=os.environ['WANDB_PROJECT'], name=f"{os.environ['WANDB_NAME']}")
        except:
            pass

    def get_model(self):
        return self.model

    def init_target_models(self, config):
        if hasattr(config, "image_gen") and "gen" in self.mode:
            if not os.path.exists(config.image_generator):
                config.image_generator = "stabilityai/stable-diffusion-2-1-unclip"
            self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(config.image_generator, torch_dtype=torch.float16, variant="fp16")
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            for p in self.pipe.image_encoder.parameters():
                p.requires_grad = False
            try:
                self.pipe = self.pipe.to("cuda")
            except:
                pass

        if hasattr(config, "image_depth") and "depth" in self.mode:
            dav2_cfg = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
            self.dav2_backbone = DepthAnythingV2(**dav2_cfg)

            if not os.path.exists(config.depth_estimator):
                url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
                local_model_path = "depth_anything_v2_vitl.pth"
                if not os.path.exists(local_model_path):
                    os.system(f"wget -O {local_model_path} {url}")
                config.depth_estimator = local_model_path

            config.depth_estimator = local_model_path
            self.dav2_backbone.load_state_dict(torch.load(config.depth_estimator, map_location='cpu'))
            for p in self.dav2_backbone.parameters():
                p.requires_grad = False

        if hasattr(config, "image_seg") and "seg" in self.mode:
            if not os.path.exists(config.image_segmentor):
                config.image_segmentor = "oneformer/oneformer_coco_swin_large"
            self.oneformer_processor = OneFormerProcessor.from_pretrained(config.image_segmentor)
            self.oneformer = OneFormerHead.from_pretrained(config.image_segmentor)
            for p in self.oneformer.parameters():
                p.requires_grad = False
            try:
                self.oneformer = self.oneformer.to("cuda")
            except:
                pass
    
    def _get_layer_loss_weight(self, config, prefix):
        layer_indices = config[f"{prefix}_layer_indices"]
        layer_indices = layer_indices.split("-")
        layer_indices = [int(i) - 1 for i in layer_indices]
        loss_weight = config[f"{prefix}_loss_weight"]
        return layer_indices, loss_weight
    
    def init_heads(self, config):
        self.mode = getattr(config, "aux_mode", "gen-depth-seg")
        self.pass_text_to_aux_head = getattr(config, "pass_text_to_aux", True)
        self.use_ce = getattr(config, "use_ce", False)
        self.contrastive_loss_weight = config.contrastive_loss_weight
        num_task_tokens = config.num_task_tokens

        if hasattr(config, "image_gen") and "gen" in self.mode:            
            self.img_layer_indices, self.img_gen_loss_weight = self._get_layer_loss_weight(config.image_gen, "img")
            if getattr(config, "use_contrastive", True):
                self.gen_logit_scale = nn.Parameter(torch.tensor(2.0))
            else:
                self.gen_logit_scale = None
            
            self.image_gen_heads = nn.ModuleList([
                TaskTokenGenHead(config.image_gen, llm_hidden_size=config.hidden_size) if num_task_tokens > 0 else GenHead(proj_config=config.image_gen, llm_hidden_size=config.hidden_size)
                for _ in self.img_layer_indices
            ])

        if hasattr(config, "image_depth") and "depth" in self.mode:
            self.depth_layer_indices, self.img_depth_loss_weight = self._get_layer_loss_weight(config.image_depth, "depth")
            self.img_depth_loss_weight = config.image_depth["depth_loss_weight"]
            
            if getattr(config, "use_contrastive", True):
                self.depth_logit_scale = nn.Parameter(torch.tensor(2.0))
            else:
                self.depth_logit_scale = None

            self.use_intermediate_depth = config.image_depth.get("use_intermediate_depth", True)

            self.image_depth_heads = nn.ModuleList([
                TaskTokenDepthHead(proj_config=config.image_depth, llm_hidden_size=config.hidden_size, use_intermediate_depth=self.use_intermediate_depth) if num_task_tokens > 0 else DepthHead(proj_config=config.image_depth, llm_hidden_size=config.hidden_size, use_intermediate_depth=self.use_intermediate_depth)
                for _ in self.depth_layer_indices
            ])
            
            self.da_v2_head = DAv2_Head()

            if not os.path.exists(config.depth_estimator):
                url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
                local_model_path = "depth_anything_v2_vitl.pth"
                if not os.path.exists(local_model_path):
                    os.system(f"wget -O {local_model_path} {url}")
                config.depth_estimator = local_model_path

            self.da_v2_head.load_state_dict(torch.load(config.depth_estimator), strict=False)
            
            for p in self.da_v2_head.parameters():
                p.requires_grad = False

        if hasattr(config, "image_seg") and "seg" in self.mode:
            self.seg_layer_indices, self.img_seg_loss_weight = self._get_layer_loss_weight(config.image_seg, "seg")

            self.seg_teacher = config.image_seg.get("seg_teacher", "sam")

            assert self.seg_teacher in ["sam", "oneformer"]

            if getattr(config, "use_contrastive", True):
                self.seg_logit_scale = nn.Parameter(torch.tensor(2.0))
            else:
                self.seg_logit_scale = None

            self.image_seg_heads = nn.ModuleList([
                OneFormerTaskTokenSegHead(config.image_seg, llm_hidden_size=config.hidden_size) if num_task_tokens > 0 else OneFormerSegHead(config.image_seg, llm_hidden_size=config.hidden_size)
                for _ in self.seg_layer_indices
            ])
    

    def log_gen(self, img_embeds, pil_images, layer_idx, is_train=False):
        pipe = self.pipe.to("cuda")

        images = []

        for img_embed in img_embeds:
            image = pipe(image_embeds=img_embed.float().detach(),
                    num_inference_steps=25,
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

        n = len(pred_masks)
        c = min(n, 16)
        r = n // c
        pred_masks = pred_masks[:c*r]
        gt_masks = gt_masks[:c*r]
        masks_grid = make_grid(pred_masks, gt_masks)
        
        wandb.log({
            f"val_seg_images/step_{self.steps}": wandb.Image(masks_grid, caption=f"Layer-{layer_idx}")
        })
    

    def _emb_loss(self, emb_preds, emb_mask, emb_targets, logit_scale):
        emb_targets = emb_targets.to(emb_preds.dtype).to(emb_preds.device)

        if emb_targets.shape[0] != emb_preds.shape[0]:
            repeat_factor = emb_preds.shape[0] // emb_targets.shape[0]
            emb_targets = emb_targets.repeat(repeat_factor, 1, 1)
            emb_mask = emb_mask.repeat(repeat_factor, 1, 1)

            if emb_targets.shape[0] != emb_preds.shape[0]:
                emb_targets = emb_targets[:emb_preds.shape[0]]
                emb_mask = emb_mask[:emb_preds.shape[0]]

        if emb_preds.ndim == 3:
            emb_mask = emb_mask.view(emb_preds.shape[0], 1, 1)
        else:
            emb_mask = emb_mask.view(emb_preds.shape[0], 1, 1, 1)

        sl1_loss = F.smooth_l1_loss(
            emb_preds.float(), emb_targets.float(), reduction="none"
        )

        if logit_scale is not None:
            contrastive_loss = calculate_contrastive_loss(emb_preds, emb_targets, logit_scale)
        else:
            contrastive_loss = 0

        sl1_loss = (sl1_loss * emb_mask.float()).mean()
        contrastive_loss = (self.contrastive_loss_weight * contrastive_loss * emb_mask.float()).mean()
        
        emb_loss = sl1_loss + contrastive_loss

        return emb_loss, sl1_loss, contrastive_loss


    def _get_gen_feats(self, pil_images, device):
        gen_feats = []
        for img in pil_images:
            with torch.no_grad():
                clip_ims = self.pipe.feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)
                feat = self.pipe.image_encoder(clip_ims).image_embeds
                gen_feats.append(feat)

        gen_feats = torch.stack(gen_feats, dim=0)
        return gen_feats
    
    def _forward_gen(self, gen_preds, layer_index, pil_images, gen_mask, gen_targets):        
        gen_loss, gen_sl1_loss, gen_cont_loss = self._emb_loss(gen_preds, gen_mask, gen_targets, self.gen_logit_scale)

        if dist.get_rank() == 0:
            if self.steps % 4000 == 0:
                try:
                    self.log_gen(gen_preds.detach(), pil_images, layer_index, is_train=True)
                except:
                    pass
    
        return gen_loss, gen_cont_loss, gen_sl1_loss
    

    def _get_dav2_feats(self, pil_images, device):
        dav2_gts = []
        depth_targets = [[]]
        for img in pil_images:
            img = img.resize((336, 336))
            img = np.array(img)
            with torch.no_grad():
                feat = self.dav2_backbone.infer_image(img, is_dsg=True)
                ft_gt = (feat[0][0] + feat[1][0] + feat[2][0] + feat[3][0]) / 4
                depth_gts = self.da_v2_head([(ft_gt, None)] * 4)
                depth_targets[0].append(ft_gt)
            min_val = depth_gts.amin(dim=(1, 2), keepdim=True)
            max_val = depth_gts.amax(dim=(1, 2), keepdim=True)
            depth_gts = (depth_gts - min_val) / (max_val - min_val)
            dav2_gts.append(depth_gts.to(device))
        dav2_gts = torch.stack(dav2_gts, dim=0).squeeze(1)
        for i in range(len(depth_targets)):
            depth_targets[i] = (torch.stack(depth_targets[i], dim=0).squeeze(1), None)
        return depth_targets, dav2_gts
    
    def _forward_depth(self, all_depth_feats, layer_index, depth_mask, all_depth_targets, depth_pred_maps, depth_gts):                

        depth_feats, depth_targets = all_depth_feats[0][0], all_depth_targets[0][0]
        depth_loss, sl1_loss, cont_loss = self._emb_loss(depth_feats, depth_mask, depth_targets, self.depth_logit_scale)

        if dist.get_rank() == 0:
            if self.steps % 1000 == 0:
                try:
                    self.log_depth(depth_pred_maps.detach(), layer_index, depth_gts, is_train=True)
                except:
                    pass

        return depth_loss, sl1_loss, cont_loss


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

    def _forward_seg(self, seg_preds, layer_index, pil_images, seg_targets, seg_mask):
        
        seg_loss, sl1_loss, cont_loss = self._emb_loss(seg_preds, seg_mask, seg_targets, self.seg_logit_scale)

        if dist.get_rank() == 0:
            if self.steps % 1000 == 0:
                try:
                    self.log_seg(seg_preds.detach(), pil_images, layer_index, seg_targets, is_train=True)
                except:
                    pass
    
        return seg_loss, sl1_loss, cont_loss


    def forward_emb_predictor(self, layer_states, idx, i, task, heads, special_tokens):
        task_idx = self.token_order.index(task)
        task_start_idx =  self.NUM_SYS_TOKENS + 576 + (self.num_task_tokens * task_idx)
        task_end_idx = task_start_idx + self.num_task_tokens
        end_idx = self.NUM_SYS_TOKENS + 576 + (self.num_task_tokens * len(self.token_order))

        inp_tokens = layer_states[idx][:, :self.NUM_SYS_TOKENS+576]

        if self.num_task_tokens == 0 or layer_states[idx].shape[1] < 600:
            if self.pass_text_to_aux_head:
                inp_tokens = layer_states[idx]
        else:
            inp_tokens = torch.cat([inp_tokens, layer_states[idx][:, task_start_idx:task_end_idx]], dim=1)
            if self.pass_text_to_aux_head:
                inp_tokens = torch.cat([inp_tokens, layer_states[idx][:, end_idx:]], dim=1)
        
        if self.num_task_tokens == 0:
            task_emb = heads[i](inp_tokens)
        else:
            task_tokens = special_tokens
            if task != "gen":
                task_tokens = task_tokens.repeat(inp_tokens.shape[0], 1, 1)
            else:
                if not self.pass_text_to_aux_head:
                    task_tokens = inp_tokens[:, -self.num_task_tokens:]
                else:
                    task_tokens = inp_tokens[:, self.NUM_SYS_TOKENS+576:self.NUM_SYS_TOKENS+576+self.num_task_tokens]
                
            task_emb = heads[i](inp_tokens, task_tokens)

        return task_emb

    def depth_emb_forward(self, pil_images, layer_states, depth_mask):
        depth_preds = []
        depth_embs = []
        depth_loss = 0
        depth_l1_loss = 0
        depth_cont_loss = 0
        if "depth" in self.mode and layer_states[0].shape[1] > self.NUM_SYS_TOKENS: 
            if pil_images is not None:
                depth_targets, depth_gts = self._get_dav2_feats(pil_images, layer_states[0].device)
            else:
                depth_targets, depth_gts = None, None
            
            for i, idx in enumerate(self.depth_layer_indices):
                
                depth_feats = self.forward_emb_predictor(layer_states, idx, i, "depth", self.image_depth_heads, self.depth_tokens)
                depth_embs.append(depth_feats)

                with torch.no_grad():
                    if self.use_intermediate_depth:
                        depth_pred = self.da_v2_head(depth_feats)
                    else:
                        depth_pred = self.da_v2_head([depth_feats[0]] * 4)
                    min_val = depth_pred.amin(dim=(1, 2), keepdim=True)
                    max_val = depth_pred.amax(dim=(1, 2), keepdim=True)
                    depth_pred = (depth_pred - min_val) / (max_val - min_val)
                    depth_preds.append(depth_pred)

                if depth_mask is not None:
                    depth_mask.zero_()

                if depth_targets is not None:
                    layer_depth_loss, layer_l1_loss, layer_cont_loss = self._forward_depth(depth_feats, idx+1, depth_mask, depth_targets, depth_pred, depth_gts)
                    depth_loss += layer_depth_loss * self.img_depth_loss_weight
                    depth_l1_loss += layer_l1_loss * self.img_depth_loss_weight
                    depth_cont_loss += layer_cont_loss * self.img_depth_loss_weight
    
        return depth_preds, depth_embs, depth_loss, depth_l1_loss, depth_cont_loss
    
    def seg_emb_forward(self, pil_images, hidden_states, layer_states, seg_mask):
        seg_embs = []
        seg_loss = 0
        seg_l1_loss = 0
        seg_contrastive_loss = 0
        if "seg" in self.mode and layer_states[0].shape[1] > self.NUM_SYS_TOKENS:
            if pil_images is not None:
                seg_targets = self._get_seg_targets(pil_images, hidden_states)
            else:
                seg_targets = None
            for i, idx in enumerate(self.seg_layer_indices):

                seg_emb = self.forward_emb_predictor(layer_states, idx, i, "seg", self.image_seg_heads, self.seg_tokens)
                seg_embs.append(seg_emb)

                if seg_mask is not None:
                    seg_mask.zero_()

                if seg_targets is not None:
                    layer_seg_loss, seg_l1_loss, seg_contrastive_loss = self._forward_seg(seg_emb, idx+1, pil_images, seg_targets, seg_mask)
                    seg_loss += layer_seg_loss * self.img_seg_loss_weight
                    seg_l1_loss += seg_l1_loss * self.img_seg_loss_weight
                    seg_contrastive_loss += seg_contrastive_loss * self.img_seg_loss_weight
        
        return seg_embs, seg_loss, seg_l1_loss, seg_contrastive_loss
    
    def gen_emb_forward(self, pil_images, hidden_states, layer_states, gen_mask):
        img_embs = []
        gen_loss = 0
        gen_con_loss = 0
        gen_mse_loss = 0
        if "gen" in self.mode and layer_states[0].shape[1] > self.NUM_SYS_TOKENS:
            if pil_images is not None:
                gen_targets = self._get_gen_feats(pil_images, hidden_states.device)
            else:
                gen_targets = None
            
            for i, idx in enumerate(self.img_layer_indices):
                
                img_emb = self.forward_emb_predictor(layer_states, idx, i, "gen", self.image_gen_heads, self.gen_tokens)
                img_embs.append(img_emb)

                if gen_mask is not None:
                    gen_mask.zero_()

                if gen_targets is not None:
                    layer_gen_loss, layer_gen_con_loss, layer_gen_mse_loss = self._forward_gen(img_emb, idx+1, pil_images, gen_mask, gen_targets)
                    gen_loss += layer_gen_loss * self.img_gen_loss_weight
                    gen_con_loss += layer_gen_con_loss * self.img_gen_loss_weight
                    gen_mse_loss += layer_gen_mse_loss * self.img_gen_loss_weight
        
        return img_embs, gen_loss, gen_mse_loss, gen_con_loss
    

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
        
        depth_mask = kwargs.pop("seg_mask", None)
        gen_mask = kwargs.pop("seg_mask", None)
        seg_mask = kwargs.pop("seg_mask", None)
        
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if pil_images is not None:
            inputs['pil_images'] = pil_images
        if depth_mask is not None:
            inputs['depth_mask'] = depth_mask
        if gen_mask is not None:
            inputs['gen_mask'] = gen_mask
        if seg_mask is not None:
            inputs['seg_mask'] = seg_mask
        return inputs
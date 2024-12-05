import torch
import torch.nn as nn
from ola_vlm.model.multimodal_encoder.openclip_utils import create_model_from_pretrained
from open_clip.model import CLIPVisionCfg, CLIPTextCfg, _build_vision_tower
from timm.models.convnext import ConvNeXt
import torch
from torch import nn
import torch.nn.functional as F
from .base_encoder import BaseVisionTower, ProcessorWrapper
from typing import Optional

class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            drop_path: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        # Fix drop path during training
        if not drop_path:
            print('Not using drop path during training.')
            vision_cfg['timm_drop_path'] = 0.0

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

def extract_res_interp(model_name):
    valid_model_prefixes = {
        "CLIP-convnext_large":"hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
        "CLIP-convnext_xxlarge":"hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
    }


    res = None
    interp = None

    for prefix in valid_model_prefixes:
        if model_name.split("/")[-1].startswith(prefix):
            base_model_name = valid_model_prefixes[prefix]
            break
    else:
        raise ValueError(f"Unknown vision tower: {model_name}")

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])
    return base_model_name, res, interp


class CLIPConvNextVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        """
        Initialize the CLIPConvNextTower.

        Args:
            vision_tower (str): The name of the vision tower model in the format "clip-convnext-resXXX-interpYYY".
            args (argparse.Namespace): The arguments parsed from the command line.
            delay_load (bool, optional): Whether to delay loading the model. Defaults to False.
        """
        super().__init__(vision_tower, args, delay_load)

        self.is_multi_stage = "multi-stage" in vision_tower
        base_model_name, res, interp = extract_res_interp(vision_tower)
        self.vision_tower_name = base_model_name
        self.ckpt_path = vision_tower.split("-res")[0]
        self._image_size = res if res is not None else 768
        self._interp_size = interp
        self._reduction = 32

        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.unfreeze_mm_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.is_loaded = False

        if not delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            assert "CLIP-convnext_large" in vision_tower or "CLIP-convnext_xxlarge" in vision_tower
            if "CLIP-convnext_large" in vision_tower:
                if "multi-stage" in  vision_tower:
                    self._hidden_size = sum([192, 384, 768, 1536])
                else:
                    self._hidden_size = 1536
            else:
                if "multi-stage" in  vision_tower:
                    self._hidden_size = sum([384, 768, 1536, 3072])
                else:
                    self._hidden_size = 3072
    

    def load_model(self, device_map=None):
        """
        Load the CLIP-ConvNext model.
        """

        assert "clip-convnext" in self.vision_tower_name.lower()
        self.vision_model = "convnext"
        try:
            clip_model, processor = create_model_from_pretrained(self.vision_tower_name, load_ckpt=True)
        except:
            clip_model, processor = create_model_from_pretrained(self.vision_tower_name, load_ckpt=False)
        processor.transforms[0].size = self._image_size
        processor.transforms[1].size = (self._image_size, self._image_size)
        self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)
 
        self.vision_tower: ConvNeXt = clip_model.visual.trunk
        self.vision_tower.output_tokens = True
        feature_info = self.vision_tower.feature_info
        if self.is_multi_stage:
            self._hidden_size = sum([stage['num_chs'] for stage in feature_info])
        else:
            self._hidden_size = feature_info[-1]['num_chs']
        self.is_loaded = True

    def interpolate(self, image_forward_outs):
        """
        Interpolate the image features to the desired number of patches.

        Args:
            image_forward_outs (torch.Tensor): The output features from the vision tower.

        Returns:
            torch.Tensor: The interpolated image features.
        """
        if self._interp_size is None:
            return image_forward_outs

        image_features = F.interpolate(
            image_forward_outs.float(),
            size=(self.num_patches_per_side, self.num_patches_per_side),
            mode='bilinear',
            align_corners=False
        ).to(dtype=image_forward_outs.dtype)
        image_features = image_features.flatten(2, 3).permute(0, 2, 1).contiguous()
        return image_features

    def _forward(self, images):
        """
        Perform the forward pass of the CLIPConvNextTower.

        Args:
            images (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The output features from the vision tower after interpolation.
        """
        image_features_stages = []
        x = self.vision_tower.stem(images.to(device=self.device, dtype=self.dtype))
        for stage in self.vision_tower.stages:
            x = stage(x)
            image_features_stages.append(x)
        image_features = self.vision_tower.norm_pre(x).contiguous()
        # if not self.is_multi_stage:
        #     image_features_stages = image_features_stages[-1:]
        # image_features_stages_rescaled = []
        # for image_features_single_stage in image_features_stages:
        #     image_features_single_stage_rescaled = self.interpolate(image_features_single_stage)
        #     image_features_stages_rescaled.append(image_features_single_stage_rescaled)
        # image_features = torch.cat(image_features_stages_rescaled, -1)
        image_features = image_features.flatten(2, 3).permute(0, 2, 1).contiguous()
        return image_features

    @property
    def image_size(self):
        return self._image_size

    @property
    def num_patches_per_side(self):
        """
        Get the number of patches per side.

        Returns:
            int: The number of patches per side.
        """
        if self._interp_size is None:
            return self._image_size // self._reduction
        else:
            return int(self._interp_size ** 0.5)

    @property
    def num_patches(self):
        """
        Get the total number of patches.

        Default: 256

        Returns:
            int: The total number of patches.
        """
        if self._interp_size is None:
            return (self._image_size // self._reduction) ** 2
        else:
            return self._interp_size
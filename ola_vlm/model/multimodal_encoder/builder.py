import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .clip_convnext_encoder import CLIPConvNextVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if "clip" in vision_tower and "convnext" not in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "convnext" in vision_tower.lower():
        return CLIPConvNextVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "sam" in vision_tower.lower():
        return SAMVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

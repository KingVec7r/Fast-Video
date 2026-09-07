from .clip_encoder import CLIPVisionTower
from .mobileclip_encoder import MobileCLIPVisionTower

def build_vision_tower(config):
    vision_tower = config.mm_vision_tower
    if isinstance(vision_tower, str):
        vision_tower_name = vision_tower.lower()
    else:
        vision_tower_name = getattr(vision_tower, 'model_type', None)
        
    if "mobileclip" in vision_tower_name:
        return MobileCLIPVisionTower(vision_tower)
    
    if "fast" in vision_tower_name:
        return MobileCLIPVisionTower(vision_tower)

    if "clip" in vision_tower_name:
        raise NotImplementedError(f'CLIP vision tower is not yet implemented: {vision_tower}')
        return CLIPVisionTower(vision_tower)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

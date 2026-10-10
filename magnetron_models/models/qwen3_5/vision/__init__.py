# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""The Qwen3.5 vision tower.

The MoE checkpoints reuse this encoder unchanged, which is why it sits in the dense package rather
than being copied: `transformers` encodes the same relationship as `Qwen3_5MoeVisionModel(Qwen3_5VisionModel)`.
"""

from .config import VisionConfig, vision_config_from_checkpoint
from .image import PreprocessedImage, load_image, smart_resize
from .model import VisionModel
from .multimodal import (
    MRotaryEmbedding,
    apply_rope,
    decode_positions,
    mrope_axis_of_slot,
    prefill_positions,
    scatter_image_embeds,
    vision_placeholders,
)
from .rope import mrope_position_ids, vision_bilinear_pos_embed_index, vision_cu_seqlens, vision_position_ids

__all__ = [
    'VisionConfig',
    'vision_config_from_checkpoint',
    'PreprocessedImage',
    'VisionModel',
    'MRotaryEmbedding',
    'apply_rope',
    'decode_positions',
    'mrope_axis_of_slot',
    'prefill_positions',
    'scatter_image_embeds',
    'vision_placeholders',
    'load_image',
    'smart_resize',
    'mrope_position_ids',
    'vision_bilinear_pos_embed_index',
    'vision_cu_seqlens',
    'vision_position_ids',
]

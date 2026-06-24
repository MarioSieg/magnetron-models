# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from dataclasses import dataclass


class RopeParameters:
    rope_theta: float = 10000000
    partial_rotary_factor: float = 1.0
    mrope_section: list[int] = [11, 11, 10]


@dataclass
class TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_headers = 16
    num_key_value_heads = 2
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters = RopeParameters()
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 8
    num_experts: int = 256
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None


@dataclass
class VisionConfig:
    Depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02


class Config:
    text_config: TextConfig
    vision_config: VisionConfig
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

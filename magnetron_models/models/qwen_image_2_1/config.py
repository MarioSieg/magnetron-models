# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from dataclasses import dataclass

REPO_ID: str = 'Qwen/Qwen-Image-2.1'

ARCH_TEXT_ENCODER: str = 'qwen_image_2_1_text_encoder'
ARCH_TRANSFORMER: str = 'qwen_image_2_1_transformer'
ARCH_VAE: str = 'qwen_image_2_1_vae'


@dataclass
class TextEncoderConfig:
    repo_id: str = REPO_ID
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    image_token_id: int = 151655
    pad_token_id: int = 151643


@dataclass
class TransformerConfig:
    repo_id: str = REPO_ID
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    rope_theta: float = 10000.0
    timestep_dim: int = 256
    eps: float = 1e-6
    causal_condition: bool = True

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim


@dataclass
class SchedulerConfig:
    num_train_timesteps: int = 1000
    base_image_seq_len: int = 256
    max_image_seq_len: int = 8192
    base_shift: float = 0.5
    max_shift: float = 0.9
    shift_terminal: float = 0.02
    default_num_inference_steps: int = 40


_LATENTS_MEAN: tuple[float, ...] = (
    0.5126,
    0.7721,
    -0.0631,
    1.3506,
    -0.7855,
    -2.1025,
    -0.3458,
    1.3722,
    1.8873,
    -1.7177,
    -0.6510,
    0.2732,
    0.7562,
    -0.6163,
    -1.0277,
    3.8363,
    2.0210,
    0.0472,
    0.9320,
    2.0087,
    2.4954,
    -0.1391,
    -1.4249,
    1.8464,
    -0.5236,
    1.2826,
    3.7046,
    -1.3035,
    2.7286,
    -1.4518,
    -1.9036,
    -1.9955,
    -0.0342,
    -1.0265,
    -0.7636,
    3.0555,
    0.0746,
    -3.0751,
    -0.1076,
    1.7376,
    -1.0914,
    -1.9435,
    -0.2784,
    -1.3680,
    0.4809,
    -0.4433,
    0.3764,
    0.5729,
    -2.0595,
    1.0960,
    -1.3260,
    -2.0211,
    -5.0179,
    0.5275,
    4.0162,
    1.8505,
    0.3026,
    1.9373,
    1.4937,
    0.2632,
    0.5547,
    -1.7121,
    -0.1562,
    0.0304,
)
_LATENTS_STD: tuple[float, ...] = (
    3.2001,
    3.2936,
    3.4321,
    3.0091,
    3.1061,
    4.0379,
    4.0705,
    3.7910,
    3.0785,
    3.6500,
    3.9308,
    3.0904,
    2.8778,
    3.7675,
    3.7320,
    5.0756,
    3.2864,
    4.0397,
    3.1317,
    4.0443,
    2.9249,
    3.9454,
    3.0988,
    4.2489,
    3.4896,
    3.8513,
    3.9323,
    3.4719,
    3.7498,
    4.2830,
    3.5694,
    4.2467,
    3.9037,
    3.2947,
    5.0770,
    3.5075,
    3.2700,
    3.4767,
    2.8063,
    5.1125,
    3.5327,
    4.7833,
    3.1286,
    4.1819,
    3.8527,
    3.8312,
    3.5605,
    4.3875,
    3.9624,
    4.0168,
    3.5643,
    4.0550,
    5.5614,
    4.2963,
    4.4080,
    3.4959,
    3.8747,
    3.7608,
    3.5735,
    3.1490,
    3.7662,
    3.6746,
    3.4563,
    3.8161,
)


@dataclass
class VAEConfig:
    repo_id: str = REPO_ID
    z_dim: int = 64
    decoder_base_dim: int = 144
    dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8)
    num_res_blocks: int = 2
    out_channels: int = 4
    scale_factor_spatial: int = 16
    temporal_downsample: tuple[bool, ...] = (False, True, True, True)
    latents_mean: tuple[float, ...] = _LATENTS_MEAN
    latents_std: tuple[float, ...] = _LATENTS_STD

    @property
    def temporal_upsample(self) -> tuple[bool, ...]:
        return tuple(reversed(self.temporal_downsample))

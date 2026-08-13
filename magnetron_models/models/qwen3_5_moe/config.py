# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from enum import Enum, unique
from dataclasses import dataclass


@unique
class SamplingStrategy(Enum):
    GREEDY = 'greedy'
    TOPK = 'topk'


@unique
class LayerType(Enum):
    LINEAR_ATTENTION = 'linear_attention'
    FULL_ATTENTION = 'full_attention'


@dataclass
class Config:
    repo_id: str = 'Qwen/Qwen3.5-35B-A3B'
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    max_position_embeddings: int = 8192  # 262144
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    delta_chunk_size: int = 64
    moe_intermediate_size: int = 512  # Per-expert MLP width.
    shared_expert_intermediate_size: int = 512
    num_experts: int = 256
    num_experts_per_tok: int = 8
    bos_token_id: int = 248044  # <|endoftext|>
    eos_token_id: int = 248046  # <|im_end|>
    stop_token_ids: frozenset[int] = frozenset({248044, 248046})
    enable_thinking: bool = False
    sampling_strategy: SamplingStrategy = SamplingStrategy.GREEDY

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def linear_value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def linear_conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

    def layer_type(self, layer_idx: int) -> LayerType:
        return LayerType.LINEAR_ATTENTION if (layer_idx + 1) % self.full_attention_interval else LayerType.FULL_ATTENTION

    @property
    def layer_types(self) -> list[LayerType]:
        return [self.layer_type(i) for i in range(self.num_hidden_layers)]


CONFIGS: dict[str, Config] = {
    'Qwen/Qwen3.5-35B-A3B': Config(),
    'Qwen/Qwen3.5-122B-A10B': Config(
        repo_id='Qwen/Qwen3.5-122B-A10B',
        hidden_size=3072,
        num_hidden_layers=48,
        num_attention_heads=32,
        linear_num_value_heads=64,
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=1024,
    ),
    'Qwen/Qwen3.5-397B-A17B': Config(
        repo_id='Qwen/Qwen3.5-397B-A17B',
        hidden_size=4096,
        num_hidden_layers=60,
        num_attention_heads=32,
        linear_num_value_heads=64,
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=1024,
        num_experts=512,
        num_experts_per_tok=10,
    ),
}

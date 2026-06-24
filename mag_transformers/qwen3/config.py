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


@dataclass
class Config:
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 8192  # 262144
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_theta: float = 5_000_000.0
    sliding_window: int | None = None
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    sampling_strategy: SamplingStrategy = SamplingStrategy.GREEDY

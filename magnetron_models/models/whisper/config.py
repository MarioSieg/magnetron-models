# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class Config:
    repo_id: str = 'openai/whisper-base'
    vocab_size: int = 51865
    num_mel_bins: int = 80
    d_model: int = 512
    encoder_layers: int = 6
    encoder_attention_heads: int = 8
    encoder_ffn_dim: int = 2048
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_ffn_dim: int = 2048
    max_source_positions: int = 1500
    max_target_positions: int = 448

    @classmethod
    def from_hf_config(cls, repo_id: str, raw: dict[str, Any]) -> Config:
        return cls(repo_id=repo_id, **{f.name: raw[f.name] for f in fields(cls) if f.name != 'repo_id' and f.name in raw})

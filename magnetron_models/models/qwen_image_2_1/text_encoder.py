# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from magnetron import Tensor, nn
from magnetron_models.models import SnapshotModule
from magnetron_models.models.qwen3.config import Config as Qwen3Config
from magnetron_models.models.qwen3.model import Block, _precompute_freq_cache
from .config import TextEncoderConfig


class QwenImageTextEncoder(SnapshotModule):
    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        block_cfg = Qwen3Config(
            repo_id=cfg.repo_id,
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            max_position_embeddings=cfg.max_position_embeddings,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            sliding_window=None,
        )
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, weight_init=nn.init.EmptyInitStrategy())
        self.layers = nn.ModuleList([Block(block_cfg) for _ in range(cfg.num_hidden_layers)])
        self.cos_cache, self.sin_cache = _precompute_freq_cache(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)

    def forward(self, input_ids: Tensor) -> Tensor:
        B, T = input_ids.shape
        if T > self.cfg.max_position_embeddings:
            raise ValueError(f'Prompt is {T} tokens, the encoder was built for at most {self.cfg.max_position_embeddings}')
        idx = Tensor.arange(stop=T).reshape(1, T).expand(B, T)
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, self.cos_cache, self.sin_cache, idx, cache=None)
        return h

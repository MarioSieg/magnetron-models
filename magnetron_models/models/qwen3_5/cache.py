# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from magnetron import Tensor, dtype
from magnetron_models.kvcache import KVLayerCache
from .config import Config, LayerType


class LinearLayerCache:
    def __init__(self, cfg: Config, batch_size: int = 1) -> None:
        self.conv: Tensor = Tensor.zeros(batch_size, cfg.linear_conv_dim, cfg.linear_conv_kernel_dim, dtype=dtype.float32)
        self.state: Tensor = Tensor.zeros(
            batch_size,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            dtype=dtype.float32,
        )
        self.primed: bool = False

    def clear(self) -> None:
        self.conv.zero_()
        self.state.zero_()
        self.primed = False


class HybridCache:
    def __init__(self, cfg: Config, batch_size: int = 1) -> None:
        self.layers: list[KVLayerCache | LinearLayerCache] = [
            KVLayerCache(
                batch_size=batch_size,
                num_kv_heads=cfg.num_key_value_heads,
                max_seq_len=cfg.max_position_embeddings,
                head_dim=cfg.head_dim,
            )
            if cfg.layer_type(i) is LayerType.FULL_ATTENTION
            else LinearLayerCache(cfg, batch_size=batch_size)
            for i in range(cfg.num_hidden_layers)
        ]
        self.cache_pos: int = 0

    def __getitem__(self, idx: int) -> KVLayerCache | LinearLayerCache:
        return self.layers[idx]

    def advance(self, num_tokens: int) -> None:
        self.cache_pos += num_tokens

    def clear(self) -> None:
        for layer in self.layers:
            layer.clear()
        self.cache_pos = 0

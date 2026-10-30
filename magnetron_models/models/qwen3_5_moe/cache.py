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
    """Recurrent state of a linear attention layer. Each step replaces conv and state with new tensors instead of writing them in place, so a checkpoint is just the old references."""

    def __init__(self, cfg: Config, batch_size: int = 1) -> None:
        self.conv: Tensor = Tensor.zeros(batch_size, cfg.linear_conv_dim, cfg.linear_conv_kernel_dim, dtype=dtype.float32)
        self.state: Tensor = Tensor.zeros(
            batch_size, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim, dtype=dtype.float32
        )
        self.primed: bool = False

    def clear(self) -> None:  # Fresh tensors rather than zero_(): a checkpoint may still reference the old ones
        self.conv = Tensor.zeros(*self.conv.shape, dtype=dtype.float32)
        self.state = Tensor.zeros(*self.state.shape, dtype=dtype.float32)
        self.primed = False


class HybridCache:
    def __init__(self, cfg: Config, batch_size: int = 1) -> None:
        self.layers: list[KVLayerCache | LinearLayerCache] = [
            KVLayerCache(batch_size=batch_size, num_kv_heads=cfg.num_key_value_heads, max_seq_len=cfg.max_position_embeddings, head_dim=cfg.head_dim)
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

    def checkpoint(self) -> tuple[int, list[int | tuple[Tensor, Tensor, bool]]]:
        return self.cache_pos, [layer.pos if isinstance(layer, KVLayerCache) else (layer.conv, layer.state, layer.primed) for layer in self.layers]

    def restore(self, checkpoint: tuple[int, list[int | tuple[Tensor, Tensor, bool]]]) -> None:
        self.cache_pos, layers = checkpoint
        for layer, saved in zip(self.layers, layers, strict=True):
            if isinstance(layer, KVLayerCache):
                layer.pos = saved  # Entries past pos are overwritten by the next append
            else:
                layer.conv, layer.state, layer.primed = saved

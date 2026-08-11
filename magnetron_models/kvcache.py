# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from magnetron import Tensor


class KVLayerCache:
    def __init__(self, batch_size: int, num_kv_heads: int, max_seq_len: int, head_dim: int) -> None:
        self.k: Tensor = Tensor.zeros(batch_size, num_kv_heads, max_seq_len, head_dim)
        self.v: Tensor = Tensor.zeros(batch_size, num_kv_heads, max_seq_len, head_dim)
        self.pos: int = 0

    def append(self, curr_k: Tensor, curr_v: Tensor, sliding_window: int | None = None) -> tuple[Tensor, Tensor]:
        T: int = curr_k.shape[2]
        start: int = self.pos
        end: int = start + T
        if end > self.k.shape[2]:
            raise RuntimeError(f'KV cache overflow: {end} > {self.k.shape[2]}')
        self.k[:, :, start:end, :] = curr_k
        self.v[:, :, start:end, :] = curr_v
        self.pos = end
        view_start: int = 0
        if sliding_window is not None:
            view_start = max(end - sliding_window, 0)
        return self.k[:, :, view_start:end, :], self.v[:, :, view_start:end, :]

    def clear(self) -> None:
        self.pos = 0


class KVCache:
    def __init__(self, num_key_value_heads: int, num_hidden_layers: int, max_seq_len: int, head_dim: int, batch_size: int = 1) -> None:
        self.layers: list[KVLayerCache] = [
            KVLayerCache(batch_size=batch_size, num_kv_heads=num_key_value_heads, max_seq_len=max_seq_len, head_dim=head_dim)
            for _ in range(num_hidden_layers)
        ]
        assert all(layer.pos == self.cache_pos for layer in self.layers)

    def __getitem__(self, idx: int) -> KVLayerCache:
        return self.layers[idx]

    @property
    def cache_pos(self) -> int:
        return self.layers[0].pos

    def clear(self) -> None:
        for layer in self.layers:
            layer.clear()

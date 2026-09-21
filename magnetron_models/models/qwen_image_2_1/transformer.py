# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg               |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import math

from magnetron import Tensor, nn, dtype
from magnetron_models.models import SnapshotModule
from .config import TransformerConfig

_EMPTY = nn.init.EmptyInitStrategy()


def _linear(in_features: int, out_features: int) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=False, weight_init=_EMPTY, bias_init=_EMPTY)


def _layer_norm(x: Tensor, eps: float) -> Tensor:
    xf = x.cast(dtype.float32)
    xm = xf - xf.mean(dim=-1, keepdim=True)
    return (xm * (xm.sqr().mean(dim=-1, keepdim=True) + eps).rsqrt()).cast(x.dtype)


class HeadRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(Tensor.empty(dim))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.cast(dtype.float32)
        h = xf * (xf.sqr().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return h.cast(self.weight.dtype) * self.weight


class ZeroCenterRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(Tensor.empty(dim))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.cast(dtype.float32)
        h = xf * (xf.sqr().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return (h * (self.weight.cast(dtype.float32) + 1.0)).cast(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_in_dim, eps)
        self.in_layer = _linear(context_in_dim, hidden_size)
        self.out_layer = _linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.in_layer(self.text_norm(x)).gelu_approx())


class TimestepEmbedder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = _linear(in_channels, embedding_dim)
        self.linear_2 = _linear(embedding_dim, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.linear_1(x).silu())


class TimeTextEmbed(nn.Module):
    def __init__(self, timestep_dim: int, embedding_dim: int, max_period: float = 10000.0, time_factor: float = 1000.0) -> None:
        super().__init__()
        if timestep_dim % 2:
            raise ValueError(f'timestep_dim must be even, got {timestep_dim}')
        half: int = timestep_dim // 2
        self.time_factor = time_factor
        self.freqs = (Tensor.arange(stop=half, dtype=dtype.float32) * (-math.log(max_period) / half)).exp().reshape(1, half)
        self.timestep_embedder = TimestepEmbedder(timestep_dim, embedding_dim)

    def forward(self, timestep: Tensor) -> Tensor:
        args = (timestep.cast(dtype.float32) * self.time_factor).reshape(-1, 1) * self.freqs
        emb = Tensor.cat([args.cos(), args.sin()], dim=-1).cast(self.timestep_embedder.linear_1.weight.dtype)
        return self.timestep_embedder(emb)


class Modulation(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = _linear(dim, 4 * dim)

    def forward(self, temb: Tensor) -> Tensor:
        return self.linear(temb.silu())


class AdaLayerNormOut(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.linear = _linear(dim, dim)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        scale = self.linear(temb.silu())
        return _layer_norm(x, self.eps) * (1.0 + scale.reshape(scale.shape[0], 1, scale.shape[1]))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int) -> None:
        super().__init__()
        self.proj = _linear(hidden_size, mlp_hidden_size)
        self.out = _linear(mlp_hidden_size, hidden_size)
        self.gate_layer = _linear(hidden_size, mlp_hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.gate_layer(x).silu() * self.proj(x))


class TransformerKVCache:
    def __init__(self, num_layers: int) -> None:
        self.k: list[Tensor | None] = [None] * num_layers
        self.v: list[Tensor | None] = [None] * num_layers

    def store(self, layer: int, k: Tensor, v: Tensor) -> None:
        self.k[layer] = k
        self.v[layer] = v

    def get(self, layer: int) -> tuple[Tensor, Tensor]:
        k, v = self.k[layer], self.v[layer]
        if k is None or v is None:
            raise RuntimeError(f'KV cache of block {layer} was never filled, run a prefill step first')
        return k, v

    def clear(self) -> None:
        for i in range(len(self.k)):
            self.k[i] = self.v[i] = None


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    B, S, H, D = x.shape
    xf = x.cast(dtype.float32).reshape(B, S, H, D // 2, 2)
    xr = xf[..., 0]
    xi = xf[..., 1]
    c = cos.reshape(1, S, 1, D // 2)
    s = sin.reshape(1, S, 1, D // 2)
    out = Tensor.stack([xr * c - xi * s, xr * s + xi * c], dim=4)
    return out.reshape(B, S, H, D).cast(x.dtype)


def _sdpa(q: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
    scores = (q * (1.0 / math.sqrt(q.shape[-1]))) @ k.transpose(2, 3)
    if causal:
        q_len, k_len = q.shape[2], k.shape[2]
        k_pos = Tensor.arange(stop=k_len).reshape(1, -1)
        q_pos = Tensor.arange(start=k_len - q_len, stop=k_len).reshape(-1, 1)
        mask = Tensor.where(k_pos <= q_pos, 0.0, -1e4).cast(scores.dtype).reshape(1, 1, q_len, k_len)
        scores = scores + mask
    return scores.softmax(dim=-1) @ v


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = _linear(dim, heads * dim_head)
        self.to_k = _linear(dim, heads * dim_head)
        self.to_v = _linear(dim, heads * dim_head)
        self.to_out = _linear(heads * dim_head, dim)
        self.norm_q = HeadRMSNorm(dim_head, eps)
        self.norm_k = HeadRMSNorm(dim_head, eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        text_len: int | None,
        layer_cache: tuple[TransformerKVCache, int] | None,
        cache_mode: str | None,
    ) -> Tensor:
        B, S, _ = x.shape
        H, D = self.heads, self.dim_head
        q = _apply_rope(self.norm_q(self.to_q(x).reshape(B, S, H, D)), cos, sin).transpose(1, 2)
        k = _apply_rope(self.norm_k(self.to_k(x).reshape(B, S, H, D)), cos, sin).transpose(1, 2)
        v = self.to_v(x).reshape(B, S, H, D).transpose(1, 2)
        if cache_mode == 'cached':
            if layer_cache is None:
                raise ValueError('cache_mode="cached" needs a cache')
            cache, layer = layer_cache
            k_txt, v_txt = cache.get(layer)
            out = _sdpa(q, Tensor.cat([k_txt, k], dim=2), Tensor.cat([v_txt, v], dim=2), causal=False)
        else:
            if text_len is None:
                raise ValueError('A prefill needs the text length to split the joint sequence')
            T: int = text_len
            k_txt, v_txt = k[:, :, :T].contiguous(), v[:, :, :T].contiguous()
            if cache_mode == 'extract':
                if layer_cache is None:
                    raise ValueError('cache_mode="extract" needs a cache to fill')
                cache, layer = layer_cache
                cache.store(layer, k_txt, v_txt)
            out_txt = _sdpa(q[:, :, :T].contiguous(), k_txt, v_txt, causal=True)
            out_img = _sdpa(q[:, :, T:].contiguous(), k, v, causal=False)
            out = Tensor.cat([out_txt, out_img], dim=2)
        return self.to_out(out.transpose(1, 2).reshape(B, S, H * D))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        dim: int = cfg.inner_dim
        self.eps = cfg.eps
        self.attn = Attention(dim, cfg.num_attention_heads, cfg.attention_head_dim, cfg.eps)
        self.img_mlp = SwiGLU(dim, dim * cfg.mlp_ratio)

    def forward(
        self,
        x: Tensor,
        rows: tuple[Tensor, Tensor, Tensor, Tensor],
        cos: Tensor,
        sin: Tensor,
        text_len: int | None,
        layer_cache: tuple[TransformerKVCache, int] | None,
        cache_mode: str | None,
    ) -> Tensor:
        scale1, gate1, scale2, gate2 = rows
        h = _layer_norm(x, self.eps) * (1.0 + scale1)
        x = x + gate1.tanh() * self.attn(h, cos, sin, text_len, layer_cache, cache_mode)
        h = _layer_norm(x, self.eps) * (1.0 + scale2)
        return x + gate2.tanh() * self.img_mlp(h)


class QwenImageTransformer(SnapshotModule):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim: int = cfg.inner_dim
        self.time_text_embed = TimeTextEmbed(cfg.timestep_dim, dim)
        self.txt_in = TextProjection(cfg.context_in_dim, dim, cfg.eps)
        self.img_in = _linear(cfg.in_channels, dim)
        self.modulation = Modulation(dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm_out = AdaLayerNormOut(dim, cfg.eps)
        self.proj_out = _linear(dim, cfg.out_channels)
        self._inv_freqs: list[Tensor] = [
            (cfg.rope_theta ** -(Tensor.arange(0, d, 2, dtype=dtype.float32) / d)).reshape(1, d // 2) for d in cfg.axes_dims_rope
        ]
        self._rope_cache: dict[tuple[int, int, int], tuple[Tensor, Tensor]] = {}

    def new_kv_cache(self) -> TransformerKVCache:
        return TransformerKVCache(self.cfg.num_layers)

    def rope_tables(self, text_len: int, height: int, width: int) -> tuple[Tensor, Tensor]:
        key = (text_len, height, width)
        if key in self._rope_cache:
            return self._rope_cache[key]
        text = list(range(text_len))
        rows = [y - (height - height // 2) for y in range(height) for _ in range(width)]
        cols = [x - (width - width // 2) for _ in range(height) for x in range(width)]
        axes = (text + [text_len] * (height * width), text + rows, text + cols)
        angles = [Tensor([float(p) for p in pos], dtype=dtype.float32).reshape(-1, 1) * inv_freq for pos, inv_freq in zip(axes, self._inv_freqs)]
        freqs = Tensor.cat(angles, dim=-1)
        tables = (freqs.cos(), freqs.sin())
        self._rope_cache[key] = tables
        return tables

    def _modulation_rows(self, params: Tensor, batch_size: int, text_len: int, image_len: int, prefill: bool) -> Tensor:
        B, C = batch_size, params.shape[-1]
        real = params[:B].contiguous().reshape(B, 1, C)
        if not self.cfg.causal_condition or not prefill:
            return real
        zero = params[B:].contiguous().reshape(1, 1, C)
        return Tensor.cat([zero.expand(B, text_len, C), real.expand(B, image_len, C)], dim=1)

    def forward(
        self,
        latents: Tensor,
        text: Tensor,
        timestep: Tensor,
        height: int,
        width: int,
        kv_cache: TransformerKVCache | None = None,
        cache_mode: str | None = None,
    ) -> Tensor:
        B, HW, _ = latents.shape
        T: int = text.shape[1]
        if HW != height * width:
            raise ValueError(f'latents carry {HW} tokens but the grid is {height}x{width}')
        if cache_mode not in (None, 'extract', 'cached'):
            raise ValueError(f'cache_mode must be None, "extract" or "cached", got {cache_mode!r}')
        if cache_mode is not None and kv_cache is None:
            raise ValueError(f'cache_mode={cache_mode!r} needs a kv_cache')
        if kv_cache is not None and not self.cfg.causal_condition:
            raise ValueError('The KV cache is only exact with causal_condition, which makes the text activations timestep independent')
        prefill: bool = cache_mode != 'cached'

        if self.cfg.causal_condition:
            timestep = Tensor.cat([timestep, Tensor.zeros(1, dtype=timestep.dtype)], dim=0)
        temb = self.time_text_embed(timestep)
        C: int = self.cfg.inner_dim
        rows = tuple(self._modulation_rows(p, B, T, HW, prefill) for p in self.modulation(temb).split(C, dim=-1))

        cos, sin = self.rope_tables(T, height, width)
        img = self.img_in(latents)
        if prefill:
            x = Tensor.cat([self.txt_in(text), img], dim=1)
        else:
            x = img
            cos, sin = cos[T:].contiguous(), sin[T:].contiguous()
        for i, block in enumerate(self.transformer_blocks):
            layer_cache = (kv_cache, i) if kv_cache is not None else None
            x = block(x, rows, cos, sin, T if prefill else None, layer_cache, cache_mode)
        if prefill:
            x = x[:, T:].contiguous()
        return self.proj_out(self.norm_out(x, temb[:B].contiguous()))

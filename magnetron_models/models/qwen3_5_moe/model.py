# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import math
from collections.abc import Iterator
from typing import override

from magnetron import Tensor, nn, dtype, context
from magnetron_models.kvcache import KVLayerCache
from magnetron_models.tokenizer import TokenizerBase
from magnetron_models.models import ModelBase
from .cache import HybridCache, LinearLayerCache
from .config import Config, LayerType, SamplingStrategy

_EMPTY = nn.init.EmptyInitStrategy()


def _linear(in_features: int, out_features: int) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=False, weight_init=_EMPTY, bias_init=_EMPTY)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(Tensor.empty(dim))

    def forward(self, x: Tensor) -> Tensor:
        in_dt = x.dtype
        h = x.cast(dtype.float32)
        h = h * (h.sqr().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return (h * (1.0 + self.weight.cast(dtype.float32))).cast(in_dt)


class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(Tensor.empty(dim))

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        in_dt = x.dtype
        h = x.cast(dtype.float32)
        h = h * (h.sqr().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        h = self.weight * h.cast(in_dt)
        return (h.cast(dtype.float32) * gate.cast(dtype.float32).silu()).cast(in_dt)


class MLP(nn.Module):
    def __init__(self, cfg: Config, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = _linear(cfg.hidden_size, intermediate_size)
        self.up_proj = _linear(cfg.hidden_size, intermediate_size)
        self.down_proj = _linear(intermediate_size, cfg.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


class TopKRouter(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.top_k: int = cfg.num_experts_per_tok
        self.weight = nn.Parameter(Tensor.empty(cfg.num_experts, cfg.hidden_size))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = Tensor.einsum('th, eh -> te', x, self.weight)
        probs = logits.cast(dtype.float32).softmax(dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1, largest=True, sorted=True)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.cast(x.dtype), indices


class Experts(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.intermediate_size: int = cfg.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(Tensor.empty(cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size))
        self.down_proj = nn.Parameter(Tensor.empty(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size))

    def forward(self, x: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
        top_k: int = top_k_index.shape[-1]
        buckets: dict[int, tuple[list[int], list[int]]] = {}
        for token, experts in enumerate(top_k_index.reshape(-1, top_k).tolist()):
            for slot, expert in enumerate(experts):
                rows, slots = buckets.setdefault(expert, ([], []))
                rows.append(token)
                slots.append(token * top_k + slot)
        weights = top_k_weights.reshape(-1)
        out = Tensor.zeros_like(x)
        for expert, (rows, slots) in sorted(buckets.items()):
            row_idx = Tensor(rows, dtype=dtype.int64)
            h = x[row_idx] @ self.gate_up_proj[expert].transpose(-1, -2)
            gate, up = h[:, : self.intermediate_size], h[:, self.intermediate_size :]
            h = (gate.silu() * up) @ self.down_proj[expert].transpose(-1, -2)
            out.index_add_(0, row_idx, h * weights[Tensor(slots, dtype=dtype.int64)].reshape(-1, 1))
        return out


class SparseMoeBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.gate = TopKRouter(cfg)
        self.experts = Experts(cfg)
        self.shared_expert = MLP(cfg, cfg.shared_expert_intermediate_size)
        self.shared_expert_gate = _linear(cfg.hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        shared = self.shared_expert(flat)
        weights, indices = self.gate(flat)
        out = self.experts(flat, indices, weights)
        return (out + self.shared_expert_gate(flat).sigmoid() * shared).reshape(B, T, H)


def _softplus(x: Tensor) -> Tensor:
    return x.relu() + x.abs().neg().exp().log1p()  # TODO: add clamp cuda kernel


def _l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return x * ((x**2).sum(dim=dim, keepdim=True) + eps).rsqrt()


class DepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(Tensor.empty(channels, 1, kernel_size))

    def forward(self, x: Tensor) -> Tensor:  # TODO: use actual conv from magnetron when we added them (conv1d, conv2d etc9
        weight = self.weight.reshape(self.channels, self.kernel_size)
        out_len: int = x.shape[-1] - self.kernel_size + 1
        acc: Tensor | None = None
        for k in range(self.kernel_size):
            term = x[:, :, k : k + out_len] * weight[:, k].reshape(1, self.channels, 1)
            acc = term if acc is None else acc + term
        return acc


def _precompute_freq_cache(dim: int, theta: float, max_seq_len: int) -> tuple[Tensor, Tensor]:
    inv_freq = (theta ** -(Tensor.arange(0, dim, 2, dtype=dtype.float32) / dim)).reshape(1, -1)
    freqs = Tensor.arange(stop=max_seq_len, dtype=dtype.float32).reshape(max_seq_len, 1) * inv_freq
    cos_half = freqs.cos()
    sin_half = freqs.sin()
    cos = Tensor.cat([cos_half, cos_half], dim=-1).cast(context.get_default_dtype())
    sin = Tensor.cat([sin_half, sin_half], dim=-1).cast(context.get_default_dtype())
    return cos, sin


def _apply_rope(q: Tensor, k: Tensor, freq_cos: Tensor, freq_sin: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:

    def _rot_half(x: Tensor) -> Tensor:
        half: int = x.shape[-1] >> 1
        return Tensor.cat([-x[:, :, :, half:], x[:, :, :, :half]], dim=-1)

    cos = freq_cos[idx]
    sin = freq_sin[idx]
    batch_size, seq_len, rotary_dim = cos.shape
    cos = cos.reshape(batch_size, 1, seq_len, rotary_dim)
    sin = sin.reshape(batch_size, 1, seq_len, rotary_dim)
    q_rot, q_pass = q[:, :, :, :rotary_dim], q[:, :, :, rotary_dim:]
    k_rot, k_pass = k[:, :, :, :rotary_dim], k[:, :, :, rotary_dim:]
    q_embed = Tensor.cat([(q_rot * cos) + (_rot_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = Tensor.cat([(k_rot * cos) + (_rot_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


def _chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = False,
    output_final_state: bool = False,
) -> tuple[Tensor, Tensor | None]:
    init_dt = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().cast(dtype.float32) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim: int = value.shape[-1]
    pad_size: int = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = query.pad((0, 0, 0, pad_size))
    key = key.pad((0, 0, 0, pad_size))
    value = value.pad((0, 0, 0, pad_size))
    beta = beta.pad((0, pad_size))
    g = g.pad((0, pad_size))
    total_sequence_length: int = sequence_length + pad_size
    query *= 1 / (query.shape[-1] ** 0.5)
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    g = g.cusum(dim=-1)
    mask = Tensor.ones(chunk_size, chunk_size).cast(dtype.boolean).triu(diagonal=0)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().cast(dtype.float32)).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):  # Forward substitution: invert the unit lower triangular matrix.
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn += Tensor.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cudecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        Tensor.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype) if initial_state is None else initial_state.cast(value.dtype)
    )
    core_attn_out = Tensor.zeros_like(value)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cudecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length].transpose(1, 2).contiguous().cast(init_dt)
    return core_attn_out, last_recurrent_state


def _recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    output_final_state: bool = False,
) -> tuple[Tensor, Tensor | None]:
    init_dt = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().cast(dtype.float32) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim: int = value.shape[-1]
    query *= 1 / (query.shape[-1] ** 0.5)
    core_attn_out = Tensor.zeros(batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype)
    last_recurrent_state = (
        Tensor.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype) if initial_state is None else initial_state.cast(value.dtype)
    )
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().cast(init_dt)
    return core_attn_out, last_recurrent_state


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_k_heads: int = cfg.linear_num_key_heads
        self.num_v_heads: int = cfg.linear_num_value_heads
        self.head_k_dim: int = cfg.linear_key_head_dim
        self.head_v_dim: int = cfg.linear_value_head_dim
        self.key_dim: int = cfg.linear_key_dim
        self.value_dim: int = cfg.linear_value_dim
        self.conv_dim: int = cfg.linear_conv_dim
        self.conv_kernel_size: int = cfg.linear_conv_kernel_dim
        self.n_rep: int = self.num_v_heads // self.num_k_heads
        self.conv1d = DepthwiseConv1d(self.conv_dim, self.conv_kernel_size)
        self.dt_bias = nn.Parameter(Tensor.empty(self.num_v_heads, dtype=dtype.float32))
        self.A_log = nn.Parameter(Tensor.empty(self.num_v_heads, dtype=dtype.float32))
        self.norm = RMSNormGated(self.head_v_dim, eps=cfg.rms_norm_eps)
        self.in_proj_qkv = _linear(cfg.hidden_size, self.conv_dim)
        self.in_proj_z = _linear(cfg.hidden_size, self.value_dim)
        self.in_proj_b = _linear(cfg.hidden_size, self.num_v_heads)
        self.in_proj_a = _linear(cfg.hidden_size, self.num_v_heads)
        self.out_proj = _linear(self.value_dim, cfg.hidden_size)

    def _conv(self, mixed_qkv: Tensor, cache: LinearLayerCache | None) -> Tensor:
        seq_len: int = mixed_qkv.shape[-1]
        kernel: int = self.conv_kernel_size
        if cache is not None and cache.primed:
            padded = Tensor.cat([cache.conv.cast(mixed_qkv.dtype), mixed_qkv], dim=-1)
        else:
            padded = mixed_qkv.pad((kernel - 1, 0))
        if cache is not None:
            state = mixed_qkv if seq_len >= kernel else Tensor.cat([cache.conv.cast(mixed_qkv.dtype), mixed_qkv], dim=-1)
            cache.conv.copy_(state[:, :, -kernel:].cast(dtype.float32))
        return self.conv1d(padded)[:, :, -seq_len:].silu()

    def forward(self, x: Tensor, cache: LinearLayerCache | None = None) -> Tensor:
        B, T, _ = x.shape
        mixed_qkv = self._conv(self.in_proj_qkv(x).transpose(1, 2), cache).transpose(1, 2)
        query = mixed_qkv[:, :, : self.key_dim].reshape(B, T, -1, self.head_k_dim)
        key = mixed_qkv[:, :, self.key_dim : 2 * self.key_dim].reshape(B, T, -1, self.head_k_dim)
        value = mixed_qkv[:, :, 2 * self.key_dim :].reshape(B, T, -1, self.head_v_dim)
        z = self.in_proj_z(x).reshape(B, T, -1, self.head_v_dim)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.exp() * _softplus(self.in_proj_a(x).cast(dtype.float32) + self.dt_bias)
        if self.n_rep > 1:
            query = query.repeat_interleave(self.n_rep, dim=2)
            key = key.repeat_interleave(self.n_rep, dim=2)
        initial_state = cache.state if cache is not None and cache.primed else None
        rule = _recurrent_gated_delta_rule if initial_state is not None and T == 1 else _chunk_gated_delta_rule
        kwargs = {} if rule is _recurrent_gated_delta_rule else {'chunk_size': self.cfg.delta_chunk_size}
        core_attn_out, last_state = rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=True,
            output_final_state=cache is not None,
            **kwargs,
        )
        if cache is not None:
            cache.state.copy_(last_state.cast(dtype.float32))
            cache.primed = True
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core_attn_out.reshape(B, T, -1))


class GatedAttention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.head_dim: int = cfg.head_dim
        self.num_heads: int = cfg.num_attention_heads
        self.num_kv_heads: int = cfg.num_key_value_heads
        self.n_rep: int = self.num_heads // self.num_kv_heads
        self.q_proj = _linear(cfg.hidden_size, self.num_heads * self.head_dim * 2)  # query and output gate
        self.k_proj = _linear(cfg.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = _linear(cfg.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = _linear(self.num_heads * self.head_dim, cfg.hidden_size)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x: Tensor, cos_freq: Tensor, sin_freq: Tensor, idx: Tensor, cache: KVLayerCache | None = None) -> Tensor:
        B, T, _ = x.shape
        qg = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim * 2)
        q = self.q_norm(qg[:, :, :, : self.head_dim]).transpose(1, 2)
        gate = qg[:, :, :, self.head_dim :].reshape(B, T, -1)
        curr_k = self.k_norm(self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        curr_v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, curr_k = _apply_rope(q, curr_k, cos_freq, sin_freq, idx)
        if cache is not None:
            k, v = cache.append(curr_k, curr_v)
        else:
            k, v = curr_k, curr_v
        qg_heads = q.reshape(B, self.num_kv_heads, self.n_rep, T, self.head_dim)
        scores = Tensor.einsum('bkgtd, bksd -> bkgts', qg_heads, k)
        scores *= 1.0 / math.sqrt(self.head_dim)
        q_len: int = q.shape[2]
        k_len: int = k.shape[2]
        if q_len == 1:
            attn = scores.softmax(dim=-1)
        else:
            k_pos_indices = Tensor.arange(k_len).reshape(1, -1)
            q_pos_indices = Tensor.arange(start=(k_len - q_len), stop=k_len).reshape(-1, 1)
            mask = Tensor.where(k_pos_indices <= q_pos_indices, 0.0, -1e4)
            mask = mask.cast(scores.dtype).reshape(1, 1, 1, q_len, k_len)
            attn = (scores + mask).softmax(dim=-1)
        out = Tensor.einsum('bkgts, bksd -> bkgtd', attn, v)
        out = out.reshape(B, self.num_heads, T, self.head_dim)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out * gate.sigmoid())


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = cfg.layer_type(layer_idx)
        if self.layer_type is LayerType.LINEAR_ATTENTION:
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            self.self_attn = GatedAttention(cfg)
        self.mlp = SparseMoeBlock(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        freq_cos: Tensor,
        freq_sin: Tensor,
        idx: Tensor,
        cache: KVLayerCache | LinearLayerCache | None = None,
    ) -> Tensor:
        normed = self.input_layernorm(x)
        if self.layer_type is LayerType.LINEAR_ATTENTION:
            h = x + self.linear_attn(normed, cache)
        else:
            h = x + self.self_attn(normed, freq_cos, freq_sin, idx, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen35MoeModel(ModelBase):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, weight_init=_EMPTY)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = None if cfg.tie_word_embeddings else _linear(cfg.hidden_size, cfg.vocab_size)
        cos_cache, sin_cache = _precompute_freq_cache(cfg.rotary_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        self.cache = HybridCache(cfg)

    def forward(self, x: Tensor, idx: Tensor) -> Tensor:
        h = self.embed_tokens(x)
        for i, layer in enumerate(self.layers):
            h = layer(h, self.cos_cache, self.sin_cache, idx, cache=self.cache[i] if self.cache is not None else None)
        h = self.norm(h)
        if self.cache is not None:
            self.cache.advance(x.shape[1])
        return Tensor.einsum('...h, vh -> ...v', h, self.embed_tokens.weight) if self.cfg.tie_word_embeddings else self.lm_head(h)

    @override
    def generate_stream(
        self,
        idx: Tensor,
        tokenizer: TokenizerBase,
        max_tokens: int,
        temp: float = 1.0,
        top_k: int = 10,
        reset_cache: bool = False,
    ) -> Iterator[str]:
        def sample(logits: Tensor, strategy: SamplingStrategy) -> int:  # Sample according to strategy
            match strategy:
                case SamplingStrategy.GREEDY:
                    return int(logits.argmax(dim=0).item())
                case SamplingStrategy.TOPK:
                    top_vals, top_idx = logits.topk(top_k, dim=0, largest=True, sorted=False)
                    return int(top_idx[top_vals.softmax(dim=-1).reshape(1, -1).multinomial(num_samples=1)[0, 0]].item())
                case _:
                    raise RuntimeError(f'Invalid sampling strategy: {strategy}')

        if reset_cache:
            self.cache.clear()
        idx = idx.reshape(1, -1)
        start_pos: int = self.cache.cache_pos
        T: int = idx.shape[1]
        logits = self(idx, idx=Tensor.arange(start=start_pos, stop=start_pos + T).reshape(1, -1))
        next_logits = logits[:, -1, :] / temp
        curr_len: int = start_pos + T
        pending: list[int] = []
        for _ in range(max_tokens):
            tok_id: int = sample(next_logits.reshape(-1), self.cfg.sampling_strategy)
            if tok_id == self.cfg.eos_token_id or tok_id in self.cfg.stop_token_ids:
                return
            pending.append(tok_id)
            delta: str = tokenizer.decode(pending)
            if delta and '�' not in delta:
                yield delta
                pending.clear()
            input_ids = Tensor([tok_id], dtype=dtype.int64).reshape(1, 1)
            logits = self(input_ids, idx=Tensor([curr_len], dtype=dtype.int64).reshape(1, 1))
            next_logits = logits[:, -1, :] / temp
            curr_len += 1

    def _assistant_header(self) -> str:
        return '<|im_start|>assistant\n<think>\n' if self.cfg.enable_thinking else '<|im_start|>assistant\n<think>\n\n</think>\n\n'

    @override
    def build_system(self, system: str) -> str:
        instructions: str = self.cfg.reasoning_instructions
        return f'<|im_start|>system\n{f"{instructions}\n\n" if instructions else ""}{system}<|im_end|>\n'

    @override
    def build_prompt(self, system: str, messages: list[tuple[str, str]]) -> str:
        out = [self.build_system(system)]
        for role, content in messages:
            out.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
        out.append(self._assistant_header())
        return ''.join(out)

    @override
    def build_user_turn(self, user: str) -> str:
        return f'<|im_start|>user\n{user}<|im_end|>\n{self._assistant_header()}'

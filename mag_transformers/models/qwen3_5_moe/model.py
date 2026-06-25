# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from magnetron import nn, context, dtype, Tensor
from mag_transformers.models.qwen3_5_moe.config import TextConfig, Config

if context.is_device_available('cuda:0'):
    context.set_default_device('cuda:0')


class VisionRotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.inv_freq = 1.0 / (theta ** (Tensor.arange(0, dim, 2).cast(dtype.float32) / dim))

    def forward(self, position_ids: Tensor) -> Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class TextRotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, cfg: TextConfig) -> None:
        super().__init__()
        self.cfg: TextConfig = cfg
        self.max_seq_len_cached: int = cfg.max_position_embeddings
        self.original_max_seq_len: int = cfg.max_position_embeddings
        self.inv_freq, self.attention_scaling = self._compute_default_rope_params(self.cfg)
        print(self.inv_freq)

    def _compute_default_rope_params(self, cfg: TextConfig) -> tuple[Tensor, float]:
        base = cfg.rope_parameters.rope_theta
        partial_rotary = cfg.rope_parameters.partial_rotary_factor
        dim = int(cfg.head_dim * partial_rotary)
        attn_factor = 1.0  # Unused in this rope
        inv_freq = 1.0 / (base ** (Tensor.arange(0, dim, 2).cast(dtype.float32) / dim))
        return inv_freq, attn_factor

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        if position_ids.rank == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].cast(dtype.float32).expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].cast(dtype.float32)
        freqs = self._apply_interleaved_mrope((inv_freq_expanded @ position_ids_expanded).transpose(2, 3))
        emb = Tensor.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.cast(x.dtype), sin.cast(x.dtype)

    def _apply_interleaved_mrope(self, freqs: Tensor) -> Tensor:
        sections: list[int] = self.cfg.rope_parameters.mrope_section
        freqs_t = freqs[0]
        for dim, offs in enumerate((1, 2), start=1):
            len = sections[dim] * 3
            idx = slice(offs, len, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor.ones(hidden_size))
        self.variance_eps = eps

    def forward(self, hidden_states: Tensor, gate: Tensor) -> Tensor:
        in_type = hidden_states.dtype
        hidden_states = hidden_states.cast(dtype.float32)
        variance = (hidden_states**2).mean(-1, keepdim=True)
        hidden_states *= (variance + self.variance_eps).rsqrt()
        hidden_states = self.weight * hidden_states.cast(in_type)
        hidden_states *= gate.cast(dtype.float32).silu()
        return hidden_states.cast(in_type)


def _causal_conv1d_update(hidden_states: Tensor, conv_state: Tensor, weight: Tensor, bias: Tensor | None, act: Tensor | None) -> Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = Tensor.cat((conv_state, hidden_states), dim=-1).cast(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    return hidden_states_new.conv1d(weight.unsqueeze(1), bias, padding=0, groups=hidden_size)[:, :, -seq_len:].silu().cast(hidden_states.dtype)


def _l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return x * ((x**2).sum(dim=dim, keepdim=True) + eps).rsqrt()


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
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = query.pad((0, 0, 0, pad_size))
    key = key.pad((0, 0, 0, pad_size))
    value = value.pad((0, 0, 0, pad_size))
    beta = beta.pad((0, pad_size))
    g = g.pad((0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query *= scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    g = g.cusum(dim=-1)
    mask = Tensor.ones(chunk_size, chunk_size).cast(dtype.boolean).triu(diagonal=0)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().cast(dtype.float32)).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
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
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query *= scale
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
        last_recurrent_state *= g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().cast(init_dt)
    return core_attn_out, last_recurrent_state


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = cfg.text_config.hidden_size
        self.num_v_heads = cfg.text_config.linear_num_value_heads


# Quick shape smoketest :§
B = 1
T = 128
H = 4
K = 32
V = 32
CHUNK = 64
query = Tensor.normal(B, T, H, K).cast(dtype.bfloat16)
key = Tensor.normal(B, T, H, K).cast(dtype.bfloat16)
value = Tensor.normal(B, T, H, V).cast(dtype.bfloat16)
beta = Tensor.normal(B, T, H).sigmoid().cast(dtype.bfloat16)
g = Tensor.normal(B, T, H).cast(dtype.bfloat16)
out, final_state = _chunk_gated_delta_rule(
    query=query,
    key=key,
    value=value,
    g=g,
    beta=beta,
    initial_state=None,
    chunk_size=CHUNK,
    use_qk_l2norm_in_kernel=False,
    output_final_state=True,
)
print('out:', out.shape, out.dtype)
print(out)
print('final_state:', None if final_state is None else (final_state.shape, final_state.dtype))

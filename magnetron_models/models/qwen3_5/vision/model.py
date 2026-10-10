# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""The Qwen3.5 vision encoder.

Parameter names mirror the checkpoint's `model.visual.*` subtree so the converter is a straight rename.
Attention runs unfused, one segment at a time, walking the query axis in blocks: correctness first, the
fused path can replace `_attend` once magnetron grows a variable-length kernel.
"""

from magnetron import Tensor, context, dtype, nn

from .config import VisionConfig
from .image import PreprocessedImage
from .rope import Grid, vision_bilinear_pos_embed_index, vision_cu_seqlens, vision_position_ids

_EMPTY = nn.init.EmptyInitStrategy()


def _linear(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=bias, weight_init=_EMPTY, bias_init=_EMPTY)


class LayerNorm(nn.Module):
    """LayerNorm that normalizes in float32, matching torch's accumulation for half-precision inputs."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(Tensor.empty(dim))
        self.bias = nn.Parameter(Tensor.empty(dim))

    def forward(self, x: Tensor) -> Tensor:
        in_dt = x.dtype
        h = x.cast(dtype.float32)
        h = h - h.mean(dim=-1, keepdim=True)
        h = h * (h.sqr().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return (h * self.weight.cast(dtype.float32) + self.bias.cast(dtype.float32)).cast(in_dt)


class VisionMLP(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = _linear(cfg.hidden_size, cfg.intermediate_size)
        self.linear_fc2 = _linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_fc2(self.linear_fc1(x).gelu_approx())  # hidden_act is gelu_pytorch_tanh.


class PatchEmbed(nn.Module):
    """The reference Conv3d has stride equal to its kernel, so it is a matmul over already flattened patches."""

    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.proj = _linear(cfg.patch_dim, cfg.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class PatchMerger(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.merged_size = cfg.merger_hidden_size
        self.norm = LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)  # Pre-shuffle: normalizes single patches.
        self.linear_fc1 = _linear(cfg.merger_hidden_size, cfg.merger_hidden_size)
        self.linear_fc2 = _linear(cfg.merger_hidden_size, cfg.out_hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x).reshape(-1, self.merged_size)
        return self.linear_fc2(self.linear_fc1(h).gelu())  # nn.GELU() in the reference, so the exact erf form.


def _rotate_half(x: Tensor) -> Tensor:
    half: int = x.shape[-1] >> 1
    return Tensor.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate (tokens, heads, head_dim) by per-token angles, in float32 like the reference."""
    in_dt = x.dtype
    h = x.cast(dtype.float32)
    return (h * cos + _rotate_half(h) * sin).cast(in_dt)


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scaling = cfg.head_dim**-0.5
        self.cfg = cfg
        self.qkv = _linear(cfg.hidden_size, cfg.hidden_size * 3)
        self.proj = _linear(cfg.hidden_size, cfg.hidden_size)

    def _attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Full attention over one segment, (heads, len, dim) in and (len, heads, dim) out."""
        seq_len: int = q.shape[1]
        block: int = self.cfg.query_block(seq_len)
        chunks: list[Tensor] = []
        for start in range(0, seq_len, block):
            length: int = min(block, seq_len - start)
            scores = (q.narrow(1, start, length) @ k.transpose(-1, -2)) * self.scaling
            attn = scores.cast(dtype.float32).softmax(dim=-1).cast(v.dtype)
            chunks.append((attn @ v).transpose(0, 1))  # (len, heads, dim)
        return chunks[0] if len(chunks) == 1 else Tensor.cat(chunks, dim=0)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cu_seqlens: list[int]) -> Tensor:
        seq_len: int = x.shape[0]
        qkv = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q = _apply_rope(qkv.select(1, 0), cos, sin).transpose(0, 1).contiguous()  # (heads, tokens, dim)
        k = _apply_rope(qkv.select(1, 1), cos, sin).transpose(0, 1).contiguous()
        v = qkv.select(1, 2).transpose(0, 1).contiguous()
        # One segment per frame: attention never crosses an image or frame boundary.
        outputs = [
            self._attend(
                q.narrow(1, start, end - start),
                k.narrow(1, start, end - start),
                v.narrow(1, start, end - start),
            )
            for start, end in zip(cu_seqlens, cu_seqlens[1:])
        ]
        out = outputs[0] if len(outputs) == 1 else Tensor.cat(outputs, dim=0)
        return self.proj(out.reshape(seq_len, -1))


class VisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.norm1 = LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.norm2 = LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cu_seqlens: list[int]) -> Tensor:
        h = x + self.attn(self.norm1(x), cos, sin, cu_seqlens)
        return h + self.mlp(self.norm2(h))


class VisionModel(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size, weight_init=_EMPTY)
        self.blocks = nn.ModuleList([VisionBlock(cfg) for _ in range(cfg.depth)])
        self.merger = PatchMerger(cfg)

    def _rotary(self, grids: list[Grid]) -> tuple[Tensor, Tensor]:
        """cos/sin of shape (tokens, 1, head_dim), the row and column angles concatenated then duplicated."""
        cfg = self.cfg
        half: int = cfg.rotary_dim // 2
        inv_freq = (cfg.rope_theta ** -(Tensor.arange(0, cfg.rotary_dim, 2, dtype=dtype.float32) / cfg.rotary_dim)).reshape(1, 1, half)
        position_ids = vision_position_ids(grids, cfg.spatial_merge_size)
        flat = [float(v) for pair in position_ids for v in pair]
        positions = Tensor(flat, dtype=dtype.float32).reshape(len(position_ids), 2, 1)
        freqs = (positions * inv_freq).reshape(len(position_ids), cfg.rotary_dim)  # (tokens, 2 * half)
        emb = Tensor.cat([freqs, freqs], dim=-1)
        return emb.cos().reshape(-1, 1, cfg.head_dim), emb.sin().reshape(-1, 1, cfg.head_dim)

    def _position_embeddings(self, grids: list[Grid]) -> Tensor:
        """Bilinear resample of the learned grid, done as four table lookups and a weighted sum."""
        cfg = self.cfg
        indices, weights = vision_bilinear_pos_embed_index(grids, cfg.num_grid_per_side, cfg.spatial_merge_size)
        count: int = len(indices[0])
        index = Tensor([v for corner in indices for v in corner], dtype=dtype.int64).reshape(4, count)
        weight = Tensor([v for corner in weights for v in corner], dtype=dtype.float32).reshape(4, count, 1)
        corners = self.pos_embed(index).cast(dtype.float32)  # (4, tokens, hidden)
        return (corners * weight).sum(dim=0)

    def forward(self, pixel_values: Tensor, grids: list[Grid]) -> Tensor:
        """Encode packed patches into LLM-side tokens, (total_patches, patch_dim) -> (total_tokens, out_hidden_size)."""
        h = self.patch_embed(pixel_values)
        h = h + self._position_embeddings(grids).cast(h.dtype)
        cos, sin = self._rotary(grids)
        cu_seqlens = vision_cu_seqlens(grids)
        for block in self.blocks:
            h = block(h, cos, sin, cu_seqlens)
        return self.merger(h)

    def encode(self, images: list[PreprocessedImage]) -> Tensor:
        """Run every image in one packed batch and return their tokens concatenated in prompt order."""
        if not images:
            raise ValueError('No images to encode')
        pixels = images[0].pixel_values if len(images) == 1 else Tensor.cat([img.pixel_values for img in images], dim=0)
        return self.forward(pixels.cast(context.get_default_dtype()), [img.grid_thw for img in images])

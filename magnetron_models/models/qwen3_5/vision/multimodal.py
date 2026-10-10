# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Everything a Qwen3.5 text tower needs to accept image tokens.

The dense and MoE towers differ only in their feed-forward block, so the rotary embedding, the
placeholder splice and the position bookkeeping are identical and live here rather than in both.
"""

from magnetron import Tensor, context, dtype

from .config import VisionConfig
from .image import PreprocessedImage
from .rope import Grid, mrope_position_ids


class MRotaryEmbedding:
    """Interleaved 3D M-RoPE.

    Text and vision tokens share one rotary space, but a vision token carries a (t, h, w) triple instead of a
    single offset. Each frequency slot is driven by exactly one of those axes, in the THWTHW... order the
    checkpoint expects. A text-only prompt feeds the same value on all three axes, which collapses this back
    to plain RoPE, so there is only one code path.
    """

    def __init__(self, rotary_dim: int, rope_theta: float, mrope_section: tuple[int, int, int]) -> None:
        if sum(mrope_section) * 2 != rotary_dim:
            raise ValueError(f'mrope_section {mrope_section} does not tile the rotary dim {rotary_dim}')
        self.rotary_dim = rotary_dim
        half: int = rotary_dim // 2
        self.inv_freq = (rope_theta ** -(Tensor.arange(0, rotary_dim, 2, dtype=dtype.float32) / rotary_dim)).reshape(1, 1, 1, half)
        # One-hot over the (t, h, w) axis per slot: selecting by multiply-and-sum keeps this to ops magnetron has.
        axes = mrope_axis_of_slot(rotary_dim, mrope_section)
        selector = [1.0 if axes[slot] == axis else 0.0 for axis in range(3) for slot in range(half)]
        self.selector = Tensor(selector, dtype=dtype.float32).reshape(1, 1, 3, half)

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """(3, B, T) positions -> cos/sin of shape (B, 1, T, rotary_dim)."""
        _, batch_size, seq_len = positions.shape
        pos = positions.permute(1, 2, 0).contiguous().reshape(batch_size, seq_len, 3, 1)
        freqs = ((pos * self.inv_freq) * self.selector).sum(dim=2)  # (B, T, half)
        emb = Tensor.cat([freqs, freqs], dim=-1)
        out_dt = context.get_default_dtype()
        cos = emb.cos().cast(out_dt).reshape(batch_size, 1, seq_len, self.rotary_dim)
        sin = emb.sin().cast(out_dt).reshape(batch_size, 1, seq_len, self.rotary_dim)
        return cos, sin


def mrope_axis_of_slot(rotary_dim: int, mrope_section: tuple[int, int, int]) -> list[int]:
    """Which of the (t, h, w) axes drives each rotary frequency slot.

    Qwen3.5 interleaves the M-RoPE sections instead of chunking them, so the layout is THWTHW... with the
    leftover tail falling back to t. Reproduces `apply_interleaved_mrope` as a static permutation.
    """
    axes: list[int] = [0] * (rotary_dim // 2)
    for axis, offset in ((1, 1), (2, 2)):
        for slot in range(offset, mrope_section[axis] * 3, 3):
            axes[slot] = axis
    return axes


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate the leading `cos.shape[-1]` channels of each head and pass the rest through untouched."""

    def _rot_half(x: Tensor) -> Tensor:
        half: int = x.shape[-1] >> 1
        return Tensor.cat([-x[:, :, :, half:], x[:, :, :, :half]], dim=-1)

    rotary_dim: int = cos.shape[-1]
    q_rot, q_pass = q[:, :, :, :rotary_dim], q[:, :, :, rotary_dim:]
    k_rot, k_pass = k[:, :, :, :rotary_dim], k[:, :, :, rotary_dim:]
    q_embed = Tensor.cat([(q_rot * cos) + (_rot_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = Tensor.cat([(k_rot * cos) + (_rot_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


def vision_placeholders(images: list[PreprocessedImage]) -> str:
    """The `<|vision_start|><|image_pad|>...<|vision_end|>` run each image expands into."""
    return ''.join('<|vision_start|>' + '<|image_pad|>' * img.num_tokens + '<|vision_end|>' for img in images)


def scatter_image_embeds(h: Tensor, image_embeds: Tensor, image_rows: Tensor) -> Tensor:
    """Overwrite the placeholder rows of (1, T, hidden) with vision tokens.

    magnetron has no masked_scatter and assigning through a fancy index is a no-op, so the placeholder rows
    are zeroed by a mask and the vision tokens are added back in.
    """
    batch_size, seq_len, hidden = h.shape
    if batch_size != 1:
        raise NotImplementedError('Multimodal prefill is single-batch only')
    flat = h.reshape(seq_len, hidden)
    keep = Tensor.ones(seq_len, 1, dtype=flat.dtype)
    keep.index_add_(0, image_rows, Tensor.ones(image_rows.shape[0], 1, dtype=flat.dtype), -1.0)
    flat = flat * keep
    flat.index_add_(0, image_rows, image_embeds.cast(flat.dtype))
    return flat.reshape(batch_size, seq_len, hidden)


def prefill_positions(
    token_ids: list[int],
    images: list[PreprocessedImage],
    *,
    vision: VisionConfig | None,
    image_token_id: int,
    video_token_id: int,
    start: int,
) -> tuple[Tensor, Tensor | None, int]:
    """M-RoPE positions, the rows the vision tokens belong in, and the next free rotary position.

    Text advances all three axes together. A vision span lays its tokens out on a (t, h, w) lattice and
    then advances the shared cursor by only `max(h, w) // merge_size`, so the returned cursor is what
    generation has to continue from, not the token count.
    """
    modality = [1 if tok == image_token_id else 2 if tok == video_token_id else 0 for tok in token_ids]
    if 2 in modality:
        raise NotImplementedError('Video input is not supported yet, the prompt contains <|video_pad|> tokens')
    grids: dict[int, list[Grid]] = {1: [img.grid_thw for img in images]}
    merge_size: int = vision.spatial_merge_size if vision is not None else 1
    axes, next_position = mrope_position_ids(token_ids, modality, grids, merge_size, start=start)
    positions = Tensor([float(v) for axis in axes for v in axis], dtype=dtype.float32).reshape(3, 1, len(token_ids))
    if not images:
        return positions, None, next_position
    rows = Tensor([i for i, kind in enumerate(modality) if kind != 0], dtype=dtype.int64)
    return positions, rows, next_position


def decode_positions(position: int) -> Tensor:
    """A generated token sits at one shared position on all three axes."""
    value = float(position)
    return Tensor([value, value, value], dtype=dtype.float32).reshape(3, 1, 1)

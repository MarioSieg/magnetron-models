# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import json
import os
from dataclasses import dataclass, replace

# What the checkpoints allow: 16777216 pixels, a 4096x4096 image, 65536 patches. The tower runs unfused
# attention, so the score matrix at that size is enormous and the run is very long, but attn_score_budget
# keeps peak memory flat and it does finish. Lower it with --max_pixels when the wait matters more.
CHECKPOINT_MAX_PIXELS: int = 16777216
DEFAULT_MAX_PIXELS: int = CHECKPOINT_MAX_PIXELS


@dataclass
class VisionConfig:
    """Qwen3.5 vision encoder, `vision_config` in the checkpoint's config.json.

    Unlike Qwen3-VL this tower has neither deepstack nor window attention, so a plain
    prenorm transformer over patch embeddings is the whole model.
    """

    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 2048  # Must equal the text tower's hidden_size, the merger projects straight into it.
    num_position_embeddings: int = 2304  # 48x48 learned grid, bilinearly resampled to the actual patch grid.
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    # Preprocessing, from preprocessor_config.json. Every Qwen3.5 checkpoint ships the same values.
    min_pixels: int = 65536
    max_pixels: int = DEFAULT_MAX_PIXELS
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    rescale_factor: float = 1.0 / 255.0
    # Attention is materialized in full, so long sequences are walked in query blocks to bound peak memory.
    # The block is the smaller of a fixed row count and whatever keeps one score matrix inside the budget,
    # so a 4096x4096 image costs the same peak as a small one, just far more blocks.
    attn_query_chunk: int = 256
    # Score elements per block. magnetron's allocator holds on to roughly 160x the largest allocation in a
    # loop, so peak RSS lands near 160x this budget's float32 size: 4M elements is ~16 MiB, so ~2.5 GiB peak.
    attn_score_budget: int = 1 << 22

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def rotary_dim(self) -> int:
        """The vision rotary embedding covers half the head, split again between the row and column axis."""
        return self.head_dim // 2

    @property
    def num_grid_per_side(self) -> int:
        return int(self.num_position_embeddings**0.5)

    @property
    def patch_dim(self) -> int:
        """Flattened patch width, the Conv3d kernel folded into a single matmul."""
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size * self.spatial_merge_size

    @property
    def merger_hidden_size(self) -> int:
        return self.hidden_size * self.merge_unit

    def query_block(self, seq_len: int) -> int:
        """Rows of queries to attend at once for a segment of `seq_len` patches."""
        return max(1, min(self.attn_query_chunk, self.attn_score_budget // max(1, self.num_heads * seq_len)))

    @property
    def resize_factor(self) -> int:
        """smart_resize keeps both sides divisible by this, so the merge blocks always tile evenly."""
        return self.patch_size * self.spatial_merge_size


VISION_CONFIGS: dict[str, VisionConfig] = {
    'Qwen/Qwen3.5-35B-A3B': VisionConfig(out_hidden_size=2048),
    'Qwen/Qwen3.5-122B-A10B': VisionConfig(out_hidden_size=3072),
    'Qwen/Qwen3.5-397B-A17B': VisionConfig(out_hidden_size=4096),
}


def vision_config_from_checkpoint(repo_dir: str, out_hidden_size: int, max_pixels: int | None = None) -> VisionConfig | None:
    """Build a VisionConfig from a downloaded checkpoint, or None if it is text only.

    Reads `vision_config` for the tower and `preprocessor_config.json` for the image pipeline, so a
    checkpoint whose size is not in CONFIGS still converts without a code change.
    """
    path = os.path.join(repo_dir, 'config.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    vision = raw.get('vision_config')
    if vision is None:
        return None
    if vision.get('deepstack_visual_indexes'):  # Qwen3-VL feeds intermediate blocks back into the text tower.
        raise ValueError(f'Unsupported deepstack vision tower: {vision["deepstack_visual_indexes"]}')
    cfg = VisionConfig(
        depth=vision.get('depth', 27),
        hidden_size=vision.get('hidden_size', 1152),
        intermediate_size=vision.get('intermediate_size', 4304),
        num_heads=vision.get('num_heads', 16),
        in_channels=vision.get('in_channels', 3),
        patch_size=vision.get('patch_size', 16),
        temporal_patch_size=vision.get('temporal_patch_size', 2),
        spatial_merge_size=vision.get('spatial_merge_size', 2),
        out_hidden_size=out_hidden_size,
        num_position_embeddings=vision.get('num_position_embeddings', 2304),
    )
    prep_path = os.path.join(repo_dir, 'preprocessor_config.json')
    if not os.path.exists(prep_path):
        return cfg
    with open(prep_path, encoding='utf-8') as f:
        prep = json.load(f)
    size = prep.get('size', {})
    return replace(
        cfg,
        patch_size=prep.get('patch_size', cfg.patch_size),
        temporal_patch_size=prep.get('temporal_patch_size', cfg.temporal_patch_size),
        spatial_merge_size=prep.get('merge_size', cfg.spatial_merge_size),
        min_pixels=size.get('shortest_edge', cfg.min_pixels),
        # The checkpoint's ceiling is far above what the unfused tower can chew through, so it is opt-in.
        max_pixels=max_pixels if max_pixels is not None else cfg.max_pixels,
        image_mean=tuple(prep.get('image_mean', cfg.image_mean)),
        image_std=tuple(prep.get('image_std', cfg.image_std)),
    )

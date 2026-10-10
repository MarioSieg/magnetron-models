# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Index math for the vision tower and for multimodal RoPE.

Everything here is pure Python on small integer grids, computed once per prompt. It mirrors
`transformers.vision_utils` and `Qwen3_5Model.get_rope_index` exactly, including the merge-block
reordering, which decides which patch lands in which row and must not drift.
"""

from itertools import groupby

Grid = tuple[int, int, int]


def _linspace(start: float, stop: float, num: int) -> list[float]:
    """torch.linspace: `num` points inclusive of both ends, a single point degenerates to `start`."""
    if num == 1:
        return [start]
    step: float = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def _merge_block_order(height: int, width: int, merge_size: int) -> list[tuple[int, int]]:
    """(row, col) of every patch in the order the tower consumes them.

    The image processor emits patches grouped by spatial merge block, so the four patches the merger
    folds into one LLM token are adjacent. Reproduces `reshape(h//m, m, w//m, m).transpose(1, 2)`.
    """
    return [
        (block_row * merge_size + row, block_col * merge_size + col)
        for block_row in range(height // merge_size)
        for block_col in range(width // merge_size)
        for row in range(merge_size)
        for col in range(merge_size)
    ]


def vision_cu_seqlens(grids: list[Grid]) -> list[int]:
    """Cumulative attention segment boundaries. One segment per frame, so frames never attend to each other."""
    bounds: list[int] = [0]
    for grid_t, grid_h, grid_w in grids:
        for _ in range(grid_t):
            bounds.append(bounds[-1] + grid_h * grid_w)
    return bounds


def vision_position_ids(grids: list[Grid], merge_size: int) -> list[list[int]]:
    """Per-patch (row, col) feeding the vision rotary embedding, flattened over every frame."""
    out: list[list[int]] = []
    for grid_t, grid_h, grid_w in grids:
        frame = _merge_block_order(grid_h, grid_w, merge_size)
        for _ in range(grid_t):
            out.extend([row, col] for row, col in frame)
    return out


def vision_bilinear_pos_embed_index(grids: list[Grid], num_grid_per_side: int, merge_size: int) -> tuple[list[list[int]], list[list[float]]]:
    """Bilinear resampling of the learned position grid onto the actual patch grid.

    Returns the four corner indices and their weights, each shaped (4, total_patches). The tower gathers
    `pos_embed` four times and blends, which is how the reference avoids a real interpolation kernel.
    """
    corner_indices: list[list[int]] = [[], [], [], []]
    corner_weights: list[list[float]] = [[], [], [], []]
    side: int = num_grid_per_side
    for grid_t, grid_h, grid_w in grids:
        row_grid = _linspace(0.0, side - 1, grid_h)
        col_grid = _linspace(0.0, side - 1, grid_w)
        row_floor = [int(v) for v in row_grid]
        col_floor = [int(v) for v in col_grid]
        row_ceil = [min(v + 1, side - 1) for v in row_floor]
        col_ceil = [min(v + 1, side - 1) for v in col_floor]
        row_frac = [row_grid[i] - row_floor[i] for i in range(grid_h)]
        col_frac = [col_grid[i] - col_floor[i] for i in range(grid_w)]
        frame = _merge_block_order(grid_h, grid_w, merge_size)
        for _ in range(grid_t):
            for row, col in frame:
                rf, cf = row_frac[row], col_frac[col]
                top, bottom = row_floor[row] * side, row_ceil[row] * side
                corner_indices[0].append(top + col_floor[col])
                corner_indices[1].append(top + col_ceil[col])
                corner_indices[2].append(bottom + col_floor[col])
                corner_indices[3].append(bottom + col_ceil[col])
                corner_weights[0].append((1.0 - rf) * (1.0 - cf))
                corner_weights[1].append((1.0 - rf) * cf)
                corner_weights[2].append(rf * (1.0 - cf))
                corner_weights[3].append(rf * cf)
    return corner_indices, corner_weights


def _vision_mrope_positions(start: int, grid: Grid, merge_size: int, time_interval: int = 1) -> list[list[int]]:
    """The (t, h, w) position of every LLM token produced by one image or video, offset by `start`."""
    grid_t, grid_h, grid_w = grid
    llm_h, llm_w = grid_h // merge_size, grid_w // merge_size
    temporal: list[int] = []
    height: list[int] = []
    width: list[int] = []
    for frame in range(grid_t):
        for row in range(llm_h):
            for col in range(llm_w):
                temporal.append(frame * time_interval + start)
                height.append(row + start)
                width.append(col + start)
    return [temporal, height, width]


def mrope_position_ids(
    token_ids: list[int],
    mm_token_type_ids: list[int],
    grids_by_type: dict[int, list[Grid]],
    merge_size: int,
    start: int = 0,
) -> tuple[list[list[int]], int]:
    """3D M-RoPE positions for one sequence, plus the next free 1D position.

    Text advances all three axes together, one step per token. A vision span instead lays its tokens out on
    a (t, h, w) lattice and then advances the shared cursor by only `max(h, w) // merge_size`, which is why
    generation after an image cannot simply keep counting tokens: the caller must carry the returned cursor.

    `mm_token_type_ids` is 0 for text, 1 for image, 2 for video, matching the reference's convention.
    """
    if len(token_ids) != len(mm_token_type_ids):
        raise ValueError(f'Token and modality lengths differ: {len(token_ids)} != {len(mm_token_type_ids)}')
    iterators = {key: iter(value) for key, value in grids_by_type.items()}
    positions: list[list[int]] = [[], [], []]
    cursor: int = start
    for modality, run in groupby(mm_token_type_ids):
        span: int = sum(1 for _ in run)
        if modality == 0:
            for axis in positions:
                axis.extend(range(cursor, cursor + span))
            cursor += span
        else:
            source = iterators.get(modality)
            grid = next(source, None) if source is not None else None
            if grid is None:
                raise ValueError(f'Prompt has a modality-{modality} span but no grid left to describe it')
            axes = _vision_mrope_positions(cursor, grid, merge_size)
            if len(axes[0]) != span:
                raise ValueError(f'Placeholder count {span} does not match grid {grid} (expected {len(axes[0])})')
            for axis, values in zip(positions, axes):
                axis.extend(values)
            cursor += max(grid[1], grid[2]) // merge_size
    next_position: int = max(max(axis) for axis in positions) + 1 if positions[0] else start
    return positions, next_position

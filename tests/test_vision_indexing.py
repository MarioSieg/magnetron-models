# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Index math for the vision tower and M-RoPE.

The expected values were produced by transformers (`vision_utils` and `Qwen3_5Model.get_rope_index`) and
are pinned here so the layout can be checked without torch installed. These orderings decide which patch
lands in which row, so a silent drift here shows up as a model that reads the image scrambled.
"""

from magnetron_models.models.qwen3_5.vision.rope import (
    mrope_position_ids,
    vision_bilinear_pos_embed_index,
    vision_cu_seqlens,
    vision_position_ids,
)

MERGE = 2
IMAGE_TOKEN = 248056
VISION_START = 248053
VISION_END = 248054


def test_cu_seqlens_is_one_segment_per_frame() -> None:
    assert vision_cu_seqlens([(1, 4, 6)]) == [0, 24]
    assert vision_cu_seqlens([(2, 6, 10)]) == [0, 60, 120]
    assert vision_cu_seqlens([(1, 4, 6), (1, 10, 8)]) == [0, 24, 104]


def test_position_ids_follow_the_merge_block_order() -> None:
    # A 2x4 patch grid with merge 2: the first four rows are the top-left 2x2 block, then the next block.
    assert vision_position_ids([(1, 2, 4)], MERGE) == [
        [0, 0], [0, 1], [1, 0], [1, 1],
        [0, 2], [0, 3], [1, 2], [1, 3],
    ]  # fmt: skip


def test_position_ids_repeat_per_frame() -> None:
    one_frame = vision_position_ids([(1, 2, 2)], MERGE)
    assert vision_position_ids([(3, 2, 2)], MERGE) == one_frame * 3


def test_bilinear_weights_form_a_partition_of_unity() -> None:
    indices, weights = vision_bilinear_pos_embed_index([(1, 14, 20)], 48, MERGE)
    assert len(indices) == len(weights) == 4
    assert all(len(corner) == 14 * 20 for corner in indices)
    for token in range(14 * 20):
        assert abs(sum(corner[token] for corner in weights) - 1.0) < 1e-9
    assert all(0 <= i < 48 * 48 for corner in indices for i in corner)


def test_bilinear_is_exact_on_the_grid_corners() -> None:
    # A single-patch row/column samples position 0 exactly, so all the weight sits on the first corner.
    indices, weights = vision_bilinear_pos_embed_index([(1, 2, 2)], 48, MERGE)
    assert indices[0][0] == 0
    assert abs(weights[0][0] - 1.0) < 1e-9


def _with_placeholders(spans: list[tuple[str, object]]) -> tuple[list[int], list[int], list[tuple[int, int, int]]]:
    ids: list[int] = []
    modality: list[int] = []
    grids: list[tuple[int, int, int]] = []
    for kind, payload in spans:
        if kind == 'text':
            ids += [100] * payload
            modality += [0] * payload
        else:
            grid_t, grid_h, grid_w = payload
            count = grid_t * (grid_h // MERGE) * (grid_w // MERGE)
            ids += [VISION_START] + [IMAGE_TOKEN] * count + [VISION_END]
            modality += [0] + [1] * count + [0]
            grids.append(payload)
    return ids, modality, grids


def test_text_only_mrope_is_plain_rope() -> None:
    ids, modality, _ = _with_placeholders([('text', 5)])
    axes, next_pos = mrope_position_ids(ids, modality, {1: []}, MERGE)
    assert axes == [[0, 1, 2, 3, 4]] * 3
    assert next_pos == 5


def test_image_span_lays_tokens_on_a_lattice() -> None:
    ids, modality, grids = _with_placeholders([('text', 1), ('image', (1, 4, 6))])
    axes, next_pos = mrope_position_ids(ids, modality, {1: grids}, MERGE)
    temporal, height, width = axes
    # Text token 0, then <|vision_start|> at 1, then a 2x3 lattice anchored at 2, then <|vision_end|>.
    assert temporal[:2] == [0, 1]
    assert temporal[2:8] == [2] * 6
    assert height[2:8] == [2, 2, 2, 3, 3, 3]
    assert width[2:8] == [2, 3, 4, 2, 3, 4]
    # The cursor advances by max(h, w) // merge = 3, not by the six tokens the image occupies.
    assert next_pos == max(max(axis) for axis in axes) + 1


def test_cursor_advances_by_the_longer_side() -> None:
    ids, modality, grids = _with_placeholders([('image', (1, 4, 20))])
    _, next_pos = mrope_position_ids(ids, modality, {1: grids}, MERGE)
    # <|vision_start|> at 0, lattice anchored at 1 spanning 10 columns, <|vision_end|> right after.
    assert next_pos == 1 + 20 // MERGE + 1


def test_start_offset_continues_a_conversation() -> None:
    ids, modality, grids = _with_placeholders([('text', 2), ('image', (1, 4, 6))])
    base, _ = mrope_position_ids(ids, modality, {1: grids}, MERGE, start=0)
    shifted, _ = mrope_position_ids(ids, modality, {1: grids}, MERGE, start=17)
    assert shifted == [[v + 17 for v in axis] for axis in base]


def test_placeholder_count_mismatch_is_rejected() -> None:
    ids, modality, grids = _with_placeholders([('image', (1, 4, 6))])
    try:
        mrope_position_ids(ids, modality, {1: [(1, 8, 8)]}, MERGE)
    except ValueError as e:
        assert 'does not match grid' in str(e)
    else:
        raise AssertionError('a grid that does not fill its placeholders must raise')

# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""smart_resize and the patchify layout, ported from Qwen2VLImageProcessor."""

import pytest
from magnetron import context, dtype

context.stop_grad_recorder()
context.set_default_device('cpu')
context.set_default_dtype(dtype.float32)

from magnetron_models.models.qwen3_5.vision import VisionConfig, smart_resize  # noqa: E402
from magnetron_models.models.qwen3_5.vision.image import preprocess_pixels  # noqa: E402

CFG = VisionConfig(min_pixels=65536, max_pixels=1048576)


def test_resize_snaps_to_the_merge_factor() -> None:
    for height, width in ((137, 91), (1, 1), (4000, 37), (321, 999)):
        out_h, out_w = smart_resize(height, width, CFG.resize_factor, CFG.min_pixels, CFG.max_pixels)
        assert out_h % CFG.resize_factor == 0 and out_w % CFG.resize_factor == 0
        assert out_h > 0 and out_w > 0


def test_resize_respects_the_pixel_bounds() -> None:
    tiny_h, tiny_w = smart_resize(8, 8, CFG.resize_factor, CFG.min_pixels, CFG.max_pixels)
    assert tiny_h * tiny_w >= CFG.min_pixels
    huge_h, huge_w = smart_resize(9000, 7000, CFG.resize_factor, CFG.min_pixels, CFG.max_pixels)
    assert huge_h * huge_w <= CFG.max_pixels


def test_resize_matches_the_reference_on_a_known_case() -> None:
    # 137x91 upscales to 320x224, the grid the transformers processor reports for the same input.
    assert smart_resize(91, 137, CFG.resize_factor, CFG.min_pixels, CFG.max_pixels) == (224, 320)


def test_extreme_aspect_ratio_is_rejected() -> None:
    with pytest.raises(ValueError, match='aspect ratio'):
        smart_resize(10, 5000, CFG.resize_factor, CFG.min_pixels, CFG.max_pixels)


def test_patch_matrix_shape_and_normalization() -> None:
    height, width = 32, 64
    raw = bytes([255]) * (height * width * 3)  # A white image maps to +1 under mean 0.5, std 0.5.
    pixels = preprocess_pixels(raw, height, width, CFG)
    assert tuple(pixels.shape) == ((height // 16) * (width // 16), CFG.patch_dim)
    values = pixels.reshape(-1).tolist()
    assert max(abs(v - 1.0) for v in values) < 1e-5


def test_each_patch_row_holds_one_spatial_block() -> None:
    # Paint every 16x16 patch a distinct grey so a row of the matrix must be constant.
    height, width = 32, 32
    raw = bytearray()
    for y in range(height):
        for x in range(width):
            shade = (y // 16) * 2 + (x // 16)
            raw += bytes([shade * 60] * 3)
    pixels = preprocess_pixels(bytes(raw), height, width, CFG)
    rows = pixels.tolist()
    assert len(rows) == 4
    for row in rows:
        assert max(row) - min(row) < 1e-5, 'a patch row must not mix pixels from different patches'
    assert len({round(row[0], 4) for row in rows}) == 4, 'the four patches must stay distinct'

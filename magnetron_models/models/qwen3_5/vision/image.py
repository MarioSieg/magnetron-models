# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Image loading and patchification, a port of Qwen2VLImageProcessor.

Pillow does the decode and the bicubic resize so the numbers line up with the reference, which also
resamples through PIL. Everything after that is magnetron tensor work: the byte buffer goes in through
the buffer protocol, so no per-pixel Python loop is on the path.
"""

import math
from array import array
from dataclasses import dataclass

from magnetron import Tensor, context, dtype

from .config import VisionConfig


@dataclass(frozen=True)
class PreprocessedImage:
    """One image flattened into the (num_patches, patch_dim) layout the tower consumes."""

    pixel_values: Tensor
    grid_thw: tuple[int, int, int]
    merge_size: int
    source: str = ''

    @property
    def num_patches(self) -> int:
        grid_t, grid_h, grid_w = self.grid_thw
        return grid_t * grid_h * grid_w

    @property
    def num_tokens(self) -> int:
        """Placeholder tokens this image occupies in the prompt, after the spatial merge."""
        return self.num_patches // (self.merge_size * self.merge_size)


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Snap both sides to a multiple of `factor` while keeping the pixel count in range and the aspect ratio close."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f'Absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}')
    h_bar: int = round(height / factor) * factor
    w_bar: int = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _decode_rgb(path: str, cfg: VisionConfig) -> tuple[bytes, int, int]:
    try:
        from PIL import Image
    except ImportError as e:  # noqa: F841
        raise ImportError(
            'Vision input needs Pillow, install magnetron-models with the vision extra: pip install "magnetron-models[vision]"'
        ) from None
    with Image.open(path) as img:
        img = img.convert('RGB')  # Before the resize, matching do_convert_rgb in the reference.
        width, height = img.size
        resized_height, resized_width = smart_resize(height, width, cfg.resize_factor, cfg.min_pixels, cfg.max_pixels)
        if (resized_height, resized_width) != (height, width):
            img = img.resize((resized_width, resized_height), Image.BICUBIC)
        return img.tobytes(), resized_height, resized_width


def preprocess_pixels(raw: bytes, height: int, width: int, cfg: VisionConfig) -> Tensor:
    """Turn a packed HWC uint8 buffer into the (num_patches, patch_dim) matrix the patch embedding expects."""
    # array() gives magnetron a C-contiguous buffer to cast from, so the pixels never become a Python list.
    pixels = Tensor(array('B', raw), dtype=dtype.float32).reshape(height, width, cfg.in_channels)
    pixels = pixels.permute(2, 0, 1).contiguous()  # HWC -> CHW, the layout the reference normalizes in.
    pixels = pixels * cfg.rescale_factor
    mean = Tensor(list(cfg.image_mean), dtype=dtype.float32).reshape(cfg.in_channels, 1, 1)
    std = Tensor(list(cfg.image_std), dtype=dtype.float32).reshape(cfg.in_channels, 1, 1)
    pixels = (pixels - mean) / std
    patch, merge, temporal = cfg.patch_size, cfg.spatial_merge_size, cfg.temporal_patch_size
    grid_h, grid_w = height // patch, width // patch
    # A still image is one frame, so it is repeated to fill the temporal patch, exactly like the reference pads.
    frames = Tensor.cat([pixels.reshape(1, 1, cfg.in_channels, height, width)] * temporal, dim=1)
    blocks = frames.reshape(1, 1, temporal, cfg.in_channels, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    blocks = blocks.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
    return blocks.reshape(grid_h * grid_w, cfg.patch_dim).cast(context.get_default_dtype())


def load_image(path: str, cfg: VisionConfig) -> PreprocessedImage:
    raw, height, width = _decode_rgb(path, cfg)
    pixel_values = preprocess_pixels(raw, height, width, cfg)
    grid = (1, height // cfg.patch_size, width // cfg.patch_size)
    return PreprocessedImage(pixel_values=pixel_values, grid_thw=grid, merge_size=cfg.spatial_merge_size, source=path)

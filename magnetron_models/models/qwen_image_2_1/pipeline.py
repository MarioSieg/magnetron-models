# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from magnetron import Tensor, dtype
from magnetron_models.tokenizer import TokenizerBase
from .config import SchedulerConfig, TextEncoderConfig
from .scheduler import FlowMatchEulerScheduler
from .text_encoder import QwenImageTextEncoder
from .transformer import QwenImageTransformer, TransformerKVCache
from .vae import QwenImageVAE

SYSTEM_PROMPT: str = 'Comprehend and analyze the provided prompt.'
LATENT_TOKEN_PIXELS: int = 16
SIZE_MULTIPLE: int = 2 * LATENT_TOKEN_PIXELS


def build_prompt(prompt: str) -> str:
    if not prompt:
        prompt = ' '
    return f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'


def system_prompt_token_count(tokenizer: TokenizerBase) -> int:
    return len(tokenizer.encode(f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n'))


def snap_size(pixels: int) -> int:
    snapped: int = pixels // SIZE_MULTIPLE * SIZE_MULTIPLE
    if snapped < SIZE_MULTIPLE:
        raise ValueError(f'Image sides must be at least {SIZE_MULTIPLE} pixels, got {pixels}')
    return snapped


@dataclass(frozen=True, slots=True)
class PromptEmbedding:
    hidden: Tensor
    token_count: int


def encode_prompt(text_encoder: QwenImageTextEncoder, tokenizer: TokenizerBase, prompt: str) -> PromptEmbedding:
    ids: list[int] = tokenizer.encode(build_prompt(prompt))
    drop: int = system_prompt_token_count(tokenizer)
    cfg: TextEncoderConfig = text_encoder.cfg
    if cfg.image_token_id in ids:
        raise ValueError('The prompt contains an image placeholder token, condition images are not supported')
    hidden = text_encoder(Tensor([ids], dtype=dtype.int64))
    hidden = hidden[:, drop:]
    return PromptEmbedding(hidden=hidden.contiguous(), token_count=hidden.shape[1])


@dataclass(frozen=True, slots=True)
class LatentGrid:
    height: int  # In latent tokens
    width: int

    @property
    def tokens(self) -> int:
        return self.height * self.width

    @classmethod
    def for_image(cls, height_px: int, width_px: int) -> LatentGrid:
        return cls(snap_size(height_px) // LATENT_TOKEN_PIXELS, snap_size(width_px) // LATENT_TOKEN_PIXELS)


def initial_latents(grid: LatentGrid, channels: int, model_dtype: dtype.DType) -> Tensor:
    noise = Tensor.normal(1, channels, grid.height, grid.width, dtype=dtype.float32, mean=0.0, std=1.0)
    return noise.reshape(1, channels, grid.tokens).transpose(1, 2).contiguous().cast(model_dtype)


def denoise(
    transformer: QwenImageTransformer,
    scheduler: FlowMatchEulerScheduler,
    prompt: PromptEmbedding,
    latents: Tensor,
    grid: LatentGrid,
    num_inference_steps: int,
    negative_prompt: PromptEmbedding | None = None,
    guidance_scale: float = 1.0,
    use_kv_cache: bool = True,
    on_step: Callable[[int, int], None] | None = None,
) -> Tensor:
    timesteps: Tensor = scheduler.set_timesteps(num_inference_steps, grid.tokens) / scheduler.cfg.num_train_timesteps
    use_cfg: bool = negative_prompt is not None and guidance_scale > 1.0
    cache_ok: bool = use_kv_cache and transformer.cfg.causal_condition
    cond_cache: TransformerKVCache | None = transformer.new_kv_cache() if cache_ok else None
    neg_cache: TransformerKVCache | None = transformer.new_kv_cache() if cache_ok and use_cfg else None
    model_dtype = latents.dtype

    for i in range(num_inference_steps):
        mode: str | None = None if not cache_ok else ('extract' if i == 0 else 'cached')
        timestep = timesteps[i].cast(model_dtype)
        pred = transformer(latents, prompt.hidden, timestep, grid.height, grid.width, cond_cache, mode)
        if use_cfg:
            assert negative_prompt is not None
            neg = transformer(latents, negative_prompt.hidden, timestep, grid.height, grid.width, neg_cache, mode)
            pred = neg + (pred - neg) * guidance_scale
        latents = scheduler.step(pred, i, latents)
        if on_step is not None:
            on_step(i + 1, num_inference_steps)
    return latents


def decode_latents(vae: QwenImageVAE, latents: Tensor, grid: LatentGrid) -> Tensor:
    z = latents.transpose(1, 2).reshape(1, vae.cfg.z_dim, grid.height, grid.width)
    z = vae.denormalize_latents(z.contiguous())
    return vae.decode(z)[0]


def to_uint8(pixels: Tensor) -> Tensor:
    return ((pixels.cast(dtype.float32) * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round().cast(dtype.uint8)


def composite_over_white(rgba: Tensor) -> Tensor:
    if rgba.shape[0] != 4:
        return rgba
    f = rgba.cast(dtype.float32) / 255.0
    alpha = f[3:4]
    rgb = f[:3] * alpha + (1.0 - alpha)
    return (rgb * 255.0).round().cast(dtype.uint8)


__all__ = [
    'SYSTEM_PROMPT',
    'LATENT_TOKEN_PIXELS',
    'SIZE_MULTIPLE',
    'PromptEmbedding',
    'LatentGrid',
    'build_prompt',
    'system_prompt_token_count',
    'snap_size',
    'encode_prompt',
    'initial_latents',
    'denoise',
    'decode_latents',
    'to_uint8',
    'composite_over_white',
    'SchedulerConfig',
]

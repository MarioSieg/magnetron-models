# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import gc
import time

from dataclasses import dataclass
from typing import Any

from magnetron import Tensor, context, dtype
from magnetron.snapshot import deserialize
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from magnetron_models.inference import AUTO_DVC, DTYPES, resolve_device
from magnetron_models.models import DIFFUSION_MODELS_MAP, DiffusionModelSpec, SnapshotModule, decode_config, load_component_snapshot
from magnetron_models.models.qwen_image_2_1 import (
    ARCH_TEXT_ENCODER,
    ARCH_TRANSFORMER,
    ARCH_VAE,
    FlowMatchEulerScheduler,
    QwenImageTextEncoder,
    QwenImageTransformer,
    QwenImageVAE,
    SchedulerConfig,
    pipeline,
)
from magnetron_models.tokenizer import HFTokenizer
from magnetron_models.utils import console

_ARCH_BY_COMPONENT: dict[str, str] = {'text-encoder': ARCH_TEXT_ENCODER, 'transformer': ARCH_TRANSFORMER, 'vae': ARCH_VAE}


@dataclass
class ImageGenConfig:
    device: str = AUTO_DVC
    dtype: str = 'bfloat16'
    model: str | None = 'qwen-image-2.1'
    # Local snapshots override the download of the named model, component by component.
    text_encoder_snapshot: str | None = None
    transformer_snapshot: str | None = None
    vae_snapshot: str | None = None
    seed: int = 3407
    height: int = 1024
    width: int = 1024
    num_inference_steps: int | None = None  # None takes the checkpoint's default
    negative_prompt: str | None = None
    guidance_scale: float = 1.0  # Qwen-Image 2.1 is meant to run without guidance
    use_kv_cache: bool = True
    offload: bool = True  # Load each network when needed and drop it afterwards, the three together are ~31 GiB in bf16


class ImageGenEngine:
    def __init__(self, cfg: ImageGenConfig) -> None:
        start = time.perf_counter()
        context.stop_grad_recorder()
        context.manual_seed(cfg.seed)
        self.config = cfg
        self.device: str = resolve_device(cfg.device)
        context.set_default_device(self.device)
        self.model_dtype: dtype.DType = DTYPES[cfg.dtype]
        context.set_default_dtype(self.model_dtype)
        spec: DiffusionModelSpec | None = DIFFUSION_MODELS_MAP[cfg.model] if cfg.model is not None else None
        overrides: dict[str, str | None] = {
            'text-encoder': cfg.text_encoder_snapshot,
            'transformer': cfg.transformer_snapshot,
            'vae': cfg.vae_snapshot,
        }
        self.snapshots: dict[str, str] = {}
        for component, override in overrides.items():
            if override is not None:
                self.snapshots[component] = override
            elif spec is not None:
                self.snapshots[component] = spec.download_snapshot(component, self.model_dtype.short_name)
            else:
                raise ValueError(f'Must specify either a model name or a snapshot file for the {component}')
        self.expect_repo_id: str | None = spec.checkpoint_repo_id if spec is not None else None
        self._resident: dict[str, SnapshotModule] = {}
        self.tokenizer: HFTokenizer | None = None
        self.scheduler = FlowMatchEulerScheduler(self._scheduler_config())
        console.print(f'Ready on {self.device} in {time.perf_counter() - start:.2f}s', style='dim')

    def _scheduler_config(self) -> SchedulerConfig:
        """The schedule ships in the transformer snapshot's metadata, reading it costs a header, not the weights."""
        _, metadata = deserialize(self.snapshots['transformer'])
        raw: dict[str, Any] | None = metadata.get('scheduler_config')
        if raw is None:
            console.print('Transformer snapshot carries no scheduler config, using the Qwen-Image 2.1 defaults', style='yellow')
            return SchedulerConfig()
        return decode_config(SchedulerConfig, raw)  # type: ignore[return-value]

    def bind_thread(self) -> None:
        context.stop_grad_recorder()
        context.set_default_device(self.device)
        context.set_default_dtype(self.model_dtype)

    def _load(self, component: str) -> SnapshotModule:
        if component in self._resident:
            return self._resident[component]
        snapshot: str = self.snapshots[component]
        console.print(f'Loading {component} from snapshot: {snapshot}', style='dim')
        start = time.perf_counter()
        module = load_component_snapshot(snapshot, _ARCH_BY_COMPONENT[component], expect_repo_id=self.expect_repo_id)
        console.print(f'{component} ready in {time.perf_counter() - start:.2f}s', style='dim')
        if component == 'text-encoder' and self.tokenizer is None:
            self.tokenizer = HFTokenizer.from_snapshot_metadata(module.snapshot_metadata)
            if self.tokenizer is None:
                repo_id: str = module.snapshot_metadata.get('source_repo') or module.cfg.repo_id
                console.print(f'Snapshot carries no tokenizer, falling back to {repo_id}', style='yellow')
                self.tokenizer = HFTokenizer.from_repo(repo_id)
        if not self.config.offload:
            self._resident[component] = module
        return module

    def _collect(self) -> None:
        """With offloading, a network dies with the frame that loaded it. Collect right away so its memory is free
        before the next one is loaded rather than whenever the collector gets around to it."""
        if self.config.offload:
            gc.collect()

    def encode_prompt(self, prompt: str) -> pipeline.PromptEmbedding:
        self.bind_thread()
        embedding = self._encode(prompt)
        self._collect()
        return embedding

    def _encode(self, prompt: str) -> pipeline.PromptEmbedding:
        text_encoder = self._load('text-encoder')
        assert isinstance(text_encoder, QwenImageTextEncoder) and self.tokenizer is not None
        return pipeline.encode_prompt(text_encoder, self.tokenizer, prompt)

    def _denoise(
        self,
        prompt_embedding: pipeline.PromptEmbedding,
        negative_embedding: pipeline.PromptEmbedding | None,
        grid: pipeline.LatentGrid,
        steps: int,
        seed: int,
        guidance: float,
    ) -> Tensor:
        transformer = self._load('transformer')
        assert isinstance(transformer, QwenImageTransformer)
        context.manual_seed(seed)
        latents = pipeline.initial_latents(grid, transformer.cfg.in_channels, self.model_dtype)
        with Progress(TextColumn('denoising'), BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), console=console) as progress:
            task = progress.add_task('denoise', total=steps)
            return pipeline.denoise(
                transformer,
                self.scheduler,
                prompt_embedding,
                latents,
                grid,
                steps,
                negative_prompt=negative_embedding,
                guidance_scale=guidance,
                use_kv_cache=self.config.use_kv_cache,
                on_step=lambda done, total: progress.update(task, completed=done),
            )

    def _decode(self, latents: Tensor, grid: pipeline.LatentGrid) -> Tensor:
        vae = self._load('vae')
        assert isinstance(vae, QwenImageVAE)
        return pipeline.to_uint8(pipeline.decode_latents(vae, latents, grid))

    def generate(
        self,
        prompt: str,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        guidance_scale: float | None = None,
    ) -> Tensor:
        """Sample one image. Returns uint8 pixels of shape [4, H, W] (RGBA), ready for Tensor.save_image."""
        self.bind_thread()
        cfg = self.config
        grid = pipeline.LatentGrid.for_image(height or cfg.height, width or cfg.width)
        steps: int = num_inference_steps or cfg.num_inference_steps or self.scheduler.cfg.default_num_inference_steps
        negative_prompt = cfg.negative_prompt if negative_prompt is None else negative_prompt
        guidance: float = cfg.guidance_scale if guidance_scale is None else guidance_scale

        prompt_embedding = self._encode(prompt)
        negative_embedding = self._encode(negative_prompt) if negative_prompt is not None and guidance > 1.0 else None
        self._collect()
        latents = self._denoise(prompt_embedding, negative_embedding, grid, steps, cfg.seed if seed is None else seed, guidance)
        self._collect()
        pixels = self._decode(latents, grid)
        self._collect()
        return pixels

    @staticmethod
    def save(pixels: Tensor, path: str) -> None:
        """Write uint8 [C, H, W] pixels. PNG keeps the alpha channel, JPEG gets the image composited over white."""
        if path.lower().endswith(('.jpg', '.jpeg')):
            pixels = pipeline.composite_over_white(pixels)
        pixels.save_image(path)

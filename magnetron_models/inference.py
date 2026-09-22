# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import asyncio
import gc
import time

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from magnetron import Tensor, context, dtype
from rich.console import Console
from magnetron_models.tokenizer import HFTokenizer
from magnetron_models.models import MODELS_MAP, ModelBase, ModelSpec, load_snapshot

console = Console()

DTYPES: dict[str, dtype.DType] = {'float16': dtype.float16, 'bfloat16': dtype.bfloat16, 'float32': dtype.float32}
AUTO_DVC = 'auto'
_AUTO_DVC_ACCELERATORS: tuple[str, ...] = ('cuda', 'cpu')  # Sorted from best to worst backend


def resolve_device(device: str) -> str:
    if device == AUTO_DVC:
        chosen = next((b for b in _AUTO_DVC_ACCELERATORS if context.is_device_available(b)), None)
        return context.best_device(chosen) if chosen is not None else 'cpu'
    if not context.is_device_available(device):
        raise RuntimeError(f'Requested device {device} is not available')
    return device if ':' in device else context.best_device(device)


@dataclass
class InferenceConfig:
    device: str = AUTO_DVC
    max_tokens: int = 1024
    temp: float = 0.6
    top_k: int = 200
    seed: int = 3407
    model: str | None = None
    dtype: str = 'bfloat16'
    repo_id: str | None = None
    snapshot: str | None = None


class InferenceEngine:
    def __init__(self, cfg: InferenceConfig) -> None:
        start = time.perf_counter()
        context.stop_grad_recorder()
        context.manual_seed(cfg.seed)
        self.device: str = resolve_device(cfg.device)
        context.set_default_device(self.device)
        spec: ModelSpec | None = MODELS_MAP[cfg.model] if cfg.model is not None else None
        if cfg.snapshot is not None:
            snapshot: str = cfg.snapshot
        elif spec is not None:
            snapshot = spec.download_snapshot(DTYPES[cfg.dtype].short_name)
        else:
            raise ValueError('Must specify either a model name or a snapshot file')
        console.print(f'Loading model from snapshot: {snapshot}', style='dim')
        self.model: ModelBase = load_snapshot(snapshot, expect_repo_id=spec.checkpoint_repo_id if spec is not None else None)
        self.tokenizer = self._load_tokenizer(cfg)
        self.config = cfg
        self.snapshot = snapshot
        self.model_dtype = context.get_default_dtype()
        end = time.perf_counter()
        console.print(f'Ready on {self.device} in {end - start:.2f}s', style='dim')
        gc.collect()  # Loads of stuff allocated on startup, clean up a bit

    def _load_tokenizer(self, cfg: InferenceConfig) -> HFTokenizer:
        if cfg.repo_id is not None:
            return HFTokenizer.from_repo(cfg.repo_id)
        tokenizer = HFTokenizer.from_snapshot_metadata(self.model.snapshot_metadata)
        if tokenizer is not None:
            return tokenizer
        repo_id: str = self.model.tokenizer_repo_id
        console.print(f'Snapshot carries no tokenizer, falling back to {repo_id}', style='yellow')
        return HFTokenizer.from_repo(repo_id)

    def bind_thread(self) -> None:
        context.stop_grad_recorder()
        context.set_default_device(self.device)
        context.set_default_dtype(self.model_dtype)

    def gen_stream(
        self, prompt: str, max_tokens: int | None = None, temp: float | None = None, top_k: int | None = None, reset_cache: bool = False
    ) -> Iterator[str]:
        self.bind_thread()
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if temp is None:
            temp = self.config.temp
        if top_k is None:
            top_k = self.config.top_k
        model_input_ids = Tensor([self.tokenizer.encode(prompt)], dtype=dtype.int64)
        yield from self.model.generate_stream(model_input_ids, self.tokenizer, max_tokens=max_tokens, temp=temp, top_k=top_k, reset_cache=reset_cache)
        gc.collect()

    async def gen_stream_async(
        self, prompt: str, max_tokens: int | None = None, temp: float | None = None, top_k: int | None = None, reset_cache: bool = False
    ) -> AsyncIterator[str]:
        for chunk in self.gen_stream(prompt, max_tokens, temp, top_k, reset_cache):
            yield chunk
            await asyncio.sleep(0)

    def gen_one_shot(
        self, prompt: str, max_tokens: int | None = None, temp: float | None = None, top_k: int | None = None, reset_cache: bool = False
    ) -> str:
        return ''.join(self.gen_stream(prompt, max_tokens, temp, top_k, reset_cache))

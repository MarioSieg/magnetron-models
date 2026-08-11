# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import argparse
import gc
import time

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from magnetron import Tensor, context, dtype
from rich.console import Console
from magnetron_models.tokenizer import HFTokenizer
from magnetron_models.models import MODELS_MAP, ModelBase

console = Console()


@dataclass
class InferenceConfig:
    system: str = 'You are a helpful assistant.'
    device: str = 'cuda'
    max_ctx: int = 4096
    max_tokens: int = 1024
    temp: float = 0.6
    top_k: int = 200
    seed: int = 3407
    model: str = 'qwen3'
    repo_id: str | None = None
    snapshot: str | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> InferenceConfig:
        return cls(
            system=args.system,
            device=args.device,
            max_ctx=args.max_ctx,
            max_tokens=args.max_tokens,
            temp=args.temp,
            top_k=args.top_k,
            seed=args.seed,
            model=args.model,
            repo_id=args.repo_id,
            snapshot=args.snapshot,
        )


class InferenceEngine:
    def __init__(self, cfg: InferenceConfig) -> None:
        if cfg.snapshot is None:
            raise ValueError('Snapshot file not provided')
        start = time.perf_counter()
        context.stop_grad_recorder()
        context.set_default_dtype(dtype.bfloat16)
        context.manual_seed(cfg.seed)
        if context.is_device_available(cfg.device):
            context.set_default_device(cfg.device)
        console.print(f'Loading model from snapshot: {cfg.snapshot}', style='dim')
        self.model: ModelBase = MODELS_MAP[cfg.model]()
        self.model.load_from_snapshot(cfg.snapshot)
        self.tokenizer = HFTokenizer(cfg.repo_id)
        self.config = cfg
        end = time.perf_counter()
        console.print(f'Ready in {end - start:.2f}s', style='dim')
        gc.collect()

    def gen_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temp: float | None = None,
        top_k: int | None = None,
        reset_cache: bool = False,
    ) -> Iterator[str]:
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if temp is None:
            temp = self.config.temp
        if top_k is None:
            top_k = self.config.top_k
        model_input_ids = Tensor([self.tokenizer.encode(prompt)], dtype=dtype.int64)
        yield from self.model.generate_stream(
            model_input_ids,
            self.tokenizer,
            max_tokens=max_tokens,
            temp=temp,
            top_k=top_k,
            reset_cache=reset_cache,
        )
        gc.collect()

    async def gen_stream_async(
        self, prompt: str, max_tokens: int | None = None, temp: float | None = None, top_k: int | None = None
    ) -> AsyncIterator[str]:
        import asyncio

        for chunk in self.gen_stream(prompt, max_tokens, temp, top_k):
            yield chunk
            await asyncio.sleep(0)

    def gen_one_shot(self, prompt: str, max_tokens: int | None = None, temp: float | None = None, top_k: int | None = None) -> str:
        parts: list[str] = []
        for chunk in self.gen_stream(prompt, max_tokens, temp, top_k):
            parts.append(chunk)
        reply: str = ''.join(parts)
        return reply

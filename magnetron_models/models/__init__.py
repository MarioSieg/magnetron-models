# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import gc
from abc import ABC, abstractmethod
from collections.abc import Iterator, Callable
from magnetron import Snapshot, Tensor, nn, context
from magnetron_models.tokenizer import TokenizerBase


class ModelBase(ABC, nn.Module):
    def load_from_snapshot(self, snapshot_file: str) -> None:
        with Snapshot.read(snapshot_file) as snap:
            for name, param in self.named_parameters():
                tensor = snap.get_tensor(name)
                if tuple(tensor.shape) != tuple(param.shape):
                    raise RuntimeError(f'Shape mismatch for {name}: {tensor.shape} != {param.shape}')
                if tensor.dtype != param.dtype:
                    raise RuntimeError(f'Dtype mismatch for {name}: {tensor.dtype} != {param.dtype}')
                if context.get_default_device() != 'cpu:0':
                    param.data = tensor.transfer(context.get_default_device())
                else:
                    param.data = tensor
        gc.collect()

    @abstractmethod
    def build_prompt(self, system: str, messages: list[tuple[str, str]]) -> str:
        raise NotImplementedError()

    @abstractmethod
    def generate_stream(
        self,
        idx: Tensor,
        tokenizer: TokenizerBase,
        max_tokens: int,
        temp: float = 1.0,
        top_k: int = 10,
        reset_cache: bool = False,
    ) -> Iterator[str]:
        raise NotImplementedError()


def _load_qwen3() -> ModelBase:
    from magnetron_models.models.qwen3 import Qwen3Model, Config

    return Qwen3Model(Config())


MODELS_MAP: dict[str, Callable[[], ModelBase]] = {'qwen3': _load_qwen3}

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
from typing import Any
from magnetron import Tensor, nn, context
from magnetron.snapshot import deserialize
from magnetron_models.tokenizer import TokenizerBase
from magnetron_models.utils import console


class ModelBase(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot_metadata: dict[str, Any] = {}

    def load_from_snapshot(self, snapshot_file: str) -> None:
        tensors, self.snapshot_metadata = deserialize(snapshot_file)
        source_repo: str | None = self.snapshot_metadata.get('source_repo')
        if source_repo is not None and source_repo != self.cfg.repo_id:
            console.print(f'Snapshot was converted from {source_repo} but this model is configured for {self.cfg.repo_id}', style='yellow')
        device: str = context.get_default_device()
        for name, param in self.named_parameters():
            tensor = tensors.pop(name, None)
            if tensor is None:
                raise KeyError(f'Snapshot {snapshot_file} has no tensor named {name}')
            if tuple(tensor.shape) != tuple(param.shape):
                raise RuntimeError(f'Shape mismatch for {name}: {tensor.shape} != {param.shape}')
            if tensor.dtype != param.dtype:
                raise RuntimeError(f'Dtype mismatch for {name}: {tensor.dtype} != {param.dtype}')
            param.data = tensor if device.startswith('cpu') else tensor.transfer(device)
        del tensors
        gc.collect()

    @property
    def tokenizer_repo_id(self) -> str:
        return self.snapshot_metadata.get('source_repo') or self.cfg.repo_id

    def build_system(self, system: str) -> str:
        return f'<|im_start|>system\n{system}<|im_end|>\n'

    @abstractmethod
    def build_prompt(self, system: str, messages: list[tuple[str, str]]) -> str:
        raise NotImplementedError()

    def build_user_turn(self, user: str) -> str:
        return f'<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n'

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


def _load_qwen3_5(repo_id: str = 'Qwen/Qwen3.5-4B') -> ModelBase:
    from magnetron_models.models.qwen3_5 import Qwen35Model, CONFIGS

    return Qwen35Model(CONFIGS[repo_id])


def _load_qwen3_5_moe(repo_id: str = 'Qwen/Qwen3.5-35B-A3B') -> ModelBase:
    from magnetron_models.models.qwen3_5_moe import Qwen35MoeModel, CONFIGS

    return Qwen35MoeModel(CONFIGS[repo_id])


MODELS_MAP: dict[str, Callable[[], ModelBase]] = {
    'qwen3': _load_qwen3,
    'qwen3.5': _load_qwen3_5,
    'qwen3.5-0.8b': lambda: _load_qwen3_5('Qwen/Qwen3.5-0.8B'),
    'qwen3.5-2b': lambda: _load_qwen3_5('Qwen/Qwen3.5-2B'),
    'qwen3.5-4b': lambda: _load_qwen3_5('Qwen/Qwen3.5-4B'),
    'qwen3.5-9b': lambda: _load_qwen3_5('Qwen/Qwen3.5-9B'),
    'qwen3.5-27b': lambda: _load_qwen3_5('Qwen/Qwen3.5-27B'),
    'qwen3.5-moe': _load_qwen3_5_moe,
    'qwen3.5-35b-a3b': lambda: _load_qwen3_5_moe('Qwen/Qwen3.5-35B-A3B'),
    'qwen3.5-122b-a10b': lambda: _load_qwen3_5_moe('Qwen/Qwen3.5-122B-A10B'),
    'qwen3.5-397b-a17b': lambda: _load_qwen3_5_moe('Qwen/Qwen3.5-397B-A17B'),
    'qwen3.8': lambda: _load_qwen3_5('Qwen/Qwen3.8-27B'),
    'qwen3.8-27b': lambda: _load_qwen3_5('Qwen/Qwen3.8-27B'),
    'qwen3.8-moe': lambda: _load_qwen3_5_moe('Qwen/Qwen3.8-2.4T-A95B'),
    'qwen3.8-2.4t-a95b': lambda: _load_qwen3_5_moe('Qwen/Qwen3.8-2.4T-A95B'),
}

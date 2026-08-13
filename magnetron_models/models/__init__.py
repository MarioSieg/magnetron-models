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

    @property
    def tokenizer_repo_id(self) -> str:
        return self.cfg.repo_id

    @abstractmethod
    def build_prompt(self, system: str, messages: list[tuple[str, str]]) -> str:
        raise NotImplementedError()

    def build_user_turn(self, user: str) -> str:
        """Single incremental chat turn, appended to a prompt whose prefix is still in the KV cache."""
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
}

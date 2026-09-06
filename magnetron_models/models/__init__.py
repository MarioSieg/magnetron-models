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
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, get_args, get_origin, get_type_hints
from magnetron import Tensor, nn, context, dtype
from magnetron.snapshot import deserialize
from magnetron_models.tokenizer import TokenizerBase
from magnetron_models.utils import console, download_or_ensure_resource, find_snapshot_file


_DTYPES: dict[str, dtype.DType] = {'float16': dtype.float16, 'bfloat16': dtype.bfloat16, 'float32': dtype.float32}


class ModelBase(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot_metadata: dict[str, Any] = {}

    def load_from_snapshot(self, snapshot_file: str) -> None:
        self.load_tensors(*deserialize(snapshot_file), source=snapshot_file)

    def load_tensors(self, tensors: dict[str, Tensor], metadata: dict[str, Any], source: str = 'snapshot') -> None:
        """Take the weights of an already opened snapshot, so the file is read once for its config and its tensors."""
        self.snapshot_metadata = metadata
        source_repo: str | None = self.snapshot_metadata.get('source_repo')
        if source_repo is not None and source_repo != self.cfg.repo_id:
            # The shapes would clash a few lines down anyway, but never as legibly as the two names do.
            raise RuntimeError(f'Snapshot was converted from {source_repo} but this model is configured for {self.cfg.repo_id}')
        device: str = context.get_default_device()
        for name, param in self.named_parameters():
            tensor = tensors.pop(name, None)
            if tensor is None:
                raise KeyError(f'Snapshot {source} has no tensor named {name}')
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


def _qwen3() -> tuple[Callable[[Any], ModelBase], Callable[..., Any]]:
    from magnetron_models.models.qwen3 import Qwen3Model, Config
    return Qwen3Model, Config


def _qwen3_5() -> tuple[Callable[[Any], ModelBase], Callable[..., Any]]:
    from magnetron_models.models.qwen3_5 import Qwen35Model, Config
    return Qwen35Model, Config


def _qwen3_5_moe() -> tuple[Callable[[Any], ModelBase], Callable[..., Any]]:
    from magnetron_models.models.qwen3_5_moe import Qwen35MoeModel, Config
    return Qwen35MoeModel, Config


_ARCHITECTURES: dict[str, Callable[[], tuple[Callable[[Any], ModelBase], Callable[..., Any]]]] = {
    'qwen3': _qwen3,
    'qwen3_5_text': _qwen3_5,
    'qwen3_5_moe_text': _qwen3_5_moe,
}

def _decode_config(config_cls: Callable[..., Any], data: dict[str, Any]) -> object:
    hints = get_type_hints(config_cls)
    def decode(hint: object, value: object) -> object:
        for candidate in (hint, *get_args(hint)):
            if value is not None and isinstance(candidate, type) and issubclass(candidate, Enum):
                return candidate(value)
        return frozenset(value) if get_origin(hint) is frozenset else value
    return config_cls(**{f.name: decode(hints[f.name], data[f.name]) for f in fields(config_cls) if f.name in data})


def load_snapshot(snapshot_file: str, expect_repo_id: str | None = None) -> ModelBase:
    tensors, metadata = deserialize(snapshot_file)
    architecture: str = metadata.get('architecture', '')
    source_repo: str = metadata.get('source_repo', '')
    if architecture not in _ARCHITECTURES:
        raise RuntimeError(f'{snapshot_file} holds a {architecture or "nameless"} model, this build runs {", ".join(sorted(_ARCHITECTURES))}')
    if expect_repo_id is not None and expect_repo_id != source_repo:
        raise ValueError(
            f'{snapshot_file} was converted from {source_repo}, but --model asked for {expect_repo_id}. Drop --model to run the snapshot.'
        )
    model_cls, config_cls = _ARCHITECTURES[architecture]()
    context.set_default_dtype(_DTYPES[metadata['dtype']])
    model: ModelBase = model_cls(_decode_config(config_cls, metadata['model_config']))
    model.load_tensors(tensors, metadata, source=snapshot_file)
    return model


@dataclass(frozen=True, slots=True)
class ModelSpec:
    checkpoint_repo_id: str
    snapshot_repo_id: str
    snapshot_file: str | None = None

    def download_snapshot(self, dtype_short_name: str) -> str:
        filename: str = self.snapshot_file or find_snapshot_file(self.snapshot_repo_id, dtype_short_name)
        return download_or_ensure_resource(repo_id=self.snapshot_repo_id, filename=filename)


_QWEN3_4B_INSTRUCT_2507 = ModelSpec('Qwen/Qwen3-4B-Instruct-2507', 'mario-sieg/Qwen3-4B-Instruct-2507-Magnetron')
_QWEN3_5_0_8B = ModelSpec('Qwen/Qwen3.5-0.8B', 'mario-sieg/Qwen3.5-0.8B-Magnetron')
_QWEN3_5_9B = ModelSpec('Qwen/Qwen3.5-9B', 'mario-sieg/Qwen3.5-9B-Magnetron')
_QWEN3_5_27B = ModelSpec('Qwen/Qwen3.5-27B', 'mario-sieg/Qwen3.5-27B-Magnetron')
_QWEN3_5_35B_A3B = ModelSpec('Qwen/Qwen3.5-35B-A3B', 'mario-sieg/Qwen3.5-35B-A3B-Magnetron')
_QWEN3_8_27B = ModelSpec('Qwen/Qwen3.8-27B', 'mario-sieg/Qwen3.8-27B-Magnetron')

MODELS_MAP: dict[str, ModelSpec] = {
    'qwen3': _QWEN3_4B_INSTRUCT_2507,
    'qwen3.5-0.8b': _QWEN3_5_0_8B,
    'qwen3.5-9b': _QWEN3_5_9B,
    'qwen3.5-27b': _QWEN3_5_27B,
    'qwen3.5-35b-a3b': _QWEN3_5_35B_A3B,
    'qwen3.8-27b': _QWEN3_8_27B,
}

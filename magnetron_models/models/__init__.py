# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import gc
import os
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


class SnapshotModule(nn.Module):
    cfg: Any

    def __init__(self) -> None:
        super().__init__()
        self.snapshot_metadata: dict[str, Any] = {}

    def load_from_snapshot(self, snapshot_file: str) -> None:
        self.load_tensors(*deserialize(snapshot_file), source=snapshot_file)

    def load_tensors(self, tensors: dict[str, Tensor], metadata: dict[str, Any], source: str = 'snapshot') -> None:
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


class ModelBase(ABC, SnapshotModule):
    def __init__(self) -> None:
        super().__init__()

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

    def sample_token(self, logits: Tensor, temp: float, top_k: int) -> int:
        """Pick the next token out of one row of logits.

        Temperature decides how, the way every OpenAI-shaped client expects it to: 0 is argmax,
        anything above it samples the top_k. Nothing else gets a vote. A strategy that outranked
        the temperature would silently ignore it -- argmax is invariant to the scaling a
        temperature applies, so both knobs would go to the caller and neither would do anything.
        """
        if temp <= 0.0:
            return int(logits.argmax(dim=0).item())
        scaled = logits / temp
        k: int = max(1, min(top_k, scaled.shape[0]))
        top_vals, top_idx = scaled.topk(k, dim=0, largest=True, sorted=False)
        return int(top_idx[top_vals.softmax(dim=-1).reshape(1, -1).multinomial(num_samples=1)[0, 0]].item())

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
    'qwen3_5': _qwen3_5,
    'qwen3_5_moe': _qwen3_5_moe,
}


def _qwen_image_2_1_text_encoder() -> tuple[Callable[[Any], SnapshotModule], Callable[..., Any]]:
    from magnetron_models.models.qwen_image_2_1 import QwenImageTextEncoder, TextEncoderConfig

    return QwenImageTextEncoder, TextEncoderConfig


def _qwen_image_2_1_transformer() -> tuple[Callable[[Any], SnapshotModule], Callable[..., Any]]:
    from magnetron_models.models.qwen_image_2_1 import QwenImageTransformer, TransformerConfig

    return QwenImageTransformer, TransformerConfig


def _qwen_image_2_1_vae() -> tuple[Callable[[Any], SnapshotModule], Callable[..., Any]]:
    from magnetron_models.models.qwen_image_2_1 import QwenImageVAE, VAEConfig

    return QwenImageVAE, VAEConfig


# Diffusion pipelines are several networks, each in its own snapshot, so each has its own architecture tag.
_COMPONENTS: dict[str, Callable[[], tuple[Callable[[Any], SnapshotModule], Callable[..., Any]]]] = {
    'qwen_image_2_1_text_encoder': _qwen_image_2_1_text_encoder,
    'qwen_image_2_1_transformer': _qwen_image_2_1_transformer,
    'qwen_image_2_1_vae': _qwen_image_2_1_vae,
}


def decode_config(config_cls: Callable[..., Any], data: dict[str, Any]) -> object:
    hints = get_type_hints(config_cls)

    def decode(hint: object, value: object) -> object:
        for candidate in (hint, *get_args(hint)):
            if value is not None and isinstance(candidate, type) and issubclass(candidate, Enum):
                return candidate(value)
        if get_origin(hint) is frozenset:
            return frozenset(value)
        if get_origin(hint) is tuple and isinstance(value, list):
            return tuple(value)  # JSON has no tuples, the manifest stores them as lists
        return value

    return config_cls(**{f.name: decode(hints[f.name], data[f.name]) for f in fields(config_cls) if f.name in data})


_decode_config = decode_config


def load_snapshot(snapshot_file: str, expect_repo_id: str | None = None) -> ModelBase:
    tensors, metadata = deserialize(snapshot_file)
    architecture: str = metadata.get('architecture', '')
    source_repo: str = metadata.get('source_repo', '')
    if architecture not in _ARCHITECTURES:
        hint = ''
        if architecture in _COMPONENTS:
            hint = ', a diffusion component, load it with load_component_snapshot'
        elif is_pipeline_snapshot(metadata):
            hint = ', a merged diffusion pipeline, run it with generate-image --snapshot'
        raise RuntimeError(f'{snapshot_file} holds a {architecture or "nameless"} model{hint}, this build runs {", ".join(sorted(_ARCHITECTURES))}')
    if expect_repo_id is not None and expect_repo_id != source_repo:
        raise ValueError(
            f'{snapshot_file} was converted from {source_repo}, but --model asked for {expect_repo_id}. Drop --model to run the snapshot.'
        )
    model_cls, config_cls = _ARCHITECTURES[architecture]()
    context.set_default_dtype(_DTYPES[metadata['dtype']])
    model: ModelBase = model_cls(_decode_config(config_cls, metadata['model_config']))
    model.load_tensors(tensors, metadata, source=snapshot_file)
    return model


PIPELINE_COMPONENTS_KEY: str = 'components'


def is_pipeline_snapshot(metadata: dict[str, Any]) -> bool:
    """A pipeline snapshot holds several networks in one file: its tensors are named <component>.<name> and its
    metadata carries each component's own manifest under 'components'. It is the only format the image pipeline
    runs; the converter writes it directly and merge-snapshots builds one out of older per-component files."""
    return isinstance(metadata.get(PIPELINE_COMPONENTS_KEY), dict)


def require_pipeline_snapshot(snapshot_file: str, metadata: dict[str, Any]) -> None:
    if is_pipeline_snapshot(metadata):
        return
    held: str = metadata.get('component') or metadata.get('architecture') or 'nameless'
    raise RuntimeError(
        f'{snapshot_file} holds a single {held} network. The pipeline runs from one snapshot with all its networks: '
        f'convert the checkpoint again, or merge the per-network files with merge-snapshots'
    )


def pipeline_architecture(component_architectures: list[str]) -> str:
    """qwen_image_2_1_text_encoder, qwen_image_2_1_transformer, ... -> qwen_image_2_1"""
    common: str = os.path.commonprefix(component_architectures).rstrip('_')
    if not common:
        raise ValueError(f'Component architectures share no prefix: {", ".join(component_architectures)}')
    return common


def build_pipeline_metadata(components: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The top-level metadata of a pipeline snapshot, derived from its components' own metadata. Every component
    must describe the same checkpoint; dtypes may differ per component (e.g. an fp32 VAE next to a bf16
    transformer), the top-level dtype is the one most components use and each component's own entry is
    authoritative when it loads."""
    if len(components) < 2:
        raise ValueError('A pipeline snapshot needs at least two components')
    for key in ('source_repo', 'model'):
        values = {str(meta.get(key)) for meta in components.values()}
        if len(values) != 1:
            raise ValueError(f'Components disagree on {key}: {", ".join(sorted(values))}')
    for component, meta in components.items():
        if not meta.get('architecture'):
            raise ValueError(f'Component {component} carries no architecture tag')
    first: dict[str, Any] = next(iter(components.values()))
    dtypes: list[str] = [meta['dtype'] for meta in components.values()]
    return {
        'source_repo': first.get('source_repo'),
        'source_format': first.get('source_format'),
        'architecture': pipeline_architecture([meta['architecture'] for meta in components.values()]),
        'model': first['model'],
        'dtype': max(set(dtypes), key=dtypes.count),
        PIPELINE_COMPONENTS_KEY: dict(components),
    }


def pipeline_component_metadata(metadata: dict[str, Any], architecture: str) -> tuple[str, dict[str, Any]]:
    """Find the component of a merged snapshot that carries the given architecture tag, as (component, metadata)."""
    components: dict[str, dict[str, Any]] = metadata[PIPELINE_COMPONENTS_KEY]
    for component, sub in components.items():
        if sub.get('architecture') == architecture:
            return component, sub
    raise RuntimeError(f'Merged snapshot holds {", ".join(sorted(components))} but no {architecture} component')


def split_pipeline_snapshot(tensors: dict[str, Tensor], metadata: dict[str, Any], architecture: str) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Carve one component out of a merged snapshot: its tensors with the component prefix stripped and its own
    metadata, exactly what deserialize() returns for a single-component file."""
    component, sub = pipeline_component_metadata(metadata, architecture)
    prefix: str = f'{component}.'
    return {name[len(prefix) :]: tensor for name, tensor in tensors.items() if name.startswith(prefix)}, sub


def load_component_snapshot(snapshot_file: str, expect_architecture: str, expect_repo_id: str | None = None) -> SnapshotModule:
    """Load one network out of a pipeline snapshot. The file is memory-mapped, so only the requested component's
    tensors are read and moved to the device; the other networks cost a header parse and nothing else."""
    tensors, metadata = deserialize(snapshot_file)
    require_pipeline_snapshot(snapshot_file, metadata)
    tensors, metadata = split_pipeline_snapshot(tensors, metadata, expect_architecture)
    architecture: str = metadata.get('architecture', '')
    source_repo: str = metadata.get('source_repo', '')
    if architecture not in _COMPONENTS:
        raise RuntimeError(f'{snapshot_file} holds a {architecture or "nameless"} component, this build knows {", ".join(sorted(_COMPONENTS))}')
    if expect_repo_id is not None and expect_repo_id != source_repo:
        raise ValueError(f'{snapshot_file} was converted from {source_repo}, but the pipeline asked for {expect_repo_id}')
    model_cls, config_cls = _COMPONENTS[architecture]()
    context.set_default_dtype(_DTYPES[metadata['dtype']])
    model: SnapshotModule = model_cls(_decode_config(config_cls, metadata['model_config']))
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


@dataclass(frozen=True, slots=True)
class DiffusionModelSpec:
    """A text-to-image pipeline is one snapshot holding all of its networks, named <stem>-<dtype>.mag."""

    checkpoint_repo_id: str
    snapshot_repo_id: str
    snapshot_stem: str

    def snapshot_file(self, dtype_short_name: str) -> str:
        return f'{self.snapshot_stem}-{dtype_short_name}.mag'

    def download_snapshot(self, dtype_short_name: str) -> str:
        return download_or_ensure_resource(repo_id=self.snapshot_repo_id, filename=self.snapshot_file(dtype_short_name))


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

_QWEN_IMAGE_2_1 = DiffusionModelSpec('Qwen/Qwen-Image-2.1', 'mario-sieg/Qwen-Image-2.1-Magnetron', 'qwen-image-2.1')

DIFFUSION_MODELS_MAP: dict[str, DiffusionModelSpec] = {
    'qwen-image-2.1': _QWEN_IMAGE_2_1,
}

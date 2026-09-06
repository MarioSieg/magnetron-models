# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from magnetron import Tensor, dtype
from magnetron.snapshot import SnapshotWriter

from magnetron._magnetron_bindings import SnapshotStreamReader
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.table import Table
from safetensors import safe_open

import torch

console = Console()

_SKIPPED_HF_SUFFIXES: frozenset[str] = frozenset({'rotary_emb.inv_freq'})
_DROPPED_MAG_PREFIXES: tuple[str, ...] = ('visual.', 'vision_tower.', 'mtp.')

_TORCH_BY_MAG: dict[dtype.DType, torch.dtype] = {
    dtype.float16: torch.float16,
    dtype.bfloat16: torch.bfloat16,
    dtype.float32: torch.float32,
}

_MAG_BY_NAME: dict[str, dtype.DType] = {dt.name: dt for dt in _TORCH_BY_MAG}

MagKeyFor = Callable[[str], str | None]
"""Maps a Hugging Face tensor name to a Magnetron parameter name, or None to drop it."""


def mag_dtype_from_str(dtype_str: str) -> dtype.DType:
    return _MAG_BY_NAME[dtype_str]


def fmt_bytes(n: float) -> str:
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(n) < 1024.0 or unit == 'TiB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.2f} {unit}'
        n /= 1024.0
    raise AssertionError


def json_safe(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: json_safe(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(json_safe(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


def download_repo(repo: str) -> str:
    console.print(f'Downloading model {repo} from Hugging Face...', style='dim')
    return snapshot_download(repo_id=repo, ignore_patterns=['*.pt', '*.bin'])


def load_hf_config(repo_dir: str) -> dict[str, Any]:
    path = os.path.join(repo_dir, 'config.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_tokenizer_json(repo_dir: str) -> str | None:
    path = os.path.join(repo_dir, 'tokenizer.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return f.read()


def iter_safetensor_shards(repo_dir: str) -> list[str]:
    index_path = os.path.join(repo_dir, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path, encoding='utf-8') as f:
            index = json.load(f)
        shards = sorted(set(index['weight_map'].values()))
        return [os.path.join(repo_dir, s) for s in shards]
    shards = sorted(glob.glob(os.path.join(repo_dir, 'model-*.safetensors')))
    if shards:
        return shards
    single = os.path.join(repo_dir, 'model.safetensors')
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError('No safetensors weights found in repo snapshot.')


@dataclass(frozen=True, slots=True)
class TensorPlan:
    shard: str
    hf_key: str
    mag_key: str
    shape: tuple[int, ...]
    dtype: dtype.DType

    @property
    def numbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.size


def plan_tensors(repo_dir: str, *, mag_key_for: MagKeyFor, dtype_for: Callable[[str], dtype.DType]) -> list[TensorPlan]:
    """Read only the shard headers, so the plan costs nothing regardless of checkpoint size."""
    plan: list[TensorPlan] = []
    seen: dict[str, str] = {}
    for shard in iter_safetensor_shards(repo_dir):
        with safe_open(shard, framework='pt') as f:
            for hf_key in sorted(f.keys()):
                if any(hf_key.endswith(skip) for skip in _SKIPPED_HF_SUFFIXES):
                    continue
                mag_key = mag_key_for(hf_key)
                if mag_key is None or mag_key.startswith(_DROPPED_MAG_PREFIXES):
                    continue
                if mag_key in seen:
                    raise KeyError(f'{mag_key} appears in both {os.path.basename(seen[mag_key])} and {os.path.basename(shard)}')
                seen[mag_key] = shard
                plan.append(
                    TensorPlan(
                        shard=shard,
                        hf_key=hf_key,
                        mag_key=mag_key,
                        shape=tuple(f.get_slice(hf_key).get_shape()),
                        dtype=dtype_for(mag_key),
                    )
                )
    if not plan:
        raise RuntimeError('No convertible tensors found in the safetensors shards.')
    return plan


def dtype_policy(mag_dtype: dtype.DType, fp32_suffixes: Iterable[str] = ()) -> Callable[[str], dtype.DType]:
    keep = tuple(fp32_suffixes)
    return lambda mag_key: dtype.float32 if mag_key.endswith(keep) else mag_dtype


def check_shapes(plan: list[TensorPlan], expected: dict[str, tuple[int, ...]]) -> None:
    found = {entry.mag_key: entry.shape for entry in plan}
    for key, shape in expected.items():
        got = found.get(key)
        if got is None:
            raise KeyError(f'Checkpoint has no tensor for {key}')
        if got != shape:
            raise ValueError(f'{key} is {got} in the checkpoint but the config implies {shape}')


def check_layers(plan: list[TensorPlan], num_hidden_layers: int, attn_module_for: Callable[[int], str]) -> None:
    modules: dict[int, set[str]] = {}
    for entry in plan:
        parts = entry.mag_key.split('.')
        if len(parts) > 2 and parts[0] == 'layers' and parts[1].isdigit():
            modules.setdefault(int(parts[1]), set()).add(parts[2])
    indices = set(modules)
    if indices != set(range(num_hidden_layers)):
        raise ValueError(f'Checkpoint has layers {sorted(indices)[:4]}...{sorted(indices)[-4:]}, config says {num_hidden_layers} layers')
    for i in sorted(indices):
        want = attn_module_for(i)
        if want not in modules[i]:
            raise ValueError(f'Layer {i} should be a {want} layer but the checkpoint has {sorted(modules[i])}')


def _load_one(entry: TensorPlan) -> Tensor:
    with safe_open(entry.shard, framework='pt') as f:
        src = f.get_tensor(entry.hf_key)
        src = src.to(_TORCH_BY_MAG[entry.dtype]).contiguous()
        out = Tensor(src, dtype=entry.dtype)
    del src
    return out


@dataclass(frozen=True, slots=True)
class SnapshotStats:
    """What the snapshot costs on disk: measured after a conversion, planned from the tensor plan for --card-only."""

    repo: str
    snap_file: str
    mag_dtype: dtype.DType
    tensor_count: int
    payload_numbytes: int
    blob_numbytes: int
    source_numbytes: int
    tokenizer_numbytes: int
    metadata_numbytes: int | None = None  # The manifest is only known once a snapshot exists, it is never predicted.
    file_numbytes: int | None = None
    elapsed: float | None = None

    @property
    def planned(self) -> bool:
        return self.file_numbytes is None

    @property
    def padding_numbytes(self) -> int:
        return self.blob_numbytes - self.payload_numbytes

    @property
    def container_numbytes(self) -> int | None:
        if self.file_numbytes is None or self.metadata_numbytes is None:
            return None
        return self.file_numbytes - self.blob_numbytes - self.metadata_numbytes

    def rows(self) -> list[tuple[str, str]]:
        """The one description of the snapshot, rendered both to the terminal table and to the model card."""
        padding = self.padding_numbytes
        rows: list[tuple[str, str]] = [
            ('Source', self.repo),
            ('DType', self.mag_dtype.name),
            ('Tensors', f'{self.tensor_count}'),
            ('Payload', fmt_bytes(self.payload_numbytes)),
            ('Alignment padding', f'{fmt_bytes(padding)} ({padding / self.blob_numbytes:.3%})'),
            ('Data section', fmt_bytes(self.blob_numbytes)),
        ]
        tokenizer_note = f' (incl. {fmt_bytes(self.tokenizer_numbytes)} tokenizer)' if self.tokenizer_numbytes else ' (no tokenizer)'
        if self.metadata_numbytes is not None:
            rows.append(('Metadata', f'{fmt_bytes(self.metadata_numbytes)}{tokenizer_note}'))
        elif self.tokenizer_numbytes:
            rows.append(('Tokenizer in metadata', fmt_bytes(self.tokenizer_numbytes)))
        container = self.container_numbytes
        if container is not None:
            rows.append(('Container overhead', fmt_bytes(container)))
        if self.file_numbytes is not None:
            rows.append(('File size', fmt_bytes(self.file_numbytes)))
            rows.append(('Source shards', f'{fmt_bytes(self.source_numbytes)} ({self.file_numbytes / self.source_numbytes:.2f}x)'))
        else:
            rows.append(('Source shards', fmt_bytes(self.source_numbytes)))
        if self.elapsed is not None:
            rows.append(('Elapsed', f'{self.elapsed:.1f} s'))
            rows.append(('Throughput', f'{fmt_bytes(self.blob_numbytes / self.elapsed)}/s'))
        return rows


def _measure_snapshot(snap_file: str) -> tuple[int, int] | None:
    """Data section and manifest size of an already written snapshot, or None if there is no file to measure.

    The reader maps the file and reads its header, so this costs the same on a 500 GiB snapshot as on a tiny one.
    """
    if not os.path.exists(snap_file):
        return None
    with SnapshotStreamReader(snap_file) as reader:
        return reader.blob_numbytes, len(reader.metadata.encode('utf-8'))


def _plan_stats(
    snap_file: str,
    *,
    repo: str,
    mag_dtype: dtype.DType,
    plan: list[TensorPlan],
    metadata: dict[str, Any],
    source_numbytes: int,
    tokenizer_numbytes: int,
) -> SnapshotStats:
    probe = SnapshotWriter(snap_file, metadata)
    for entry in plan:
        probe.declare(entry.mag_key, entry.shape, entry.dtype)
    written = _measure_snapshot(snap_file)
    if written is not None and written[0] != probe.blob_numbytes:
        console.print(
            f'{snap_file} holds a {fmt_bytes(written[0])} data section but this plan lays out '
            f'{fmt_bytes(probe.blob_numbytes)}, so the card reports the plan',
            style='yellow',
        )
        written = None
    return SnapshotStats(
        repo=repo,
        snap_file=snap_file,
        mag_dtype=mag_dtype,
        tensor_count=probe.tensor_count,
        payload_numbytes=probe.payload_numbytes,
        blob_numbytes=probe.blob_numbytes,
        source_numbytes=source_numbytes,
        tokenizer_numbytes=tokenizer_numbytes,
        metadata_numbytes=written[1] if written is not None else None,
        file_numbytes=os.path.getsize(snap_file) if written is not None else None,
    )


def _write_model_card(
    path: str,
    *,
    stats: SnapshotStats,
    config_title: str,
    cfg: object,
    plan: list[TensorPlan],
    has_tokenizer: bool,
) -> None:
    repo = stats.repo
    model_name = repo.split('/')[-1]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# {model_name} Magnetron Snapshot\n\n')
        f.write(f'This repository contains a Magnetron snapshot converted from the original Hugging Face model `{repo}`.\n\n')
        f.write('The snapshot is intended for inference with the Magnetron runtime. ')
        f.write(f'All convertible tensors are stored using `{stats.mag_dtype.short_name}` where applicable.\n\n')
        f.write('## Model details\n\n')
        f.write(f'- **Source model:** `{repo}`\n')
        f.write(f'- **Snapshot file:** `{stats.snap_file}`\n')
        f.write(f'- **Magnetron dtype mode:** `{stats.mag_dtype.short_name}`\n')
        f.write(f'- **Tokenizer:** {"embedded in the snapshot metadata" if has_tokenizer else "not included, bring your own"}\n\n')
        f.write('## Snapshot\n\n')
        if stats.planned:
            f.write('> Sizes are planned from the conversion plan, no snapshot was written next to this card.\n\n')
        f.write('| Metric | Value |\n')
        f.write('|---|---:|\n')
        for label, value in stats.rows():
            f.write(f'| {label} | `{value}` |\n')
        f.write('\n')
        f.write(f'## {config_title}\n\n')
        f.write('| Field | Value |\n')
        f.write('|---|---:|\n')
        for k, v in json_safe(cfg).items():
            if isinstance(v, (dict, list)):
                continue
            f.write(f'| `{k}` | `{v}` |\n')
        f.write('\n')
        f.write('## Tensor manifest\n\n')
        f.write('| Name | Shape | DType |\n')
        f.write('|---|---:|---|\n')
        for entry in sorted(plan, key=lambda e: e.mag_key):
            shape_s = 'x'.join(str(x) for x in entry.shape)
            f.write(f'| `{entry.mag_key}` | `{shape_s}` | `{entry.dtype.short_name}` |\n')


def _print_stats(stats: SnapshotStats) -> None:
    table = Table(title=stats.snap_file, title_style='bold', show_header=False, box=None, pad_edge=False)
    table.add_column(style='dim')
    table.add_column(justify='right')
    for label, value in stats.rows():
        table.add_row(label, value)
    console.print()
    console.print(table)


def convert_repo(
    repo: str,
    repo_dir: str,
    plan: list[TensorPlan],
    *,
    mag_dtype: dtype.DType,
    architecture: str,
    model: str,
    cfg: object,
    config_title: str,
    out: str | None = None,
    write_model_card: bool = False,
    model_card_path: str = 'model_card.md',
    card_only: bool = False,
) -> str:
    hf_config = load_hf_config(repo_dir)
    tokenizer_json = load_tokenizer_json(repo_dir)
    if tokenizer_json is None:
        console.print(f'{repo} ships no tokenizer.json, the snapshot will need one from elsewhere', style='yellow')

    total_bytes = sum(entry.numbytes for entry in plan)
    source_numbytes = sum(os.path.getsize(shard) for shard in dict.fromkeys(entry.shard for entry in plan))
    tokenizer_numbytes = len(tokenizer_json.encode('utf-8')) if tokenizer_json else 0
    snap_file: str = out or f'{repo.split("/")[-1].lower()}-{mag_dtype.short_name}.mag'

    metadata: dict[str, Any] = {
        'source_repo': repo,
        'source_format': 'safetensors',
        'architecture': architecture,
        'model': model,
        'dtype': mag_dtype.name,
        'model_config': json_safe(cfg),
        'hf_config': hf_config,
    }
    if tokenizer_json is not None:
        metadata['tokenizer_json'] = tokenizer_json

    if card_only:
        console.print(f'Planning {len(plan)} tensors ({fmt_bytes(total_bytes)} of {mag_dtype.short_name}) for {snap_file}, no weights written', style='dim')
        stats = _plan_stats(
            snap_file,
            repo=repo,
            mag_dtype=mag_dtype,
            plan=plan,
            metadata=metadata,
            source_numbytes=source_numbytes,
            tokenizer_numbytes=tokenizer_numbytes,
        )
    else:
        console.print(f'Writing {len(plan)} tensors ({fmt_bytes(total_bytes)} of {mag_dtype.short_name}) to {snap_file}', style='dim')
        start = time.perf_counter()
        with SnapshotWriter(snap_file, metadata) as snap:
            for entry in plan:
                snap.declare(entry.mag_key, entry.shape, entry.dtype)
            with Progress(
                TextColumn('{task.fields[name]}', style='cyan'),
                BarColumn(),
                TaskProgressColumn(),
                DownloadColumn(binary_units=True),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task('convert', total=total_bytes, name='')
                for entry in plan:
                    progress.update(task, name=f'{entry.mag_key[-38:]:<38}')
                    snap.write(entry.mag_key, lambda entry=entry: _load_one(entry))
                    progress.advance(task, entry.numbytes)
        stats = SnapshotStats(
            repo=repo,
            snap_file=snap_file,
            mag_dtype=mag_dtype,
            tensor_count=snap.tensor_count,
            payload_numbytes=snap.payload_numbytes,
            blob_numbytes=snap.blob_numbytes,
            source_numbytes=source_numbytes,
            tokenizer_numbytes=tokenizer_numbytes,
            metadata_numbytes=snap.metadata_numbytes,
            file_numbytes=os.path.getsize(snap_file),
            elapsed=time.perf_counter() - start,
        )

    if write_model_card or card_only:
        _write_model_card(
            model_card_path,
            stats=stats,
            config_title=config_title,
            cfg=cfg,
            plan=plan,
            has_tokenizer=tokenizer_json is not None,
        )
        console.print(f'Model card saved to {model_card_path}', style='dim')
    _print_stats(stats)
    return snap_file


def build_arg_parser(description: str, *, default_model: str, known_models: Iterable[str] = ()) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    known = ', '.join(sorted(known_models))
    parser.add_argument(
        '--model',
        type=str,
        default=default_model,
        # Not a choices= list: the converter reads the shipped config.json, so unlisted sizes convert too.
        help=f'HF repo model name{f" (known: {known})" if known else ""}',
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Snapshot output path, defaults to <model>-<dtype>.mag in the working directory',
    )
    parser.add_argument(
        '--model-card',
        action='store_true',
        help='Write a Hugging Face-style model_card.md with tensor manifest',
    )
    parser.add_argument(
        '--model-card-path',
        type=str,
        default='model_card.md',
        help='Output path for the generated model card',
    )
    parser.add_argument(
        '--card-only',
        action='store_true',
        help='Write only the model card, no snapshot: sizes come from the conversion plan, or from the .mag already sitting at --out',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=sorted(_MAG_BY_NAME.keys()),
        help='Data type for Magnetron tensors',
    )
    return parser

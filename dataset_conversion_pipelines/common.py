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


def _write_model_card(
    path: str,
    *,
    repo: str,
    snap_file: str,
    mag_dtype: dtype.DType,
    config_title: str,
    cfg: object,
    plan: list[TensorPlan],
    has_tokenizer: bool,
) -> None:
    model_name = repo.split('/')[-1]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# {model_name} Magnetron Snapshot\n\n')
        f.write(f'This repository contains a Magnetron snapshot converted from the original Hugging Face model `{repo}`.\n\n')
        f.write('The snapshot is intended for inference with the Magnetron runtime. ')
        f.write(f'All convertible tensors are stored using `{mag_dtype.short_name}` where applicable.\n\n')
        f.write('## Model details\n\n')
        f.write(f'- **Source model:** `{repo}`\n')
        f.write(f'- **Snapshot file:** `{snap_file}`\n')
        f.write(f'- **Magnetron dtype mode:** `{mag_dtype.short_name}`\n')
        f.write(f'- **Tensor count:** `{len(plan)}`\n')
        f.write(f'- **Tokenizer:** {"embedded in the snapshot metadata" if has_tokenizer else "not included, bring your own"}\n\n')
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


def _print_stats(
    snap_file: str,
    *,
    repo: str,
    mag_dtype: dtype.DType,
    snap: SnapshotWriter,
    source_numbytes: int,
    elapsed: float,
    tokenizer_numbytes: int,
) -> None:
    file_numbytes = os.path.getsize(snap_file)
    payload = snap.payload_numbytes
    blob = snap.blob_numbytes
    meta = snap.metadata_numbytes
    padding = blob - payload
    container = file_numbytes - blob - meta
    tokenizer_note = f' (incl. {fmt_bytes(tokenizer_numbytes)} tokenizer)' if tokenizer_numbytes else ' (no tokenizer)'
    table = Table(title=snap_file, title_style='bold', show_header=False, box=None, pad_edge=False)
    table.add_column(style='dim')
    table.add_column(justify='right')
    table.add_row('Source', repo)
    table.add_row('DType', mag_dtype.name)
    table.add_row('Tensors', f'{snap.tensor_count}')
    table.add_row('Payload', fmt_bytes(payload))
    table.add_row('Alignment padding', f'{fmt_bytes(padding)} ({padding / blob:.3%})')
    table.add_row('Data section', fmt_bytes(blob))
    table.add_row('Metadata', f'{fmt_bytes(meta)}{tokenizer_note}')
    table.add_row('Container overhead', fmt_bytes(container))
    table.add_row('File size', fmt_bytes(file_numbytes))
    table.add_row('Source shards', f'{fmt_bytes(source_numbytes)} ({file_numbytes / source_numbytes:.2f}x)')
    table.add_row('Elapsed', f'{elapsed:.1f} s')
    table.add_row('Throughput', f'{fmt_bytes(blob / elapsed)}/s')
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
) -> str:
    hf_config = load_hf_config(repo_dir)
    tokenizer_json = load_tokenizer_json(repo_dir)
    if tokenizer_json is None:
        console.print(f'{repo} ships no tokenizer.json, the snapshot will need one from elsewhere', style='yellow')

    total_bytes = sum(entry.numbytes for entry in plan)
    source_numbytes = sum(os.path.getsize(shard) for shard in dict.fromkeys(entry.shard for entry in plan))
    snap_file: str = out or f'{repo.split("/")[-1].lower()}-{mag_dtype.short_name}.mag'
    console.print(f'Writing {len(plan)} tensors ({fmt_bytes(total_bytes)} of {mag_dtype.short_name}) to {snap_file}', style='dim')

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
    elapsed = time.perf_counter() - start

    if write_model_card:
        _write_model_card(
            model_card_path,
            repo=repo,
            snap_file=snap_file,
            mag_dtype=mag_dtype,
            config_title=config_title,
            cfg=cfg,
            plan=plan,
            has_tokenizer=tokenizer_json is not None,
        )
        console.print(f'Model card saved to {model_card_path}', style='dim')
    _print_stats(
        snap_file,
        repo=repo,
        mag_dtype=mag_dtype,
        snap=snap,
        source_numbytes=source_numbytes,
        elapsed=elapsed,
        tokenizer_numbytes=len(tokenizer_json.encode('utf-8')) if tokenizer_json else 0,
    )
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
        '--dtype',
        type=str,
        default='bfloat16',
        choices=sorted(_MAG_BY_NAME.keys()),
        help='Data type for Magnetron tensors',
    )
    return parser

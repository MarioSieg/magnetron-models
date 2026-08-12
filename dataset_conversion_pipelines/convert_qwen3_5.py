# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import argparse
import json
import os
import glob
import gc

from magnetron import Snapshot, Tensor, dtype, context
from magnetron_models.models.qwen3_5 import Qwen35Model, Config, CONFIGS
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import torch

_TEXT_PREFIX: str = 'model.language_model.'


def _mag_to_torch_dtype(mag_dtype: dtype.DType) -> torch.dtype:
    return {
        dtype.float16: torch.float16,
        dtype.bfloat16: torch.bfloat16,
        dtype.float32: torch.float32,
    }[mag_dtype]


def _mag_dtype_from_str(dtype_str: str) -> dtype.DType:
    return {
        'float16': dtype.float16,
        'bfloat16': dtype.bfloat16,
        'float32': dtype.float32,
    }[dtype_str]


def _iter_safetensor_shards(repo_dir: str) -> list[str]:
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


def _config_for(repo: str, repo_dir: str) -> Config:
    """Prefer the shipped config.json so unknown checkpoint sizes convert without a code change."""
    cfg = CONFIGS.get(repo, Config(repo_id=repo))
    path = os.path.join(repo_dir, 'config.json')
    if not os.path.exists(path):
        return cfg
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    text = raw.get('text_config', raw)
    rope = text.get('rope_parameters', {})
    return Config(
        repo_id=repo,
        vocab_size=text.get('vocab_size', cfg.vocab_size),
        hidden_size=text.get('hidden_size', cfg.hidden_size),
        intermediate_size=text.get('intermediate_size', cfg.intermediate_size),
        num_hidden_layers=text.get('num_hidden_layers', cfg.num_hidden_layers),
        num_attention_heads=text.get('num_attention_heads', cfg.num_attention_heads),
        num_key_value_heads=text.get('num_key_value_heads', cfg.num_key_value_heads),
        head_dim=text.get('head_dim', cfg.head_dim),
        max_position_embeddings=cfg.max_position_embeddings,  # Deliberately not the checkpoint's 262144.
        rms_norm_eps=text.get('rms_norm_eps', cfg.rms_norm_eps),
        tie_word_embeddings=text.get('tie_word_embeddings', cfg.tie_word_embeddings),
        rope_theta=rope.get('rope_theta', cfg.rope_theta),
        partial_rotary_factor=rope.get('partial_rotary_factor', cfg.partial_rotary_factor),
        full_attention_interval=text.get('full_attention_interval', cfg.full_attention_interval),
        linear_conv_kernel_dim=text.get('linear_conv_kernel_dim', cfg.linear_conv_kernel_dim),
        linear_key_head_dim=text.get('linear_key_head_dim', cfg.linear_key_head_dim),
        linear_value_head_dim=text.get('linear_value_head_dim', cfg.linear_value_head_dim),
        linear_num_key_heads=text.get('linear_num_key_heads', cfg.linear_num_key_heads),
        linear_num_value_heads=text.get('linear_num_value_heads', cfg.linear_num_value_heads),
    )


def _write_model_card(
    path: str,
    *,
    repo: str,
    snap_file: str,
    mag_dtype: dtype.DType,
    cfg: Config,
    tensor_rows: list[tuple[str, tuple[int, ...], str]],
) -> None:
    tensor_rows = sorted(tensor_rows, key=lambda x: x[0])
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
        f.write(f'- **Tensor count:** `{len(tensor_rows)}`\n\n')
        f.write('## Qwen3.5 configuration\n\n')
        f.write('| Field | Value |\n')
        f.write('|---|---:|\n')
        for k, v in vars(cfg).items():
            f.write(f'| `{k}` | `{v}` |\n')
        f.write('\n')
        f.write('## Tensor manifest\n\n')
        f.write('| Name | Shape | DType |\n')
        f.write('|---|---:|---|\n')
        for name, shape, dt in tensor_rows:
            shape_s = 'x'.join(str(x) for x in shape)
            f.write(f'| `{name}` | `{shape_s}` | `{dt}` |\n')


def _convert_model(
    repo: str,
    torch_dtype: torch.dtype,
    mag_dtype: dtype.DType,
    *,
    write_model_card: bool = False,
    model_card_path: str = 'model_card.md',
) -> None:
    print(f'Downloading model {repo} from Hugging Face...')
    repo_dir = snapshot_download(repo_id=repo, ignore_patterns=['*.pt', '*.bin'])
    context.stop_grad_recorder()
    context.set_default_dtype(mag_dtype)
    cfg = _config_for(repo, repo_dir)
    mag_model = Qwen35Model(cfg)  # Not cast as a whole: A_log and dt_bias stay float32, like the reference.
    remaining: dict[str, Tensor] = dict(mag_model.state_dict())

    def hf_key_for(mag_key: str) -> str:
        if mag_key.startswith('lm_head.'):
            return mag_key
        return _TEXT_PREFIX + mag_key

    snap_file: str = f'{repo.split("/")[1].lower()}-{mag_dtype.short_name}.mag'
    tensor_manifest: list[tuple[str, tuple[int, ...], str]] = []
    print(f'Writing snapshot to {snap_file}...')
    with Snapshot.write(snap_file) as snap:
        for shard_path in _iter_safetensor_shards(repo_dir):
            hf_state_dict: dict[str, torch.Tensor] = load_file(shard_path, device='cpu')
            processed_stack: list[str] = []
            for key in list(remaining.keys()):
                hf_key: str = hf_key_for(key)
                torch_tensor: torch.Tensor | None = hf_state_dict.get(hf_key)
                if torch_tensor is None:
                    continue
                target_dtype = remaining[key].dtype
                print(f'Converting {hf_key} -> {key} shape={tuple(torch_tensor.shape)} dtype={target_dtype.short_name}')
                cast_to = torch_dtype if target_dtype == mag_dtype else _mag_to_torch_dtype(target_dtype)
                out_tensor = Tensor(torch_tensor.to(cast_to).to('cpu').contiguous(), dtype=target_dtype)
                if tuple(out_tensor.shape) != tuple(remaining[key].shape):
                    raise RuntimeError(f'Shape mismatch for {key}: {tuple(out_tensor.shape)} != {tuple(remaining[key].shape)}')
                snap.put_tensor(key, out_tensor)
                tensor_manifest.append((key, tuple(out_tensor.shape), out_tensor.dtype.short_name))
                processed_stack.append(key)
                del out_tensor
                gc.collect()
            for k in processed_stack:
                remaining.pop(k, None)
            del hf_state_dict
            gc.collect()
        if remaining:
            raise KeyError(f'Missing HF weights for magnetron keys: {sorted(remaining.keys())}')
        snap.print_info()
    if write_model_card:
        _write_model_card(
            model_card_path,
            repo=repo,
            snap_file=snap_file,
            mag_dtype=mag_dtype,
            cfg=cfg,
            tensor_rows=tensor_manifest,
        )
        print(f'Model card saved to {model_card_path}')
    print(f'Converted model saved to {snap_file}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert a dense Hugging Face Qwen3.5 model to Magnetron file format')
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3.5-4B',
        choices=sorted(CONFIGS.keys()),
        help='HF repo model name',
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
        choices=['float16', 'bfloat16', 'float32'],
        help='Data type for Magnetron tensors',
    )
    args = parser.parse_args()
    mag_dtype = _mag_dtype_from_str(args.dtype)
    _convert_model(
        args.model,
        torch_dtype=_mag_to_torch_dtype(mag_dtype),
        mag_dtype=mag_dtype,
        write_model_card=args.model_card,
        model_card_path=args.model_card_path,
    )


if __name__ == '__main__':
    main()

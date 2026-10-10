# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Both converters' weight loops, driven over a synthetic checkpoint.

Needs torch, so it is skipped unless the dev extra is installed. What it guards is the wiring the real
conversion depends on and that a download would otherwise be the only way to reach: the model.visual.*
rename, the Conv3d kernel flattening, and that every key the model declares is found and written.
"""

import json
import os
import tempfile
from types import ModuleType

import pytest

torch = pytest.importorskip('torch')

from magnetron import Snapshot, context, dtype  # noqa: E402

context.stop_grad_recorder()
context.set_default_device('cpu')
context.set_default_dtype(dtype.bfloat16)

import dataset_conversion_pipelines.convert_qwen3_5 as dense_conv  # noqa: E402
import dataset_conversion_pipelines.convert_qwen3_5_moe as moe_conv  # noqa: E402
from magnetron_models.models.qwen3_5 import Config as DenseConfig, Qwen35Model  # noqa: E402
from magnetron_models.models.qwen3_5.vision import VisionConfig  # noqa: E402
from magnetron_models.models.qwen3_5_moe import Config as MoeConfig, Qwen35MoeModel  # noqa: E402

TINY_VISION = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4, num_position_embeddings=2304)


def _convert_synthetic(mod: ModuleType, model_cls: type, cfg: object, repo: str, tmp: str, text_prefix: str) -> set[str]:
    """Write a fake HF snapshot whose tensors match the magnetron model, convert it, return the snapshot keys."""
    os.makedirs(tmp, exist_ok=True)
    vision = dict(TINY_VISION, out_hidden_size=cfg.hidden_size, patch_size=16, temporal_patch_size=2, spatial_merge_size=2, in_channels=3)
    with open(f'{tmp}/config.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'vision_config': vision,
                'text_config': {'hidden_size': cfg.hidden_size},
                'image_token_id': 248056,
                'video_token_id': 248057,
                'vision_start_token_id': 248053,
                'vision_end_token_id': 248054,
            },
            f,
        )
    with open(f'{tmp}/preprocessor_config.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'size': {'shortest_edge': 65536, 'longest_edge': 16777216},
                'patch_size': 16,
                'temporal_patch_size': 2,
                'merge_size': 2,
                'image_mean': [0.5] * 3,
                'image_std': [0.5] * 3,
            },
            f,
        )
    model = model_cls(cfg)
    shard = {}
    for name, tensor in model.state_items():
        shape = tuple(tensor.shape)
        if name == 'visual.patch_embed.proj.weight':
            shape = (shape[0], 3, 2, 16, 16)  # The checkpoint stores the unflattened Conv3d kernel.
        key = name if name.startswith('lm_head.') else ('model.' + name if name.startswith('visual.') else text_prefix + name)
        shard[key] = torch.zeros(shape, dtype=torch.bfloat16 if tensor.dtype == dtype.bfloat16 else torch.float32)
    del model
    mod.snapshot_download = lambda **kwargs: tmp
    mod._iter_safetensor_shards = lambda repo_dir: ['synthetic']
    mod.load_file = lambda path, device='cpu': shard
    mod.CONFIGS = {**mod.CONFIGS, repo: cfg}
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        mod._convert_model(repo, torch_dtype=torch.bfloat16, mag_dtype=dtype.bfloat16)
        written = [f for f in os.listdir('.') if f.endswith('.mag')]
        assert len(written) == 1, written
        with Snapshot.read(written[0]) as snap:
            return set(snap.tensor_keys())
    finally:
        os.chdir(cwd)


def _dense_config() -> DenseConfig:
    return DenseConfig(
        repo_id='synthetic/dense',
        vocab_size=512,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,
        full_attention_interval=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        tie_word_embeddings=False,
        vision=VisionConfig(**TINY_VISION, out_hidden_size=64),
    )


def _moe_config() -> MoeConfig:
    return MoeConfig(
        repo_id='synthetic/moe',
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,
        full_attention_interval=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=False,
        vision=VisionConfig(**TINY_VISION, out_hidden_size=64),
    )


def test_dense_converter_writes_the_vision_tower() -> None:
    with tempfile.TemporaryDirectory() as root:
        keys = _convert_synthetic(dense_conv, Qwen35Model, _dense_config(), 'synthetic/dense', f'{root}/d', 'model.language_model.')
    vision = {k for k in keys if k.startswith('visual.')}
    assert 'visual.patch_embed.proj.weight' in vision
    assert 'visual.pos_embed.weight' in vision
    assert 'visual.merger.linear_fc2.bias' in vision
    assert len(vision) == 33


def test_moe_converter_writes_the_vision_tower() -> None:
    with tempfile.TemporaryDirectory() as root:
        keys = _convert_synthetic(moe_conv, Qwen35MoeModel, _moe_config(), 'synthetic/moe', f'{root}/m', 'model.language_model.')
    vision = {k for k in keys if k.startswith('visual.')}
    assert len(vision) == 33
    assert any(k.startswith('layers.0.mlp.experts') for k in keys), 'the MoE experts must still convert'


def test_both_converters_agree_on_the_vision_key_set() -> None:
    with tempfile.TemporaryDirectory() as root:
        dense = _convert_synthetic(dense_conv, Qwen35Model, _dense_config(), 'synthetic/dense', f'{root}/d', 'model.language_model.')
        moe = _convert_synthetic(moe_conv, Qwen35MoeModel, _moe_config(), 'synthetic/moe', f'{root}/m', 'model.language_model.')
    assert {k for k in dense if k.startswith('visual.')} == {k for k in moe if k.startswith('visual.')}

# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from enum import Enum
from typing import Protocol

from dataset_conversion_pipelines import common

_MULTIMODAL_TEXT_PREFIX: str = 'model.language_model.'
_CAUSAL_LM_PREFIX: str = 'model.'

FP32_SUFFIXES: tuple[str, ...] = ('.A_log', '.dt_bias')


class HybridConfig(Protocol):
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    rope_theta: float
    partial_rotary_factor: float
    full_attention_interval: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    enable_thinking: bool
    reasoning_effort: Enum | None

    @property
    def linear_conv_dim(self) -> int: ...

    def layer_type(self, layer_idx: int) -> Enum: ...

    @property
    def layer_types(self) -> list[Enum]: ...


def text_prefix(hf_config: dict) -> str:
    return _MULTIMODAL_TEXT_PREFIX if 'text_config' in hf_config else _CAUSAL_LM_PREFIX


def mag_key_for(cfg: HybridConfig, prefix: str) -> common.MagKeyFor:
    def mag_key(hf_key: str) -> str | None:
        if hf_key.startswith('lm_head.'):
            return None if cfg.tie_word_embeddings else hf_key
        return hf_key[len(prefix) :] if hf_key.startswith(prefix) else None

    return mag_key


def base_config_kwargs(repo: str, text: dict, cfg: HybridConfig) -> dict[str, object]:
    rope = text.get('rope_parameters', {})
    return dict(
        repo_id=repo,
        vocab_size=text.get('vocab_size', cfg.vocab_size),
        hidden_size=text.get('hidden_size', cfg.hidden_size),
        num_hidden_layers=text.get('num_hidden_layers', cfg.num_hidden_layers),
        num_attention_heads=text.get('num_attention_heads', cfg.num_attention_heads),
        num_key_value_heads=text.get('num_key_value_heads', cfg.num_key_value_heads),
        head_dim=text.get('head_dim', cfg.head_dim),
        max_position_embeddings=cfg.max_position_embeddings,
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
        enable_thinking=cfg.enable_thinking,
        reasoning_effort=cfg.reasoning_effort,
    )


def check_layer_types(cfg: HybridConfig, text: dict) -> None:
    layer_types: list[str] | None = text.get('layer_types')
    if layer_types is None:
        return
    ours = [t.value for t in cfg.layer_types]
    if ours != layer_types:
        raise ValueError(f'Layer type mismatch, full_attention_interval={cfg.full_attention_interval} yields {ours}, checkpoint says {layer_types}')


def validate(plan: list[common.TensorPlan], cfg: HybridConfig, layer_type: type[Enum], extra: dict[str, tuple[int, ...]]) -> None:
    linear_type, full_type = layer_type.LINEAR_ATTENTION, layer_type.FULL_ATTENTION
    expected: dict[str, tuple[int, ...]] = {
        'embed_tokens.weight': (cfg.vocab_size, cfg.hidden_size),
        'norm.weight': (cfg.hidden_size,),
    }
    if not cfg.tie_word_embeddings:
        expected['lm_head.weight'] = (cfg.vocab_size, cfg.hidden_size)
    types = cfg.layer_types
    if linear_type in types:
        i = types.index(linear_type)
        expected[f'layers.{i}.linear_attn.conv1d.weight'] = (cfg.linear_conv_dim, 1, cfg.linear_conv_kernel_dim)
        expected[f'layers.{i}.linear_attn.A_log'] = (cfg.linear_num_value_heads,)
        expected[f'layers.{i}.linear_attn.in_proj_qkv.weight'] = (cfg.linear_conv_dim, cfg.hidden_size)
    if full_type in types:
        i = types.index(full_type)
        expected[f'layers.{i}.self_attn.q_proj.weight'] = (cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size)
        expected[f'layers.{i}.self_attn.k_proj.weight'] = (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
    expected.update(extra)
    common.check_layers(plan, cfg.num_hidden_layers, lambda i: 'linear_attn' if cfg.layer_type(i) is linear_type else 'self_attn')
    common.check_shapes(plan, expected)

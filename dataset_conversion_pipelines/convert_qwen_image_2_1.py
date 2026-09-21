# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import os
from typing import Any

from magnetron import dtype
from magnetron_models.models.qwen_image_2_1 import (
    ARCH_TEXT_ENCODER,
    ARCH_TRANSFORMER,
    ARCH_VAE,
    SchedulerConfig,
    TextEncoderConfig,
    TransformerConfig,
    VAEConfig,
)

from dataset_conversion_pipelines import common

_DEFAULT_MODEL: str = 'Qwen/Qwen-Image-2.1'
_COMPONENTS: tuple[str, ...] = ('text-encoder', 'transformer', 'vae')
_ALLOW_PATTERNS: dict[str, list[str]] = {
    'text-encoder': ['text_encoder/*', 'processor/*'],
    'transformer': ['transformer/*', 'scheduler/*'],
    'vae': ['vae/*'],
}

_TEXT_PREFIX: str = 'model.language_model.'


def _text_encoder_config(repo: str, hf_config: dict[str, Any]) -> TextEncoderConfig:
    text: dict[str, Any] = hf_config.get('text_config', hf_config)
    cfg = TextEncoderConfig(repo_id=repo)
    rope: dict[str, Any] = text.get('rope_scaling') or text.get('rope_parameters') or {}
    return TextEncoderConfig(
        repo_id=repo,
        vocab_size=text.get('vocab_size', cfg.vocab_size),
        hidden_size=text.get('hidden_size', cfg.hidden_size),
        intermediate_size=text.get('intermediate_size', cfg.intermediate_size),
        num_hidden_layers=text.get('num_hidden_layers', cfg.num_hidden_layers),
        num_attention_heads=text.get('num_attention_heads', cfg.num_attention_heads),
        num_key_value_heads=text.get('num_key_value_heads', cfg.num_key_value_heads),
        head_dim=text.get('head_dim', cfg.head_dim),
        max_position_embeddings=cfg.max_position_embeddings,  # Deliberately not the checkpoint's 262144, prompts are short.
        rms_norm_eps=text.get('rms_norm_eps', cfg.rms_norm_eps),
        rope_theta=rope.get('rope_theta', text.get('rope_theta', cfg.rope_theta)),
        image_token_id=hf_config.get('image_token_id', cfg.image_token_id),
        pad_token_id=text.get('pad_token_id', cfg.pad_token_id) or cfg.pad_token_id,
    )


def _text_encoder_key(hf_key: str) -> str | None:
    if not hf_key.startswith(_TEXT_PREFIX):
        return None  # lm_head and the vision tower
    key = hf_key[len(_TEXT_PREFIX) :]
    if key == 'norm.weight':
        return None
    return key


def _validate_text_encoder(plan: list[common.TensorPlan], cfg: TextEncoderConfig) -> None:
    common.check_layers(plan, cfg.num_hidden_layers, lambda _: 'self_attn')
    common.check_shapes(
        plan,
        {
            'embed_tokens.weight': (cfg.vocab_size, cfg.hidden_size),
            'layers.0.self_attn.q_proj.weight': (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
            'layers.0.self_attn.k_proj.weight': (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
            'layers.0.self_attn.q_norm.weight': (cfg.head_dim,),
            'layers.0.mlp.gate_proj.weight': (cfg.intermediate_size, cfg.hidden_size),
        },
    )


def _transformer_config(repo: str, hf_config: dict[str, Any]) -> TransformerConfig:
    cfg = TransformerConfig(repo_id=repo)
    return TransformerConfig(
        repo_id=repo,
        in_channels=hf_config.get('in_channels', cfg.in_channels),
        out_channels=hf_config.get('out_channels') or hf_config.get('in_channels', cfg.out_channels),
        num_layers=hf_config.get('num_layers', cfg.num_layers),
        attention_head_dim=hf_config.get('attention_head_dim', cfg.attention_head_dim),
        num_attention_heads=hf_config.get('num_attention_heads', cfg.num_attention_heads),
        context_in_dim=hf_config.get('context_in_dim', cfg.context_in_dim),
        mlp_ratio=hf_config.get('mlp_ratio', cfg.mlp_ratio),
        axes_dims_rope=tuple(hf_config.get('axes_dims_rope', cfg.axes_dims_rope)),
        eps=hf_config.get('eps', cfg.eps),
        causal_condition=hf_config.get('causal_condition', cfg.causal_condition),
    )


def _scheduler_config(repo_dir: str) -> SchedulerConfig:
    raw = common.load_hf_config(os.path.join(repo_dir, 'scheduler'), filename='scheduler_config.json')
    cfg = SchedulerConfig()
    if not raw:
        common.console.print('No scheduler_config.json, the snapshot gets the Qwen-Image 2.1 defaults', style='yellow')
        return cfg
    if raw.get('time_shift_type', 'exponential') != 'exponential' or not raw.get('use_dynamic_shifting', True):
        raise ValueError('Only the exponential, resolution dependent time shift is implemented')
    return SchedulerConfig(
        num_train_timesteps=raw.get('num_train_timesteps', cfg.num_train_timesteps),
        base_image_seq_len=raw.get('base_image_seq_len', cfg.base_image_seq_len),
        max_image_seq_len=raw.get('max_image_seq_len', cfg.max_image_seq_len),
        base_shift=raw.get('base_shift', cfg.base_shift),
        max_shift=raw.get('max_shift', cfg.max_shift),
        shift_terminal=raw.get('shift_terminal') or 0.0,
    )


def _transformer_key(hf_key: str) -> str | None:
    if hf_key == 'modulation.1.weight':
        return 'modulation.linear.weight'
    if hf_key.endswith('.attn.to_out.0.weight'):
        return hf_key[: -len('.0.weight')] + '.weight'
    return hf_key


def _validate_transformer(plan: list[common.TensorPlan], cfg: TransformerConfig) -> None:
    dim: int = cfg.inner_dim
    common.check_layers(plan, cfg.num_layers, lambda _: 'attn', prefix='transformer_blocks')
    common.check_shapes(
        plan,
        {
            'img_in.weight': (dim, cfg.in_channels),
            'txt_in.text_norm.weight': (cfg.context_in_dim,),
            'txt_in.in_layer.weight': (dim, cfg.context_in_dim),
            'time_text_embed.timestep_embedder.linear_1.weight': (dim, cfg.timestep_dim),
            'modulation.linear.weight': (4 * dim, dim),
            'transformer_blocks.0.attn.to_q.weight': (dim, dim),
            'transformer_blocks.0.attn.norm_q.weight': (cfg.attention_head_dim,),
            'transformer_blocks.0.attn.to_out.weight': (dim, dim),
            'transformer_blocks.0.img_mlp.gate_layer.weight': (dim * cfg.mlp_ratio, dim),
            'norm_out.linear.weight': (dim, dim),
            'proj_out.weight': (cfg.out_channels, dim),
        },
    )


def _vae_config(repo: str, hf_config: dict[str, Any]) -> VAEConfig:
    cfg = VAEConfig(repo_id=repo)
    if hf_config.get('patch_size'):
        raise ValueError('A patchified VAE is not implemented')
    if not hf_config.get('is_residual', True):
        raise ValueError('Only the residual decoder layout is implemented')
    return VAEConfig(
        repo_id=repo,
        z_dim=hf_config.get('z_dim', cfg.z_dim),
        decoder_base_dim=hf_config.get('decoder_base_dim') or hf_config.get('base_dim', cfg.decoder_base_dim),
        dim_mult=tuple(hf_config.get('dim_mult', cfg.dim_mult)),
        num_res_blocks=hf_config.get('num_res_blocks', cfg.num_res_blocks),
        out_channels=hf_config.get('out_channels', cfg.out_channels),
        scale_factor_spatial=hf_config.get('scale_factor_spatial', cfg.scale_factor_spatial),
        temporal_downsample=tuple(hf_config.get('temperal_downsample', cfg.temporal_downsample)),
        latents_mean=tuple(hf_config.get('latents_mean', cfg.latents_mean)),
        latents_std=tuple(hf_config.get('latents_std', cfg.latents_std)),
    )


def _vae_key(hf_key: str) -> str | None:
    if hf_key.startswith(('encoder.', 'quant_conv.')):
        return None
    if '.upsampler.time_conv.' in hf_key:
        return None
    return hf_key.replace('.upsampler.resample.1.', '.upsampler.conv.')


def _validate_vae(plan: list[common.TensorPlan], cfg: VAEConfig) -> None:
    dims: list[int] = [cfg.decoder_base_dim * m for m in (cfg.dim_mult[-1], *reversed(cfg.dim_mult))]
    expected: dict[str, tuple[int, ...]] = {
        'post_quant_conv.weight': (cfg.z_dim, cfg.z_dim, 1, 1),
        'decoder.conv_in.weight': (dims[0], cfg.z_dim, 3, 3),
        'decoder.mid_block.attentions.0.to_qkv.weight': (3 * dims[0], dims[0], 1, 1),
        'decoder.mid_block.attentions.0.norm.gamma': (dims[0], 1, 1),
        'decoder.mid_block.resnets.0.norm1.gamma': (dims[0], 1, 1, 1),
        'decoder.norm_out.gamma': (dims[-1], 1, 1, 1),
        'decoder.conv_out.weight': (cfg.out_channels, dims[-1], 3, 3),
    }
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        expected[f'decoder.up_blocks.{i}.resnets.0.conv1.weight'] = (out_dim, in_dim, 3, 3)
        expected[f'decoder.up_blocks.{i}.resnets.{cfg.num_res_blocks}.conv2.weight'] = (out_dim, out_dim, 3, 3)
        if i != len(cfg.dim_mult) - 1:
            expected[f'decoder.up_blocks.{i}.upsampler.conv.weight'] = (out_dim, out_dim, 3, 3)
    common.check_shapes(plan, expected)
    leaked = [e.mag_key for e in plan if e.mag_key.startswith('encoder.') or 'time_conv' in e.mag_key]
    if leaked:
        raise ValueError(f'Encoder or temporal tensors leaked into the decoder plan: {leaked[:3]}')


# --- driver -----------------------------------------------------------------------------------------------------------


def _plan_component(component: str, repo: str, repo_dir: str, mag_dtype: dtype.DType) -> common.PipelineComponent:
    if component == 'text-encoder':
        sub = os.path.join(repo_dir, 'text_encoder')
        cfg = _text_encoder_config(repo, common.load_hf_config(sub))
        plan = common.plan_tensors(sub, mag_key_for=_text_encoder_key, dtype_for=common.dtype_policy(mag_dtype))
        _validate_text_encoder(plan, cfg)
        return common.PipelineComponent(
            component=component,
            architecture=ARCH_TEXT_ENCODER,
            repo_dir=sub,
            mag_dtype=mag_dtype,
            cfg=cfg,
            config_title='Qwen3-VL text encoder configuration',
            plan=plan,
            tokenizer_dir=os.path.join(repo_dir, 'processor'),
        )
    if component == 'transformer':
        sub = os.path.join(repo_dir, 'transformer')
        cfg = _transformer_config(repo, common.load_hf_config(sub))
        plan = common.plan_tensors(sub, mag_key_for=_transformer_key, dtype_for=common.dtype_policy(mag_dtype), shard_stem='diffusion_pytorch_model')
        _validate_transformer(plan, cfg)
        return common.PipelineComponent(
            component=component,
            architecture=ARCH_TRANSFORMER,
            repo_dir=sub,
            mag_dtype=mag_dtype,
            cfg=cfg,
            config_title='Qwen-Image 2.1 transformer configuration',
            plan=plan,
            extra_metadata={'scheduler_config': _scheduler_config(repo_dir)},
        )
    if component == 'vae':
        sub = os.path.join(repo_dir, 'vae')
        cfg = _vae_config(repo, common.load_hf_config(sub))
        plan = common.plan_tensors(sub, mag_key_for=_vae_key, dtype_for=common.dtype_policy(mag_dtype), shard_stem='diffusion_pytorch_model')
        _validate_vae(plan, cfg)
        return common.PipelineComponent(
            component=component,
            architecture=ARCH_VAE,
            repo_dir=sub,
            mag_dtype=mag_dtype,
            cfg=cfg,
            config_title='Qwen-Image 2.1 VAE decoder configuration',
            plan=plan,
        )
    raise ValueError(f'Unknown component {component}')


def main() -> None:
    parser = common.build_arg_parser(
        'Convert Hugging Face Qwen-Image-2.1 into one Magnetron pipeline snapshot holding the text encoder, transformer and VAE',
        default_model=_DEFAULT_MODEL,
    )
    parser.add_argument(
        '--vae-dtype', type=str, default=None, choices=sorted(common._MAG_BY_NAME.keys()), help='Data type for the VAE, defaults to --dtype'
    )
    parser.add_argument('--repo-dir', type=str, default=None, help='Already downloaded checkpoint directory, skips the Hugging Face download')
    args = parser.parse_args()
    mag_dtype: dtype.DType = common.mag_dtype_from_str(args.dtype)
    vae_dtype: dtype.DType = common.mag_dtype_from_str(args.vae_dtype) if args.vae_dtype else mag_dtype
    if args.repo_dir is not None:
        repo_dir: str = args.repo_dir
    else:
        patterns: list[str] = ['model_index.json', *(p for c in _COMPONENTS for p in _ALLOW_PATTERNS[c])]
        repo_dir = common.download_repo(args.model, allow_patterns=patterns)
    components: list[common.PipelineComponent] = []
    for component in _COMPONENTS:
        common.console.print(f'[bold]{component}[/bold]: planning')
        components.append(_plan_component(component, args.model, repo_dir, vae_dtype if component == 'vae' else mag_dtype))
    common.convert_pipeline(
        args.model,
        components,
        mag_dtype=mag_dtype,
        model='qwen-image-2.1',
        out=args.out,
        write_model_card=args.model_card,
        model_card_path=args.model_card_path,
        card_only=args.card_only,
    )


if __name__ == '__main__':
    main()

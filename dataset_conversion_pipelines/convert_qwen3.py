# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from magnetron import dtype
from magnetron_models.models.qwen3 import Config

from dataset_conversion_pipelines import common

_CAUSAL_LM_PREFIX: str = 'model.'


def _config_for(repo: str, raw: dict) -> Config:
    cfg = Config(repo_id=repo)
    if not raw:
        return cfg
    return Config(
        repo_id=repo,
        vocab_size=raw.get('vocab_size', cfg.vocab_size),
        hidden_size=raw.get('hidden_size', cfg.hidden_size),
        intermediate_size=raw.get('intermediate_size', cfg.intermediate_size),
        num_hidden_layers=raw.get('num_hidden_layers', cfg.num_hidden_layers),
        num_attention_heads=raw.get('num_attention_heads', cfg.num_attention_heads),
        num_key_value_heads=raw.get('num_key_value_heads', cfg.num_key_value_heads),
        head_dim=raw.get('head_dim', cfg.head_dim),
        max_position_embeddings=cfg.max_position_embeddings,  # Deliberately not the checkpoint's 262144.
        rms_norm_eps=raw.get('rms_norm_eps', cfg.rms_norm_eps),
        tie_word_embeddings=raw.get('tie_word_embeddings', cfg.tie_word_embeddings),
        rope_theta=raw.get('rope_theta', cfg.rope_theta),
        sliding_window=raw.get('sliding_window', cfg.sliding_window),
        bos_token_id=raw.get('bos_token_id', cfg.bos_token_id),
        eos_token_id=raw.get('eos_token_id', cfg.eos_token_id),
    )


def _mag_key_for(cfg: Config) -> common.MagKeyFor:
    def mag_key_for(hf_key: str) -> str | None:
        if hf_key.startswith('lm_head.'):
            return None if cfg.tie_word_embeddings else hf_key
        if hf_key.startswith(_CAUSAL_LM_PREFIX):
            return hf_key[len(_CAUSAL_LM_PREFIX) :]
        return None

    return mag_key_for


def _validate(plan: list[common.TensorPlan], cfg: Config) -> None:
    expected: dict[str, tuple[int, ...]] = {
        'embed_tokens.weight': (cfg.vocab_size, cfg.hidden_size),
        'norm.weight': (cfg.hidden_size,),
        'layers.0.self_attn.q_proj.weight': (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
        'layers.0.self_attn.k_proj.weight': (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
        'layers.0.mlp.gate_proj.weight': (cfg.intermediate_size, cfg.hidden_size),
    }
    if not cfg.tie_word_embeddings:
        expected['lm_head.weight'] = (cfg.vocab_size, cfg.hidden_size)
    common.check_layers(plan, cfg.num_hidden_layers, lambda _: 'self_attn')
    common.check_shapes(plan, expected)


def main() -> None:
    parser = common.build_arg_parser(
        'Convert a Hugging Face Qwen3 model to the Magnetron snapshot format',
        default_model='Qwen/Qwen3-4B-Instruct-2507',
    )
    args = parser.parse_args()
    mag_dtype: dtype.DType = common.mag_dtype_from_str(args.dtype)
    repo_dir: str = common.download_repo(args.model)
    hf_config = common.load_hf_config(repo_dir)
    cfg = _config_for(args.model, hf_config)
    plan = common.plan_tensors(
        repo_dir,
        mag_key_for=_mag_key_for(cfg),
        dtype_for=common.dtype_policy(mag_dtype),
    )
    _validate(plan, cfg)
    common.convert_repo(
        args.model,
        repo_dir,
        plan,
        mag_dtype=mag_dtype,
        architecture=hf_config.get('model_type', 'qwen3'),
        model='qwen3',
        cfg=cfg,
        config_title='Qwen3 configuration',
        out=args.out,
        write_model_card=args.model_card,
        model_card_path=args.model_card_path,
    )


if __name__ == '__main__':
    main()

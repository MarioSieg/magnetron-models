# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from magnetron import dtype

from dataset_conversion_pipelines import common
from magnetron_models.models.whisper import Config

_MODEL_PREFIX: str = 'model.'
_KNOWN_MODELS: tuple[str, ...] = (
    'openai/whisper-tiny',
    'openai/whisper-base',
    'openai/whisper-small',
    'openai/whisper-medium',
    'openai/whisper-large-v3',
    'openai/whisper-large-v3-turbo',
)


def _mag_key_for(hf_key: str) -> str | None:
    if hf_key.startswith(_MODEL_PREFIX):
        return hf_key[len(_MODEL_PREFIX) :]
    return None


def _validate(plan: list[common.TensorPlan], cfg: Config) -> None:
    common.check_shapes(
        plan,
        {
            'encoder.conv1.weight': (cfg.d_model, cfg.num_mel_bins, 3),
            'encoder.embed_positions.weight': (cfg.max_source_positions, cfg.d_model),
            'decoder.embed_tokens.weight': (cfg.vocab_size, cfg.d_model),
            'decoder.embed_positions.weight': (cfg.max_target_positions, cfg.d_model),
            f'encoder.layers.{cfg.encoder_layers - 1}.fc1.weight': (cfg.encoder_ffn_dim, cfg.d_model),
            f'decoder.layers.{cfg.decoder_layers - 1}.fc1.weight': (cfg.decoder_ffn_dim, cfg.d_model),
            f'decoder.layers.{cfg.decoder_layers - 1}.encoder_attn.q_proj.weight': (cfg.d_model, cfg.d_model),
        },
    )


def convert(
    repo: str,
    mag_dtype: dtype.DType,
    out: str | None = None,
    *,
    write_model_card: bool = False,
    model_card_path: str = 'model_card.md',
    card_only: bool = False,
) -> str:
    repo_dir: str = common.download_repo(repo, allow_patterns=['*.json', 'model.safetensors'])
    hf_config = common.load_hf_config(repo_dir)
    if hf_config.get('model_type') != 'whisper':
        raise ValueError(f'{repo} is not a Whisper model (model_type={hf_config.get("model_type")})')
    cfg = Config.from_hf_config(repo, hf_config)
    plan = common.plan_tensors(repo_dir, mag_key_for=_mag_key_for, dtype_for=common.dtype_policy(mag_dtype))
    _validate(plan, cfg)
    return common.convert_repo(
        repo,
        repo_dir,
        plan,
        mag_dtype=mag_dtype,
        architecture='whisper',
        model='whisper',
        cfg=cfg,
        config_title='Whisper configuration',
        out=out,
        write_model_card=write_model_card,
        model_card_path=model_card_path,
        card_only=card_only,
        extra_metadata={'generation_config': common.load_hf_config(repo_dir, 'generation_config.json')},
    )


def main() -> None:
    parser = common.build_arg_parser(
        'Convert a Hugging Face Whisper model to the Magnetron snapshot format', default_model='openai/whisper-base', known_models=_KNOWN_MODELS
    )
    args = parser.parse_args()
    convert(
        args.model,
        common.mag_dtype_from_str(args.dtype),
        args.out,
        write_model_card=args.model_card,
        model_card_path=args.model_card_path,
        card_only=args.card_only,
    )


if __name__ == '__main__':
    main()

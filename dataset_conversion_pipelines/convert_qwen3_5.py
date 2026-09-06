# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from magnetron import dtype
from magnetron_models.models.qwen3_5 import CONFIGS, Config, LayerType

from dataset_conversion_pipelines import common, qwen3_5_shared as shared

_DENSE_LAYER: int = 0


def _config_for(repo: str, hf_config: dict) -> Config:
    cfg = CONFIGS.get(repo, Config(repo_id=repo))
    if not hf_config:
        return cfg
    text = hf_config.get('text_config', hf_config)
    cfg = Config(
        **shared.base_config_kwargs(repo, text, cfg),
        intermediate_size=text.get('intermediate_size', cfg.intermediate_size),
    )
    shared.check_layer_types(cfg, text)
    return cfg


def _validate(plan: list[common.TensorPlan], cfg: Config) -> None:
    shared.validate(
        plan,
        cfg,
        LayerType,
        {
            f'layers.{_DENSE_LAYER}.mlp.gate_proj.weight': (cfg.intermediate_size, cfg.hidden_size),
            f'layers.{_DENSE_LAYER}.mlp.down_proj.weight': (cfg.hidden_size, cfg.intermediate_size),
        },
    )


def main() -> None:
    parser = common.build_arg_parser(
        'Convert a dense Hugging Face Qwen3.5 or Qwen3.8 model to the Magnetron snapshot format',
        default_model='Qwen/Qwen3.5-4B',
        known_models=CONFIGS.keys(),
    )
    args = parser.parse_args()
    mag_dtype: dtype.DType = common.mag_dtype_from_str(args.dtype)
    repo_dir: str = common.download_repo(args.model)
    hf_config = common.load_hf_config(repo_dir)
    cfg = _config_for(args.model, hf_config)
    plan = common.plan_tensors(
        repo_dir,
        mag_key_for=shared.mag_key_for(cfg, shared.text_prefix(hf_config)),
        dtype_for=common.dtype_policy(mag_dtype, shared.FP32_SUFFIXES),
    )
    _validate(plan, cfg)
    common.convert_repo(
        args.model,
        repo_dir,
        plan,
        mag_dtype=mag_dtype,
        architecture=common.text_model_type(hf_config, 'qwen3_5_text'),
        model=args.model.split('/')[-1].lower(),
        cfg=cfg,
        config_title='Qwen3.5 configuration',
        out=args.out,
        write_model_card=args.model_card,
        model_card_path=args.model_card_path,
        card_only=args.card_only,
    )


if __name__ == '__main__':
    main()

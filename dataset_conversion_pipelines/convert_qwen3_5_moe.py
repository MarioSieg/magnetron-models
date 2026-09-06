# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from magnetron import dtype
from magnetron_models.models.qwen3_5_moe import CONFIGS, Config, LayerType

from dataset_conversion_pipelines import common, qwen3_5_shared as shared

_MOE_LAYER: int = 0


def _config_for(repo: str, hf_config: dict) -> Config:
    cfg = CONFIGS.get(repo, Config(repo_id=repo))
    if not hf_config:
        return cfg
    text = hf_config.get('text_config', hf_config)
    cfg = Config(
        **shared.base_config_kwargs(repo, text, cfg),
        moe_intermediate_size=text.get('moe_intermediate_size', cfg.moe_intermediate_size),
        shared_expert_intermediate_size=text.get('shared_expert_intermediate_size', cfg.shared_expert_intermediate_size),
        num_experts=text.get('num_experts', cfg.num_experts),
        num_experts_per_tok=text.get('num_experts_per_tok', cfg.num_experts_per_tok),
        thinking_only=cfg.thinking_only,
    )
    shared.check_layer_types(cfg, text)
    if text.get('mlp_only_layers'):
        raise ValueError(f'Unsupported mlp_only_layers: {text["mlp_only_layers"]}')
    return cfg


def _validate(plan: list[common.TensorPlan], cfg: Config) -> None:
    shared.validate(
        plan,
        cfg,
        LayerType,
        {
            f'layers.{_MOE_LAYER}.mlp.gate.weight': (cfg.num_experts, cfg.hidden_size),
            f'layers.{_MOE_LAYER}.mlp.experts.gate_up_proj': (cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size),
            f'layers.{_MOE_LAYER}.mlp.experts.down_proj': (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
            f'layers.{_MOE_LAYER}.mlp.shared_expert.gate_proj.weight': (cfg.shared_expert_intermediate_size, cfg.hidden_size),
            f'layers.{_MOE_LAYER}.mlp.shared_expert_gate.weight': (1, cfg.hidden_size),
        },
    )


def main() -> None:
    parser = common.build_arg_parser(
        'Convert a Hugging Face Qwen3.5 or Qwen3.8 MoE model to the Magnetron snapshot format',
        default_model='Qwen/Qwen3.5-35B-A3B',
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
        architecture=common.text_model_type(hf_config, 'qwen3_5_moe_text'),
        model=args.model.split('/')[-1].lower(),
        cfg=cfg,
        config_title='Qwen3.5-MoE configuration',
        out=args.out,
        write_model_card=args.model_card,
        model_card_path=args.model_card_path,
        card_only=args.card_only,
    )


if __name__ == '__main__':
    main()

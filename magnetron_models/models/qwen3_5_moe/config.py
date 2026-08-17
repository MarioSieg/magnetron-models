# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from enum import Enum, unique
from dataclasses import dataclass


@unique
class SamplingStrategy(Enum):
    GREEDY = 'greedy'
    TOPK = 'topk'


@unique
class LayerType(Enum):
    LINEAR_ATTENTION = 'linear_attention'
    FULL_ATTENTION = 'full_attention'


@unique
class ReasoningEffort(Enum):
    """Qwen3.8 steers thinking depth by prepending an instruction to the system prompt."""

    LOW = 'low'
    MEDIUM = 'medium'
    XHIGH = 'xhigh'


REASONING_INSTRUCTIONS: dict[ReasoningEffort, str] = {  # Verbatim from the Qwen3.8 chat template, medium adds nothing.
    ReasoningEffort.LOW: 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.',
    ReasoningEffort.MEDIUM: '',
    ReasoningEffort.XHIGH: 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.',
}


@dataclass
class Config:
    repo_id: str = 'Qwen/Qwen3.5-35B-A3B'
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    max_position_embeddings: int = 8192  # 262144
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    delta_chunk_size: int = 64
    moe_intermediate_size: int = 512  # Per-expert MLP width.
    shared_expert_intermediate_size: int = 512
    num_experts: int = 256
    num_experts_per_tok: int = 8
    bos_token_id: int = 248044  # <|endoftext|>
    eos_token_id: int = 248046  # <|im_end|>
    stop_token_ids: frozenset[int] = frozenset({248044, 248046})
    enable_thinking: bool = False
    thinking_only: bool = False  # Qwen3.8-2.4T-A95B refuses a non-thinking prompt in its chat template.
    reasoning_effort: ReasoningEffort | None = None  # Qwen3.5 has no effort control, Qwen3.8 defaults to xhigh.
    sampling_strategy: SamplingStrategy = SamplingStrategy.GREEDY

    def __post_init__(self) -> None:
        if self.thinking_only and not self.enable_thinking:
            raise ValueError(f'{self.repo_id} cannot run with thinking disabled')

    @property
    def reasoning_instructions(self) -> str:
        """The system prompt prefix for the configured effort, empty unless the model is thinking."""
        if not self.enable_thinking or self.reasoning_effort is None:
            return ''
        return REASONING_INSTRUCTIONS[self.reasoning_effort]

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def linear_value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def linear_conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

    def layer_type(self, layer_idx: int) -> LayerType:
        return LayerType.LINEAR_ATTENTION if (layer_idx + 1) % self.full_attention_interval else LayerType.FULL_ATTENTION

    @property
    def layer_types(self) -> list[LayerType]:
        return [self.layer_type(i) for i in range(self.num_hidden_layers)]


CONFIGS: dict[str, Config] = {
    'Qwen/Qwen3.5-35B-A3B': Config(),
    'Qwen/Qwen3.5-122B-A10B': Config(
        repo_id='Qwen/Qwen3.5-122B-A10B',
        hidden_size=3072,
        num_hidden_layers=48,
        num_attention_heads=32,
        linear_num_value_heads=64,
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=1024,
    ),
        'Qwen/Qwen3.5-397B-A17B': Config(
        repo_id='Qwen/Qwen3.5-397B-A17B',
        hidden_size=4096,
        num_hidden_layers=60,
        num_attention_heads=32,
        linear_num_value_heads=64,
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=1024,
        num_experts=512,
        num_experts_per_tok=10,
    ),
    # Qwen3.8 keeps the Qwen3.5 MoE architecture (model_type qwen3_5_moe_text), so the same blocks run it.
    # Its config adds output_gate_type=swish, which transformers does not read: the gate stays sigmoid.
    # Unlike every Qwen3.5 checkpoint this one is text only, so its weights are not nested under a vision wrapper.
    'Qwen/Qwen3.8-2.4T-A95B': Config(
        repo_id='Qwen/Qwen3.8-2.4T-A95B',
        hidden_size=8192,
        num_hidden_layers=92,
        num_attention_heads=64,
        num_key_value_heads=4,
        linear_num_value_heads=128,
        moe_intermediate_size=2048,
        shared_expert_intermediate_size=2048,
        num_experts=512,
        num_experts_per_tok=10,
        enable_thinking=True,
        thinking_only=True,
        reasoning_effort=ReasoningEffort.XHIGH,
    ),
}

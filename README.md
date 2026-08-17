# magnetron-models

Training and inference for transformer LLMs (Qwen3 today, more to come) built on the
[Magnetron](https://github.com/MarioSieg/magnetron) ML framework.

Architectures are implemented directly against Magnetron's tensor and `nn` APIs — no PyTorch at
runtime. Weights are converted once from Hugging Face safetensors into Magnetron's `.mag` snapshot
format, then run on CPU or CUDA.

## Install

Requires Python >= 3.14.

```bash
uv sync                  # or: pip install -e .
pip install -e '.[dev]'  # torch + safetensors, needed only for weight conversion
```

## Convert weights

```bash
convert-qwen3 --model Qwen/Qwen3-4B-Instruct-2507 --dtype bfloat16
convert-qwen3-5 --model Qwen/Qwen3.5-9B --dtype bfloat16
convert-qwen3-5-moe --model Qwen/Qwen3.5-35B-A3B --dtype bfloat16

# Qwen3.8 keeps the Qwen3.5 architecture, so it goes through the same two converters
convert-qwen3-5 --model Qwen/Qwen3.8-27B --dtype bfloat16
convert-qwen3-5-moe --model Qwen/Qwen3.8-2.4T-A95B --dtype bfloat16
```

Downloads the HF repo and writes a `.mag` snapshot next to the working directory. Both Qwen3.5
converters read the shipped `config.json`, so checkpoint sizes that are not in `CONFIGS` still
convert. The vision tower and the MTP head in the Qwen3.5 checkpoints are skipped, and text only
checkpoints (`Qwen3.8-2.4T-A95B`) are detected from the config so their unnested weights still map.

## Run

`inference` takes a model name (see `--help` for the full list) and a snapshot:

```bash
# Qwen3.5 4B, interactive chat on CUDA
uv run inference --model qwen3.5-4b --snapshot qwen3.5-4b-bf16.mag --device cuda --repl

# Qwen3.5 9B on CPU
uv run inference --model qwen3.5-9b --snapshot qwen3.5-9b-bf16.mag --device cpu --repl

# Qwen3 4B, one shot
uv run inference --model qwen3 --snapshot qwen3-4b-instruct-2507-bf16.mag --prompt 'Explain RoPE briefly.'

# Qwen3.5 35B-A3B (MoE) on CPU, ~70 GB of weights
uv run inference --model qwen3.5-35b-a3b --snapshot qwen3.5-35b-a3b-bf16.mag --device cpu --repl

# Qwen3.8 27B, interactive chat
uv run inference --model qwen3.8-27b --snapshot qwen3.8-27b-bf16.mag --device cuda --repl
```

Sampling and context via `--temp`, `--top_k`, `--max_tokens`, `--max_ctx`, `--seed`, `--dtype`. The
tokenizer is pulled from `--repo_id`, defaulting to the repo named in the model's config.

## Qwen3.8

Qwen3.8 ships the same architectures as Qwen3.5 — the checkpoints declare `model_type`
`qwen3_5_text` and `qwen3_5_moe_text` — so it runs on the existing blocks and only needs its shapes
and prompt format. `qwen3.8-27b` is the dense model, `qwen3.8-2.4t-a95b` the 2.4T-A95B MoE. Two
details differ from Qwen3.5:

- The configs add `output_gate_type: swish`, which `transformers` does not read. The attention
  output gate stays sigmoid, matching the reference implementation.
- The chat template prepends a reasoning-effort instruction to the system prompt (`Config.reasoning_effort`,
  default `xhigh`, applied only while thinking). `Qwen3.8-2.4T-A95B` cannot run with thinking
  disabled, so its config sets `thinking_only` and rejects `enable_thinking=False`.

The MTP head and, for the 27B, the vision tower are skipped as with Qwen3.5.

## Status

Inference works end to end for Qwen3, Qwen3.5 dense and Qwen3.5 MoE (`qwen3.5-35b-a3b`,
`qwen3.5-122b-a10b`, `qwen3.5-397b-a17b`) models, and for Qwen3.8 (`qwen3.8-27b`,
`qwen3.8-2.4t-a95b`). The MoE checkpoints are large enough that CPU is the only practical device
today. `qwen3_moe` (Qwen3.0 MoE), the vision towers and the training path are still in progress.

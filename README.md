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
```

Downloads the HF repo and writes a `.mag` snapshot next to the working directory.

## Run

`inference` takes a model name (see `--help` for the full list) and a snapshot:

```bash
# Qwen3.5 4B, interactive chat on CUDA
uv run inference --model qwen3.5-4b --snapshot qwen3.5-4b-bf16.mag --device cuda --repl

# Qwen3.5 9B on CPU
uv run inference --model qwen3.5-9b --snapshot qwen3.5-9b-bf16.mag --device cpu --repl

# Qwen3 4B, one shot
uv run inference --model qwen3 --snapshot qwen3-4b-instruct-2507-bf16.mag --prompt 'Explain RoPE briefly.'
```

Sampling and context via `--temp`, `--top_k`, `--max_tokens`, `--max_ctx`, `--seed`, `--dtype`. The
tokenizer is pulled from `--repo_id`, defaulting to the repo named in the model's config.

## Status

Inference works end to end for Qwen3 and Qwen3.5 dense models. MoE variants (`qwen3_moe`,
`qwen3_5_moe`) and the training path are still in progress.

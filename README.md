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
```

Downloads the HF repo and writes a `.mag` snapshot.

## Run

```bash
python -m magnetron_models.main --repl --snapshot qwen3-4b.mag --device cuda
python -m magnetron_models.main --prompt 'Explain RoPE briefly.' --snapshot qwen3-4b.mag
```

Sampling and context via `--temp`, `--top_k`, `--max_tokens`, `--max_ctx`, `--seed`, `--dtype`. The
tokenizer is pulled from `--repo_id`.

## Status

Inference works end to end for Qwen3 dense models. MoE variants (`qwen3_moe`, `qwen3_5_moe`) and the
training path are still in progress.

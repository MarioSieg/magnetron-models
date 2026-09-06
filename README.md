# magnetron-models

Transformer LLM inference (Qwen3, Qwen3.5, Qwen3.8) on the
[Magnetron](https://github.com/MarioSieg/magnetron) ML framework — no PyTorch at runtime. Weights are
converted once from Hugging Face safetensors into a `.mag` snapshot, which carries the tokenizer and
config too, so the `.mag` file is all inference needs.

## Install

Requires Python >= 3.14.

```bash
uv sync
pip install -e '.[conversion]'  # torch, needed only for conversion
```

## Run

`inference` pulls the published `.mag` snapshot for `--model` from the Hub and drops into a chat REPL:

```bash
# Qwen3.8 27B - the flagship dense model
uv run inference --model qwen3.8-27b --device cuda --repl

# Qwen3.5 dense
uv run inference --model qwen3.5-0.8b --device cuda --repl
uv run inference --model qwen3.5-9b   --device cuda --repl
uv run inference --model qwen3.5-27b  --device cuda --repl

# Qwen3.5 35B-A3B (MoE), CPU only in practice
uv run inference --model qwen3.5-35b-a3b --device cpu --repl

# Qwen3.0 4B Instruct 2507
uv run inference --model qwen3 --device cuda --repl
```

`--prompt 'Explain RoPE briefly.'` replaces `--repl` for a one shot. Sampling and context via
`--temp`, `--top_k`, `--max_tokens`, `--max_ctx`, `--seed`, `--dtype`; `--help` lists every model
name.

A `.mag` defines its own model. The architecture, the config the weights were written with, the dtype
and the tokenizer all come out of the snapshot's manifest, so a local file runs on its own and a
checkpoint this build has no config table entry for runs like any other:

```bash
uv run inference --snapshot ./qwen3.5-35b-a3b-bf16.mag --device cpu --repl
```

`--model` and `--dtype` only pick which snapshot to download. A `--model` that disagrees with
`--snapshot` is an error, raised before any weights are allocated.

## Convert

Only needed to produce a new snapshot — done once, then uploaded to the Hub. Writes a `.mag` into the
working directory, or wherever `--out` points:

```bash
convert-qwen3-5 --model Qwen/Qwen3.8-27B --dtype bfloat16

convert-qwen3-5 --model Qwen/Qwen3.5-0.8B --dtype bfloat16
convert-qwen3-5 --model Qwen/Qwen3.5-9B   --dtype bfloat16
convert-qwen3-5 --model Qwen/Qwen3.5-27B  --dtype bfloat16

convert-qwen3-5-moe --model Qwen/Qwen3.5-35B-A3B --dtype bfloat16

convert-qwen3 --model Qwen/Qwen3-4B-Instruct-2507 --dtype bfloat16
```

Qwen3.8 reuses the Qwen3.5 architectures, so it goes through the same two converters. The conversion
is planned from the safetensors headers and streamed tensor by tensor, so a checkpoint far larger
than RAM still converts. `--model-card` writes a tensor manifest next to the snapshot.

Conversion is I/O bound and runs at about the speed of a file copy. When `--dtype` matches the
checkpoint's own dtype — the usual bf16 case — each tensor is `pread` in file order and its bytes go
into the snapshot untouched: no decode, no copy, no torch on the hot path. The stats table at the end
reports the throughput actually reached; if it is well under what the disk can do, check that the HF
cache and `--out` are not on the same busy device.

```bash
hf upload mario-sieg/Qwen3.8-27B-Magnetron qwen3.8-27b-bf16.mag qwen3.8-27b-bf16.mag
hf upload-large-folder mario-sieg/Qwen3.5-35B-A3B-Magnetron /mnt/models --type model --include '*.mag'
```

## Notes

The MoE checkpoints are large enough that CPU is the only practical device today. Vision towers, the
MTP head, `qwen3_moe` (Qwen3.0 MoE) and the training path are still in progress.

# magnetron-models

Transformer LLM modelling code (Qwen3, Qwen3.5, Qwen3.8) implemented on top of the
[Magnetron](https://github.com/MarioSieg/magnetron) ML framework. Model weights are
converted once from Hugging Face safetensors into a `.mag` snapshot, which carries the tokenizer and
config too, so the `.mag` file is all inference needs.

This package contains the modelling layer:
* Model Architectures
* Tokenizers
* KV Cache
* Streaming Generator
* Dataset Conversion Pipelines Pipelines


## Example

```python
from magnetron_models.inference import InferenceConfig, InferenceEngine

engine = InferenceEngine(InferenceConfig(model='qwen3.5-0.8b', device='cpu'))
prompt = engine.model.build_prompt('You are a helpful assistant.', [('user', 'Explain RoPE briefly.')])
for chunk in engine.gen_stream(prompt, reset_cache=True):
    print(chunk, end='', flush=True)
```

## Dataset Conversion Pipelines

> [!NOTE]
> Additional packages like PyTorch and safentensors are needed for the data conversion only. Run `uv sync --extra conversion` to install them.

Only needed to produce a new snapshot. We already provide pre-converted models ready to download on [HuggingFace](https://huggingface.co/mario-sieg/models).

```bash
convert-qwen3-5 --model Qwen/Qwen3.8-27B --dtype bfloat16

convert-qwen3-5 --model Qwen/Qwen3.5-0.8B --dtype bfloat16
convert-qwen3-5 --model Qwen/Qwen3.5-9B   --dtype bfloat16
convert-qwen3-5 --model Qwen/Qwen3.5-27B  --dtype bfloat16

convert-qwen3-5-moe --model Qwen/Qwen3.5-35B-A3B --dtype bfloat16

convert-qwen3 --model Qwen/Qwen3-4B-Instruct-2507 --dtype bfloat16
```

> [!NOTE]
> Qwen3.8 reuses the Qwen3.5 architecture, so it uses the same converters.

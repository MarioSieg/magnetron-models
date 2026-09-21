# magnetron-models

Transformer LLM modelling code (Qwen3, Qwen3.5, Qwen3.8) and the Qwen-Image 2.1 text-to-image pipeline,
implemented on top of the [Magnetron](https://github.com/MarioSieg/magnetron) ML framework. Model weights are
converted once from Hugging Face safetensors into a `.mag` snapshot, which carries the tokenizer and
config too, so the `.mag` file is all inference needs.

This package contains the modelling layer:
* Model Architectures (causal LLMs, diffusion transformer, VAE decoder)
* Tokenizers
* KV Cache
* Streaming Generator
* Flow-matching sampler and text-to-image pipeline
* Dataset Conversion Pipelines


## Example

```python
from magnetron_models.inference import InferenceConfig, InferenceEngine

engine = InferenceEngine(InferenceConfig(model='qwen3.5-0.8b', device='cpu'))
prompt = engine.model.build_prompt('You are a helpful assistant.', [('user', 'Explain RoPE briefly.')])
for chunk in engine.gen_stream(prompt, reset_cache=True):
    print(chunk, end='', flush=True)
```

## Text-to-Image (Qwen-Image 2.1)

[Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) is three networks: a Qwen3-VL-8B text encoder, a 7B
single-stream diffusion transformer and a 16x VAE. Each is its own snapshot (`qwen-image-2.1-{text-encoder,transformer,vae}-<dtype>.mag`),
so the pipeline can load them one after another and never hold all ~31 GiB (bf16) at once. Output is RGBA, the model paints
native transparency.

```python
from magnetron_models.diffusion import ImageGenConfig, ImageGenEngine

engine = ImageGenEngine(ImageGenConfig(model='qwen-image-2.1', device='cuda', height=1024, width=1024))
image = engine.generate('A capybara wearing a wizard hat, reading a book by candlelight, oil painting', seed=42)
engine.save(image, 'capybara.png')  # .png keeps alpha, .jpg composites over white
```

Or from the terminal, with a progress bar over the denoising steps:

```bash
generate-image "A capybara wearing a wizard hat, reading a book by candlelight, oil painting" -o capybara.png --seed 42
generate-image "..." --width 1344 --height 768 --steps 30 --device cuda
```

* Text-to-image only for now, no condition images or editing.
* Sampled without guidance by default like the reference; pass `negative_prompt` and `guidance_scale > 1` for classifier-free guidance.
* `offload=True` (default) loads each network just in time and frees it afterwards. Set it to `False` when everything fits in memory.
* The text keys and values are cached after the first denoising step, so later steps only run the image tokens.
* Unlike the Apache licensed Qwen3 LLMs, Qwen-Image 2.1 ships under the Qwen Research License (non-commercial).

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

```bash
convert-qwen-image-2-1 --dtype bfloat16                            # text encoder, transformer and VAE
convert-qwen-image-2-1 --component vae --vae-dtype float32          # one component, the VAE at full precision
convert-qwen-image-2-1 --repo-dir /data/Qwen-Image-2.1 --out-dir /data/snapshots
```

The text encoder snapshot keeps the Qwen3-VL language model only (no vision tower, no `lm_head`), the VAE snapshot keeps
the decoder only; both are all text-to-image needs.

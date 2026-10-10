# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Show a Qwen3.5 model an image and stream back its description.

    uv run python examples/describe_image.py \
        --model qwen3.5-4b --snapshot qwen3.5-4b-bf16.mag --image photo.jpg

Three steps matter, and they have to happen in this order:

1. Decode the image first. How many prompt tokens it occupies depends on its resized patch grid, so
   the prompt cannot be written until the image has been through the processor.
2. Put the placeholder run in front of the user's text. `vision_prefix` expands to
   `<|vision_start|>` + one `<|image_pad|>` per vision token + `<|vision_end|>`, matching the chat
   template's ordering of image before text.
3. Hand the same `PreprocessedImage` list to `gen_stream`. The engine runs the vision tower, swaps
   its output into the placeholder rows, and lays out 3D rotary positions over the result.
"""

import argparse
import time

from rich.console import Console

from magnetron_models.inference import InferenceConfig, InferenceEngine

console = Console()

DEFAULT_QUESTION: str = 'Beschreibe das Bild sehr detailliert.'


def describe(engine: InferenceEngine, image_path: str, question: str, max_tokens: int) -> str:
    """Stream one answer about one image, printing it as it arrives."""
    if not engine.supports_images:
        raise SystemExit(f'{engine.config.model} is a text only checkpoint, pick a Qwen3.5 or Qwen3.8-27B model')

    # 1. Decode and patchify. This is what tells us how many placeholder tokens the image needs.
    images = engine.load_images([image_path])

    # 2. Build the turn with the image ahead of the question, then wrap it in the chat template.
    user_turn: str = engine.vision_prefix(images) + question
    prompt: str = engine.model.build_prompt(engine.config.system, [('user', user_turn)])

    # 3. Generate. Passing the images is what makes the placeholders resolve to real vision tokens.
    console.print('\n[bold magenta]Answer[/]: ', end='')
    start = time.perf_counter()
    parts: list[str] = []
    for chunk in engine.gen_stream(prompt, images=images, reset_cache=True, max_tokens=max_tokens):
        parts.append(chunk)
        console.print(chunk, style='bold white', end='')
    elapsed = time.perf_counter() - start
    console.print(f'\n[dim]{len(parts)} chunks in {elapsed:.1f}s[/dim]')
    return ''.join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=str, default='qwen3.5-4b', help='Model name, must match the snapshot')
    parser.add_argument('--snapshot', type=str, required=True, help='Converted .mag snapshot')
    parser.add_argument('--image', type=str, required=True, help='Image file to look at')
    parser.add_argument('--prompt', type=str, default=DEFAULT_QUESTION, help='Question to ask about the image')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--temp', type=float, default=0.6)
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument(
        '--max_pixels',
        type=int,
        default=None,
        help='Shrink the image below the snapshot ceiling. The vision tower is quadratic in patch count, so this is the speed dial',
    )
    args = parser.parse_args()

    engine = InferenceEngine(
        InferenceConfig(
            model=args.model,
            snapshot=args.snapshot,
            device=args.device,
            max_tokens=args.max_tokens,
            temp=args.temp,
            top_k=args.top_k,
            seed=args.seed,
            max_pixels=args.max_pixels,
        )
    )
    describe(engine, args.image, args.prompt, args.max_tokens)


if __name__ == '__main__':
    main()

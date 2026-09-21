# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import argparse
import time

from magnetron_models.diffusion import ImageGenConfig, ImageGenEngine
from magnetron_models.inference import AUTO_DVC, DTYPES
from magnetron_models.models import DIFFUSION_MODELS_MAP
from magnetron_models.utils import console


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Text-to-image generation with Qwen-Image 2.1 on Magnetron')
    parser.add_argument('prompt', type=str, help='What to paint')
    parser.add_argument('-o', '--out', type=str, default='image.png', help='Output file, .png keeps the alpha channel, .jpg composites over white')
    parser.add_argument(
        '--model', type=str, default='qwen-image-2.1', choices=sorted(DIFFUSION_MODELS_MAP), help='Named model whose pipeline snapshot to download'
    )
    parser.add_argument('--snapshot', type=str, default=None, help='Local pipeline .mag holding all three networks, overrides the download')
    parser.add_argument('--width', type=int, default=1024, help='Image width in pixels, rounded down to a multiple of 32')
    parser.add_argument('--height', type=int, default=1024, help='Image height in pixels, rounded down to a multiple of 32')
    parser.add_argument('--steps', type=int, default=None, help='Denoising steps, defaults to the checkpoint setting (40)')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed for the initial noise')
    parser.add_argument('--negative-prompt', type=str, default=None, help='Enables classifier-free guidance together with --guidance-scale > 1')
    parser.add_argument('--guidance-scale', type=float, default=1.0, help='Classifier-free guidance scale, 1.0 disables it like the reference')
    parser.add_argument('--device', type=str, default=AUTO_DVC, help='cpu, cuda, cuda:1 or auto')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=sorted(DTYPES), help='Snapshot dtype to load')
    parser.add_argument('--no-kv-cache', action='store_true', help='Recompute the text tokens at every step instead of caching their K/V')
    parser.add_argument('--keep-loaded', action='store_true', help='Keep all three networks in memory instead of loading each just in time')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = ImageGenConfig(
        device=args.device,
        dtype=args.dtype,
        model=None if args.snapshot is not None else args.model,
        snapshot=args.snapshot,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        use_kv_cache=not args.no_kv_cache,
        offload=not args.keep_loaded,
    )
    engine = ImageGenEngine(cfg)
    console.print(f'[bold]Prompt:[/bold] {args.prompt}')
    start = time.perf_counter()
    image = engine.generate(args.prompt)
    engine.save(image, args.out)
    _, height, width = image.shape
    console.print(f'Saved {width}x{height} image to [bold]{args.out}[/bold] in {time.perf_counter() - start:.1f}s', style='green')


if __name__ == '__main__':
    main()

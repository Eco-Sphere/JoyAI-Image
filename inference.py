"""Local inference entrypoint for the clean JoyAI-Image release."""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import torch
from PIL import Image

import torch_npu
from torch_npu.contrib import transfer_to_npu

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

warnings.filterwarnings('ignore')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run local inference without FastAPI.')
    parser.add_argument('--ckpt-root', required=True, help='Checkpoint root.')
    parser.add_argument('--prompt', required=True, help='Edit prompt or T2I prompt.')
    parser.add_argument('--image', help='Optional input image path for image editing.')
    parser.add_argument('--output', default='example.png', help='Output image path.')
    parser.add_argument('--height', type=int, default=1024, help='Only used for text-to-image inference.')
    parser.add_argument('--width', type=int, default=1024, help='Only used for text-to-image inference.')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=5.0)
    parser.add_argument(
        '--cfg-size',
        type=int,
        choices=[1, 2],
        default=None,
        help='CFG parallel size. 1 disables cfg parallel, 2 enables two-rank cfg parallel.',
    )
    parser.add_argument(
        '--ulysses-size',
        type=int,
        default=None,
        help='Ulysses sequence parallel size. 1 disables sequence parallel.',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--neg-prompt', default='')
    parser.add_argument('--basesize', type=int, default=1024, help='Resize bucket base size for image editing inputs.')
    parser.add_argument(
        '--keep-image-size',
        action='store_true',
        help='For image editing, keep bucket preprocessing but resize final output back to the input image resolution.',
    )
    parser.add_argument('--rewrite-prompt', action='store_true')
    parser.add_argument('--config', help='Optional config path. Defaults to <ckpt-root>/infer_config.py.')
    parser.add_argument('--rewrite-model', default='gpt-5')
    parser.add_argument('--hsdp-shard-dim', type=int, help='Override config hsdp_shard_dim for multi-GPU FSDP inference.')
    parser.add_argument(
        '--warmup',
        action='store_true',
        help='Run one dry-run inference before timed inference to warm up kernels and runtime caches.',
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=8,
        help='Denoising steps for warmup run. Only used when --warmup is set.',
    )
    parser.add_argument(
        '--warmup-height',
        type=int,
        default=None,
        help='Warmup height for text-to-image warmup. Defaults to --height.',
    )
    parser.add_argument(
        '--warmup-width',
        type=int,
        default=None,
        help='Warmup width for text-to-image warmup. Defaults to --width.',
    )
    parser.add_argument(
        '--quant-desc-path',
        type=str,
        default=None,
        help="Path to quantization description json (e.g. quant_model_description_*.json). "
             "If set, transformer quantization will be applied at load time.",
    )
    return parser.parse_args()


def validate_quant_desc_path(quant_desc_path: str | None) -> None:
    if not quant_desc_path:
        return
    if not os.path.exists(quant_desc_path):
        raise FileNotFoundError(f"Quantization description file not found: {quant_desc_path}")
    if not quant_desc_path.endswith(".json") or "quant_model_description" not in os.path.basename(quant_desc_path):
        raise ValueError(
            f"Invalid quantization file: {quant_desc_path}. "
            "Expected format: quant_model_description_*.json"
        )


def load_input_image(image_path: str | None) -> Image.Image | None:
    if not image_path:
        return None
    return Image.open(image_path).convert('RGB')


def is_rank0() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


def resolve_device() -> torch.device:
    #if not torch.cuda.is_available():
    if not torch_npu.npu.is_available():
        return torch.device('cpu')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    #torch.cuda.set_device(local_rank)
    #return torch.device(f'cuda:{local_rank}')
    torch_npu.npu.set_device(local_rank)
    return torch.device(f'npu:{local_rank}')


def main() -> None:
    args = parse_args()
    validate_quant_desc_path(args.quant_desc_path)
    if args.warmup_steps <= 0:
        raise ValueError(f'--warmup-steps must be > 0, got {args.warmup_steps}')
    if args.ulysses_size is not None and args.ulysses_size <= 0:
        raise ValueError(f'--ulysses-size must be > 0, got {args.ulysses_size}')
    if args.cfg_size is not None:
        os.environ["CFG_SIZE"] = str(args.cfg_size)
    if args.ulysses_size is not None:
        os.environ["ULYSSES_SIZE"] = str(args.ulysses_size)

    from infer_runtime.model import InferenceParams, build_model, preprocess_edit_image
    from infer_runtime.settings import load_settings
    from modules.utils import maybe_init_distributed, clean_dist_env
    #from modules.models.attention import describe_attention_backend

    dist_initialized = False
    try:
        settings = load_settings(
            ckpt_root=args.ckpt_root,
            config_path=args.config,
            rewrite_model=args.rewrite_model,
            default_seed=args.seed,
        )
        device = resolve_device()
        dist_initialized = maybe_init_distributed()

        if is_rank0():
            print(f'Chosen device: {device}')
        #    print(f'Attention backend: {describe_attention_backend()}')
            print(f'Config path: {settings.config_path}')
            print(f'Checkpoint path: {settings.ckpt_path}')
            if args.hsdp_shard_dim is not None:
                print(f'Override hsdp_shard_dim: {args.hsdp_shard_dim}')
            if args.quant_desc_path:
                print(f'Quant desc path: {args.quant_desc_path}')
            if args.cfg_size is not None:
                print(f'CFG size: {args.cfg_size}')
            else:
                print(f'CFG size: {os.environ.get("CFG_SIZE", "auto")} (env/default)')
            if args.ulysses_size is not None:
                print(f'Ulysses size: {args.ulysses_size}')
            else:
                print(f'Ulysses size: {os.environ.get("ULYSSES_SIZE", "1")} (env/default)')
            if args.keep_image_size:
                print('Image resize mode: bucket preprocess + output resize to original')
            if args.warmup:
                print('Warmup: enabled')

        model = build_model(
            settings,
            device=device,
            hsdp_shard_dim_override=args.hsdp_shard_dim,
            quant_desc_path=args.quant_desc_path,
        )
        input_image = load_input_image(args.image)
        effective_prompt = model.maybe_rewrite_prompt(args.prompt, input_image, args.rewrite_prompt)

        if args.warmup:
            warmup_height = args.warmup_height if args.warmup_height is not None else args.height
            warmup_width = args.warmup_width if args.warmup_width is not None else args.width
            if is_rank0():
                if input_image is not None:
                    warmup_image = preprocess_edit_image(
                        input_image,
                        basesize=args.basesize,
                        keep_image_size=args.keep_image_size,
                    )
                    resize_mode = 'bucket_then_restore_size' if args.keep_image_size else 'bucket'
                    print(
                        f'Start warmup: steps={args.warmup_steps}, '
                        f'image_input={input_image.size[0]}x{input_image.size[1]}, '
                        f'image_used={warmup_image.size[0]}x{warmup_image.size[1]}, '
                        f'resize_mode={resize_mode}, '
                        f'basesize={args.basesize}'
                    )
                else:
                    print(
                        f'Start warmup: steps={args.warmup_steps}, '
                        f'height={warmup_height}, width={warmup_width}'
                    )
            warmup_start = time.time()
            _ = model.infer(
                InferenceParams(
                    prompt=effective_prompt,
                    image=input_image,
                    height=warmup_height,
                    width=warmup_width,
                    steps=args.warmup_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    neg_prompt=args.neg_prompt,
                    basesize=args.basesize,
                    keep_image_size=args.keep_image_size,
                )
            )
            warmup_elapsed = time.time() - warmup_start
            if is_rank0():
                print(f'Warmup done: {warmup_elapsed:.2f} seconds')

        start_time = time.time()
        output_image = model.infer(
            InferenceParams(
                prompt=effective_prompt,
                image=input_image,
                height=args.height,
                width=args.width,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                neg_prompt=args.neg_prompt,
                basesize=args.basesize,
                keep_image_size=args.keep_image_size,
            )
        )
        elapsed = time.time() - start_time

        if is_rank0():
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_image.save(output_path)
            print(f'Prompt used: {effective_prompt}')
            print(f'Saved output: {output_path}')
            print(f'Time taken: {elapsed:.2f} seconds')
    finally:
        if dist_initialized:
            clean_dist_env()


if __name__ == '__main__':
    main()


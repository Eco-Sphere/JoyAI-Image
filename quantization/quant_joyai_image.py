#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch_npu

try:
    from accelerate.hooks import remove_hook_from_module
except Exception:
    remove_hook_from_module = None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"

for path_str in (str(ROOT_DIR), str(SRC_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from infer_runtime.infer_config import load_infer_config_class_from_pyfile
from infer_runtime.settings import load_settings
from modules.models import load_dit
from quantization.config import get_joyai_image_quant_config
from quantization.dump_utils import get_disable_layer_names
from quantization.quantizer import JoyAIImageQuantizer


def init_npu_env(device_id: int) -> torch.device:
    if not torch_npu.npu.is_available():
        raise RuntimeError("NPU is not available. JoyAI quantization script currently requires NPU.")

    torch_npu.npu.set_device(device_id)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    if hasattr(torch, "npu") and hasattr(torch.npu, "config"):
        torch.npu.config.allow_internal_format = False
    return torch.device(f"npu:{device_id}")


def parse_quant_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JoyAI-Image transformer quantization script (FLUX.2-dev style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    core_group = parser.add_argument_group(title="Core")
    core_group.add_argument("--ckpt-root", type=str, required=True, help="JoyAI checkpoint root directory.")
    core_group.add_argument("--config", type=str, default=None, help="Optional config path; defaults to <ckpt-root>/infer_config.py.")
    core_group.add_argument("--device-id", type=int, default=0, help="NPU device id.")
    core_group.add_argument(
        "--dit-precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Transformer load precision before quantization.",
    )

    quant_group = parser.add_argument_group(title="Quantization")
    quant_group.add_argument("--quant-save-dir", type=str, default="./joyai_image_quant_weights")
    quant_group.add_argument("--quant-mode", type=str, default="w8a8", choices=["w8a8", "w8a16"])
    quant_group.add_argument("--is-dynamic", action="store_true", default=False)
    quant_group.add_argument("--w-sym", action="store_true", default=False)
    quant_group.add_argument("--act-method", type=int, default=3, help="1=min-max, 2=histogram, 3=auto-mixed")
    quant_group.add_argument(
        "--layer-include",
        type=str,
        nargs="*",
        default=["*double_blocks*"],
        help="Wildcard patterns of module names to include in quantization candidate set.",
    )
    quant_group.add_argument(
        "--layer-exclude",
        type=str,
        nargs="*",
        default=["*img_mod.1*", "*txt_mod.1*", "*condition_embedder*", "*proj_out*"],
        help="Wildcard patterns of module names to exclude from quantization.",
    )

    args = parser.parse_args()
    ckpt_root = Path(args.ckpt_root).expanduser()
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {ckpt_root}")
    Path(args.quant_save_dir).mkdir(parents=True, exist_ok=True)
    return args


def load_joyai_transformer(args: argparse.Namespace, device: torch.device):
    settings = load_settings(
        ckpt_root=args.ckpt_root,
        config_path=args.config,
    )
    config_class = load_infer_config_class_from_pyfile(settings.config_path)
    cfg = config_class()
    cfg.dit_ckpt = settings.ckpt_path
    cfg.training_mode = False
    cfg.use_fsdp_inference = False
    cfg.hsdp_shard_dim = 1
    cfg.reshard_after_forward = False
    cfg.dit_precision = args.dit_precision

    model = load_dit(cfg, device=device)
    model.eval()
    model.requires_grad_(False)
    return model, settings, cfg


def remove_accelerate_hooks(model: torch.nn.Module) -> None:
    if remove_hook_from_module is None:
        return
    for _, module in model.named_modules():
        try:
            remove_hook_from_module(module)
        except Exception:
            continue


def save_quant_metadata(
    save_dir: Path,
    args: argparse.Namespace,
    settings,
    disable_quant_layers: list[str],
) -> None:
    metadata = {
        "source_ckpt_root": str(Path(args.ckpt_root).expanduser().resolve()),
        "source_transformer_ckpt": str(settings.ckpt_path),
        "config_path": str(settings.config_path),
        "quant_mode": args.quant_mode,
        "is_dynamic": bool(args.is_dynamic),
        "w_sym": bool(args.w_sym),
        "act_method": int(args.act_method),
        "device_id": int(args.device_id),
        "dit_precision": str(args.dit_precision),
        "layer_include": list(args.layer_include),
        "layer_exclude": list(args.layer_exclude),
        "disabled_layer_count": len(disable_quant_layers),
    }
    with (save_dir / "quant_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    args = parse_quant_args()
    device = init_npu_env(args.device_id)

    print(f"Loading JoyAI transformer from: {args.ckpt_root}")
    model, settings, _ = load_joyai_transformer(args, device=device)

    disable_quant_layers = get_disable_layer_names(
        model=model,
        layer_include=args.layer_include,
        layer_exclude=args.layer_exclude,
    )
    print(
        f"Quant config: mode={args.quant_mode}, disable_layers={len(disable_quant_layers)}"
    )

    quant_config = get_joyai_image_quant_config(
        quant_mode=args.quant_mode,
        is_dynamic=args.is_dynamic,
        w_sym=args.w_sym,
        act_method=args.act_method,
        disable_names=disable_quant_layers,
        dev_id=args.device_id,
    )

    remove_accelerate_hooks(model)
    quantizer = JoyAIImageQuantizer(model=model, quant_config=quant_config)
    calibrator = quantizer.quantize()
    quantizer.save_quantized_weights(calibrator, save_dir=args.quant_save_dir)
    save_quant_metadata(
        save_dir=Path(args.quant_save_dir),
        args=args,
        settings=settings,
        disable_quant_layers=disable_quant_layers,
    )
    print("JoyAI-Image quantization completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Quantization failed: {exc}", flush=True)
        raise

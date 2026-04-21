#!/usr/bin/env bash
set -euo pipefail

ckpt_root=${1:-/path/to/JoyAI-Image-Edit}
device_id=${2:-0}

python ./quantization/quant_joyai_image.py \
  --ckpt-root "${ckpt_root}" \
  --device-id "${device_id}" \
  --quant-mode w8a8 \
  --w-sym \
  --act-method 3 \
  --is-dynamic \
  --quant-save-dir ./joyai_image_quant_w8a8


#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}

export COND_CACHE=${COND_CACHE:-0}
export UNCOND_CACHE=${UNCOND_CACHE:-0}
export ATTENTION_CACHE=${ATTENTION_CACHE:-0}

export ADALN_FUSE=${ADALN_FUSE:-1}
export ROPE_FUSE=${ROPE_FUSE:-1}

ckpt_root=${1:?Usage: bash server_w8a8.sh <ckpt-root> <quant-desc-path> [image-path] [output-path]}
quant_desc_path=${2:?Usage: bash server_w8a8.sh <ckpt-root> <quant-desc-path> [image-path] [output-path]}
image_path=${3:-../test2.png}
output_path=${4:-outputs/result.png}

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} inference.py \
  --ckpt-root "${ckpt_root}" \
  --prompt "I want the human from the second figure to lovingly play with the dog, petting its fluffy, light tan and white coat while the puppy looks up with its dark, expressive eyes." \
  --image "${image_path}" \
  --output "${output_path}" \
  --seed 123 \
  --steps 50 \
  --guidance-scale 4.0 \
  --basesize 1024 \
  --cfg-size 2 \
  --keep-image-size \
  --warmup \
  --quant-desc-path "${quant_desc_path}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEDSB_DIR="$ROOT_DIR/data/external/fed-sb/fed_sb"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
EPSILON="${EPSILON:-3}"
LORA_R="${LORA_R:-64}"
DATA_DIR="${DATA_DIR:-DP/SNLI/data}"

cd "$FEDSB_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python DP/SNLI/trainer.py \
  --data_dir "$DATA_DIR" \
  --dataset_not_processed \
  --lora_r "$LORA_R" \
  --epsilon "$EPSILON"

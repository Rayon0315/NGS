#!/usr/bin/env bash
set -euo pipefail
cd /home/data2/MaLei/NGS
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python train_nam.py \
  --model pafr_ecm \
  --train-region-partitions datasets/EORSSD/train-nam-npz \
  --region-consistency-weight 0.25 \
  --seed 42 \
  --run-dir runs/pafr_smt_ecm_nam_w025

#!/usr/bin/env bash
set -euo pipefail
cd /home/data2/MaLei/NGS
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python train_nam.py \
  --model pafr_ecm_deep \
  --train-images datasets/EORSSD/train-images \
  --train-scribble-gt datasets/EORSSD/train-scribble-gt \
  --train-scribble-mask datasets/EORSSD/train-scribble-mask \
  --train-region-partitions datasets/EORSSD/train-nam-npz \
  --region-consistency-weight 0.75 \
  --seed 42 \
  --run-dir runs/pafr_smt_ecm_deep_nam_w075

#!/usr/bin/env bash
set -euo pipefail
cd /home/data2/MaLei/NGS
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python train.py \
  --model pafr \
  --seed 42 \
  --run-dir runs/pafr_smt

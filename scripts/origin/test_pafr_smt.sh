#!/usr/bin/env bash
set -euo pipefail
cd /home/data2/MaLei/NGS
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python test.py \
  --model auto \
  --checkpoint runs/pafr_smt/checkpoints/final.pth \
  --output-dir runs/pafr_smt/test/EORSSD \
  --seed 42

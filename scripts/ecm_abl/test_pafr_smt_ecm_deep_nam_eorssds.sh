#!/usr/bin/env bash
set -euo pipefail
cd /home/data2/MaLei/NGS
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python test.py \
  --model auto \
  --checkpoint runs/pafr_smt_ecm_deep_nam_w075/checkpoints/final.pth \
  --test-images datasets/EORSSD/test-images \
  --test-masks datasets/EORSSD/test-labels \
  --output-dir runs/pafr_smt_ecm_deep_nam_w075/test/EORSSD \
  --seed 42

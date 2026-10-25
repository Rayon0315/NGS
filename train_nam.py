from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from engine.run_training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NGS training with NAMLab region-consistency loss."
    )
    parser.add_argument(
        "--model",
        choices=("pafr", "pafr_ecm", "pafr_ecm_deep", "pafr_ecm_full"),
        default="pafr_ecm",
        help="ECM placement: pafr=none, pafr_ecm=shallow, pafr_ecm_deep=deep, pafr_ecm_full=four-scale.",
    )
    parser.add_argument("--train-images", default="datasets/EORSSD/train-images")
    parser.add_argument(
        "--train-scribble-gt",
        default="datasets/EORSSD/train-scribble-gt",
    )
    parser.add_argument(
        "--train-scribble-mask",
        default="datasets/EORSSD/train-scribble-mask",
    )
    parser.add_argument(
        "--train-region-partitions",
        default="datasets/EORSSD/train-region-partitions",
    )
    parser.add_argument("--region-consistency-weight", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--crf-weight", type=float, default=0.3)
    parser.add_argument("--crf-radius", type=int, default=5)
    parser.add_argument("--crf-xy-sigma", type=float, default=6.0)
    parser.add_argument("--crf-rgb-sigma", type=float, default=0.1)
    parser.add_argument("--src-alpha", type=float, default=0.85)
    parser.add_argument("--src-scale", type=float, default=0.3)
    parser.add_argument("--crf-scale", type=float, default=0.25)
    parser.add_argument(
        "--augment-8way",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--run-dir", default="runs/ngs_ecm_nam_eorssd")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args(), nam=True)

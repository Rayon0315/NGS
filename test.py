from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.eval_dataset import EvalDataset
from engine.common import (
    configure_reproducibility,
    get_amp_dtype,
    make_generator,
    seed_worker,
)
from metrics.sod_metrics import (
    evaluate_prediction_directory,
    get_metric_library_version,
    save_metric_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an NGS checkpoint.")
    parser.add_argument(
        "--model",
        choices=("auto", "pafr", "pafr_ecm"),
        default="auto",
        help="auto reads the model variant from checkpoint['args']['model'].",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/ngs_ecm_nam_eorssd/checkpoints/final.pth",
    )
    parser.add_argument("--test-images", default="datasets/EORSSD/test-images")
    parser.add_argument("--test-masks", default="datasets/EORSSD/test-labels")
    parser.add_argument(
        "--output-dir",
        default="runs/ngs_ecm_nam_eorssd/test/EORSSD",
    )
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(model_name: str):
    if model_name == "pafr":
        from models.networks.smt_t_pafr_sod import build_model as builder
    elif model_name == "pafr_ecm":
        from models.networks.smt_t_pafr_shallow_ecm_sod import (
            build_model as builder,
        )
    else:
        raise ValueError(f"Unsupported model variant: {model_name}")
    return builder()


def main() -> None:
    args = parse_args()
    configure_reproducibility(args.seed, True)
    device = torch.device(args.device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_args = checkpoint.get("args", {})
    model_name = args.model
    if model_name == "auto":
        model_name = checkpoint_args.get("model", "pafr_ecm")

    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
    )
    logging.info("Model variant: %s", model_name)

    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    dataset = EvalDataset(
        args.test_images,
        args.test_masks,
        (args.image_size, args.image_size),
    )
    generator = make_generator(args.seed)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )

    amp_dtype = get_amp_dtype(
        device,
        args.amp and device.type == "cuda",
    )
    start = time.perf_counter()

    with torch.inference_mode():
        for batch in tqdm(
            data_loader,
            desc="Inference",
            unit="batch",
        ):
            image = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if amp_dtype else torch.float32,
                enabled=amp_dtype is not None,
            ):
                probability = torch.sigmoid(
                    model(image)["pred"].float()
                )

            for index, name in enumerate(batch["name"]):
                height = int(batch["original_size"][index, 0])
                width = int(batch["original_size"][index, 1])
                prediction = F.interpolate(
                    probability[index : index + 1],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].clamp(0, 1)
                Image.fromarray(
                    (prediction.cpu().numpy() * 255.0)
                    .round()
                    .astype(np.uint8)
                ).save(prediction_dir / f"{name}.png")

    metrics, curves, count = evaluate_prediction_directory(
        prediction_dir,
        args.test_masks,
    )
    save_metric_curves(output_dir / "metric_curves.npz", curves)
    summary = {
        "samples": count,
        "model": model_name,
        "checkpoint": args.checkpoint,
        "pysodmetrics": get_metric_library_version(),
        "inference_seconds": time.perf_counter() - start,
        **metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

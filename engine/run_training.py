from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from data.scribble_dataset import ScribbleSODDataset
from data.scribble_region_dataset import ScribbleRegionSODDataset
from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.common import configure_reproducibility, make_generator, seed_worker
from engine.pafr_region_trainer import train_one_epoch as train_nam_epoch
from engine.pafr_trainer import train_one_epoch as train_base_epoch
from losses.pafr_loss import PAFRWeakLoss
from losses.pafr_region_consistency_loss import PAFRRegionConsistencyLoss


MODEL_BUILDERS = {
    "pafr": "models.networks.smt_t_pafr_sod",
    "pafr_ecm": "models.networks.smt_t_pafr_shallow_ecm_sod",
    "pafr_ecm_deep": "models.networks.smt_t_pafr_deep_ecm_sod",
    "pafr_ecm_full": "models.networks.smt_t_pafr_full_ecm_sod",
}


def build_model(model_name: str):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Expected one of: {', '.join(MODEL_BUILDERS)}"
        )

    if model_name == "pafr":
        from models.networks.smt_t_pafr_sod import build_model as builder
    elif model_name == "pafr_ecm":
        from models.networks.smt_t_pafr_shallow_ecm_sod import (
            build_model as builder,
        )
    elif model_name == "pafr_ecm_deep":
        from models.networks.smt_t_pafr_deep_ecm_sod import build_model as builder
    else:
        from models.networks.smt_t_pafr_full_ecm_sod import build_model as builder

    return builder()


def setup_logging(path: Path, resume: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                path,
                mode="a" if resume else "w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def prepare_csv(path: Path, nam: bool, resume: bool) -> None:
    if resume and path.exists():
        return

    fields = [
        "epoch",
        "global_step",
        "train_loss",
        "train_loss_main",
        "train_loss_pce",
        "train_loss_crf",
        "train_loss_src",
    ]
    if nam:
        fields.append("train_loss_region_consistency")
    fields += [
        "scribble_valid_ratio",
        "scribble_fg_ratio_on_valid",
        "learning_rate",
        "train_time_seconds",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(fields)


def append_csv(
    path: Path,
    epoch: int,
    step: int,
    statistics: dict,
    nam: bool,
) -> None:
    row = [
        epoch,
        step,
        statistics["loss"],
        statistics.get("loss_main", ""),
        statistics.get("loss_pce", ""),
        statistics.get("loss_crf", ""),
        statistics.get("loss_src", ""),
    ]
    if nam:
        row.append(statistics.get("loss_region_consistency", ""))
    row += [
        statistics.get("stat_valid_ratio", ""),
        statistics.get("stat_fg_ratio_on_valid", ""),
        statistics["lr"],
        statistics["time_seconds"],
    ]

    with path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(row)


def snapshot_sources(dst: Path, nam: bool) -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        "train.py",
        "train_nam.py",
        "engine/run_training.py",
        "engine/common.py",
        "engine/checkpoint.py",
        "engine/pafr_trainer.py",
        "models/backbones/smt.py",
        "models/networks/smt_t_pafr_sod.py",
        "models/networks/smt_t_pafr_shallow_ecm_sod.py",
        "models/networks/smt_t_pafr_deep_ecm_sod.py",
        "models/networks/smt_t_pafr_full_ecm_sod.py",
        "losses/pafr_loss.py",
        "losses/pafr_gated_crf.py",
        "losses/pafr_ssim.py",
        "data/scribble_dataset.py",
    ]
    if nam:
        paths += [
            "engine/pafr_region_trainer.py",
            "losses/pafr_region_consistency_loss.py",
            "data/scribble_region_dataset.py",
        ]

    dst.mkdir(parents=True, exist_ok=True)
    for relative_path in paths:
        source = root / relative_path
        if source.exists():
            shutil.copy2(
                source,
                dst / relative_path.replace("/", "__"),
            )


def run_training(args, nam: bool) -> None:
    configure_reproducibility(args.seed, args.deterministic)

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    source_dir = run_dir / "source_snapshot"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir / "train.log", args.resume is not None)
    logger = logging.getLogger(__name__)

    (run_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    snapshot_sources(source_dir, nam)

    logger.info(
        "NGS | model: %s | ECM: %s | NAM loss: %s",
        args.model,
        args.model.startswith("pafr_ecm"),
        nam,
    )
    logger.info(
        "Device: %s | AMP: %s | deterministic: %s | seed: %d",
        device,
        use_amp,
        args.deterministic,
        args.seed,
    )

    model = build_model(args.model).to(device)

    dataset_kwargs = dict(
        image_dir=args.train_images,
        scribble_gt_dir=args.train_scribble_gt,
        scribble_mask_dir=args.train_scribble_mask,
        image_size=(args.image_size, args.image_size),
        augment_8way=args.augment_8way,
    )

    if nam:
        dataset = ScribbleRegionSODDataset(
            **dataset_kwargs,
            region_partition_dir=args.train_region_partitions,
        )
    else:
        dataset = ScribbleSODDataset(**dataset_kwargs)

    if args.max_train_samples is not None:
        dataset = Subset(
            dataset,
            range(min(args.max_train_samples, len(dataset))),
        )

    generator = make_generator(args.seed)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    common_loss_kwargs = dict(
        crf_weight=args.crf_weight,
        crf_radius=args.crf_radius,
        crf_xy_sigma=args.crf_xy_sigma,
        crf_rgb_sigma=args.crf_rgb_sigma,
        src_alpha=args.src_alpha,
        crf_scale=args.crf_scale,
    )

    if nam:
        criterion = PAFRRegionConsistencyLoss(
            region_consistency_weight=args.region_consistency_weight,
            **common_loss_kwargs,
        )
    else:
        criterion = PAFRWeakLoss(**common_loss_kwargs)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            generator,
        )
        logger.info(
            "Resumed from %s | epoch %d | step %d",
            args.resume,
            start_epoch,
            global_step,
        )

    metrics_path = log_dir / "metrics.csv"
    prepare_csv(metrics_path, nam, args.resume is not None)
    trainer = train_nam_epoch if nam else train_base_epoch

    logger.info(
        "Samples: %d | batches/epoch: %d | epochs: %d",
        len(dataset),
        len(data_loader),
        args.epochs,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        statistics, global_step = trainer(
            model,
            data_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            global_step,
            use_amp,
            args.log_interval,
            args.src_scale,
        )
        append_csv(
            metrics_path,
            epoch,
            global_step,
            statistics,
            nam,
        )
        scheduler.step()

        save_checkpoint(
            checkpoint_dir / "latest.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            args,
            epoch,
            global_step,
            generator,
        )
        if epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                epoch,
                global_step,
                generator,
            )

        logger.info(
            "Epoch %03d complete | loss %.6f | lr %.8f | %.1fs",
            epoch,
            statistics["loss"],
            statistics["lr"],
            statistics["time_seconds"],
        )

    save_checkpoint(
        checkpoint_dir / "final.pth",
        model,
        optimizer,
        scheduler,
        scaler,
        args,
        args.epochs,
        global_step,
        generator,
    )
    logger.info("Training complete: %s", checkpoint_dir / "final.pth")
from __future__ import annotations

import logging
import time

import torch
import torchvision.transforms.functional as TF

from engine.common import MAX_GRAD_NORM, get_amp_dtype

logger = logging.getLogger(__name__)

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _denorm(x):
    return x * x.new_tensor(_STD).view(1, 3, 1, 1) + x.new_tensor(_MEAN).view(
        1, 3, 1, 1
    )


def _norm(x):
    return (x - x.new_tensor(_MEAN).view(1, 3, 1, 1)) / x.new_tensor(_STD).view(
        1, 3, 1, 1
    )


def make_pafr_color_jitter(image):
    rgb = _denorm(image.float()).clamp(0, 1)
    b = float(torch.empty(1).uniform_(0.8, 1.2))
    c = float(torch.empty(1).uniform_(0.8, 1.2))
    s = float(torch.empty(1).uniform_(0.8, 1.2))
    h = float(torch.empty(1).uniform_(-0.1, 0.1))
    funcs = [
        lambda x: TF.adjust_brightness(x, b),
        lambda x: TF.adjust_contrast(x, c),
        lambda x: TF.adjust_saturation(x, s),
        lambda x: TF.adjust_hue(x, h),
    ]
    for i in torch.randperm(4).tolist():
        rgb = funcs[i](rgb)
    return _norm(rgb.clamp(0, 1))


def _report(outputs, loss_dict):
    pred = outputs["pred"].detach().float()
    return (
        f"pred=[{pred.min().item():.4g},{pred.max().item():.4g}] losses="
        + ",".join(
            f"{k}={float(v.detach().float()):.4g}"
            for k, v in loss_dict.items()
            if torch.is_tensor(v) and v.numel() == 1
        )
    )


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    global_step,
    use_amp,
    log_interval,
    src_scale=0.3,
):
    model.train()
    criterion = criterion.to(device)
    criterion.train()
    dtype = get_amp_dtype(device, use_amp)
    use_scaler = (
        use_amp and dtype == torch.float16 and scaler.is_enabled()
    )
    total = 0
    sums = {}
    start = time.perf_counter()

    for bi, batch in enumerate(data_loader, 1):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["scribble_gt"].to(device, non_blocking=True)
        valid = batch["scribble_mask"].to(device, non_blocking=True)
        scaled = torch.nn.functional.interpolate(
            make_pafr_color_jitter(image),
            scale_factor=src_scale,
            mode="bilinear",
            align_corners=True,
            recompute_scale_factor=False,
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=dtype if dtype is not None else torch.float32,
            enabled=dtype is not None,
        ):
            outputs = model(image)
            scaled_outputs = model(scaled)
            ld = criterion(outputs, scaled_outputs, image, target, valid)
            loss = ld["loss"]

        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                f"Non-finite loss epoch={epoch} batch={bi} {_report(outputs, ld)}"
            )

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=False
        )

        if not torch.isfinite(grad):
            raise FloatingPointError(
                f"Non-finite grad epoch={epoch} batch={bi}"
            )

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        bs = image.shape[0]
        total += bs
        global_step += 1

        for k, v in ld.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach().float()) * bs

        if bi % log_interval == 0 or bi == len(data_loader):
            logger.info(
                "Epoch %03d | Batch %05d/%05d | Step %07d | Loss %.6f | PCE %.6f | CRF %.6f | SRC %.6f | Grad %.4f",
                epoch,
                bi,
                len(data_loader),
                global_step,
                float(loss.detach().float()),
                float(ld["loss_pce"]),
                float(ld["loss_crf"]),
                float(ld["loss_src"]),
                float(grad.detach().float()),
            )

    stats = {k: v / total for k, v in sums.items()}
    stats["lr"] = optimizer.param_groups[0]["lr"]
    stats["time_seconds"] = time.perf_counter() - start
    return stats, global_step
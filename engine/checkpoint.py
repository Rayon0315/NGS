from __future__ import annotations

import torch

from engine.common import get_rng_state, set_rng_state


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    epoch,
    global_step,
    data_generator,
):
    torch.save(
        {
            "format_version": 2,
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "rng_state": get_rng_state(data_generator),
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    data_generator,
):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    set_rng_state(ckpt.get("rng_state"), data_generator)
    return ckpt["epoch"] + 1, ckpt["global_step"]
from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gaussian(window_size: int, sigma: float) -> Tensor:
    k = torch.tensor(
        [
            exp(-((x - window_size // 2) ** 2) / (2.0 * sigma * sigma))
            for x in range(window_size)
        ],
        dtype=torch.float32,
    )
    return k / k.sum()


def _create_window(
    window_size: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    one = _gaussian(window_size, 1.5).unsqueeze(1)
    two = one.mm(one.t()).unsqueeze(0).unsqueeze(0)
    return (
        two.expand(channels, 1, window_size, window_size)
        .contiguous()
        .to(device=device, dtype=dtype)
    )


class SSIM(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, image1: Tensor, image2: Tensor) -> Tensor:
        c = image1.shape[1]
        w = _create_window(
            self.window_size, c, image1.device, image1.dtype
        )
        p = self.window_size // 2

        mu1 = F.conv2d(image1, w, padding=p, groups=c)
        mu2 = F.conv2d(image2, w, padding=p, groups=c)
        m1 = mu1.square()
        m2 = mu2.square()
        m12 = mu1 * mu2

        s1 = F.conv2d(image1 * image1, w, padding=p, groups=c) - m1
        s2 = F.conv2d(image2 * image2, w, padding=p, groups=c) - m2
        s12 = F.conv2d(image1 * image2, w, padding=p, groups=c) - m12

        c1 = 0.01**2
        c2 = 0.03**2

        return (
            ((2 * m12 + c1) * (2 * s12 + c2))
            / ((m1 + m2 + c1) * (s1 + s2 + c2))
        ).mean()
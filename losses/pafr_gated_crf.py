from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GatedCRF(nn.Module):
    def forward(
        self,
        prediction: Tensor,
        kernels_desc: list[dict[str, float]],
        kernels_radius: int,
        sample: dict[str, Tensor],
    ) -> Tensor:
        b, c, h, w = prediction.shape
        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, b, h, w, prediction.device
        )
        u = self._unfold(prediction, kernels_radius)
        center = u[:, :, kernels_radius, kernels_radius, :, :].view(b, c, 1, 1, h, w)
        d = (center - u).abs()
        diameter = kernels_radius * 2 + 1
        return (
            (kernels * d)
            .view(b, c, diameter * diameter, h, w)
            .sum(dim=2)
            .mean()
        )

    @classmethod
    def _create_kernels(
        cls,
        descs,
        radius,
        sample,
        batch,
        height,
        width,
        device,
    ):
        kernels = None
        for desc in descs:
            features = []
            weight = float(desc["weight"])
            for modality, sigma_value in desc.items():
                if modality == "weight":
                    continue
                sigma = float(sigma_value)
                if modality == "xy":
                    feature = cls._get_mesh(batch, height, width, device)
                else:
                    feature = sample[modality]
                    if feature.shape[-2:] != (height, width):
                        feature = F.interpolate(
                            feature,
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        )
                features.append(feature / sigma)
            kernel = weight * cls._create_kernels_from_features(
                torch.cat(features, dim=1), radius
            )
            kernels = kernel if kernels is None else kernels + kernel
        if kernels is None:
            raise ValueError("kernels_desc must contain at least one kernel")
        return kernels

    @classmethod
    def _create_kernels_from_features(cls, features: Tensor, radius: int) -> Tensor:
        b, c, h, w = features.shape
        u = cls._unfold(features, radius)
        center = u[:, :, radius, radius, :, :].view(b, c, 1, 1, h, w)
        return (-0.5 * (u - center).square()).sum(dim=1, keepdim=True).exp()

    @staticmethod
    def _get_mesh(batch, height, width, device):
        x = (
            torch.arange(width, dtype=torch.float32, device=device)
            .view(1, 1, 1, width)
            .repeat(batch, 1, height, 1)
        )
        y = (
            torch.arange(height, dtype=torch.float32, device=device)
            .view(1, 1, height, 1)
            .repeat(batch, 1, 1, width)
        )
        return torch.cat((x, y), dim=1)

    @staticmethod
    def _unfold(image: Tensor, radius: int) -> Tensor:
        b, c, h, w = image.shape
        d = 2 * radius + 1
        return F.unfold(image, kernel_size=d, padding=radius).view(b, c, d, d, h, w)
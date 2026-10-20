from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from losses.pafr_gated_crf import GatedCRF
from losses.pafr_ssim import SSIM


class PAFRWeakLoss(nn.Module):
    def __init__(
        self,
        crf_weight: float = 0.3,
        crf_radius: int = 5,
        crf_xy_sigma: float = 6.0,
        crf_rgb_sigma: float = 0.1,
        src_alpha: float = 0.85,
        crf_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.crf_weight = crf_weight
        self.crf_radius = crf_radius
        self.src_alpha = src_alpha
        self.crf_scale = crf_scale
        self.kernels_desc = [
            {"weight": 1.0, "xy": crf_xy_sigma, "rgb": crf_rgb_sigma}
        ]
        self.gated_crf = GatedCRF()
        self.ssim = SSIM(11)

    def forward(
        self,
        outputs: dict[str, Any],
        scaled_outputs: dict[str, Any],
        image: Tensor,
        target: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        logits_list = outputs["pafr_logits"]
        scaled = scaled_outputs["pafr_logits"]
        image_float = image.float()
        crf_image = F.interpolate(
            image_float,
            scale_factor=self.crf_scale,
            mode="bilinear",
            align_corners=True,
        )
        sample = {"rgb": crf_image}
        pces = []
        crfs = []
        outs = []

        for logits in logits_list:
            lf = logits.float()
            pce = self.masked_bce_with_logits(lf, target, valid_mask)
            prob = torch.sigmoid(lf)
            cp = F.interpolate(
                prob,
                size=crf_image.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            crf = self.gated_crf(cp, self.kernels_desc, self.crf_radius, sample)
            pces.append(pce)
            crfs.append(crf)
            outs.append(pce + self.crf_weight * crf)

        src_refined = self._src_loss(scaled[0], logits_list[0])
        src_side1 = self._src_loss(scaled[1], logits_list[1])
        src = src_refined + src_side1
        total = torch.stack(outs).sum() + src

        valid = valid_mask.detach().float()
        tgt = target.detach().float()
        count = valid.sum().clamp_min(1.0)

        return {
            "loss": total,
            "loss_main": outs[0] + src,
            "loss_pce": torch.stack(pces).sum(),
            "loss_crf": torch.stack(crfs).sum(),
            "loss_src": src,
            "loss_src_refined": src_refined,
            "loss_src_side1": src_side1,
            "stat_valid_ratio": valid.mean(),
            "stat_fg_ratio_on_valid": (tgt * valid).sum() / count,
        }

    def _src_loss(self, scaled_logits: Tensor, native_logits: Tensor) -> Tensor:
        with torch.autocast(
            device_type=scaled_logits.device.type, enabled=False
        ):
            sp = torch.sigmoid(scaled_logits.float())
            np = torch.sigmoid(native_logits.float())
            np = F.interpolate(
                np, size=sp.shape[-2:], mode="bilinear", align_corners=True
            )
            ssim = 1.0 - self.ssim(sp, np)
            l1 = (sp - np).abs().mean()
            return self.src_alpha * ssim + (1.0 - self.src_alpha) * l1

    @staticmethod
    def masked_bce_with_logits(
        logits: Tensor, target: Tensor, valid_mask: Tensor
    ) -> Tensor:
        target = target.float()
        valid_mask = valid_mask.float()
        lm = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        return (lm * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
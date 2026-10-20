from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.smt_t_pafr_sod import PRETRAINED_PATH, SMTTinyPAFRSOD


class _GroupedActivation5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                5,
                padding=2,
                groups=math.gcd(in_channels, out_channels),
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _UpsampleConv3x3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, target_size):
        return self.relu(
            self.conv(
                F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            )
        )


class ShallowFeatureECM(nn.Module):
    def __init__(self, input_channels: tuple[int, int], edge_channels: int = 96):
        super().__init__()
        self.stage1_activation = _GroupedActivation5x5(input_channels[0], edge_channels)
        self.stage2_activation = _GroupedActivation5x5(input_channels[1], edge_channels)
        self.stage2_to_stage1 = _UpsampleConv3x3(edge_channels)
        self.final_structure = nn.Sequential(
            nn.Conv2d(edge_channels, edge_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        s1, s2 = features
        a = self.stage1_activation(s1)
        b = self.stage2_to_stage1(self.stage2_activation(s2), a.shape[-2:])
        return self.final_structure(a + b)


class SMTTinyPAFRShallowECMSOD(SMTTinyPAFRSOD):
    input_keys = ("image",)

    def __init__(self, pretrained_path: str | Path | None, ecm_channels: int = 96):
        super().__init__(pretrained_path)
        self.ecm = ShallowFeatureECM(
            (self.backbone.out_channels[0], self.backbone.out_channels[1]),
            ecm_channels,
        )
        self.structure_fusion = nn.Sequential(
            nn.Conv2d(64 + ecm_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _resize_local(x, size):
        return (
            x
            if x.shape[-2:] == size
            else F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        )

    def forward(self, image):
        size = image.shape[-2:]
        s1, s2, s3, s4 = self.backbone(image)
        s5 = self.stage5(self.stage5_projection(self.pool4(s4)))
        opm = self.opm(s5)
        d5, _ = self.crm5(opm, s5)
        d4, _ = self.crm4(self._resize_local(d5, s4.shape[-2:]), s4)
        d3, _ = self.crm3(self._resize_local(d4, s3.shape[-2:]), s3)
        d2, _ = self.crm2(self._resize_local(d3, s2.shape[-2:]), s2)
        d1, _ = self.crm1(self._resize_local(d2, s1.shape[-2:]), s1)
        d1 = self.structure_fusion(
            torch.cat((d1, self.ecm((s1, s2))), dim=1)
        )
        side1 = self._resize_local(self.out1(d1), size)
        logits = [
            self.refunet(side1),
            side1,
            self._resize_local(self.out2(d2), size),
            self._resize_local(self.out3(d3), size),
            self._resize_local(self.out4(d4), size),
            self._resize_local(self.out5(d5), size),
            self._resize_local(self.out_opm(opm), size),
        ]
        return {"pred": logits[0], "aux": logits[1:], "pafr_logits": logits}


def build_model():
    return SMTTinyPAFRShallowECMSOD(PRETRAINED_PATH)
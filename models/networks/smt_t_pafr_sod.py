from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.smt import SMTTinyBackbone, smt_tiny

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_PATH = PROJECT_ROOT / "pretrained" / "smt" / "smt_tiny.pth"


class PAFRBasicBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        r = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + r)


class OPM(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.convx0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bnx0 = nn.BatchNorm2d(channels)
        self.convx1 = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.convx2 = nn.Conv2d(channels, channels, 1)
        self.bnx2 = nn.BatchNorm2d(channels)
        hidden = max(channels // 16, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x0 = F.relu(self.bnx0(self.convx0(x)), inplace=True)
        weight, bias = self.convx1(x).chunk(2, dim=1)
        transformed = F.relu(weight * x0 + bias, inplace=True)
        projected = F.relu(self.bnx2(self.convx2(x)), inplace=True)
        return projected * self.conv_du(self.avg_pool(transformed))


class CRM(nn.Module):
    def __init__(self, left_channels, encoder_channels, out_channels):
        super().__init__()
        self.conv_l1 = nn.Conv2d(left_channels, out_channels, 3, padding=1)
        self.bn_l1 = nn.BatchNorm2d(out_channels)
        self.conv_d1 = nn.Conv2d(encoder_channels, out_channels, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(out_channels // 16, 1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(out_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
        )
        self.conv_d2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, left, encoder_feature):
        lf = F.relu(self.bn_l1(self.conv_l1(left)), inplace=True)
        df = F.relu(self.bn_d1(self.conv_d1(encoder_feature)), inplace=True)
        df = self.conv_d2(df)
        ca = torch.sigmoid(self.conv_du(self.avg_pool(lf)))
        cf = ca * lf
        ad = df

        if ad.shape[-2:] != lf.shape[-2:]:
            ad = F.interpolate(ad, size=lf.shape[-2:], mode="bilinear", align_corners=False)

        sf = F.relu(lf * ad, inplace=True)
        out = F.relu(self.bn3(self.conv3(torch.cat((cf + sf, sf), dim=1))), inplace=True)
        return out, df


class RefUnet(nn.Module):
    def __init__(self, in_channels=1, stem_channels=64):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, stem_channels, 3, padding=1)

        for i in range(1, 6):
            setattr(self, f"conv{i}", nn.Conv2d(stem_channels if i == 1 else 64, 64, 3, padding=1))
            setattr(self, f"bn{i}", nn.BatchNorm2d(64))

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        for i in range(1, 5):
            setattr(self, f"conv_d{i}", nn.Conv2d(128, 64, 3, padding=1))
            setattr(self, f"bn_d{i}", nn.BatchNorm2d(64))

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

    @staticmethod
    def _up_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        residual = x
        x = self.conv0(x)
        x1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.pool(x1)
        x2 = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool(x2)
        x3 = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.pool(x3)
        x4 = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.pool(x4)
        x5 = F.relu(self.bn5(self.conv5(x)), inplace=True)

        x = self._up_to(x5, x4)
        d4 = F.relu(self.bn_d4(self.conv_d4(torch.cat((x, x4), 1))), inplace=True)
        x = self._up_to(d4, x3)
        d3 = F.relu(self.bn_d3(self.conv_d3(torch.cat((x, x3), 1))), inplace=True)
        x = self._up_to(d3, x2)
        d2 = F.relu(self.bn_d2(self.conv_d2(torch.cat((x, x2), 1))), inplace=True)
        x = self._up_to(d2, x1)
        d1 = F.relu(self.bn_d1(self.conv_d1(torch.cat((x, x1), 1))), inplace=True)
        return residual + self.conv_d0(d1)


class SMTTinyPAFRSOD(nn.Module):
    input_keys = ("image",)

    def __init__(self, pretrained_path):
        super().__init__()
        self.backbone: SMTTinyBackbone = smt_tiny(pretrained_path=pretrained_path)
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage5_projection = nn.Sequential(
            nn.Conv2d(512, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.stage5 = nn.Sequential(
            PAFRBasicBlock(512),
            PAFRBasicBlock(512),
            PAFRBasicBlock(512),
        )
        self.opm = OPM(512)
        self.crm5 = CRM(512, 512, 512)
        self.crm4 = CRM(512, 512, 256)
        self.crm3 = CRM(256, 256, 128)
        self.crm2 = CRM(128, 128, 64)
        self.crm1 = CRM(64, 64, 64)
        self.out_opm = nn.Conv2d(512, 1, 3, padding=1)
        self.out5 = nn.Conv2d(512, 1, 3, padding=1)
        self.out4 = nn.Conv2d(256, 1, 3, padding=1)
        self.out3 = nn.Conv2d(128, 1, 3, padding=1)
        self.out2 = nn.Conv2d(64, 1, 3, padding=1)
        self.out1 = nn.Conv2d(64, 1, 3, padding=1)
        self.refunet = RefUnet(1, 64)

    @staticmethod
    def _resize(x, size):
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, image):
        size = image.shape[-2:]
        s1, s2, s3, s4 = self.backbone(image)
        s5 = self.stage5(self.stage5_projection(self.pool4(s4)))
        opm = self.opm(s5)
        d5, _ = self.crm5(opm, s5)
        d4, _ = self.crm4(self._resize(d5, s4.shape[-2:]), s4)
        d3, _ = self.crm3(self._resize(d4, s3.shape[-2:]), s3)
        d2, _ = self.crm2(self._resize(d3, s2.shape[-2:]), s2)
        d1, _ = self.crm1(self._resize(d2, s1.shape[-2:]), s1)
        logits = [
            self.refunet(self._resize(self.out1(d1), size)),
            self._resize(self.out1(d1), size),
            self._resize(self.out2(d2), size),
            self._resize(self.out3(d3), size),
            self._resize(self.out4(d4), size),
            self._resize(self.out5(d5), size),
            self._resize(self.out_opm(opm), size),
        ]
        return {"pred": logits[0], "aux": logits[1:], "pafr_logits": logits}


def build_model():
    return SMTTinyPAFRSOD(PRETRAINED_PATH)
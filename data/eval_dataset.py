from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from data.scribble_dataset import ScribbleSODDataset


class EvalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(352, 352)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.images = ScribbleSODDataset._collect(self.image_dir)
        self.masks = ScribbleSODDataset._collect(self.mask_dir)
        self.names = sorted(self.images)
        missing = sorted(set(self.names) - set(self.masks))
        if missing:
            raise FileNotFoundError(f"Missing test masks: {missing[:20]}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        with Image.open(self.images[name]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        ow, oh = image.size
        th, tw = self.image_size
        resized = image.resize((tw, th), Image.Resampling.BILINEAR)
        return {
            "image": ScribbleSODDataset._image_tensor(resized),
            "name": name,
            "original_size": torch.tensor([oh, ow], dtype=torch.long),
        }
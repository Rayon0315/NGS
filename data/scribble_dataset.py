from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class ScribbleSODDataset(Dataset):
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    IMAGE_MEAN = torch.tensor(
        [0.485, 0.456, 0.406], dtype=torch.float32
    ).view(3, 1, 1)
    IMAGE_STD = torch.tensor(
        [0.229, 0.224, 0.225], dtype=torch.float32
    ).view(3, 1, 1)

    def __init__(
        self,
        image_dir,
        scribble_gt_dir,
        scribble_mask_dir,
        image_size=(352, 352),
        augment_8way=True,
    ):
        self.image_dir = Path(image_dir)
        self.gt_dir = Path(scribble_gt_dir)
        self.mask_dir = Path(scribble_mask_dir)
        self.image_size = image_size
        self.augment_8way = augment_8way

        self.image_map = self._collect(self.image_dir)
        self.gt_map = self._collect(self.gt_dir)
        self.mask_map = self._collect(self.mask_dir)
        self.names = sorted(self.image_map)

        if not self.names:
            raise RuntimeError(f"No images found in {self.image_dir}")

        for label, m in [
            ("scribble GT", self.gt_map),
            ("scribble mask", self.mask_map),
        ]:
            missing = sorted(set(self.names) - set(m))
            if missing:
                raise FileNotFoundError(f"Missing {label}: {missing[:20]}")

    def __len__(self):
        return len(self.names) * (8 if self.augment_8way else 1)

    @classmethod
    def _collect(cls, directory: Path):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        return {
            p.stem: p
            for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() in cls.IMAGE_SUFFIXES
        }

    @staticmethod
    def _transform(img: Image.Image, i: int):
        if i == 0:
            return img
        if i == 1:
            return img.transpose(Image.Transpose.ROTATE_90)
        if i == 2:
            return img.transpose(Image.Transpose.ROTATE_180)
        if i == 3:
            return img.transpose(Image.Transpose.ROTATE_270)

        f = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if i == 4:
            return f
        if i == 5:
            return f.transpose(Image.Transpose.ROTATE_90)
        if i == 6:
            return f.transpose(Image.Transpose.ROTATE_180)
        if i == 7:
            return f.transpose(Image.Transpose.ROTATE_270)

        raise ValueError(i)

    @staticmethod
    def _read_rgb(p):
        with Image.open(p) as im:
            return ImageOps.exif_transpose(im).convert("RGB")

    @staticmethod
    def _read_l(p):
        with Image.open(p) as im:
            return ImageOps.exif_transpose(im).convert("L")

    @classmethod
    def _image_tensor(cls, img):
        a = np.array(img, dtype=np.float32, copy=True)
        t = torch.from_numpy(a).permute(2, 0, 1).contiguous() / 255.0
        return (t - cls.IMAGE_MEAN) / cls.IMAGE_STD

    @staticmethod
    def _binary_tensor(img):
        a = np.array(img, dtype=np.float32, copy=True)
        return (torch.from_numpy(a).unsqueeze(0) / 255.0 >= 0.5).float()

    def __getitem__(self, index):
        base, ti = divmod(index, 8) if self.augment_8way else (index, 0)
        name = self.names[base]

        image = self._read_rgb(self.image_map[name])
        gt = self._read_l(self.gt_map[name])
        mask = self._read_l(self.mask_map[name])
        ow, oh = image.size

        image = self._transform(image, ti)
        gt = self._transform(gt, ti)
        mask = self._transform(mask, ti)

        th, tw = self.image_size
        image = image.resize((tw, th), Image.Resampling.BILINEAR)
        gt = gt.resize((tw, th), Image.Resampling.NEAREST)
        mask = mask.resize((tw, th), Image.Resampling.NEAREST)

        return {
            "image": self._image_tensor(image),
            "scribble_gt": self._binary_tensor(gt),
            "scribble_mask": self._binary_tensor(mask),
            "name": name,
            "original_size": torch.tensor([oh, ow], dtype=torch.long),
        }
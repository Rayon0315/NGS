from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.scribble_dataset import ScribbleSODDataset


class ScribbleRegionSODDataset(ScribbleSODDataset):
    def __init__(self, *args, region_partition_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.region_partition_dir = Path(region_partition_dir)
        self.region_map = {
            p.stem: p
            for p in sorted(self.region_partition_dir.glob("*.npz"))
        }
        missing = sorted(set(self.names) - set(self.region_map))
        if missing:
            raise FileNotFoundError(f"Missing NAMLab partitions: {missing[:20]}")

    @staticmethod
    def _load_partition(path):
        with np.load(path, allow_pickle=False) as data:
            if "partition" not in data:
                raise KeyError(f"{path} lacks 'partition'")
            p = np.array(data["partition"], dtype=np.int32, copy=True)
        if p.ndim != 2 or p.min() < 0:
            raise ValueError(f"Invalid partition: {path}, shape={p.shape}")
        return p

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        base, ti = divmod(index, 8) if self.augment_8way else (index, 0)
        name = self.names[base]
        p = Image.fromarray(
            self._load_partition(self.region_map[name]), mode="I"
        )
        p = self._transform(p, ti)
        th, tw = self.image_size
        p = p.resize((tw, th), Image.Resampling.NEAREST)
        sample["region_partition"] = torch.from_numpy(
            np.array(p, dtype=np.int64, copy=True)
        )
        return sample
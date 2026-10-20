from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
from py_sod_metrics import (
    Emeasure,
    FmeasureHandler,
    FmeasureV2,
    MAE,
    PrecisionHandler,
    RecallHandler,
    Smeasure,
    WeightedFmeasure,
)
from tqdm import tqdm

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class SODMetricRecorder:
    def __init__(self):
        self.mae = MAE()
        self.smeasure = Smeasure()
        self.wfm = WeightedFmeasure()
        self.em = Emeasure()
        self.fm = FmeasureV2(
            metric_handlers={
                "fmeasure": FmeasureHandler(
                    beta=0.3, with_adaptive=True, with_dynamic=True
                ),
                "precision": PrecisionHandler(
                    with_adaptive=True, with_dynamic=True
                ),
                "recall": RecallHandler(with_adaptive=True, with_dynamic=True),
            }
        )

    def step(self, prediction, ground_truth):
        self.mae.step(prediction, ground_truth)
        self.smeasure.step(prediction, ground_truth)
        self.wfm.step(prediction, ground_truth)
        self.em.step(prediction, ground_truth)
        self.fm.step(prediction, ground_truth)

    def get_results(self):
        mae = self.mae.get_results()["mae"]
        sm = self.smeasure.get_results()["sm"]
        wfm = self.wfm.get_results()["wfm"]
        em = self.em.get_results()["em"]
        fmr = self.fm.get_results()
        fm = fmr["fmeasure"]
        pr = fmr["precision"]
        rc = fmr["recall"]
        fc = np.asarray(fm["dynamic"], dtype=np.float64)
        pc = np.asarray(pr["dynamic"], dtype=np.float64)
        rcv = np.asarray(rc["dynamic"], dtype=np.float64)
        ec = np.asarray(em["curve"], dtype=np.float64)

        metrics = {
            "mae": float(mae),
            "smeasure": float(sm),
            "weighted_fmeasure": float(wfm),
            "max_fmeasure": float(fc.max()),
            "mean_fmeasure": float(fc.mean()),
            "adaptive_fmeasure": float(fm["adaptive"]),
            "max_emeasure": float(ec.max()),
            "mean_emeasure": float(ec.mean()),
            "adaptive_emeasure": float(em["adp"]),
        }
        curves = {
            "thresholds": np.linspace(0, 1, 256, dtype=np.float64),
            "precision": np.flip(pc).copy(),
            "recall": np.flip(rcv).copy(),
            "fmeasure": np.flip(fc).copy(),
            "emeasure": np.flip(ec).copy(),
        }
        return metrics, curves


def _collect(d):
    d = Path(d)
    files = [
        p
        for p in sorted(d.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise RuntimeError(f"No images found in {d}")
    return {p.stem: p for p in files}


def _read(p):
    with Image.open(p) as im:
        return np.array(im.convert("L"), dtype=np.uint8, copy=True)


def evaluate_prediction_directory(
    prediction_dir, ground_truth_dir, show_progress=True
):
    pm = _collect(prediction_dir)
    gm = _collect(ground_truth_dir)
    rec = SODMetricRecorder()
    names = sorted(pm)

    for n in tqdm(
        names, desc="Evaluation", unit="image", disable=not show_progress
    ):
        pred = _read(pm[n])
        gt = _read(gm[n])
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred).resize(
                    (gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
                copy=True,
            )
        rec.step(pred, gt)

    m, c = rec.get_results()
    return m, c, len(names)


def save_metric_curves(path, curves):
    np.savez_compressed(path, **curves)


def get_metric_library_version():
    return version("pysodmetrics")
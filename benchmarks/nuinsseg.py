"""Benchmark detect_nuclei's method on NuInsSeg (665 H&E images, 31 human and mouse tissues, CC BY 4.0).

Reproduces what the service does: StarDist's pretrained 2D_versatile_he model with a detection threshold of
0.45, the image resized so nuclei come out at the size the model handles best (from --mpp, the images'
microns per pixel, when given; otherwise from the median nucleus of a first pass), outlines reduced to 16
points, and the enlarged-first ranking capped at 500 nuclei. The outlines are compared with NuInsSeg's
expert label masks. --previous runs the settings before 0.5.0 (threshold 0.69, keep the largest of 1x, 2x
and 3x that finds over 15% more nuclei).

The threshold was chosen on the even-numbered images of each tissue (--split tune): the lowest that keeps
precision at 0.8 or more. Report results on the odd-numbered ones (--split test, the default).

  pip install "tensorflow==2.15.1" stardist==0.9.1 numpy==1.26.4 opencv-python-headless pillow tifffile
  # download NuInsSeg.zip from https://zenodo.org/records/10518968 and unzip it
  python nuinsseg.py /path/to/NuInsSeg [--mpp 0.1375] [--previous] [--split test|tune|all] [--out results.json]

Measures (an object counts as found when its IoU with the expert outline is at least 0.5): precision,
recall, F1, panoptic quality (PQ), the mean IoU of matched nuclei, and pixel Dice of nucleus vs background.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

MAX_SCALED_SIDE = 2400
POLYGON_POINTS = 16
MAX_NUCLEI = 500
PROB_THRESH = 0.45
MODEL_MPP = 0.25
NUCLEUS_AREA = 520


def load_model():
    from stardist.models import StarDist2D

    return StarDist2D.from_pretrained("2D_versatile_he")


def predict(model, image: np.ndarray, scale: float, prob: float | None) -> tuple[np.ndarray, np.ndarray]:
    from csbdeep.utils import normalize

    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    _, details = model.predict_instances(normalize(image, 1, 99.8, axis=(0, 1, 2)), **({"prob_thresh": prob} if prob else {}))
    coords = details["coord"]
    polygons = np.stack([coords[:, 1, :], coords[:, 0, :]], axis=2) / scale if len(coords) else np.zeros((0, 32, 2))
    return polygons, details["prob"]


def bounded(scale: float, w: int, h: int) -> float:
    return max(0.3, min(scale, 3.0, MAX_SCALED_SIDE / max(w, h)))


def area(polygon: np.ndarray) -> float:
    return abs(cv2.contourArea(polygon.astype(np.float32)))


def detect(model, image: np.ndarray, mpp: float | None = None, previous: bool = False) -> list[np.ndarray]:
    """The service's outlines for one image, as (16, 2) x/y arrays."""
    h, w = image.shape[:2]
    if previous:
        best = None
        for scale in [s for s in (1, 2, 3) if s == 1 or max(w, h) * s <= MAX_SCALED_SIDE]:
            polygons, probs = predict(model, image, scale, None)
            if best is None or len(probs) > 1.15 * len(best[1]):
                best = (polygons, probs)
        polygons, probs = best
    elif mpp:
        polygons, probs = predict(model, image, bounded(mpp / MODEL_MPP, w, h), PROB_THRESH)
    else:
        polygons, probs = predict(model, image, 1, PROB_THRESH)
        if len(polygons) >= 5:
            median = float(np.median([area(p) for p in polygons]))
            wanted = bounded(math.sqrt(NUCLEUS_AREA / median), w, h) if median else 1.0
            if abs(math.log(wanted)) > math.log(1.25):
                polygons, probs = predict(model, image, wanted, PROB_THRESH)
    areas = np.array([area(p) for p in polygons])
    median = float(np.median(areas)) if len(areas) else 0.0
    enlarged = (areas > 2.5 * median) if median else np.zeros(len(areas), bool)
    order = np.lexsort((-probs, ~enlarged))[:MAX_NUCLEI]
    step = max(1, polygons.shape[1] // POLYGON_POINTS) if len(polygons) else 1
    return [np.clip(polygons[i][::step], [0, 0], [w - 1, h - 1]) for i in order]


def rasterize(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    labels = np.zeros(shape, np.int32)
    for i, polygon in enumerate(polygons, start=1):
        cv2.fillPoly(labels, [np.round(polygon).astype(np.int32)], i)
    return labels


def images(root: Path) -> list[tuple[str, Path, Path]]:
    found = []
    for tissue in sorted(p for p in root.iterdir() if p.is_dir()):
        for image in sorted((tissue / "tissue images").glob("*.png")):
            mask = tissue / "label masks modify" / f"{image.stem}.tif"
            if mask.exists():
                found.append((tissue.name, image, mask))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="The unzipped NuInsSeg folder (one folder per tissue)")
    parser.add_argument("--limit", type=int, help="Only the first N images")
    parser.add_argument("--split", choices=("test", "tune", "all"), default="test",
                        help="test: odd-numbered images (default); tune: even-numbered, used to choose the threshold")
    parser.add_argument("--mpp", type=float, help="The images' microns per pixel (NuInsSeg: about 0.1375)")
    parser.add_argument("--previous", action="store_true", help="The settings before 0.5.0")
    parser.add_argument("--out", type=Path, default=Path("nuinsseg-results.json"))
    args = parser.parse_args()

    from stardist.matching import matching_dataset

    pairs = images(args.root)
    if args.split != "all":
        parity = 1 if args.split == "test" else 0
        pairs = [p for p in pairs if int(p[1].stem.rsplit("_", 1)[1]) % 2 == parity]
    pairs = pairs[: args.limit]
    if not pairs:
        sys.exit(f"No NuInsSeg images under {args.root}")
    model = load_model()
    truths, preds, tissues, seconds, dice = [], [], [], [], []
    for n, (tissue, image_path, mask_path) in enumerate(pairs, start=1):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        truth = tifffile.imread(mask_path).astype(np.int32)
        started = time.perf_counter()
        pred = rasterize(detect(model, image, args.mpp, args.previous), truth.shape)
        seconds.append(time.perf_counter() - started)
        truths.append(truth)
        preds.append(pred)
        tissues.append(tissue)
        t, p = truth > 0, pred > 0
        dice.append(2 * (t & p).sum() / max(1, t.sum() + p.sum()))
        if n % 50 == 0:
            print(f"{n}/{len(pairs)}", file=sys.stderr)

    def scores(idx: list[int]) -> dict[str, float] | None:
        if not idx:
            return None
        m = matching_dataset([truths[i] for i in idx], [preds[i] for i in idx], thresh=0.5, show_progress=False)
        return {
            "images": len(idx),
            "nuclei_expert": int(m.n_true),
            "nuclei_found": int(m.n_pred),
            "precision": round(m.precision, 3),
            "recall": round(m.recall, 3),
            "f1": round(m.f1, 3),
            "panoptic_quality": round(m.panoptic_quality, 3),
            "mean_matched_iou": round(m.mean_matched_score, 3),
            "pixel_dice": round(float(np.mean([dice[i] for i in idx])), 3),
        }

    everything = list(range(len(pairs)))
    result = {
        "dataset": "NuInsSeg v1.0 (https://zenodo.org/records/10518968, CC BY 4.0)",
        "method": "previous (threshold 0.69, 1x/2x/3x by count)" if args.previous else
                  f"threshold {PROB_THRESH}, " + (f"resized from {args.mpp} to {MODEL_MPP} um/px" if args.mpp else "resized by nucleus size"),
        "split": args.split,
        "iou_threshold": 0.5,
        "overall": scores(everything),
        "human": scores([i for i in everything if tissues[i].startswith("human")]),
        "mouse": scores([i for i in everything if tissues[i].startswith("mouse")]),
        "by_tissue": {t: scores([i for i in everything if tissues[i] == t]) for t in sorted(set(tissues))},
        "median_seconds_per_image": round(statistics.median(seconds), 2),
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("overall", "human", "mouse", "median_seconds_per_image")}, indent=2))


if __name__ == "__main__":
    main()

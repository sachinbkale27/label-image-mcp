"""Python port of exportCoco / exportYolo from src/services/exportUtils.ts."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def _normalize(value: float, dimension: float) -> float:
    return round(value / dimension * 10000) / 10000


def _polygon_area(points: list[dict[str, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        area += p["x"] * q["y"] - q["x"] * p["y"]
    return abs(area / 2)


def _circle_points(cx: float, cy: float, r: float, segments: int = 32) -> list[dict[str, float]]:
    return [
        {"x": cx + r * math.cos(2 * math.pi * i / segments), "y": cy + r * math.sin(2 * math.pi * i / segments)}
        for i in range(segments)
    ]


def _exportable(ann: dict[str, Any]) -> bool:
    # Like the frontend, only area annotations (polygons and circles) are exported
    if ann.get("type") == "polygon":
        return len(ann.get("points") or []) > 2
    return ann.get("type") == "circle" and bool(ann.get("radius"))


def _label_ids(annotations: list[dict[str, Any]], start: int) -> dict[str, int]:
    ids: dict[str, int] = {}
    for ann in annotations:
        ids.setdefault(ann["label"], len(ids) + start)
    return ids


def export_coco(image: dict[str, Any]) -> dict[str, Any]:
    annotations = image.get("annotations") or []
    category_ids = _label_ids(annotations, start=1)

    coco_annotations = []
    for ann in filter(_exportable, annotations):
        if ann["type"] == "circle":
            r = ann["radius"]
            points = _circle_points(ann["x"], ann["y"], r)
            area = math.pi * r * r
            bbox = [round(ann["x"] - r), round(ann["y"] - r), round(r * 2), round(r * 2)]
        else:
            points = ann["points"]
            area = _polygon_area(points)
            bbox = [round(ann["x"]), round(ann["y"]), round(ann["width"]), round(ann["height"])]

        entry: dict[str, Any] = {
            "id": len(coco_annotations) + 1,
            "image_id": 1,
            "category_id": category_ids[ann["label"]],
            "bbox": bbox,
            "area": round(area),
            "iscrowd": 0,
            "segmentation": [[c for p in points for c in (round(p["x"]), round(p["y"]))]],
        }
        attributes = {k: ann[k] for k in ("confidence", "notes") if ann.get(k)}
        if attributes:
            entry["attributes"] = attributes
        coco_annotations.append(entry)

    now = datetime.now(timezone.utc)
    return {
        "info": {
            "description": "Dataset exported from Image Label for Vision Transformer training",
            "version": "1.0",
            "year": now.year,
            "date_created": now.isoformat(),
        },
        "images": [{"id": 1, "file_name": image["filename"], "width": image["width"], "height": image["height"]}],
        "annotations": coco_annotations,
        "categories": [{"id": i, "name": name, "supercategory": "object"} for name, i in category_ids.items()],
    }


def export_yolo(image: dict[str, Any]) -> dict[str, Any]:
    annotations = image.get("annotations") or []
    class_ids = _label_ids(annotations, start=0)
    w, h = image["width"], image["height"]

    lines = []
    for ann in filter(_exportable, annotations):
        if ann["type"] == "circle":
            d = ann["radius"] * 2
            xc, yc, bw, bh = ann["x"], ann["y"], d, d
        else:
            xc, yc = ann["x"] + ann["width"] / 2, ann["y"] + ann["height"] / 2
            bw, bh = ann["width"], ann["height"]
        values = (_normalize(xc, w), _normalize(yc, h), _normalize(bw, w), _normalize(bh, h))
        lines.append(" ".join([str(class_ids[ann["label"]])] + [_fmt(v) for v in values]))

    return {"classes": list(class_ids), "annotations": "\n".join(lines)}


def _rgb(color: str | None) -> list[int]:
    value = (color or "#ff0000").lstrip("#")
    if len(value) == 3:  # short form, e.g. #0f0
        value = "".join(c * 2 for c in value)
    try:
        return [int(value[i : i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return [255, 0, 0]


def export_qupath(image: dict[str, Any]) -> dict[str, Any]:
    """GeoJSON FeatureCollection that QuPath imports (File > Import objects): polygons, circles and rectangles
    as polygons, pins as points, each classified by its label. Nuclei found by StarDist are detections,
    everything else annotations, as QuPath distinguishes them."""
    features = []
    for ann in image.get("annotations") or []:
        kind = ann.get("type")
        if kind == "pin":
            geometry: dict[str, Any] = {"type": "Point", "coordinates": [ann["x"], ann["y"]]}
        elif _exportable(ann) or (kind == "rectangle" and ann.get("width") and ann.get("height")):
            if kind == "circle":
                points = _circle_points(ann["x"], ann["y"], ann["radius"])
            elif kind == "rectangle":
                x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
                points = [{"x": x, "y": y}, {"x": x + w, "y": y}, {"x": x + w, "y": y + h}, {"x": x, "y": y + h}]
            else:
                points = ann["points"]
            ring = [[round(p["x"], 1), round(p["y"], 1)] for p in points]
            geometry = {"type": "Polygon", "coordinates": [ring + [ring[0]]]}  # GeoJSON rings are closed
        else:
            continue  # arrows have no area or position QuPath can use
        properties: dict[str, Any] = {
            "objectType": "detection" if "StarDist" in (ann.get("notes") or "") else "annotation",
            "classification": {"name": ann["label"], "color": _rgb(ann.get("color"))},
        }
        if ann.get("notes"):
            properties["name"] = ann["notes"][:200]
        features.append({"type": "Feature", "id": ann.get("id"), "geometry": geometry, "properties": properties})
    return {"type": "FeatureCollection", "features": features}


def _fmt(value: float) -> str:
    # Match JS number formatting (0.5 not 0.50, 1 not 1.0)
    return str(int(value)) if value == int(value) else repr(value)

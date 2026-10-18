#!/usr/bin/env python3
"""Merge per-image Image Label exports into one consistent dataset.

export_annotations exports one image at a time and numbers classes by first appearance in
that image, so ids differ between images. This script re-maps every image onto one global
class list.

  coco: merge_exports.py coco a.json b.json -o annotations.json
  yolo: merge_exports.py yolo scan_01.json scan_02.json -o dataset_dir
        (each YOLO file is named after its image: scan_01.png -> scan_01.json)

--classes "a,b,c" fixes the class order; labels not listed are appended in first-seen order.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(paths: list[str]) -> list[tuple[Path, dict]]:
    exports = []
    for raw in paths:
        path = Path(raw)
        try:
            exports.append((path, json.loads(path.read_text())))
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"error: cannot read {path}: {exc}")
    if not exports:
        sys.exit("error: no export files given")
    return exports


class ClassMap:
    def __init__(self, fixed: list[str]):
        self.names: list[str] = []
        for name in fixed:
            self.add(name)

    def add(self, name: str) -> int:
        if name not in self.names:
            self.names.append(name)
        return self.names.index(name)


def merge_coco(exports: list[tuple[Path, dict]], classes: ClassMap) -> dict:
    images, annotations = [], []
    for path, data in exports:
        if "images" not in data or "categories" not in data:
            sys.exit(f"error: {path} is not a COCO export (expected 'images' and 'categories')")
        names = {c["id"]: c["name"] for c in data["categories"]}
        image_id = len(images) + 1
        for image in data["images"]:  # Image Label exports one image per file
            images.append({**image, "id": image_id})
        for ann in data.get("annotations", []):
            annotations.append({
                **ann,
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": classes.add(names[ann["category_id"]]) + 1,
            })
    info = dict(exports[0][1].get("info", {}))
    info["description"] = f"Merged from {len(exports)} Image Label exports"
    return {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n, "supercategory": "object"} for i, n in enumerate(classes.names)],
    }


def merge_yolo(exports: list[tuple[Path, dict]], classes: ClassMap, out: Path) -> tuple[int, int]:
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for path, data in exports:
        if "classes" not in data or "annotations" not in data:
            sys.exit(f"error: {path} is not a YOLO export (expected 'classes' and 'annotations')")
        lines = []
        for line in filter(None, data["annotations"].splitlines()):
            local_id, *coords = line.split()
            global_id = classes.add(data["classes"][int(local_id)])
            lines.append(" ".join([str(global_id), *coords]))
        (labels_dir / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        total += len(lines)
    (out / "classes.txt").write_text("\n".join(classes.names) + "\n")
    names = "\n".join(f"  {i}: {json.dumps(n)}" for i, n in enumerate(classes.names))
    (out / "data.yaml").write_text(
        "# Put the images in images/ with the same file stems as labels/*.txt\n"
        f"path: .\ntrain: images\nval: images\nnames:\n{names}\n"
    )
    return len(exports), total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("format", choices=["coco", "yolo"])
    parser.add_argument("files", nargs="+", help="per-image export JSON files")
    parser.add_argument("-o", "--output", required=True, help="COCO: output .json file. YOLO: output directory")
    parser.add_argument("--classes", default="", help="comma-separated class order to pin")
    args = parser.parse_args()

    classes = ClassMap([c.strip() for c in args.classes.split(",") if c.strip()])
    exports = load(args.files)
    out = Path(args.output)

    if args.format == "coco":
        merged = merge_coco(exports, classes)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, indent=2))
        summary = f"{len(merged['images'])} images, {len(merged['annotations'])} annotations"
    else:
        images, annotations = merge_yolo(exports, classes, out)
        summary = f"{images} label files, {annotations} annotations"
    print(f"Wrote {out}: {summary}, {len(classes.names)} classes: {', '.join(classes.names)}")


if __name__ == "__main__":
    main()

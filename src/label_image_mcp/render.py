"""Pictures of an image for the assistant to look at: scaled down, with a pixel grid and, optionally, the
annotations drawn on it, so it can place boxes for segment_objects and check outlines."""

from __future__ import annotations

import io
import math
from collections import Counter
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

GRID_COLOR = (255, 255, 0)
LABELS_UP_TO = 40  # with more annotations (e.g. hundreds of nuclei) the outlines are drawn without names


def _rgb(color: str | None) -> tuple[int, int, int]:
    value = (color or "#ff0000").lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return (255, 0, 0)


def grid_step(span: float) -> int:
    """A round grid spacing in image pixels giving about 8 lines across the shown span."""
    raw = max(span / 8, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    return int(min((m * magnitude for m in (1, 2, 5, 10)), key=lambda step: abs(math.log(step / raw))))


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has one fixed size
        return ImageFont.load_default()


def _shape_points(ann: dict[str, Any]) -> list[tuple[float, float]]:
    kind = ann.get("type")
    if kind == "polygon":
        return [(p["x"], p["y"]) for p in ann.get("points") or []]
    if kind == "rectangle":
        x, y, w, h = ann["x"], ann["y"], ann.get("width") or 0, ann.get("height") or 0
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if kind == "circle":
        r = ann.get("radius") or 0
        return [(ann["x"] + r * math.cos(2 * math.pi * i / 32), ann["y"] + r * math.sin(2 * math.pi * i / 32)) for i in range(32)]
    return []


def render(
    content: bytes,
    annotations: list[dict[str, Any]] | None = None,
    region: tuple[float, float, float, float] | None = None,
    max_size: int = 1024,
    grid: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """A JPEG of the image (or a region x, y, width, height of it) at most max_size pixels on its longer side.

    Grid labels and everything in the returned info are in the ORIGINAL image's pixel coordinates, so the
    assistant can read positions straight off the picture.
    """
    with Image.open(io.BytesIO(content)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    full_w, full_h = image.size
    left, top, right, bottom = 0.0, 0.0, float(full_w), float(full_h)
    if region:
        x, y, w, h = region
        left, top = max(0.0, x), max(0.0, y)
        right, bottom = min(float(full_w), x + w), min(float(full_h), y + h)
        if right - left < 4 or bottom - top < 4:
            raise ValueError(f"The region is outside the image ({full_w} x {full_h} pixels)")
        image = image.crop((round(left), round(top), round(right), round(bottom)))
    scale = min(1.0, max_size / max(image.size))
    if scale < 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)

    def to_view(px: float, py: float) -> tuple[float, float]:
        return ((px - left) * scale, (py - top) * scale)

    draw = ImageDraw.Draw(image)
    width = max(1, round(max(image.size) / 500))
    font = _font(max(11, round(max(image.size) / 70)))
    drawn = annotations or []
    for ann in drawn:
        color = _rgb(ann.get("color"))
        points = [to_view(*p) for p in _shape_points(ann)]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=color, width=width)
        elif ann.get("type") in ("pin", "arrow") and ann.get("x") is not None:
            cx, cy = to_view(ann["x"], ann["y"])
            if ann.get("type") == "arrow" and ann.get("arrowTarget"):
                draw.line([(cx, cy), to_view(ann["arrowTarget"]["x"], ann["arrowTarget"]["y"])], fill=color, width=width + 1)
            r = 3 * width + 2
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=width + 1)
    if 0 < len(drawn) <= LABELS_UP_TO:
        for ann in drawn:
            points = _shape_points(ann)
            px, py = (min(points)[0], min(p[1] for p in points)) if points else (ann.get("x"), ann.get("y"))
            if px is None:
                continue
            vx, vy = to_view(px, py)
            text = str(ann.get("label", ""))[:40]
            box = draw.textbbox((vx, vy), text, font=font)
            ty = vy - (box[3] - box[1]) - 4 if vy > (box[3] - box[1]) + 4 else vy + 2
            box = draw.textbbox((vx, ty), text, font=font)
            draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(0, 0, 0))
            draw.text((vx, ty), text, fill=_rgb(ann.get("color")), font=font)

    step = grid_step(max(right - left, bottom - top)) if grid else 0
    if grid:
        small = _font(max(10, round(max(image.size) / 85)))
        for gx in range(math.ceil(left / step) * step, int(right) + 1, step):
            vx, _ = to_view(gx, top)
            draw.line([(vx, 0), (vx, image.height)], fill=GRID_COLOR, width=1)
            draw.text((vx + 2, 2), str(gx), fill=GRID_COLOR, font=small, stroke_width=2, stroke_fill=(0, 0, 0))
        for gy in range(math.ceil(top / step) * step, int(bottom) + 1, step):
            _, vy = to_view(left, gy)
            draw.line([(0, vy), (image.width, vy)], fill=GRID_COLOR, width=1)
            draw.text((2, vy + 2), str(gy), fill=GRID_COLOR, font=small, stroke_width=2, stroke_fill=(0, 0, 0))

    out = io.BytesIO()
    image.save(out, "JPEG", quality=82)
    info: dict[str, Any] = {
        "image_size": [full_w, full_h],
        "shown_region": [round(left), round(top), round(right - left), round(bottom - top)],
        "picture_size": list(image.size),
        "scale": round(scale, 4),
    }
    if grid:
        info["grid_every"] = step
    if drawn:
        counts = Counter(str(a.get("label", "")) for a in drawn)
        colors = {str(a.get("label", "")): a.get("color") for a in drawn}
        info["annotations_drawn"] = [{"label": k, "count": v, "color": colors[k]} for k, v in counts.most_common()]
    return out.getvalue(), info

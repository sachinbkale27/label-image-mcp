"""MCP server exposing the Image Label REST API as tools (stdio, or Streamable HTTP with --http)."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import re
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, model_validator

from .client import ApiError, LabelImageClient, load_secret
from .exporters import export_coco, export_dataset as build_dataset, export_qupath, export_yolo
from .render import render

DEFAULT_COLOR = "#ff0000"


class Point(BaseModel):
    x: float
    y: float


class AnnotationInput(BaseModel):
    """One annotation in image pixel coordinates (origin at top-left)."""

    type: Literal["pin", "polygon", "arrow", "circle", "rectangle"]
    label: str
    x: float | None = Field(None, description="Pin point, circle center, arrow label position or rectangle left edge")
    y: float | None = Field(None, description="Pin point, circle center, arrow label position or rectangle top edge")
    width: float | None = Field(None, description="Rectangle width")
    height: float | None = Field(None, description="Rectangle height")
    radius: float | None = Field(None, description="Circle radius")
    points: list[Point] | None = Field(None, description="Polygon vertices (3 or more)")
    arrowTarget: Point | None = Field(None, description="Point the arrow points to")
    color: str = DEFAULT_COLOR
    notes: str | None = None
    confidence: Literal["high", "medium", "low"] | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> AnnotationInput:
        if self.type == "polygon":
            if not self.points or len(self.points) < 3:
                raise ValueError("polygon needs at least 3 points")
        elif self.x is None or self.y is None:
            raise ValueError(f"{self.type} needs x and y")
        if self.type == "circle" and not self.radius:
            raise ValueError("circle needs radius")
        if self.type == "rectangle" and (not self.width or not self.height):
            raise ValueError("rectangle needs width and height")
        if self.type == "arrow" and self.arrowTarget is None:
            raise ValueError("arrow needs arrowTarget")
        return self

    def to_record(self) -> dict[str, Any]:
        record = self.model_dump(exclude_none=True)
        if self.type == "polygon":
            # The canvas stores a polygon's bounding box in x/y/width/height
            xs = [p.x for p in self.points]
            ys = [p.y for p in self.points]
            record.update(x=min(xs), y=min(ys), width=max(xs) - min(xs), height=max(ys) - min(ys))
        record["id"] = str(uuid.uuid4())
        record["createdAt"] = datetime.now(timezone.utc).isoformat()
        return record


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[LabelImageClient]:
    client = LabelImageClient()
    try:
        yield client
    finally:
        await client.aclose()


INSTRUCTIONS = """Annotate images saved in Image Label, including H&E histopathology: slide regions and
snapshots (up to 20 MB), and whole-slide files (SVS, NDPI, pyramidal TIFF, up to 5 GB).

- Look before you outline: view_image shows the image with a pixel grid, so boxes and points you give
  segment_objects are in the right place. After annotating, view_image(with_annotations=true) to check
  the outlines, and fix wrong ones with remove_annotations.
- Histology: check_image_quality first; call warm_up("nuclei") early, since the first run after a quiet
  spell takes up to a minute; then detect_nuclei. Pass microns_per_pixel (often 0.25 at 40x, 0.5 at 20x)
  when known, for sizes in square microns.
- Other objects: warm_up("sam"), or "medsam" for medical images, then segment_objects with a rough box
  per object. Prefer it to drawing polygons yourself.
- Whole slides: upload_slide (or upload_slide_from_url) starts a background analysis of sampled tissue
  tiles; analyze_slide waits for it (or restarts it), get_slide_analysis reads the results. The slide's
  image is then its overview with the analysed tiles marked, the densest tiles are saved as images of
  their own (annotate or view them in detail), and all nuclei are in a QuPath GeoJSON in slide coordinates.
- Training data: export_annotations for one image, export_dataset for a category or a list of images
  (one class list, so class ids match across images).
- Annotations are drafts for a person to review. replace_annotations and remove_annotations delete work:
  use them only when asked."""

mcp = FastMCP("histopathology-annotation", instructions=INSTRUCTIONS, lifespan=lifespan)

HOSTED = False  # set by configure_hosted (--hosted): no local files on a shared server

READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
ADD = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
DELETE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)


def _client(ctx: Context) -> LabelImageClient:
    return ctx.request_context.lifespan_context


def _summary(image: dict[str, Any]) -> dict[str, Any]:
    keys = ("imageId", "filename", "name", "category", "subCategory", "description", "width", "height", "updatedAt")
    summary = {k: image.get(k) for k in keys} | {"annotationCount": len(image.get("annotations") or [])}
    if image.get("kind") == "slide":
        analysis = image.get("slideAnalysis") or {}
        summary |= {"kind": "slide", "analysis": analysis.get("status", "not started")}
    return summary


@mcp.tool(title="List images", annotations=READ)
async def list_images(ctx: Context, category: str | None = None) -> list[dict[str, Any]]:
    """List images saved in Image Label, optionally filtered by category (case-insensitive).
    Returns summaries; use get_image for annotations and the viewing URL."""
    images = await _client(ctx).list_images()
    if category:
        images = [i for i in images if (i.get("category") or "").lower() == category.lower()]
    return [_summary(i) for i in images]


@mcp.tool(title="Get image details", annotations=READ)
async def get_image(ctx: Context, image_id: str) -> dict[str, Any]:
    """Get one image's metadata, all annotations, and a signed viewing URL (valid 24h)."""
    image = await _client(ctx).get_image(image_id)
    image.pop("downloadUrl", None)
    return image


class Region(BaseModel):
    x: float = Field(description="Left edge, in image pixels")
    y: float = Field(description="Top edge, in image pixels")
    width: float = Field(gt=0)
    height: float = Field(gt=0)


def _overlaps(ann: dict[str, Any], region: Region) -> bool:
    x, y = ann.get("x"), ann.get("y")
    if x is None or y is None:
        return False
    if ann.get("type") == "circle":
        r = ann.get("radius") or 0
        x, y, w, h = x - r, y - r, 2 * r, 2 * r
    else:
        w, h = ann.get("width") or 0, ann.get("height") or 0
    return x <= region.x + region.width and x + w >= region.x and y <= region.y + region.height and y + h >= region.y


@mcp.tool(title="View image", annotations=READ, structured_output=False)
async def view_image(
    ctx: Context,
    image_id: str,
    with_annotations: bool = False,
    region: Region | None = None,
    grid: bool = True,
    max_size: Annotated[int, Field(ge=256, le=1568)] = 1024,
):
    """Look at an image: returns the picture itself, scaled to at most max_size pixels, with a yellow grid
    labelled in the ORIGINAL image's pixel coordinates.

    Use it before segment_objects or add_annotations to read where objects are (boxes and points are in
    those coordinates), and with with_annotations=true afterwards to check the outlines. region zooms into
    part of a large image (x, y, width, height in image pixels), e.g. to see individual nuclei.
    """
    client = _client(ctx)
    record = await client.get_image(image_id)
    if record.get("kind") == "slide" and not record.get("thumbnailKey"):
        raise ValueError("This whole slide has no overview yet: analyze_slide makes one")
    image, content = await client.fetch_image(image_id, record)
    annotations = (image.get("annotations") or []) if with_annotations else []
    if region:
        annotations = [a for a in annotations if _overlaps(a, region)]
    box = (region.x, region.y, region.width, region.height) if region else None
    picture, info = render(content, annotations, box, max_size, grid)
    info = {"image_id": image_id, "name": image.get("name") or image.get("filename"), **info}
    if with_annotations:
        info["annotation_count"] = len(image.get("annotations") or [])
    return [json.dumps(info), Image(data=picture, format="jpeg")]


@mcp.tool(title="Update image details", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
async def update_image_metadata(
    ctx: Context,
    image_id: str,
    name: str | None = None,
    category: str | None = None,
    sub_category: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Update an image's name, category, sub-category or description. Omitted fields are left unchanged."""
    updates = {
        k: v
        for k, v in {"name": name, "category": category, "subCategory": sub_category, "description": description}.items()
        if v is not None
    }
    if not updates:
        raise ValueError("Provide at least one field to update")
    return _summary(await _client(ctx).update_image(image_id, updates))


@mcp.tool(title="Add annotations", annotations=ADD)
async def add_annotations(ctx: Context, image_id: str, annotations: list[AnnotationInput]) -> dict[str, Any]:
    """Append annotations to an image, keeping the existing ones."""
    client = _client(ctx)
    image = await client.get_image(image_id)
    new = [a.to_record() for a in annotations]
    result = await client.save_annotations(image_id, (image.get("annotations") or []) + new)
    return result | {"addedIds": [a["id"] for a in new]}


@mcp.tool(title="Replace all annotations", annotations=DELETE)
async def replace_annotations(ctx: Context, image_id: str, annotations: list[AnnotationInput]) -> dict[str, Any]:
    """Replace ALL annotations on an image with the given list. Pass an empty list to clear them."""
    return await _client(ctx).save_annotations(image_id, [a.to_record() for a in annotations])


@mcp.tool(title="Remove annotations", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False))
async def remove_annotations(ctx: Context, image_id: str, annotation_ids: list[str]) -> dict[str, Any]:
    """Remove specific annotations from an image by their ids."""
    client = _client(ctx)
    existing = (await client.get_image(image_id)).get("annotations") or []
    to_remove = set(annotation_ids)
    remaining = [a for a in existing if a.get("id") not in to_remove]
    result = await client.save_annotations(image_id, remaining)
    return result | {"removed": len(existing) - len(remaining)}


@mcp.tool(title="Upload a local image", annotations=ADD)
async def upload_image(
    ctx: Context,
    path: str,
    name: str | None = None,
    category: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Upload a local image file (path on the machine running this server) to Image Label."""
    fields = {k: v for k, v in {"name": name, "category": category, "description": description}.items() if v}
    return _summary(await _client(ctx).upload_image(path, **fields))


@mcp.tool(title="Upload an image from a URL", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
async def upload_image_from_url(
    ctx: Context,
    url: str,
    name: str | None = None,
    category: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Download an image from an https URL and upload it to Image Label (PNG, JPEG, WebP, GIF, BMP or
    TIFF, up to 20 MB). Use this for images that are online, e.g. ones attached in a chat."""
    client = _client(ctx)
    content, filename, content_type = await client.download_image(url)
    fields = {k: v for k, v in {"name": name, "category": category, "description": description}.items() if v}
    return _summary(await client.upload_bytes(content, filename, content_type, **fields))


class Box(BaseModel):
    x: float = Field(description="Left edge")
    y: float = Field(description="Top edge")
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class SegmentTarget(BaseModel):
    """One object to outline: a rough box around it (best), or a point on it."""

    label: str
    box: Box | None = Field(None, description="Rough box around the object, in image pixels; a loose box is fine")
    point: Point | None = Field(None, description="A point on the object, when a box is hard to give")
    color: str = DEFAULT_COLOR
    notes: str | None = None

    @model_validator(mode="after")
    def _one_prompt(self) -> SegmentTarget:
        if (self.box is None) == (self.point is None):
            raise ValueError("give either a box or a point")
        return self

    def prompt(self) -> dict[str, list[float]]:
        if self.box:
            b = self.box
            return {"box": [b.x, b.y, b.x + b.width, b.y + b.height]}
        return {"point": [self.point.x, self.point.y]}  # type: ignore[union-attr]


def _confidence(score: float) -> Literal["high", "medium", "low"]:
    return "high" if score >= 0.95 else "medium" if score >= 0.85 else "low"


@mcp.tool(title="Outline objects (Segment Anything)", annotations=ADD)
async def segment_objects(
    ctx: Context,
    image_id: str,
    objects: list[SegmentTarget],
    model: Literal["sam", "medsam"] = "sam",
    save: bool = True,
) -> dict[str, Any]:
    """Trace precise outlines with Segment Anything and add them as polygon annotations.

    Give each object a rough box (or a point on it) in image pixels; the model finds the object's exact
    edges, so the box only has to contain the object. Prefer this to drawing polygons yourself.
    model: "sam" for photos and general images; "medsam" for medical images (CT, MRI, X-ray, ultrasound,
    endoscopy, pathology), which takes boxes only. Up to 20 objects per call. With save=false the outlines
    are returned without being saved. Each result has the model's quality score (0-1).
    """
    if not 1 <= len(objects) <= 20:
        raise ValueError("Give 1 to 20 objects")
    if model == "medsam" and any(o.box is None for o in objects):
        raise ValueError("MedSAM needs a box for every object")
    client = _client(ctx)
    traced = await client.segment(image_id, [o.prompt() for o in objects], model)
    new, failed, added = [], [], []
    for target, result in zip(objects, traced["results"], strict=True):
        if "polygon" not in result:
            failed.append({"label": target.label, "reason": result.get("error", "no outline found")})
            continue
        score = result["score"]
        annotation = AnnotationInput(
            type="polygon",
            label=target.label,
            points=[Point(x=x, y=y) for x, y in result["polygon"]],
            color=target.color,
            confidence=_confidence(score),
            notes=" ".join(filter(None, [target.notes, f"Traced by {model.upper()} (score {score:.2f})."])),
        ).to_record()
        new.append(annotation)
        added.append(
            {"label": target.label, "id": annotation["id"], "score": score, "points": len(result["polygon"]),
             "area": result["area"], "bbox": result["bbox"]}
        )
    summary: dict[str, Any] = {"model": model, "added": added, "failed": failed}
    if save and new:
        image = await client.get_image(image_id)
        saved = await client.save_annotations(image_id, (image.get("annotations") or []) + new)
        summary["annotationCount"] = saved.get("annotationCount")
    elif not save:
        summary["outlines"] = [{"label": a["label"], "points": a["points"]} for a in new]
    return summary


NUCLEUS_COLOR = "#00b8d4"


def in_microns(stats: dict[str, Any], width: int | None, height: int | None, mpp: float) -> dict[str, Any]:
    """The pixel measurements in physical units, for a scanner resolution of mpp microns per pixel."""
    out: dict[str, Any] = {"microns_per_pixel": mpp, "median_area_um2": round(stats["median_area"] * mpp**2, 1)}
    if width and height:
        out["per_mm2"] = round(stats["count"] / (width * height * mpp**2 / 1e6), 1)
    return out
ENLARGED_COLOR = "#ff1744"


def _nucleus_confidence(score: float) -> Literal["high", "medium", "low"]:
    # StarDist only keeps candidates above about 0.69
    return "high" if score >= 0.85 else "medium" if score >= 0.75 else "low"


@mcp.tool(title="Detect nuclei (StarDist)", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
async def detect_nuclei(
    ctx: Context,
    image_id: str,
    max_nuclei: Annotated[int, Field(ge=1, le=400)] = 300,
    mark_enlarged: bool = True,
    replace_previous: bool = True,
    save: bool = True,
    microns_per_pixel: Annotated[float | None, Field(gt=0.01, le=10)] = None,
) -> dict[str, Any]:
    """Find and outline every cell nucleus in an H&E-stained histology image with StarDist, automatically.

    No boxes needed: the model finds the nuclei itself (it is trained on H&E; other stains and non-histology
    images give poor results). Returns the count, density, median nuclear area and how much nuclear size
    varies (a rough measure of pleomorphism). With mark_enlarged, nuclei over 2.5x the median area are
    labelled "enlarged nucleus" (in red), the rest "nucleus". The most confident max_nuclei outlines are
    saved; replace_previous removes nuclei saved by an earlier run first. With microns_per_pixel (the
    scanner's resolution: often 0.25 at 40x, 0.5 at 20x) sizes are also given in square microns and density
    per square millimetre. The first run after a quiet spell takes up to a minute while the model starts
    (warm_up("nuclei") avoids the wait).
    """
    client = _client(ctx)
    found = await client.pathology(image_id, "nuclei", max_nuclei=max_nuclei)
    new = []
    for nucleus in found["nuclei"]:
        enlarged = mark_enlarged and nucleus["enlarged"]
        new.append(
            AnnotationInput(
                type="polygon",
                label="enlarged nucleus" if enlarged else "nucleus",
                points=[Point(x=x, y=y) for x, y in nucleus["polygon"]],
                color=ENLARGED_COLOR if enlarged else NUCLEUS_COLOR,
                confidence=_nucleus_confidence(nucleus["score"]),
                notes=f"Detected by StarDist (score {nucleus['score']:.2f}, area {nucleus['area']} px"
                + (f", {nucleus['area'] * microns_per_pixel ** 2:.1f} \u00b5m\u00b2" if microns_per_pixel else "") + ").",
            ).to_record()
        )
    stats = dict(found["summary"])
    if microns_per_pixel:
        stats |= in_microns(stats, found.get("width"), found.get("height"), microns_per_pixel)
    summary: dict[str, Any] = {
        "summary": stats,
        "scale": found.get("scale"),
        "labels": {
            "nucleus": sum(1 for a in new if a["label"] == "nucleus"),
            "enlarged nucleus": sum(1 for a in new if a["label"] == "enlarged nucleus"),
        },
    }
    if save and new:
        image = await client.get_image(image_id)
        kept = [
            a for a in image.get("annotations") or []
            if not (replace_previous and "StarDist" in (a.get("notes") or ""))
        ]
        saved = await client.save_annotations(image_id, kept + new)
        summary["annotationCount"] = saved.get("annotationCount")
        summary["removedPrevious"] = len(image.get("annotations") or []) - len(kept)
    elif not save:
        summary["outlines"] = [{"label": a["label"], "points": a["points"]} for a in new[:50]]
    return summary


@mcp.tool(title="Normalize H&E stain", annotations=ADD)
async def normalize_stain(ctx: Context, image_id: str) -> dict[str, Any]:
    """Make an H&E image's colours standard (Macenko stain normalization), saved as a NEW image.

    Slides from different labs and scanners differ in pink and purple tones; normalizing them makes images
    comparable and helps models such as detect_nuclei. The original is unchanged. Returns the new image.
    """
    result = await _client(ctx).pathology(image_id, "normalize")
    return {"original": image_id, "normalized": _summary(result["normalized"])}


@mcp.tool(title="Check slide quality", annotations=READ)
async def check_image_quality(ctx: Context, image_id: str) -> dict[str, Any]:
    """Check a histology image's quality before analysing it (HistoQC-style checks).

    Measures how much of the image is tissue, blur, marker-pen ink, over- and under-exposure and faint
    staining. Returns a verdict (good, check or poor), the measurements and plain-language warnings.
    """
    result = await _client(ctx).pathology(image_id, "quality")
    return {k: result[k] for k in ("verdict", "warnings", "metrics", "width", "height")}


@mcp.tool(title="Warm up a model", annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False))
async def warm_up(ctx: Context, model: Literal["sam", "medsam", "nuclei"] = "sam") -> dict[str, Any]:
    """Start loading a model ahead of use, without waiting for it: "sam" or "medsam" for segment_objects,
    "nuclei" for detect_nuclei.

    After a quiet spell a model takes a minute or two to start; call this as soon as you know you will
    need it (for example right after uploading), and do other work meanwhile.
    """
    client = _client(ctx)
    if model == "nuclei":
        # The API starts the model before it looks up an image, so any id will do
        return await client.pathology("warmup", "warmup")
    return await client.warm_segmenter(model)


SLIDE_POLL_SECONDS = 5


def _slide_report(image: dict[str, Any]) -> dict[str, Any]:
    analysis = image.get("slideAnalysis") or {"status": "not started"}
    report: dict[str, Any] = {"imageId": image.get("imageId"), "name": image.get("name") or image.get("filename"), **analysis}
    if analysis.get("status") == "done":
        report["slide"] = image.get("slide")
        report["overview"] = {"width": image.get("width"), "height": image.get("height"),
                              "note": "view_image shows the overview with the analysed tiles; densest tiles are red"}
        report["nuclei_geojson_url"] = image.get("nucleiUrl")
    if image.get("slideUrl"):
        report["slide_url"] = image["slideUrl"]
    return report


@mcp.tool(title="Analyse a whole slide", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
async def analyze_slide(
    ctx: Context,
    image_id: str,
    max_tiles: Annotated[int, Field(ge=1, le=48)] = 16,
    restart: bool = False,
    wait_seconds: Annotated[int, Field(ge=0, le=240)] = 50,
) -> dict[str, Any]:
    """Analyse a whole-slide image in the background and wait up to wait_seconds for the result.

    Finds the tissue, samples up to max_tiles tiles of it (512 x 512 um each at 20x) and runs the quality
    check and StarDist nuclei on each: nuclei per mm2, nuclear size in um2, size variation and enlarged nuclei
    across the slide, plus per-tile values. Takes about 10 seconds per tile, plus a minute when the model
    starts. If it's still running when the wait ends, call get_slide_analysis later. An analysis already
    running is waited for; restart=true starts a new one (e.g. with more tiles).
    """
    client = _client(ctx)
    image = await client.get_image(image_id)
    status = (image.get("slideAnalysis") or {}).get("status")
    if restart or status not in ("queued", "running", "done"):
        try:
            await client.start_slide_analysis(image_id, max_tiles)
        except ApiError as exc:
            if exc.status != 409:  # 409: already running, so wait for it
                raise
    elif status == "done":
        return _slide_report(image)
    waited = 0
    while waited < wait_seconds:
        await asyncio.sleep(min(SLIDE_POLL_SECONDS, wait_seconds - waited))
        waited += SLIDE_POLL_SECONDS
        image = await client.get_image(image_id)
        analysis = image.get("slideAnalysis") or {}
        if analysis.get("status") in ("done", "failed"):
            break
        step = analysis.get("step", "")
        if m := re.search(r"tile (\d+) of (\d+)", step):
            await ctx.report_progress(int(m.group(1)), int(m.group(2)), step)
    return _slide_report(image)


@mcp.tool(title="Get a slide's analysis", annotations=READ)
async def get_slide_analysis(ctx: Context, image_id: str) -> dict[str, Any]:
    """A whole slide's analysis: its status (queued, running with the current step, done or failed), and when done
    the slide-level summary, each analysed tile (position, nuclei per mm2, enlarged nuclei, quality), the images
    made from the densest tiles, and a link to every nucleus as QuPath GeoJSON."""
    return _slide_report(await _client(ctx).get_image(image_id))


@mcp.tool(title="Upload a whole slide", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
async def upload_slide(
    ctx: Context,
    path: str,
    name: str | None = None,
    category: str | None = None,
    analyze: bool = True,
    max_tiles: Annotated[int, Field(ge=1, le=48)] = 16,
) -> dict[str, Any]:
    """Upload a whole-slide file (.svs, .ndpi, pyramidal .tif/.tiff, up to 5 GB) from the machine running this
    server, and start its analysis in the background (see analyze_slide)."""
    fields = {k: v for k, v in {"name": name, "category": category}.items() if v}
    client = _client(ctx)
    image = await client.upload_slide(path, **fields)
    if analyze:
        await client.start_slide_analysis(image["imageId"], max_tiles)
    return _summary(image) | {"analysis": "started" if analyze else "not started"}


@mcp.tool(title="Upload a whole slide from a URL", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
async def upload_slide_from_url(
    ctx: Context,
    url: str,
    name: str | None = None,
    category: str | None = None,
    analyze: bool = True,
    max_tiles: Annotated[int, Field(ge=1, le=48)] = 16,
) -> dict[str, Any]:
    """Copy a whole-slide file (.svs, .ndpi, pyramidal .tif/.tiff, up to 5 GB) from an https URL into Image Label,
    and start its analysis in the background (see analyze_slide)."""
    fields = {k: v for k, v in {"name": name, "category": category}.items() if v}
    client = _client(ctx)
    image = await client.upload_slide_from_url(url, **fields)
    if analyze:
        await client.start_slide_analysis(image["imageId"], max_tiles)
    return _summary(image) | {"analysis": "started" if analyze else "not started"}


@mcp.tool(title="Export annotations", annotations=READ)
async def export_annotations(
    ctx: Context, image_id: str, format: Literal["coco", "yolo", "qupath"] = "coco"
) -> dict[str, Any]:
    """Export an image's annotations: COCO JSON or YOLO (polygons and circles, as the web app exports), or
    QuPath GeoJSON (File > Import objects in QuPath; nuclei found by StarDist come in as detections)."""
    image = await _client(ctx).get_image(image_id)
    if image.get("kind") == "slide" and (image.get("slide") or {}).get("thumbnailScale"):
        return _slide_export(image, format)
    if format == "qupath":
        return export_qupath(image)
    return export_coco(image) if format == "coco" else export_yolo(image)


def _slide_export(image: dict[str, Any], format: str) -> dict[str, Any]:
    """A whole slide's own annotations are drawn on its overview: QuPath gets them in slide coordinates (with the
    link to every nucleus), COCO and YOLO describe the overview image."""
    if format != "qupath":
        return (export_coco(image) if format == "coco" else export_yolo(image)) | {
            "note": "These annotations are on the slide's overview image; for nuclei use format qupath or the densest-tile images"
        }
    factor = 1 / float(image["slide"]["thumbnailScale"])
    scaled = []
    for a in image.get("annotations") or []:
        b = dict(a)
        for k in ("x", "y", "width", "height", "radius"):
            if b.get(k) is not None:
                b[k] = round(float(b[k]) * factor, 1)
        if b.get("points"):
            b["points"] = [{"x": round(float(p["x"]) * factor, 1), "y": round(float(p["y"]) * factor, 1)} for p in b["points"]]
        scaled.append(b)
    return export_qupath({**image, "annotations": scaled}) | {
        "nuclei_geojson_url": image.get("nucleiUrl"),
        "note": "Regions are in slide (level 0) coordinates. Every detected nucleus is in the GeoJSON at nuclei_geojson_url "
        "(link valid 24 h): in QuPath open the slide, then File > Import objects for each file.",
    }


MAX_DATASET_IMAGES = 100
MAX_INLINE_BYTES = 1_500_000


@mcp.tool(title="Export a dataset", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
async def export_dataset(
    ctx: Context,
    format: Literal["coco", "yolo", "qupath"] = "coco",
    category: str | None = None,
    image_ids: list[str] | None = None,
    classes: list[str] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Export several images as ONE training dataset, with one class list so ids mean the same everywhere.

    Choose the images by category (case-insensitive) or image_ids (up to 100). coco: one COCO file for all
    images; yolo: classes plus a label file per image, named after it; qupath: a GeoJSON file per image.
    classes fixes the class order; other labels follow in first-seen order. With output_dir (a folder on
    the machine running this server; not available when hosted) the files are written there and only a
    summary is returned, which suits large datasets such as nuclei.
    """
    client = _client(ctx)
    if image_ids:
        chosen = list(dict.fromkeys(image_ids))
    elif category:
        chosen = [i["imageId"] for i in await client.list_images() if (i.get("category") or "").lower() == category.lower()]
    else:
        raise ValueError("Give a category or image_ids")
    if not chosen:
        raise ValueError(f"No images in category {category!r}")
    if len(chosen) > MAX_DATASET_IMAGES:
        raise ValueError(f"{len(chosen)} images; export at most {MAX_DATASET_IMAGES} at a time (pass image_ids)")
    images = [await client.get_image(i) for i in chosen]
    dataset = build_dataset(images, format, classes)
    summary: dict[str, Any] = {
        "format": format,
        "images": len(images),
        "annotations": sum(len(i.get("annotations") or []) for i in images),
    }
    if "classes" in dataset or "categories" in dataset:
        summary["classes"] = dataset.get("classes") or [c["name"] for c in dataset["categories"]]
    if output_dir:
        if HOSTED:
            raise ValueError("output_dir is not available on a hosted server; leave it out to get the files")
        files = dataset.get("files") or {"annotations.json": dataset}
        if format == "yolo":
            files = {f"labels/{k}": v for k, v in files.items()} | {"classes.txt": "\n".join(dataset["classes"]) + "\n"}
        root = os.path.abspath(os.path.expanduser(output_dir))
        for name, body in files.items():
            path = os.path.join(root, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(body if isinstance(body, str) else json.dumps(body))
        return summary | {"written_to": root, "files": sorted(files)}
    if len(json.dumps(dataset)) > MAX_INLINE_BYTES:
        raise ValueError(
            "This dataset is too large to return in a message; pass output_dir to write it to files, "
            "or export fewer images"
        )
    return summary | {"dataset": dataset}


# ---------------------------------------------------------------- prompts


@mcp.prompt(title="Analyse a histology slide")
def analyse_slide(image: str, microns_per_pixel: str = "") -> str:
    """Quality check, nuclei and a look at the result, for one H&E image or whole slide (an id, a URL or a path)."""
    mpp = f" with microns_per_pixel={microns_per_pixel}" if microns_per_pixel else ""
    return (
        f"Analyse this H&E histology image: {image}\n"
        "If it is a whole-slide file (.svs, .ndpi, pyramidal .tif): upload it with upload_slide or "
        "upload_slide_from_url, wait with analyze_slide, then view_image the overview, summarise the slide-level "
        "and per-tile results, and offer the densest-tile images and the QuPath GeoJSON. Otherwise:\n"
        "1. If it is a URL, upload it with upload_image_from_url. Call warm_up('nuclei') straight away.\n"
        "2. check_image_quality, and tell me about any warnings before going on.\n"
        f"3. detect_nuclei{mpp}.\n"
        "4. view_image with with_annotations=true and describe what you see: where the nuclei are crowded, "
        "and where the enlarged ones are.\n"
        "5. Summarise the count, density, size variation and enlarged nuclei in plain language. These are "
        "measurements for research, not a diagnosis. Offer QuPath, COCO or YOLO exports."
    )


@mcp.prompt(title="Outline objects in an image")
def outline_objects(image: str, objects: str) -> str:
    """Outline named objects with Segment Anything, checking the result by looking at it."""
    return (
        f"Outline these objects: {objects}\nin this image: {image}\n"
        "1. If it is a URL, upload it with upload_image_from_url. Call warm_up('sam') (or 'medsam' for a "
        "medical image).\n"
        "2. view_image to find each object, and read a rough box around it off the grid.\n"
        "3. segment_objects with those boxes.\n"
        "4. view_image with with_annotations=true; remove and redo any outline on the wrong object.\n"
        "5. Tell me what was outlined and how confident each outline is."
    )


@mcp.prompt(title="Build a training dataset")
def build_training_dataset(category: str, format: str = "coco") -> str:
    """Check a category's annotations and export them as one dataset."""
    return (
        f"Build a {format} training dataset from the images in category '{category}'.\n"
        "1. list_images for the category, and tell me how many images and annotations it has, and which "
        "images have none.\n"
        "2. Look for inconsistent labels (e.g. 'Nucleus' and 'nuclei'); if any, propose a fix and wait for me.\n"
        f"3. export_dataset with format='{format}', and summarise the classes and counts."
    )


class BearerAuth:
    """ASGI middleware for hosted servers: every request needs `Authorization: Bearer <token>`.

    The token comes from MCP_AUTH_TOKEN, or the `mcp_token` field of LABEL_CREDENTIALS_SECRET (read
    on first use, so the secret can be created after the server is deployed). With no token
    configured, every request is refused: a hosted server never runs open.
    """

    def __init__(self, app):
        self.app = app
        self.token: str | None = os.environ.get("MCP_AUTH_TOKEN")

    def _expected(self) -> str | None:
        if not self.token and (secret_id := os.environ.get("LABEL_CREDENTIALS_SECRET")):
            try:
                self.token = load_secret(secret_id).get("mcp_token")
            except Exception:  # noqa: BLE001 - secret not created yet: stay closed, retry next request
                return None
        return self.token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        expected = self._expected()
        given = dict(scope["headers"]).get(b"authorization", b"").decode()
        if not expected:
            return await _deny(send, 503, "Server is not configured: no access token")
        if not hmac.compare_digest(given.encode(), f"Bearer {expected}".encode()):
            return await _deny(send, 401, "Missing or invalid bearer token")
        return await self.app(scope, receive, send)


async def _deny(send, status: int, message: str) -> None:
    body = json.dumps({"error": message}).encode()
    await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": body})


def configure_hosted() -> None:
    """A hosted server has no local files of its users: upload_image would read the server's own disk."""
    global HOSTED
    HOSTED = True
    mcp.remove_tool("upload_image")
    mcp.remove_tool("upload_slide")


def main() -> None:
    parser = argparse.ArgumentParser(description="Histopathology Annotation MCP server")
    parser.add_argument("--http", action="store_true", help="Serve Streamable HTTP at http://HOST:PORT/mcp instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--hosted",
        action="store_true",
        help="For remote hosting (e.g. AWS Lambda): require a bearer token, keep no session state, answer with JSON",
    )
    args = parser.parse_args()

    if not args.http:
        mcp.run()
        return
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    if not args.hosted:
        mcp.run(transport="streamable-http")
        return

    import uvicorn
    from mcp.server.transport_security import TransportSecuritySettings

    configure_hosted()
    mcp.settings.stateless_http = True
    mcp.settings.json_response = True
    # DNS-rebinding protection guards local servers from web pages; a browser page can't send the
    # bearer token, so behind BearerAuth it isn't needed (and the public host isn't known in advance).
    mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    uvicorn.run(BearerAuth(mcp.streamable_http_app()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

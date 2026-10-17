"""MCP server exposing the Image Label REST API as tools (stdio, or Streamable HTTP with --http)."""

from __future__ import annotations

import argparse
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field, model_validator

from .client import LabelImageClient
from .exporters import export_coco, export_yolo

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


mcp = FastMCP("label-image", lifespan=lifespan)


def _client(ctx: Context) -> LabelImageClient:
    return ctx.request_context.lifespan_context


def _summary(image: dict[str, Any]) -> dict[str, Any]:
    keys = ("imageId", "filename", "name", "category", "subCategory", "description", "width", "height", "updatedAt")
    return {k: image.get(k) for k in keys} | {"annotationCount": len(image.get("annotations") or [])}


@mcp.tool()
async def list_images(ctx: Context, category: str | None = None) -> list[dict[str, Any]]:
    """List images saved in Image Label, optionally filtered by category (case-insensitive).
    Returns summaries; use get_image for annotations and the viewing URL."""
    images = await _client(ctx).list_images()
    if category:
        images = [i for i in images if (i.get("category") or "").lower() == category.lower()]
    return [_summary(i) for i in images]


@mcp.tool()
async def get_image(ctx: Context, image_id: str) -> dict[str, Any]:
    """Get one image's metadata, all annotations, and a signed viewing URL (valid 24h)."""
    image = await _client(ctx).get_image(image_id)
    image.pop("downloadUrl", None)
    return image


@mcp.tool()
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


@mcp.tool()
async def add_annotations(ctx: Context, image_id: str, annotations: list[AnnotationInput]) -> dict[str, Any]:
    """Append annotations to an image, keeping the existing ones."""
    client = _client(ctx)
    image = await client.get_image(image_id)
    new = [a.to_record() for a in annotations]
    result = await client.save_annotations(image_id, (image.get("annotations") or []) + new)
    return result | {"addedIds": [a["id"] for a in new]}


@mcp.tool()
async def replace_annotations(ctx: Context, image_id: str, annotations: list[AnnotationInput]) -> dict[str, Any]:
    """Replace ALL annotations on an image with the given list. Pass an empty list to clear them."""
    return await _client(ctx).save_annotations(image_id, [a.to_record() for a in annotations])


@mcp.tool()
async def remove_annotations(ctx: Context, image_id: str, annotation_ids: list[str]) -> dict[str, Any]:
    """Remove specific annotations from an image by their ids."""
    client = _client(ctx)
    existing = (await client.get_image(image_id)).get("annotations") or []
    to_remove = set(annotation_ids)
    remaining = [a for a in existing if a.get("id") not in to_remove]
    result = await client.save_annotations(image_id, remaining)
    return result | {"removed": len(existing) - len(remaining)}


@mcp.tool()
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


@mcp.tool()
async def export_annotations(ctx: Context, image_id: str, format: Literal["coco", "yolo"] = "coco") -> dict[str, Any]:
    """Export an image's polygon and circle annotations as COCO JSON or YOLO, same as the web app's export."""
    image = await _client(ctx).get_image(image_id)
    return export_coco(image) if format == "coco" else export_yolo(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Label MCP server")
    parser.add_argument("--http", action="store_true", help="Serve Streamable HTTP at http://HOST:PORT/mcp instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()

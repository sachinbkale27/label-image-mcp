"""MCP server exposing the Image Label REST API as tools (stdio, or Streamable HTTP with --http)."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field, model_validator

from .client import LabelImageClient, load_secret
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


@mcp.tool()
async def export_annotations(ctx: Context, image_id: str, format: Literal["coco", "yolo"] = "coco") -> dict[str, Any]:
    """Export an image's polygon and circle annotations as COCO JSON or YOLO, same as the web app's export."""
    image = await _client(ctx).get_image(image_id)
    return export_coco(image) if format == "coco" else export_yolo(image)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Label MCP server")
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

    mcp.settings.stateless_http = True
    mcp.settings.json_response = True
    # DNS-rebinding protection guards local servers from web pages; a browser page can't send the
    # bearer token, so behind BearerAuth it isn't needed (and the public host isn't known in advance).
    mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    uvicorn.run(BearerAuth(mcp.streamable_http_app()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

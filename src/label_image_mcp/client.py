"""Async HTTP client for the Image Label REST API (backend/template.yaml)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import mimetypes
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from PIL import Image

DEFAULT_API_URL = "https://labels.sachin-kale.com/api"
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp", "image/tiff"}


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@lru_cache
def load_secret(secret_id: str) -> dict[str, str]:
    """A JSON secret from AWS Secrets Manager (used when hosted: LABEL_CREDENTIALS_SECRET).
    Cached per process; a failed read (secret not created yet) is retried on the next call."""
    import boto3  # only needed on AWS, where the Lambda runtime provides it

    return json.loads(boto3.client("secretsmanager").get_secret_value(SecretId=secret_id)["SecretString"])


class LabelImageClient:
    def __init__(
        self,
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LABEL_API_URL") or DEFAULT_API_URL).rstrip("/")
        self.username = username if username is not None else os.environ.get("LABEL_USERNAME")
        self.password = password if password is not None else os.environ.get("LABEL_PASSWORD")
        self._token: str | None = None
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=35.0, transport=transport)  # the API itself gives up at 30 s
        # S3 presigned uploads go to a different host, so use a separate client
        self._s3 = httpx.AsyncClient(timeout=120.0, transport=transport)

    async def aclose(self) -> None:
        await self._http.aclose()
        await self._s3.aclose()

    def _load_credentials(self) -> None:
        # Hosted servers keep the account in Secrets Manager instead of environment variables
        secret_id = os.environ.get("LABEL_CREDENTIALS_SECRET")
        if secret_id and not (self.username and self.password):
            secret = load_secret(secret_id)
            self.username, self.password = secret.get("username"), secret.get("password")

    async def _ensure_token(self) -> None:
        # Without credentials, requests are anonymous and see no images: the API only
        # returns, changes or accepts images for the logged-in account.
        if self._token:
            return
        self._load_credentials()
        if not (self.username and self.password):
            return
        data = await self._request(
            "POST", "/admin/login", json={"username": self.username, "password": self.password}, auth=False
        )
        self._token = data["token"]

    async def _request(self, method: str, path: str, *, auth: bool = True, **kwargs: Any) -> Any:
        if auth:
            await self._ensure_token()
        headers = {"Authorization": f"Bearer {self._token}"} if auth and self._token else {}
        resp = await self._http.request(method, path, headers=headers, **kwargs)
        if resp.status_code == 401 and auth and self._token:
            # Sessions expire (12 hours): log in again once and retry.
            self._token = None
            await self._ensure_token()
            headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
            resp = await self._http.request(method, path, headers=headers, **kwargs)
        try:
            body = resp.json()
        except ValueError:
            raise ApiError(f"{method} {path} returned HTTP {resp.status_code} with a non-JSON body", resp.status_code)
        if resp.status_code >= 400 or not body.get("success", False):
            error = body.get("error") or body.get("message") or "unknown error"
            raise ApiError(f"{method} {path} failed (HTTP {resp.status_code}): {error}", resp.status_code)
        return body.get("data")

    async def list_images(self) -> list[dict[str, Any]]:
        return await self._request("GET", "/images") or []

    async def get_image(self, image_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/images/{image_id}")

    async def update_image(self, image_id: str, updates: dict[str, str]) -> dict[str, Any]:
        return await self._request("PATCH", f"/images/{image_id}", json=updates)

    async def segment(
        self, image_id: str, prompts: list[dict[str, list[float]]], model: str = "sam", *, retry_delay: float = 10.0
    ) -> dict[str, Any]:
        """Trace outlines for box or point prompts with Segment Anything. The model can take longer than
        the API's 30 seconds to start after a quiet spell (503/504), so that is retried twice."""
        for attempt in range(3):
            try:
                return await self._request(
                    "POST", f"/images/{image_id}/segment", json={"model": model, "prompts": prompts}
                )
            except ApiError as exc:
                if exc.status not in (503, 504) or attempt == 2:
                    raise
                await asyncio.sleep(retry_delay)
        raise AssertionError("unreachable")

    async def save_annotations(self, image_id: str, annotations: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._request("PUT", f"/images/{image_id}/annotations", json={"annotations": annotations})

    async def upload_image(self, path: str | Path, **fields: str) -> dict[str, Any]:
        """Upload a local file."""
        file_path = Path(path).expanduser()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        return await self.upload_bytes(file_path.read_bytes(), file_path.name, content_type, **fields)

    async def download_image(self, url: str) -> tuple[bytes, str, str]:
        """Fetch an image to upload: (content, filename, content type).

        Only https URLs, or hosts listed in LABEL_UPLOAD_URL_HOSTS when that is set (which also
        allows http for them, e.g. localhost). Redirects are not followed, so an allowed host
        can't bounce the request elsewhere.
        """
        parsed = urlparse(url)
        allowed = [h.strip().lower() for h in os.environ.get("LABEL_UPLOAD_URL_HOSTS", "").split(",") if h.strip()]
        host = (parsed.hostname or "").lower()
        if allowed and host not in allowed:
            raise ApiError(f"Downloads are only allowed from: {', '.join(allowed)}")
        if parsed.scheme != "https" and not (allowed and parsed.scheme == "http"):
            raise ApiError("Image URLs must use https")
        async with self._s3.stream("GET", url, follow_redirects=False) as resp:
            if resp.status_code != 200:
                raise ApiError(f"Downloading the image failed (HTTP {resp.status_code})")
            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type not in IMAGE_TYPES:
                raise ApiError(f"Not an image: {content_type or 'unknown content type'}")
            chunks, size = [], 0
            async for chunk in resp.aiter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ApiError(f"Images larger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB are not accepted")
                chunks.append(chunk)
        filename = Path(parsed.path).name or "image"
        return b"".join(chunks), filename, content_type

    async def upload_bytes(self, content: bytes, filename: str, content_type: str, **fields: str) -> dict[str, Any]:
        """Same three-step flow as the frontend: presigned URL -> S3 PUT -> confirm."""
        with Image.open(io.BytesIO(content)) as img:
            width, height = img.size

        upload = await self._request(
            "POST", "/images/upload-url", json={"filename": filename, "contentType": content_type}
        )
        put = await self._s3.put(upload["uploadUrl"], content=content, headers={"Content-Type": content_type})
        if put.status_code >= 400:
            raise ApiError(f"S3 upload failed (HTTP {put.status_code})")

        return await self._request(
            "POST",
            f"/images/{upload['imageId']}/confirm",
            json={
                "width": width,
                "height": height,
                "filename": filename,
                "fileSize": len(content),
                "contentType": content_type,
                "imageHash": hashlib.sha256(content).hexdigest(),
                **fields,
            },
        )

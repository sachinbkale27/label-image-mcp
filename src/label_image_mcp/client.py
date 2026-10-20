"""Async HTTP client for the Image Label REST API (backend/template.yaml)."""

from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

DEFAULT_API_URL = "https://labels.sachin-kale.com/api"


class ApiError(RuntimeError):
    pass


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
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=30.0, transport=transport)
        # S3 presigned uploads go to a different host, so use a separate client
        self._s3 = httpx.AsyncClient(timeout=120.0, transport=transport)

    async def aclose(self) -> None:
        await self._http.aclose()
        await self._s3.aclose()

    async def _ensure_token(self) -> None:
        # Without credentials, requests are anonymous and see no images: the API only
        # returns, changes or accepts images for the logged-in account.
        if self._token or not (self.username and self.password):
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
            raise ApiError(f"{method} {path} returned HTTP {resp.status_code} with a non-JSON body")
        if resp.status_code >= 400 or not body.get("success", False):
            raise ApiError(f"{method} {path} failed (HTTP {resp.status_code}): {body.get('error', 'unknown error')}")
        return body.get("data")

    async def list_images(self) -> list[dict[str, Any]]:
        return await self._request("GET", "/images") or []

    async def get_image(self, image_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/images/{image_id}")

    async def update_image(self, image_id: str, updates: dict[str, str]) -> dict[str, Any]:
        return await self._request("PATCH", f"/images/{image_id}", json=updates)

    async def save_annotations(self, image_id: str, annotations: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._request("PUT", f"/images/{image_id}/annotations", json={"annotations": annotations})

    async def upload_image(self, path: str | Path, **fields: str) -> dict[str, Any]:
        """Same three-step flow as the frontend: presigned URL -> S3 PUT -> confirm."""
        file_path = Path(path).expanduser()
        content = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with Image.open(file_path) as img:
            width, height = img.size

        upload = await self._request(
            "POST", "/images/upload-url", json={"filename": file_path.name, "contentType": content_type}
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
                "filename": file_path.name,
                "fileSize": len(content),
                "contentType": content_type,
                "imageHash": hashlib.sha256(content).hexdigest(),
                **fields,
            },
        )

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


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("label-image-mcp")
    except PackageNotFoundError:
        # Run from source (the hosted server): the version in pyproject.toml
        import re

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        match = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.M) if pyproject.exists() else None
        return match.group(1) if match else "dev"


VERSION = _version()
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_SLIDE_BYTES = 5 * 1024**3  # one S3 upload
SLIDE_EXTENSIONS = (".svs", ".ndpi", ".tif", ".tiff", ".scn", ".bif")
CHUNK = 8 * 1024 * 1024
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
        # The AI client calling this server (from the MCP handshake), sent with each API call so usage can be
        # counted by client and version; nothing else about the conversation is sent
        self.client_name = "unknown"
        # On the hosted server only: where its caller's visitor is (x-viewer-* headers: city, country, time zone
        # and a daily anonymous visitor id), passed on so usage counts that person, not the hosting server
        self.viewer: dict[str, str] = {}
        self.guest: dict[str, Any] | None = None  # set when working as a guest (no account)
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
        # With an account, log in. Without one, use a guest identity (no sign-up: limited, kept 7 days), saved
        # locally so the same images are there next time. LABEL_GUEST=0 turns guests off.
        if self._token:
            return
        self._load_credentials()
        if self.username and self.password:
            data = await self._request(
                "POST", "/admin/login", json={"username": self.username, "password": self.password}, auth=False
            )
            self._token = data["token"]
            return
        if self.guest_allowed():
            self._token = await self._guest_token()

    @staticmethod
    def guest_allowed() -> bool:
        hosted = bool(os.environ.get("LABEL_CREDENTIALS_SECRET"))
        return not hosted and os.environ.get("LABEL_GUEST", "1").lower() not in ("0", "false", "no")

    def _guest_file(self) -> Path:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        return base / "label-image-mcp" / f"guest-{hashlib.sha256(self.base_url.encode()).hexdigest()[:8]}.json"

    async def _guest_token(self, renew: bool = False) -> str | None:
        """The saved guest token while it is valid (with a day to spare), otherwise a new guest."""
        import time

        path = self._guest_file()
        if not renew and path.exists():
            try:
                saved = json.loads(path.read_text())
                if saved.get("expires", 0) > time.time() + 86400:
                    self.guest = saved
                    return saved["token"]
            except (OSError, ValueError, KeyError):
                pass
        try:
            data = await self._request("POST", "/guest", auth=False)
        except ApiError:
            return None  # no guests (e.g. too many from this address today): anonymous
        from datetime import datetime

        expires = datetime.fromisoformat(data["expiresAt"].replace("Z", "+00:00")).timestamp()
        self.guest = {"username": data["username"], "token": data["token"], "expires": expires, "limits": data.get("limits", {})}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.guest))
            path.chmod(0o600)  # the token is the guest's key to its images
        except OSError:
            pass
        return data["token"]

    async def _request(self, method: str, path: str, *, auth: bool = True, **kwargs: Any) -> Any:
        if auth:
            await self._ensure_token()
        agent = {"User-Agent": f"label-image-mcp/{VERSION} ({self.client_name})"} | self.viewer
        headers = agent | ({"Authorization": f"Bearer {self._token}"} if auth and self._token else {})
        resp = await self._http.request(method, path, headers=headers, **kwargs)
        if resp.status_code == 401 and auth and self._token:
            # Sessions expire (12 hours): log in again once and retry. A guest that is no longer valid starts over.
            self._token = None
            if self.guest:
                self._token = await self._guest_token(renew=True)
            await self._ensure_token()
            headers = agent | ({"Authorization": f"Bearer {self._token}"} if self._token else {})
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

    async def warm_segmenter(self, model: str = "sam") -> dict[str, Any]:
        """Start loading the segmentation model in the background; returns at once."""
        return await self._request("POST", "/segment/warmup", json={"model": model})

    async def segment(
        self,
        image_id: str,
        prompts: list[dict[str, list[float]]],
        model: str = "sam",
        *,
        attempts: int = 6,
        retry_delay: float = 15.0,
    ) -> dict[str, Any]:
        """Trace outlines for box or point prompts with Segment Anything. After a quiet spell the model takes
        a minute or two to start, longer than the API's 30 seconds (503/504), so that is retried for a while."""
        for attempt in range(attempts):
            try:
                return await self._request(
                    "POST", f"/images/{image_id}/segment", json={"model": model, "prompts": prompts}
                )
            except ApiError as exc:
                if exc.status not in (503, 504) or attempt == attempts - 1:
                    raise
                await asyncio.sleep(retry_delay)
        raise AssertionError("unreachable")

    async def pathology(
        self, image_id: str, action: str, *, attempts: int = 8, retry_delay: float = 15.0, **params: Any
    ) -> dict[str, Any]:
        """Run a histopathology tool on an image: "nuclei", "normalize" or "quality" ("warmup" starts the model
        without waiting). Like segment, a cold start is longer than the API's 30 seconds, so 503/504 is retried."""
        for attempt in range(attempts):
            try:
                return await self._request("POST", f"/images/{image_id}/pathology", json={"action": action, **params})
            except ApiError as exc:
                if exc.status not in (503, 504) or attempt == attempts - 1:
                    raise
                await asyncio.sleep(retry_delay)
        raise AssertionError("unreachable")

    async def fetch_image(self, image_id: str, image: dict[str, Any] | None = None) -> tuple[dict[str, Any], bytes]:
        """An image's record (read unless given) and its file, from the record's signed URL."""
        image = image or await self.get_image(image_id)
        url = image.get("downloadUrl") or image.get("url")
        if not url:
            raise ApiError(f"Image {image_id} has no download URL")
        resp = await self._s3.get(url)
        if resp.status_code != 200:
            raise ApiError(f"Downloading image {image_id} failed (HTTP {resp.status_code})")
        return image, resp.content

    async def _upload_stream(self, chunks: Any, size: int, filename: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Presigned upload of a file of known size, streamed in chunks, then confirmed as a whole slide."""
        if size > MAX_SLIDE_BYTES:
            raise ApiError(f"Slides larger than {MAX_SLIDE_BYTES // 1024**3} GB are not accepted")
        content_type = "application/octet-stream"
        upload = await self._request("POST", "/images/upload-url", json={"filename": filename, "contentType": content_type})
        put = await self._s3.put(
            upload["uploadUrl"], content=chunks, headers={"Content-Type": content_type, "Content-Length": str(size)},
            timeout=httpx.Timeout(60.0, write=600.0),
        )
        if put.status_code >= 400:
            raise ApiError(f"S3 upload failed (HTTP {put.status_code})")
        return await self._request(
            "POST", f"/images/{upload['imageId']}/confirm",
            json={"width": 0, "height": 0, "filename": filename, "fileSize": size, "contentType": content_type, "kind": "slide", **fields},
        )

    async def upload_slide(self, path: str | Path, **fields: str) -> dict[str, Any]:
        """Upload a whole-slide file (SVS, NDPI, pyramidal TIFF) without reading it into memory."""
        file_path = Path(path).expanduser()
        if file_path.suffix.lower() not in SLIDE_EXTENSIONS:
            raise ApiError(f"Not a whole-slide file: expected one of {', '.join(SLIDE_EXTENSIONS)}")

        async def chunks():
            with file_path.open("rb") as f:
                while chunk := f.read(CHUNK):
                    yield chunk

        return await self._upload_stream(chunks(), file_path.stat().st_size, file_path.name, fields)

    async def upload_slide_from_url(self, url: str, **fields: str) -> dict[str, Any]:
        """Copy a whole-slide file from an https URL straight into storage, without keeping it locally."""
        parsed = urlparse(url)
        name = Path(parsed.path).name
        if not name.lower().endswith(SLIDE_EXTENSIONS):
            raise ApiError(f"Not a whole-slide file: expected one of {', '.join(SLIDE_EXTENSIONS)}")
        allowed = [h.strip().lower() for h in os.environ.get("LABEL_UPLOAD_URL_HOSTS", "").split(",") if h.strip()]
        if allowed and (parsed.hostname or "").lower() not in allowed:
            raise ApiError(f"Downloads are only allowed from: {', '.join(allowed)}")
        if parsed.scheme != "https" and not (allowed and parsed.scheme == "http"):
            raise ApiError("Slide URLs must use https")
        async with self._s3.stream("GET", url, follow_redirects=False, timeout=httpx.Timeout(60.0, read=600.0)) as resp:
            if resp.status_code != 200:
                raise ApiError(f"Downloading the slide failed (HTTP {resp.status_code})")
            size = int(resp.headers.get("content-length") or 0)
            if not size:
                raise ApiError("The server didn't say how large the slide is; download it and use upload_slide")
            return await self._upload_stream(resp.aiter_bytes(CHUNK), size, name, fields)

    async def start_slide_analysis(
        self, image_id: str, max_tiles: int | None = None, cell_types: bool | None = None, coverage: str = "sample"
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"action": "slide", "coverage": coverage}
        if cell_types is not None:
            body["cell_types"] = cell_types
        if max_tiles:
            body["max_tiles"] = max_tiles
        return await self._request("POST", f"/images/{image_id}/pathology", json=body)

    async def share_image(self, image_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/images/{image_id}/share")

    async def start_classification(self, image_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/images/{image_id}/pathology", json={"action": "classify"})

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
            "POST", "/images/upload-url", json={"filename": filename, "contentType": content_type, "fileSize": len(content)}
        )
        put = await self._s3.put(
            upload["uploadUrl"], content=content, headers={"Content-Type": content_type, **upload.get("uploadHeaders", {})}
        )
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

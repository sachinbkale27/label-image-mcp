import json

import httpx
import pytest
import respx
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image

from label_image_mcp import server
from label_image_mcp.client import ApiError, LabelImageClient
from label_image_mcp.exporters import export_coco, export_yolo

API = "https://api.test/api"

IMAGE = {
    "imageId": "img-1",
    "filename": "cells.png",
    "width": 200,
    "height": 100,
    "category": "Histopathology",
    "annotations": [
        {"id": "a1", "type": "pin", "label": "nucleus", "x": 5, "y": 5, "color": "#f00", "createdAt": "t"},
        {
            "id": "a2", "type": "polygon", "label": "tumor", "x": 10, "y": 20, "width": 40, "height": 30,
            "points": [{"x": 10, "y": 20}, {"x": 50, "y": 20}, {"x": 50, "y": 50}, {"x": 10, "y": 50}],
            "color": "#0f0", "createdAt": "t", "confidence": "high",
        },
        {"id": "a3", "type": "circle", "label": "cell", "x": 100, "y": 50, "radius": 10, "color": "#00f", "createdAt": "t"},
    ],
}


def ok(data):
    return httpx.Response(200, json={"success": True, "data": data})


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("LABEL_API_URL", API)
    monkeypatch.setenv("LABEL_USERNAME", "alice")
    monkeypatch.setenv("LABEL_PASSWORD", "secret")
    with respx.mock(assert_all_called=False) as mock:
        mock.post(f"{API}/admin/login").mock(return_value=ok({"token": "tok", "username": "alice", "expiresIn": 3600}))
        mock.get(f"{API}/images").mock(return_value=ok([IMAGE, {**IMAGE, "imageId": "img-2", "category": "Other"}]))
        mock.get(f"{API}/images/img-1").mock(return_value=ok(IMAGE))
        yield mock


async def call_raw(name, args):
    async with create_connected_server_and_client_session(server.mcp._mcp_server) as session:
        return await session.call_tool(name, args)


async def call(name, args):
    result = await call_raw(name, args)
    assert not result.isError, result.content
    data = result.structuredContent
    # Non-object return values are wrapped as {"result": ...}
    return data["result"] if set(data) == {"result"} else data


async def test_list_images_filters_by_category(api):
    images = await call("list_images", {"category": "histopathology"})
    assert [i["imageId"] for i in images] == ["img-1"]
    assert images[0]["annotationCount"] == 3


async def test_requests_send_bearer_token(api):
    await call("get_image", {"image_id": "img-1"})
    assert api.routes[0].call_count == 1  # logged in once
    assert api.calls.last.request.headers["Authorization"] == "Bearer tok"


async def test_add_annotations_appends_and_computes_polygon_bbox(api):
    save = api.put(f"{API}/images/img-1/annotations").mock(return_value=ok({"imageId": "img-1", "annotationCount": 4}))
    result = await call("add_annotations", {
        "image_id": "img-1",
        "annotations": [{"type": "polygon", "label": "stroma",
                         "points": [{"x": 1, "y": 2}, {"x": 9, "y": 2}, {"x": 5, "y": 12}]}],
    })
    sent = json.loads(save.calls.last.request.content)["annotations"]
    assert [a["id"] for a in sent[:3]] == ["a1", "a2", "a3"]
    assert sent[3] | {"id": None, "createdAt": None} == {
        "type": "polygon", "label": "stroma", "color": "#ff0000", "id": None, "createdAt": None,
        "points": [{"x": 1, "y": 2}, {"x": 9, "y": 2}, {"x": 5, "y": 12}],
        "x": 1, "y": 2, "width": 8, "height": 10,
    }
    assert result["addedIds"] == [sent[3]["id"]]


async def test_invalid_annotation_is_rejected(api):
    result = await call_raw("add_annotations", {
        "image_id": "img-1", "annotations": [{"type": "circle", "label": "c", "x": 1, "y": 1}],
    })
    assert result.isError
    assert "circle needs radius" in result.content[0].text


async def test_remove_annotations(api):
    save = api.put(f"{API}/images/img-1/annotations").mock(return_value=ok({"imageId": "img-1", "annotationCount": 2}))
    result = await call("remove_annotations", {"image_id": "img-1", "annotation_ids": ["a1", "missing"]})
    assert result["removed"] == 1
    assert [a["id"] for a in json.loads(save.calls.last.request.content)["annotations"]] == ["a2", "a3"]


async def test_api_error_surfaces(api):
    api.put(f"{API}/images/img-1/annotations").mock(
        return_value=httpx.Response(403, json={"success": False, "error": "Permission denied"})
    )
    client = LabelImageClient()
    with pytest.raises(ApiError, match="Permission denied"):
        await client.save_annotations("img-1", [])
    await client.aclose()


async def test_upload_image_flow(api, tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (64, 32)).save(path)
    api.post(f"{API}/images/upload-url").mock(
        return_value=ok({"imageId": "new", "uploadUrl": "https://s3.test/put", "imageUrl": "u"})
    )
    s3 = api.put("https://s3.test/put").mock(return_value=httpx.Response(200))
    confirm = api.post(f"{API}/images/new/confirm").mock(return_value=ok({"imageId": "new", "filename": "photo.png"}))

    result = await call("upload_image", {"path": str(path), "category": "Test"})

    body = json.loads(confirm.calls.last.request.content)
    assert result["imageId"] == "new"
    assert (body["width"], body["height"], body["contentType"], body["category"]) == (64, 32, "image/png", "Test")
    assert len(body["imageHash"]) == 64
    assert s3.calls.last.request.headers["Content-Type"] == "image/png"


def test_export_coco_matches_frontend_rules():
    coco = export_coco(IMAGE)
    # Pins are not exported, but every label still gets a category (same as exportUtils.ts)
    assert [c["name"] for c in coco["categories"]] == ["nucleus", "tumor", "cell"]
    poly, circle = coco["annotations"]
    assert poly["bbox"] == [10, 20, 40, 30] and poly["area"] == 1200 and poly["category_id"] == 2
    assert poly["attributes"] == {"confidence": "high"}
    assert circle["bbox"] == [90, 40, 20, 20] and len(circle["segmentation"][0]) == 64


def test_export_yolo_normalizes():
    yolo = export_yolo(IMAGE)
    assert yolo["classes"] == ["nucleus", "tumor", "cell"]
    assert yolo["annotations"].splitlines() == ["1 0.15 0.35 0.2 0.3", "2 0.5 0.5 0.1 0.2"]


async def test_upload_image_from_url(api, monkeypatch):
    monkeypatch.setenv("LABEL_UPLOAD_URL_HOSTS", "blog.test")
    buf = __import__("io").BytesIO()
    Image.new("RGB", (40, 30)).save(buf, format="PNG")
    api.get("https://blog.test/media/cells.png").mock(
        return_value=httpx.Response(200, content=buf.getvalue(), headers={"content-type": "image/png"})
    )
    api.post(f"{API}/images/upload-url").mock(return_value=ok({"imageId": "new", "uploadUrl": "https://s3.test/put"}))
    api.put("https://s3.test/put").mock(return_value=httpx.Response(200))
    confirm = api.post(f"{API}/images/new/confirm").mock(return_value=ok({"imageId": "new", "filename": "cells.png"}))

    result = await call("upload_image_from_url", {"url": "https://blog.test/media/cells.png", "category": "Histo"})

    body = json.loads(confirm.calls.last.request.content)
    assert result["imageId"] == "new"
    assert (body["filename"], body["width"], body["height"], body["category"]) == ("cells.png", 40, 30, "Histo")


@pytest.mark.parametrize(
    ("url", "response", "error"),
    [
        ("https://evil.test/x.png", None, "only allowed from: blog.test"),
        ("https://blog.test/page", httpx.Response(200, text="<html>", headers={"content-type": "text/html"}), "Not an image"),
        ("https://blog.test/moved.png", httpx.Response(302, headers={"location": "http://169.254.169.254/"}), "HTTP 302"),
    ],
)
async def test_upload_image_from_url_rejects_unsafe_input(api, monkeypatch, url, response, error):
    monkeypatch.setenv("LABEL_UPLOAD_URL_HOSTS", "blog.test")
    if response is not None:
        api.get(url).mock(return_value=response)
    result = await call_raw("upload_image_from_url", {"url": url})
    assert result.isError
    assert error in result.content[0].text


async def test_hosted_mode_requires_the_bearer_token(monkeypatch):
    from starlette.testclient import TestClient

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    monkeypatch.delenv("LABEL_CREDENTIALS_SECRET", raising=False)
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    assert TestClient(server.BearerAuth(app)).post("/mcp").status_code == 503  # no token configured: closed

    monkeypatch.setenv("MCP_AUTH_TOKEN", "s3cret")
    client = TestClient(server.BearerAuth(app))
    assert client.post("/mcp").status_code == 401
    assert client.post("/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/mcp", headers={"Authorization": "Bearer s3cret"}).status_code == 200


SEGMENTED = {
    "imageId": "img-1", "width": 200, "height": 100, "model": "sam",
    "results": [
        {"polygon": [[10, 10], [60, 10], [60, 40], [10, 40]], "score": 0.97, "area": 1500, "bbox": [10, 10, 51, 31], "regions": 1},
        {"error": "nothing found in this box or at this point", "score": 0.2},
    ],
}


async def test_segment_objects_sends_prompts_and_saves_traced_polygons(api):
    seg = api.post(f"{API}/images/img-1/segment").mock(return_value=ok(SEGMENTED))
    save = api.put(f"{API}/images/img-1/annotations").mock(return_value=ok({"imageId": "img-1", "annotationCount": 4}))
    result = await call("segment_objects", {
        "image_id": "img-1",
        "objects": [
            {"label": "cell", "box": {"x": 5, "y": 5, "width": 60, "height": 40}},
            {"label": "speck", "point": {"x": 150, "y": 80}},
        ],
    })
    assert json.loads(seg.calls.last.request.content) == {
        "model": "sam", "prompts": [{"box": [5, 5, 65, 45]}, {"point": [150, 80]}],
    }
    sent = json.loads(save.calls.last.request.content)["annotations"]
    assert len(sent) == 4  # three existing + one traced
    traced = sent[3]
    assert traced["type"] == "polygon" and traced["label"] == "cell" and traced["confidence"] == "high"
    assert traced["points"][0] == {"x": 10, "y": 10} and traced["width"] == 50
    assert "SAM (score 0.97)" in traced["notes"]
    assert result["added"][0]["id"] == traced["id"] and result["annotationCount"] == 4
    assert result["failed"] == [{"label": "speck", "reason": "nothing found in this box or at this point"}]


async def test_segment_objects_without_saving(api):
    api.post(f"{API}/images/img-1/segment").mock(return_value=ok(SEGMENTED))
    save = api.put(f"{API}/images/img-1/annotations")
    result = await call("segment_objects", {
        "image_id": "img-1", "save": False, "objects": [{"label": "cell", "box": {"x": 5, "y": 5, "width": 60, "height": 40}}, {"label": "b", "point": {"x": 1, "y": 1}}],
    })
    assert not save.called
    assert result["outlines"][0]["label"] == "cell" and len(result["outlines"][0]["points"]) == 4


async def test_medsam_needs_boxes(api):
    result = await call_raw("segment_objects", {
        "image_id": "img-1", "model": "medsam", "objects": [{"label": "lesion", "point": {"x": 5, "y": 5}}],
    })
    assert result.isError and "MedSAM needs a box" in result.content[0].text


async def test_segment_retries_while_the_model_starts():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/admin/login"):
            return ok({"token": "tok"})
        if len(calls) < 4:
            return httpx.Response(503, json={"success": False, "error": "The segmentation model is starting up."})
        return ok(SEGMENTED)

    client = LabelImageClient(API, "alice", "secret", transport=httpx.MockTransport(handler))
    result = await client.segment("img-1", [{"box": [0, 0, 5, 5]}], attempts=3, retry_delay=0)
    assert result["model"] == "sam" and len(calls) == 4  # login, 503, 503, success
    await client.aclose()


async def test_warm_up_segmentation(api):
    warm = api.post(f"{API}/segment/warmup").mock(return_value=ok({"warming": "medsam"}))
    assert await call("warm_up_segmentation", {"model": "medsam"}) == {"warming": "medsam"}
    assert json.loads(warm.calls.last.request.content) == {"model": "medsam"}


NUCLEI = {
    "imageId": "img-1", "action": "nuclei", "width": 200, "height": 100, "scale": 2,
    "summary": {"count": 2, "per_megapixel": 100.0, "median_area": 30, "area_variation": 0.4, "enlarged": 1, "returned": 2},
    "nuclei": [
        {"polygon": [[10, 10], [14, 10], [14, 14], [10, 14]], "score": 0.9, "area": 16, "enlarged": False},
        {"polygon": [[50, 50], [70, 50], [70, 70], [50, 70]], "score": 0.72, "area": 400, "enlarged": True},
    ],
}


async def test_detect_nuclei_saves_outlines_and_replaces_an_earlier_run(api):
    earlier = {"id": "old", "type": "polygon", "label": "nucleus", "x": 0, "y": 0, "width": 1, "height": 1,
               "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}], "notes": "Detected by StarDist (score 0.8)."}
    api.get(f"{API}/images/img-1").mock(return_value=ok({**IMAGE, "annotations": IMAGE["annotations"] + [earlier]}))
    run = api.post(f"{API}/images/img-1/pathology").mock(return_value=ok(NUCLEI))
    save = api.put(f"{API}/images/img-1/annotations").mock(return_value=ok({"imageId": "img-1", "annotationCount": 5}))
    result = await call("detect_nuclei", {"image_id": "img-1"})
    assert json.loads(run.calls.last.request.content) == {"action": "nuclei", "max_nuclei": 300}
    sent = json.loads(save.calls.last.request.content)["annotations"]
    assert [a["id"] for a in sent[:3]] == ["a1", "a2", "a3"] and len(sent) == 5  # the earlier StarDist nucleus is gone
    assert sent[3]["label"] == "nucleus" and sent[3]["confidence"] == "high"
    assert sent[4]["label"] == "enlarged nucleus" and sent[4]["confidence"] == "low"
    assert result["labels"] == {"nucleus": 1, "enlarged nucleus": 1} and result["removedPrevious"] == 1
    assert result["summary"]["count"] == 2


async def test_normalize_stain_and_quality(api):
    normalized = {"imageId": "img-9", "filename": "cells-normalized.jpg", "name": "cells (stain-normalized)", "annotations": []}
    run = api.post(f"{API}/images/img-1/pathology")
    run.side_effect = [
        ok({"imageId": "img-1", "action": "normalize", "normalized": normalized}),
        ok({"imageId": "img-1", "action": "quality", "verdict": "check", "warnings": ["Faint staining"], "metrics": {}, "width": 200, "height": 100}),
    ]
    result = await call("normalize_stain", {"image_id": "img-1"})
    assert result["original"] == "img-1" and result["normalized"]["imageId"] == "img-9"
    quality = await call("check_image_quality", {"image_id": "img-1"})
    assert quality["verdict"] == "check" and quality["warnings"] == ["Faint staining"]


def test_export_qupath_geojson():
    from label_image_mcp.exporters import export_qupath

    image = {**IMAGE, "annotations": IMAGE["annotations"] + [
        {"id": "n1", "type": "polygon", "label": "nucleus", "x": 0, "y": 0, "width": 4, "height": 4, "color": "#00b8d4",
         "points": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4}], "notes": "Detected by StarDist (score 0.9)."},
        {"id": "r1", "type": "arrow", "label": "look", "x": 1, "y": 1, "arrowTarget": {"x": 5, "y": 5}},
    ]}
    features = export_qupath(image)["features"]
    assert [f["id"] for f in features] == ["a1", "a2", "a3", "n1"]  # the arrow is left out
    assert features[0]["geometry"] == {"type": "Point", "coordinates": [5, 5]}
    ring = features[1]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and len(ring) == 5
    assert features[1]["properties"]["classification"] == {"name": "tumor", "color": [0, 255, 0]}
    assert features[1]["properties"]["objectType"] == "annotation" and features[3]["properties"]["objectType"] == "detection"

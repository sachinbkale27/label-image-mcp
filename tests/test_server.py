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


async def test_warm_up_starts_the_chosen_model(api):
    warm = api.post(f"{API}/segment/warmup").mock(return_value=ok({"warming": "medsam"}))
    assert await call("warm_up", {"model": "medsam"}) == {"warming": "medsam"}
    assert json.loads(warm.calls.last.request.content) == {"model": "medsam"}
    nuclei = api.post(f"{API}/images/warmup/pathology").mock(return_value=ok({"warming": True}))
    await call("warm_up", {"model": "nuclei"})
    assert json.loads(nuclei.calls.last.request.content) == {"action": "warmup"}


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


async def test_detect_nuclei_in_microns(api):
    run = api.post(f"{API}/images/img-1/pathology").mock(return_value=ok(NUCLEI))
    save = api.put(f"{API}/images/img-1/annotations").mock(return_value=ok({"imageId": "img-1", "annotationCount": 5}))
    result = await call("detect_nuclei", {"image_id": "img-1", "microns_per_pixel": 0.5})
    assert json.loads(run.calls.last.request.content)["microns_per_pixel"] == 0.5
    # 30 px x 0.25 um2/px; 2 nuclei in 200 x 100 px = 5,000 um2 = 0.005 mm2
    assert result["summary"]["median_area_um2"] == 7.5 and result["summary"]["per_mm2"] == 400.0
    assert "4.0 \u00b5m\u00b2" in json.loads(save.calls.last.request.content)["annotations"][3]["notes"]


def _png(width=200, height=100) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (240, 200, 220)).save(buffer, "PNG")
    return buffer.getvalue()


async def test_view_image_returns_the_picture_with_a_grid_and_annotations(api):
    api.get(f"{API}/images/img-1").mock(return_value=ok({**IMAGE, "downloadUrl": "https://s3.test/cells.png"}))
    api.get("https://s3.test/cells.png").mock(return_value=httpx.Response(200, content=_png()))
    result = await call_raw("view_image", {"image_id": "img-1", "with_annotations": True})
    assert not result.isError, result.content
    info, picture = result.content
    data = json.loads(info.text)
    assert picture.type == "image" and picture.mimeType == "image/jpeg"
    assert data["image_size"] == [200, 100] and data["scale"] == 1 and data["grid_every"] == 20
    assert {d["label"]: d["count"] for d in data["annotations_drawn"]} == {"nucleus": 1, "tumor": 1, "cell": 1}

    zoom = await call_raw("view_image", {"image_id": "img-1", "with_annotations": True, "region": {"x": 0, "y": 0, "width": 60, "height": 60}})
    data = json.loads(zoom.content[0].text)
    assert data["shown_region"] == [0, 0, 60, 60] and {d["label"] for d in data["annotations_drawn"]} == {"nucleus", "tumor"}

    plain = await call_raw("view_image", {"image_id": "img-1"})
    assert "annotations_drawn" not in json.loads(plain.content[0].text)


async def test_export_dataset_uses_one_class_list(api):
    second = {**IMAGE, "imageId": "img-2", "filename": "cells.png", "annotations": [
        {"id": "b1", "type": "circle", "label": "cell", "x": 20, "y": 20, "radius": 5, "color": "#00f", "createdAt": "t"},
    ]}
    api.get(f"{API}/images/img-2").mock(return_value=ok(second))
    coco = (await call("export_dataset", {"image_ids": ["img-1", "img-2"], "classes": ["cell"]}))["dataset"]
    assert [c["name"] for c in coco["categories"]] == ["cell", "tumor"]
    assert [(a["image_id"], a["category_id"]) for a in coco["annotations"]] == [(1, 2), (1, 1), (2, 1)]
    yolo = await call("export_dataset", {"image_ids": ["img-1", "img-2"], "format": "yolo"})
    assert yolo["classes"] == ["tumor", "cell"]
    assert sorted(yolo["dataset"]["files"]) == ["cells.txt", "cells_2.txt"]  # same file name, kept apart
    assert yolo["dataset"]["files"]["cells_2.txt"].startswith("1 ")
    by_category = await call("export_dataset", {"category": "histopathology", "format": "qupath"})
    assert by_category["images"] == 1 and list(by_category["dataset"]["files"]) == ["cells.geojson"]


async def test_export_dataset_writes_files_locally(api, tmp_path):
    result = await call("export_dataset", {"image_ids": ["img-1"], "format": "yolo", "output_dir": str(tmp_path)})
    assert result["files"] == ["classes.txt", "labels/cells.txt"] and "dataset" not in result
    assert (tmp_path / "classes.txt").read_text() == "tumor\ncell\n"


async def test_tools_say_what_they_change():
    tools = {t.name: t for t in await server.mcp.list_tools()}
    assert all(t.title and t.annotations for t in tools.values())
    hints = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")
    missing = {n: [h for h in hints if not isinstance(getattr(t.annotations, h), bool)] for n, t in tools.items()}
    assert not {n: m for n, m in missing.items() if m}, "every tool declares all four hints"
    assert tools["replace_annotations"].annotations.destructiveHint is True
    assert tools["view_image"].annotations.readOnlyHint is True
    assert "view_image" in server.mcp.instructions


async def test_hosted_server_has_no_local_file_tools(monkeypatch):
    monkeypatch.setattr(server, "HOSTED", False)
    tools = {name: server.mcp._tool_manager.get_tool(name) for name in ("upload_image", "upload_slide")}
    try:
        server.configure_hosted()
        assert not set(tools) & {t.name for t in await server.mcp.list_tools()}
    finally:
        server.mcp._tool_manager._tools.update(tools)  # restore for the other tests


async def test_prompts_are_offered():
    prompts = {p.name for p in await server.mcp.list_prompts()}
    assert prompts == {"analyse_slide", "outline_objects", "build_training_dataset"}
    result = await server.mcp.get_prompt("analyse_slide", {"image": "img-1", "microns_per_pixel": "0.25"})
    assert "detect_nuclei with microns_per_pixel=0.25" in result.messages[0].content.text


SLIDE = {**IMAGE, "imageId": "s-1", "filename": "case.svs", "kind": "slide", "width": 2000, "height": 1500,
         "thumbnailKey": "images/s-1/thumbnail.jpg", "nucleiUrl": "https://s3.test/nuclei.geojson", "slideUrl": "https://s3.test/case.svs",
         "slide": {"width": 46000, "height": 34500, "mpp": 0.5, "thumbnailScale": 0.0434783},
         "annotations": [{"id": "t1", "type": "rectangle", "label": "densest tile", "x": 100, "y": 50, "width": 44.5, "height": 44.5, "color": "#ff1744"}]}


async def test_upload_slide_streams_the_file_and_starts_the_analysis(api, tmp_path):
    path = tmp_path / "case.svs"
    path.write_bytes(b"x" * 3000)
    api.post(f"{API}/images/upload-url").mock(return_value=ok({"imageId": "s-1", "uploadUrl": "https://s3.test/put"}))
    put = api.put("https://s3.test/put").mock(return_value=httpx.Response(200))
    confirm = api.post(f"{API}/images/s-1/confirm").mock(return_value=ok({**SLIDE, "slideAnalysis": None}))
    start = api.post(f"{API}/images/s-1/pathology").mock(return_value=ok({"slideAnalysis": {"status": "queued"}}))
    result = await call("upload_slide", {"path": str(path), "category": "Histopathology", "max_tiles": 8})
    assert put.calls.last.request.headers["content-length"] == "3000"
    sent = json.loads(confirm.calls.last.request.content)
    assert sent["kind"] == "slide" and sent["fileSize"] == 3000 and sent["category"] == "Histopathology"
    assert json.loads(start.calls.last.request.content) == {"action": "slide", "coverage": "sample", "max_tiles": 8}
    assert result["kind"] == "slide" and result["analysis"] == "started"
    photo = tmp_path / "cat.png"
    photo.write_bytes(b"x")
    refused = await call_raw("upload_slide", {"path": str(photo)})
    assert refused.isError and "Not a whole-slide file" in refused.content[0].text


async def test_analyze_slide_waits_for_the_background_run(api, monkeypatch):
    monkeypatch.setattr(server, "SLIDE_POLL_SECONDS", 0.01)
    running = {**SLIDE, "slideAnalysis": {"status": "running", "step": "analysing tile 3 of 8", "startedAt": "t"}}
    done = {**SLIDE, "slideAnalysis": {"status": "done", "summary": {"nuclei": 4200, "per_mm2": 1050.0}, "tileImages": [{"imageId": "t-1"}]}}
    api.get(f"{API}/images/s-1").side_effect = [ok({**SLIDE, "slideAnalysis": None}), ok(running), ok(done)]
    start = api.post(f"{API}/images/s-1/pathology").mock(return_value=ok({}))
    result = await call("analyze_slide", {"image_id": "s-1", "wait_seconds": 5})
    assert start.called and result["status"] == "done" and result["summary"]["per_mm2"] == 1050.0
    assert result["nuclei_geojson_url"] == "https://s3.test/nuclei.geojson" and result["tileImages"] == [{"imageId": "t-1"}]

    # already running elsewhere (409): it waits instead of failing
    api.get(f"{API}/images/s-1").side_effect = None
    api.get(f"{API}/images/s-1").mock(return_value=ok(running))
    start.mock(return_value=httpx.Response(409, json={"success": False, "error": "already being analysed"}))
    result = await call("analyze_slide", {"image_id": "s-1", "restart": True, "wait_seconds": 0})
    assert result["status"] == "running" and result["step"] == "analysing tile 3 of 8"


async def test_slide_exports_and_views(api):
    api.get(f"{API}/images/s-1").mock(return_value=ok(SLIDE))
    geo = await call("export_annotations", {"image_id": "s-1", "format": "qupath"})
    ring = geo["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == [2300.0, 1150.0] and geo["nuclei_geojson_url"] == "https://s3.test/nuclei.geojson"  # overview px / 0.0435
    coco = await call("export_annotations", {"image_id": "s-1", "format": "coco"})
    assert "overview image" in coco["note"]
    api.get(f"{API}/images/s-2").mock(return_value=ok({**SLIDE, "imageId": "s-2", "thumbnailKey": None}))
    blocked = await call_raw("view_image", {"image_id": "s-2"})
    assert blocked.isError and "no overview yet" in blocked.content[0].text


async def test_classify_cells_starts_and_waits(api, monkeypatch):
    monkeypatch.setattr(server, "SLIDE_POLL_SECONDS", 0.01)
    done = {"status": "done", "counts": {"tumour cell": 236, "immune cell": 14}, "immune_to_tumour": 0.06, "note": "CC BY-NC-SA"}
    api.get(f"{API}/images/img-1").side_effect = [ok(IMAGE), ok({**IMAGE, "cellTypes": {"status": "running"}}), ok({**IMAGE, "cellTypes": done})]
    start = api.post(f"{API}/images/img-1/pathology").mock(return_value=ok({"cellTypes": {"status": "queued"}}))
    result = await call("classify_cells", {"image_id": "img-1", "wait_seconds": 5})
    assert json.loads(start.calls.last.request.content) == {"action": "classify"}
    assert result["status"] == "done" and result["counts"]["tumour cell"] == 236


async def test_api_calls_name_the_ai_client(api):
    """The client and server versions go with each API call (counted in usage), nothing else."""
    listing = api.get(f"{API}/images").mock(return_value=ok([IMAGE]))
    await call("list_images", {})
    agent = listing.calls.last.request.headers["user-agent"]
    assert agent.startswith("label-image-mcp/") and "(mcp/" in agent  # the in-memory test client calls itself "mcp"
    assert not any(h.startswith("x-viewer-") for h in listing.calls.last.request.headers)  # only the hosted server forwards these


async def test_share_image_returns_a_link(api):
    share = api.post(f"{API}/images/img-1/share").mock(return_value=ok({"imageId": "img-1", "url": "https://labels.test/?shared=t0k", "expiresAt": "x"}))
    result = await call("share_image", {"image_id": "img-1"})
    assert share.called and result["url"].endswith("?shared=t0k")


async def test_without_an_account_it_works_as_a_saved_guest(monkeypatch, tmp_path):
    monkeypatch.setenv("LABEL_API_URL", API)
    monkeypatch.delenv("LABEL_USERNAME", raising=False)
    monkeypatch.delenv("LABEL_PASSWORD", raising=False)
    monkeypatch.delenv("LABEL_CREDENTIALS_SECRET", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    guest = {"username": "guest-abc", "token": "g-tok", "expiresAt": "2099-01-01T00:00:00Z", "limits": {"images": 50}}
    with respx.mock(assert_all_called=False) as mock:
        made = mock.post(f"{API}/guest").mock(return_value=ok(guest))
        listing = mock.get(f"{API}/images").mock(return_value=ok([]))
        client = LabelImageClient()
        await client.list_images()
        assert made.call_count == 1 and listing.calls.last.request.headers["authorization"] == "Bearer g-tok"
        saved = next(tmp_path.glob("label-image-mcp/guest-*.json"))
        assert json.loads(saved.read_text())["username"] == "guest-abc" and oct(saved.stat().st_mode)[-3:] == "600"
        await client.aclose()

        again = LabelImageClient()  # a new run reuses the saved guest
        await again.list_images()
        assert made.call_count == 1
        await again.aclose()

        # uploads say their size and send the headers the API signed (the guest retention tag)
        url = mock.post(f"{API}/images/upload-url").mock(return_value=ok({"imageId": "i1", "uploadUrl": "https://s3.test/put", "uploadHeaders": {"x-amz-tagging": "retention=guest"}}))
        put = mock.put("https://s3.test/put").mock(return_value=httpx.Response(200))
        mock.post(f"{API}/images/i1/confirm").mock(return_value=ok({"imageId": "i1"}))
        uploader = LabelImageClient()
        await uploader.upload_bytes(_png(), "a.png", "image/png")
        assert json.loads(url.calls.last.request.content)["fileSize"] == len(_png())
        assert put.calls.last.request.headers["x-amz-tagging"] == "retention=guest"
        await uploader.aclose()

    monkeypatch.setenv("LABEL_GUEST", "0")
    assert not LabelImageClient.guest_allowed()

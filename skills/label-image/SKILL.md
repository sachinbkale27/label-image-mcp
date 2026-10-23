---
name: label-image
description: Build image-annotation training datasets with the Image Label MCP server (label-image-mcp, labels.sachin-kale.com). Use when the user wants to upload images, add, fix or clean up annotations (polygons, circles, rectangles, pins, arrows), normalize labels, or export COCO JSON / YOLO training data from Image Label. Also use when the user mentions Image Label, labels.sachin-kale.com, or the list_images / add_annotations / export_annotations tools.
license: MIT
---

# Image Label datasets

This skill drives the `label-image` MCP server, which manages images and annotations in
[Image Label](https://labels.sachin-kale.com) and exports them as COCO JSON or YOLO.

## Before you start

1. Check that the tools are available: `list_images`, `get_image`, `update_image_metadata`,
   `add_annotations`, `replace_annotations`, `remove_annotations`, `upload_image`,
   `upload_image_from_url`, `export_annotations` (and `segment_objects` on newer servers). If they are missing, tell the user to install the server
   (credentials go in environment variables; never ask the user to paste a password into chat):
   ```bash
   claude mcp add label-image -e LABEL_USERNAME=... -e LABEL_PASSWORD=... -- uvx label-image-mcp
   ```
   Other clients: https://github.com/sachinbkale27/label-image-mcp#install
2. The tools need an account: they only see and change images saved to it, and without
   credentials they see none. An `HTTP 401`/`403` error means missing or wrong credentials.
   Don't retry; tell the user.
3. Only images saved to the account are visible. Work kept only in the browser is not.

## Data model you must respect

Coordinates are **image pixels, origin top-left**. Always call `get_image` first and keep
every coordinate inside `0..width` and `0..height`.

| type | required fields | exported to COCO/YOLO? |
|---|---|---|
| `polygon` | `points` (3 or more `{x, y}`) | **yes** (segmentation + bbox) |
| `circle` | `x`, `y` (center), `radius` | **yes** (as a 32-point polygon) |
| `rectangle` | `x`, `y` (top-left), `width`, `height` | no |
| `pin` | `x`, `y` | no |
| `arrow` | `x`, `y` (label position), `arrowTarget` `{x, y}` | no |

Every annotation also takes `label` (the class name), and optionally `color` (default
`#ff0000`), `notes` and `confidence` (`high` / `medium` / `low`).

**Training data must be polygons or circles.** Rectangles, pins and arrows are silently left
out of exports. For a box, use a 4-point polygon:
`[{x:L,y:T},{x:R,y:T},{x:R,y:B},{x:L,y:B}]`. Use pins and arrows only for human-facing notes.

## Workflows

### Explore
`list_images(category?)` returns summaries with `annotationCount`. `get_image(image_id)` returns
the annotations (with their `id`s) and a signed viewing URL valid for 24 hours. Show the user a
compact table rather than raw JSON.

### Upload
`upload_image(path, name?, category?, description?)` reads the file **on the machine running the
MCP server**. Use an absolute path. With a remote HTTP server, a local path won't exist: use
`upload_image_from_url(url, ...)` for images that are online. Set a
`category` at upload: it's the only filter `list_images` has. Upload one file per call and
report each new `imageId`.

### Trace outlines (preferred)
Use `segment_objects(image_id, objects)` when the server has it: give each object a `label` and a rough
`box` (x, y, width, height; loose is fine, it only has to contain the object) or a `point` on it, and
Segment Anything traces the exact edges and saves polygons. Use `model: "medsam"` for medical images
(CT, MRI, X-ray, ultrasound, endoscopy, pathology), with a box for every object. Check `failed` in the
result, and report the scores: an outline can be precise but of the wrong object if the box was off.

### Add annotations
Use `add_annotations(image_id, [...])` for pins, arrows and shapes the user specifies, or when
`segment_objects` isn't available. It keeps existing annotations and returns `addedIds`.
- Send all annotations for one image in a single call.
- When you place shapes from your own view of the image (not coordinates the user or another
  tool gave you), set `confidence: "low"` or `"medium"`, say that the shapes are approximate,
  and ask the user to review them in the web app. Never present estimated boundaries as precise.
- Reuse the exact label strings already used in the dataset. Check with `list_images` +
  `get_image` before inventing a new class name.

### Clean up
1. Audit: for the images in scope, collect every label with its count, and flag:
   - near-duplicate labels (case, spacing, plurals, typos, synonyms such as `Tumor` / `tumour`)
   - polygons with fewer than 3 distinct points or zero area, and circles with radius 0
   - coordinates outside the image, and exact duplicate shapes
   - rectangles / pins / arrows that look like they were meant as training labels
2. Propose a plan (label mapping, shapes to fix, convert or remove) and **wait for the user's
   confirmation** before changing anything.
3. Apply it:
   - deleting only: `remove_annotations(image_id, ids)`
   - renaming labels, converting rectangles to polygons, clipping coordinates:
     `replace_annotations` with the **complete** corrected list. Start from `get_image`'s
     annotations and carry over every field (`type`, `label`, coordinates, `color`, `notes`,
     `confidence`). Anything you leave out is lost. Annotation ids and `createdAt` are
     regenerated; mention that if the user tracks ids.
4. Call `get_image` again and report the before → after counts.

### Export a dataset
`export_annotations(image_id, format)` exports **one image**, and numbers classes by first
appearance **in that image**. Class ids and COCO category ids therefore differ between images.
Never concatenate per-image exports by hand.

1. Optionally fix the class order with the user (for example to match an existing model).
2. For each image, call `export_annotations` and save the raw tool result to a JSON file:
   - COCO: any file name (`file_name` is inside the export)
   - YOLO: name it after the image file's stem, so `scan_01.png` → `scan_01.json`
3. Merge with the bundled script:
   ```bash
   python scripts/merge_exports.py coco exports/*.json -o dataset/annotations.json
   python scripts/merge_exports.py yolo exports/*.json -o dataset --classes "tumor,stroma"
   ```
   `--classes` pins the order; labels it doesn't list are appended. For YOLO it writes
   `labels/<stem>.txt`, `classes.txt` and `data.yaml`. Copy the image files into
   `dataset/images/` yourself (download them from the `get_image` viewing URLs if needed).
4. Report the image, annotation and class counts, and how many annotations were skipped for
   being non-exportable types.

YOLO output is **detection** format (`class xc yc w h`, normalized). For segmentation masks use
COCO (`segmentation` polygons).

## Rules

- `replace_annotations` overwrites everything on an image. Show what will change and get
  confirmation first. Prefer `add_annotations` / `remove_annotations` when they're enough.
- There is no tool for deleting images, on purpose. If asked, explain that it has to be done
  in the web app.
- Batch per image, and don't loop over hundreds of images without telling the user the scope.
- Tool errors read `METHOD /path failed (HTTP status): message`. Report the message. Validation
  errors (for example `polygon needs at least 3 points`) mean fixing the input, not retrying it.

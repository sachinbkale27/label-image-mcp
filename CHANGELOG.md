# Changelog

All notable changes to Histopathology Annotation MCP (`label-image-mcp`). Versions follow
[semantic versioning](https://semver.org): until 1.0, a minor version (0.x.0) may change or remove tools,
and the change is listed here under **Changed** or **Removed**.

## 0.7.3 (2026-10-25)

### Added
- `share_image`: a 7-day link that opens the image and its annotations in the Image Label web app, in
  anyone's browser, as their own copy to edit, save and export.

## 0.7.2 (2026-10-25)

### Fixed
- Usage counts on hosted servers: the version comes from pyproject.toml when the package isn't installed, and
  the client from the HTTP caller when there is no MCP handshake (stateless mode).

## 0.7.1 (2026-10-25)

### Added
- Each call to the service names the AI client and server version (`User-Agent: label-image-mcp/<version>
  (<client>)`), so usage can be counted by client; nothing from the conversation is sent. See "Your data".
- A JOSS paper draft (`paper/`).

## 0.7.0 (2026-10-25)

### Added
- `analyze_slide(coverage="full")`: every tissue tile of a slide (up to 3,000), analysed by parallel workers,
  with the overview tinted as a nuclei-density heatmap and the full per-tile table (`tiles_url`). Cell types
  are opt-in for full coverage.

## 0.6.1 (2026-10-25)

### Changed
- `classify_cells` explains that it needs a 20x or 40x image; coarser images are refused by the service.

## 0.6.0 (2026-10-25)

### Added
- `classify_cells`: types detected nuclei as tumour, immune, connective tissue, dead or normal epithelial
  cells with HoVer-Net (PanNuke weights, CC BY-NC-SA 4.0: research and education only), in the background.
  Nuclei are relabelled by type; the result has counts, percentages and the immune-to-tumour ratio. Images
  coarser than about 20x are refused: enlarging them can't add the detail the model reads.
- Whole slides get cell types too (`analyze_slide(cell_types=True)`, the default): per tile and for the slide,
  in the densest-tile images and in the QuPath GeoJSON. With cell types on, `max_tiles` defaults to 12.

## 0.5.0 (2026-10-24)

### Added
- Whole-slide images: `upload_slide` (streamed, up to 5 GB), `upload_slide_from_url`, `analyze_slide` and
  `get_slide_analysis`. Aperio SVS (JPEG and JPEG 2000), Hamamatsu NDPI, pyramidal TIFF and OME-TIFF.
  A background analysis samples up to 48 tissue tiles at 0.25 µm/px (40x) and reports nuclei per mm²,
  nuclear area in µm², size variation, enlarged nuclei and quality per tile.
- After an analysis the slide's image is its overview with the analysed tiles marked, the three densest
  tiles are saved as images with their nuclei outlined, and every nucleus is in a QuPath GeoJSON in
  slide coordinates.
- `export_annotations(format="qupath")` on a slide gives its regions in slide coordinates and the link to
  the nuclei GeoJSON.
- A benchmark on NuInsSeg (benchmarks/nuinsseg.py) with results in the README; CITATION.cff.

### Changed
- `detect_nuclei` finds about two thirds more nuclei (F1 0.47 to 0.64 on held-out NuInsSeg images): a
  detection threshold of 0.45 instead of StarDist's 0.69, and the image is resized so nuclei come out at
  the size the model handles best (from `microns_per_pixel` when given, otherwise from the nuclei's own
  size) instead of keeping whichever enlargement found most. It is also several times faster.

## 0.4.0 (2026-10-24)

### Added
- `view_image`: the picture itself, with a grid in image pixels, optionally with the annotations drawn on
  or zoomed into a region, so the assistant can place boxes and check its outlines.
- `export_dataset`: many images as one COCO, YOLO or QuPath dataset with one class list; `output_dir`
  writes the files.
- `detect_nuclei(microns_per_pixel=...)`: sizes in µm² and density per mm².
- Every tool has a title and read-only / destructive annotations; server instructions; three prompts
  (Analyse a histology slide, Outline objects in an image, Build a training dataset).

### Changed
- `warm_up(model)` with `sam`, `medsam` or `nuclei`.

### Removed
- `warm_up_segmentation` and `warm_up_pathology` (use `warm_up`).
- `upload_image` on hosted servers, where it would read the server's own disk.

## 0.3.1 to 0.3.6 (2026-10-23 to 2026-10-24)

### Changed
- Renamed to Histopathology Annotation MCP; the package and registry names stay `label-image-mcp`.
- Descriptions and links point to the browser version at https://agent.sachin-kale.com/#annotate.

## 0.3.0

### Added
- Histopathology tools: `detect_nuclei` (StarDist), `normalize_stain` (Macenko), `check_image_quality`
  (HistoQC-style checks) and QuPath GeoJSON export.

## 0.2.0 (2026-10-22)

### Added
- `segment_objects`: exact outlines from rough boxes or points with Segment Anything, or MedSAM for
  medical images.
- `upload_image_from_url`.

## 0.1.0 (2026-10-16)

### Added
- First release: list, read, upload and annotate images in Image Label, and export COCO or YOLO; a
  LangGraph agent.

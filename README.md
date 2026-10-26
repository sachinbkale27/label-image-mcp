# Histopathology Annotation MCP

<!-- mcp-name: io.github.sachinbkale27/label-image-mcp -->

**Image Annotation, including Histopathology slides; From H&E slide to training data in one conversation: nuclei, outlines and slide QC, exported to QuPath, COCO or YOLO. No GPU needed.**

[![PyPI](https://img.shields.io/pypi/v/label-image-mcp)](https://pypi.org/project/label-image-mcp/)
[![CI](https://github.com/sachinbkale27/label-image-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/sachinbkale27/label-image-mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![MCP Registry](https://img.shields.io/badge/MCP%20Registry-io.github.sachinbkale27%2Flabel--image--mcp-6f42c1)](https://registry.modelcontextprotocol.io/v0/servers?search=io.github.sachinbkale27/label-image-mcp)
[![M8ven Score](https://m8ven.ai/badge/mcp/sachinbkale27-label-image-mcp-1v4i0w?v=953b1eb82140527ad1bb9e6422dc6df3)](https://m8ven.ai/mcp/sachinbkale27-label-image-mcp-1v4i0w?s=readme)

![Loading an H&E breast cancer slide, outlining every nucleus with StarDist, Diagnose: an AI interpretation written as a report, problematic regions outlined on the image and the agent's report, then Ask AI: a question about the results answered with an illustration labelled from the answer](https://raw.githubusercontent.com/sachinbkale27/label-image-mcp/main/docs/demo.gif)

**Accuracy:** see [Benchmark](#benchmark). **Your data:** see [Your data](#your-data). **Changes:** see [CHANGELOG](CHANGELOG.md).

An [MCP](https://modelcontextprotocol.io) server for annotating histopathology and other images in
[Image Label](https://labels.sachin-kale.com), a free annotation tool. It lets Claude and other AI
assistants upload images, outline every nucleus in an H&E slide automatically (StarDist), trace regions
from rough boxes (Segment Anything, or MedSAM for medical images), check slide quality, normalize stain
colours, clean up annotations, and export training data as COCO JSON, YOLO labels, YOLO classes or QuPath GeoJSON. A LangGraph agent is included.

**Try it in the browser:** [https://agent.sachin-kale.com/#annotate](https://agent.sachin-kale.com/#annotate): upload a slide and annotate it, detect nuclei, check
its quality or get an AI interpretation, with no setup.

Published as `label-image-mcp` (PyPI) and `io.github.sachinbkale27/label-image-mcp` (MCP Registry).

Use cases and setup guide: https://labels.sachin-kale.com/mcp

```
LangGraph agent (label-image-agent)      Claude Desktop / Claude Code / any MCP client
              │  langchain-mcp-adapters             │
              └──────────────┬──────────────────────┘
                             ▼
                    label-image-mcp (stdio or HTTP)
                             │
                             ▼
                 Image Label API  (/api/images ...)
```

## Account (optional)

**No account needed to try it.** Without credentials the server works as a **guest**: it gets a guest
identity on first use, saves it in `~/.config/label-image-mcp/`, and keeps using it, so your images are
there next time.

| | Guest (no sign-up) | Account ([request one, free](https://labels.sachin-kale.com/contact)) |
|---|---|---|
| Images kept | 7 days, up to 50 | 90 days |
| Upload size | 20 MB per image | 20 MB per image; whole slides up to 5 GB |
| Analyses (nuclei, outlines, cell types, …) | 200 a day | No daily cap |
| Whole slides | No | Yes |

The tools see, change and upload only the images of the identity they use (guest or account). Set
`LABEL_USERNAME` and `LABEL_PASSWORD` to use an account, or `LABEL_GUEST=0` to turn guests off.

## Install

**Claude Desktop:** download `label-image-mcp.mcpb` from the
[latest release](https://github.com/sachinbkale27/label-image-mcp/releases/latest) and open it. Leave
username and password empty to start as a guest.

**Claude Code:**

```bash
claude mcp add label-image -e LABEL_USERNAME=... -e LABEL_PASSWORD=... -- uvx label-image-mcp
```

**Any MCP client** (needs [uv](https://docs.astral.sh/uv/)):

```json
{
  "mcpServers": {
    "label-image": {
      "command": "uvx",
      "args": ["label-image-mcp"],
      "env": { "LABEL_USERNAME": "...", "LABEL_PASSWORD": "..." }
    }
  }
}
```

## Benchmark

`detect_nuclei` checked against expert outlines on [NuInsSeg](https://zenodo.org/records/10518968) (CC BY 4.0):
H&E images of 31 human and mouse tissues, 30,698 hand-segmented nuclei. The detection threshold was chosen on
the even-numbered images of each tissue; these results are on the other 332 images (15,701 nuclei), which were
not used for any choice. A nucleus counts as found when its outline overlaps the expert's by IoU ≥ 0.5.

| Settings | Precision | Recall | F1 | Panoptic quality | Pixel Dice |
|---|---|---|---|---|---|
| 0.5.0, resized by nucleus size (default) | 0.78 | 0.55 | **0.64** | 0.48 | 0.65 |
| 0.5.0, with `microns_per_pixel` | 0.81 | 0.52 | **0.64** | 0.47 | 0.66 |
| Before 0.5.0 | 0.81 | 0.33 | 0.47 | 0.35 | 0.43 |

- Better on human tissue (F1 0.67) than mouse (0.50 to 0.53). Best on pancreas, cerebellum, placenta and
  tongue (0.73 to 0.81); weakest on umbilical cord, brain, mouse heart and bone (0.13 to 0.34), where nuclei
  are sparse, thin or faint. Per-tissue results are in [benchmarks/results](benchmarks/results).
- The model is StarDist's pretrained H&E model, not trained on NuInsSeg: models trained on a dataset score
  higher on it. Read these as what to expect on new images, and review outlines before training on them.
- Reproduce: [benchmarks/nuinsseg.py](benchmarks/nuinsseg.py) runs the same method locally.
- Cell types (`classify_cells`) are not benchmarked here yet: NuInsSeg has no cell-type labels. HoVer-Net's
  published results on PanNuke are the reference until then.

## Your data

What happens to images you upload through this server (hosted on AWS in us-east-1):

- **Private to your account.** Only you can list, read or change your images; the storage bucket blocks all
  public access, and viewing links are signed and expire after 24 hours.
- **Deleted after 90 days**, automatically: files, slides, annotations and analysis results. Deleting an image
  removes it straight away.
- **Encrypted at rest** (AES-256) and in transit (HTTPS).
- **Not used to train models**, and not sent to other AI services by this server: StarDist, Segment Anything
  and MedSAM run on the service itself. Your AI assistant sees what its tools return (`view_image` returns
  the picture), under that assistant's own terms.
- **Usage counts.** Each call to the service names the AI client (for example `claude-ai/0.1.0`, from the MCP
  handshake) and the server version, so usage can be counted by client; the account is logged only as a hash.
  The service also logs the approximate place of each call (city, region, country and time zone, from its CDN)
  and the local hour; IP addresses are not kept. Nothing from your conversation is sent.
- **No patient identifiers, please.** Remove names, record numbers and slide labels (crop the label area or
  use a de-identified copy) before uploading. Results are for research and education, not diagnosis.

## Agent skill

[`skills/label-image/SKILL.md`](skills/label-image/SKILL.md) teaches the agent when and how to
use these tools: which annotation types end up in exports, cleaning up labels safely, and
merging per-image exports into one COCO or YOLO dataset with consistent class ids (bundled
`scripts/merge_exports.py`). It follows the open SKILL.md format, so it works in Claude Code and
other agents that support skills.

```bash
# Claude Code: install for all your projects (or copy into a project's .claude/skills/)
git clone --depth 1 https://github.com/sachinbkale27/label-image-mcp /tmp/label-image-mcp
mkdir -p ~/.claude/skills && cp -r /tmp/label-image-mcp/skills/label-image ~/.claude/skills/
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LABEL_USERNAME` / `LABEL_PASSWORD` | unset | Account credentials. Without them the tools see no images |
| `LABEL_API_URL` | `https://labels.sachin-kale.com/api` | API base URL, for another deployment or `sam local start-api` |

## Tools

| Tool | What it does |
|---|---|
| `list_images(category?)` | Image summaries with annotation counts |
| `get_image(image_id)` | Metadata, annotations and a signed viewing URL |
| `view_image(image_id, with_annotations?, region?, grid?, max_size?)` | The picture itself, for the assistant to look at: scaled down, with a grid labelled in image pixels, optionally with the annotations drawn on, or zoomed into a region. Use it to place boxes for `segment_objects` and to check outlines |
| `update_image_metadata(image_id, name?, category?, sub_category?, description?)` | Edit image details |
| `add_annotations(image_id, annotations)` | Append pins, polygons, circles, rectangles or arrows |
| `replace_annotations(image_id, annotations)` | Replace all annotations |
| `remove_annotations(image_id, annotation_ids)` | Remove annotations by id |
| `upload_image(path, name?, category?, description?)` | Upload a local image file |
| `upload_image_from_url(url, name?, category?, description?)` | Download an image from an https URL (up to 20 MB) and upload it |
| `segment_objects(image_id, objects, model?, save?)` | Trace exact outlines with Segment Anything from a rough box or a point per object, and save them as polygons. `model`: `sam` (photos) or `medsam` (CT, MRI, X-ray, ultrasound, endoscopy, pathology; boxes only) |
| `detect_nuclei(image_id, max_nuclei?, mark_enlarged?, replace_previous?, save?, microns_per_pixel?)` | Find and outline every nucleus in an H&E image with StarDist, automatically (no boxes). Returns count, density, median area and size variation; nuclei over 2.5x the median area are labelled `enlarged nucleus`. With `microns_per_pixel` (often 0.25 at 40x, 0.5 at 20x) also in µm² and nuclei per mm² |
| `classify_cells(image_id, wait_seconds?)` | Types the nuclei `detect_nuclei` saved as **tumour, immune, connective tissue, dead or normal epithelial** cells (HoVer-Net trained on PanNuke, 19 tissues), relabelling each nucleus and returning counts, percentages and the immune-to-tumour ratio. Runs in the background, 30 to 90 s. Needs a 20x or 40x image (coarser ones are refused). **Research and education only** (CC BY-NC-SA 4.0 weights) |
| `normalize_stain(image_id)` | Macenko stain normalization: a copy with standard H&E colours, saved as a new image |
| `check_image_quality(image_id)` | HistoQC-style checks (tissue fraction, blur, marker pen, exposure, faint staining) with a verdict and warnings |
| `warm_up(model?)` | Start loading a model without waiting: `sam`, `medsam` or `nuclei` (each takes a minute or two after a quiet spell) |
| `share_image(image_id)` | A link (valid 7 days) that opens the image and its annotations in the Image Label web app in anyone's browser, copied into their own workspace to edit and export; no account needed |
| `export_annotations(image_id, format)` | COCO JSON or YOLO, same output as the web app's export, or `qupath`: GeoJSON for QuPath's File > Import objects (StarDist nuclei come in as detections) |
| `export_dataset(format?, category?, image_ids?, classes?, output_dir?)` | Many images (a category, or up to 100 ids) as ONE dataset with one class list: a COCO file, YOLO classes plus a label file per image, or a QuPath GeoJSON per image. `output_dir` writes the files (local server only) |

Every tool is marked read-only or destructive (`replace_annotations`, `remove_annotations`), so clients can
ask before the destructive ones. The server also offers three prompts: **Analyse a histology slide**,
**Outline objects in an image** and **Build a training dataset**.

### Whole slides

| Tool | What it does |
|---|---|
| `upload_slide(path, name?, category?, analyze?, max_tiles?)` | Upload a whole-slide file (.svs, .ndpi, pyramidal .tif/.tiff, up to 5 GB; streamed, never loaded into memory) and start its analysis. Local server only |
| `upload_slide_from_url(url, ...)` | The same from an https URL, copied straight into storage |
| `analyze_slide(image_id, coverage?, max_tiles?, cell_types?, restart?, wait_seconds?)` | Background analysis: finds the tissue, then runs the quality check, StarDist nuclei and cell types on tiles of 256 × 256 µm at 40x. `coverage: "sample"` (default) takes up to 48 tiles spread over the tissue, with cell types; `"full"` analyses **every tissue tile** (up to 3,000, about 200 mm²) in parallel and tints the overview as a nuclei-density heatmap (cell types opt-in there: about 15× slower). Waits up to `wait_seconds` and reports progress |
| `get_slide_analysis(image_id)` | Status, or the results: nuclei per mm², median nuclear area in µm², size variation, enlarged nuclei and cell-type percentages across the slide, per-tile values and quality |

After the analysis the slide's image in the library is its overview, with the analysed tiles marked (the
three densest in red; with full coverage, a density heatmap and the ten densest tiles), so `view_image` and
the web app show it. The three densest tiles are saved as
images of their own, with their nuclei outlined, for a close look or further annotation. Every nucleus
is in a QuPath GeoJSON in slide coordinates: open the slide in QuPath, then File > Import objects.
Slides are read with tifffile (Aperio SVS including JPEG 2000, Hamamatsu NDPI, pyramidal TIFF and
OME-TIFF), decoding only the tiles needed. Other images are up to 20 MB.

There is deliberately no tool for deleting images.

`segment_objects` calls the API's `POST /images/{id}/segment`, which runs SAM
([facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)) or MedSAM
([wanglab/medsam-vit-base](https://huggingface.co/wanglab/medsam-vit-base)) in a Lambda container
(`backend/segment`). After a quiet spell the model takes a while to start; the client retries then.
Results are for research and dataset building, not diagnosis.

The histopathology tools call `POST /images/{id}/pathology`, which runs
[StarDist](https://github.com/stardist/stardist) (BSD-3; its pretrained `2D_versatile_he` model was trained
on public H&E datasets, some under non-commercial terms, so check them before commercial use),
[torchstain](https://github.com/EIDOSLAB/torchstain) (MIT) and OpenCV quality measures in a second Lambda
container (`backend/pathology`). Microscope photos come at many magnifications, so `detect_nuclei` also
tries the image enlarged 2x and 3x and keeps the scale that finds the most nuclei.

## HTTP mode

`label-image-mcp --http [--host 127.0.0.1] [--port 8765]` serves Streamable HTTP at
`http://HOST:PORT/mcp`, so several clients can share one running server:

```bash
claude mcp add --transport http label-image http://127.0.0.1:8765/mcp
```

## Hosted mode

`label-image-mcp --http --hosted --host 0.0.0.0` is for running the server remotely, for example on
AWS Lambda behind the Lambda Web Adapter. It keeps no session state, answers with JSON, and requires
`Authorization: Bearer <token>` on every request; with no token configured it refuses all requests.

| Variable | Purpose |
|---|---|
| `MCP_AUTH_TOKEN` | The bearer token clients must send |
| `LABEL_CREDENTIALS_SECRET` | AWS Secrets Manager secret with `username`, `password` and `mcp_token`, instead of the variables above |
| `LABEL_UPLOAD_URL_HOSTS` | Comma-separated hosts `upload_image_from_url` may download from (recommended when hosted) |

## LangGraph agent

A terminal chat agent that uses these tools. The graph is `START → agent ⇄ tools`, with
conversation memory kept for the session.

```bash
uv tool install "label-image-mcp[agent]"
export ANTHROPIC_API_KEY=... LABEL_USERNAME=... LABEL_PASSWORD=...
label-image-agent
```

| Variable | Default | Purpose |
|---|---|---|
| `LABEL_AGENT_MODEL` | `anthropic:claude-sonnet-5-5` | Model as `provider:model`, e.g. `openai:gpt-4.1` or `ollama:llama3.1` (install `langchain-openai` / `langchain-ollama`) |
| `LABEL_MCP_URL` | unset | Connect to a running HTTP server instead of starting one |

## Development

```bash
uv sync --extra agent
uv run pytest
npx @modelcontextprotocol/inspector uv run label-image-mcp
```

In the Image Label app repo, `./start.sh` runs this server in HTTP mode next to the web app.
It reads credentials from `mcp-server/.env`. Releases are described in [PUBLISHING.md](PUBLISHING.md).

## License

MIT

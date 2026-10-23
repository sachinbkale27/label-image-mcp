# Image Label MCP server

<!-- mcp-name: io.github.sachinbkale27/label-image-mcp -->

An [MCP](https://modelcontextprotocol.io) server for [Image Label](https://labels.sachin-kale.com),
a free image annotation tool. It lets Claude and other AI assistants list and upload your images,
add or clean up annotations, and export COCO or YOLO training data. A LangGraph agent is included.

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

## Account

The tools need an Image Label account ([request one](https://labels.sachin-kale.com/contact)).
They see, change and upload only images saved to that account; without one they see no images.
Work kept only in your browser is not visible to them.

## Install

**Claude Desktop:** download `label-image-mcp.mcpb` from the
[latest release](https://github.com/sachinbkale27/label-image-mcp/releases/latest), open it, and
enter your username and password when asked.

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
| `update_image_metadata(image_id, name?, category?, sub_category?, description?)` | Edit image details |
| `add_annotations(image_id, annotations)` | Append pins, polygons, circles, rectangles or arrows |
| `replace_annotations(image_id, annotations)` | Replace all annotations |
| `remove_annotations(image_id, annotation_ids)` | Remove annotations by id |
| `upload_image(path, name?, category?, description?)` | Upload a local image file |
| `upload_image_from_url(url, name?, category?, description?)` | Download an image from an https URL (up to 20 MB) and upload it |
| `segment_objects(image_id, objects, model?, save?)` | Trace exact outlines with Segment Anything from a rough box or a point per object, and save them as polygons. `model`: `sam` (photos) or `medsam` (CT, MRI, X-ray, ultrasound, endoscopy, pathology; boxes only) |
| `export_annotations(image_id, format)` | COCO JSON or YOLO, same output as the web app's export |

There is deliberately no tool for deleting images.

`segment_objects` calls the API's `POST /images/{id}/segment`, which runs SAM
([facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)) or MedSAM
([wanglab/medsam-vit-base](https://huggingface.co/wanglab/medsam-vit-base)) in a Lambda container
(`backend/segment`). After a quiet spell the model takes a while to start; the client retries then.
Results are for research and dataset building, not diagnosis.

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

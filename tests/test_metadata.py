"""Keep the packaging files (pyproject, server.json, manifest.json, README) in sync."""

import json
import re
import tomllib
from pathlib import Path

from label_image_mcp import server

ROOT = Path(__file__).parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
SERVER_JSON = json.loads((ROOT / "server.json").read_text())
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def test_versions_match():
    version = PYPROJECT["version"]
    assert SERVER_JSON["version"] == version
    assert SERVER_JSON["packages"][0]["version"] == version
    assert MANIFEST["version"] == version


def test_registry_ownership_line_in_readme():
    # The MCP Registry checks the PyPI description for this exact line
    readme = (ROOT / "README.md").read_text()
    assert re.search(rf"mcp-name: {re.escape(SERVER_JSON['name'])}\s", readme)
    assert SERVER_JSON["packages"][0]["identifier"] == PYPROJECT["name"]


async def test_manifest_lists_every_tool():
    tools = {t.name for t in await server.mcp.list_tools()}
    assert {t["name"] for t in MANIFEST["tools"]} == tools

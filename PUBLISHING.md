# Publishing

The source of truth is `mcp-server/` in the private `label-image` repo. The public repo
`sachinbkale27/label-image-mcp` gets that folder's history via `git subtree`:

```bash
# from the label-image repo root
git subtree push --prefix mcp-server mcp-public main
```

(`mcp-public` is a git remote pointing at https://github.com/sachinbkale27/label-image-mcp.git)

## One-time setup

1. **PyPI trusted publisher:** on https://pypi.org/manage/account/publishing/ add a pending
   publisher: project `label-image-mcp`, owner `sachinbkale27`, repo `label-image-mcp`,
   workflow `release.yml`, environment `pypi`.
2. **GitHub environment:** in the public repo, Settings → Environments → create `pypi`.
3. **MCP Registry:** nothing to set up. The release workflow logs in with GitHub OIDC, which
   owns the `io.github.sachinbkale27/*` namespace.

## Release

1. Bump the version in `pyproject.toml`, `server.json` (both `version` fields) and
   `manifest.json`. `tests/test_metadata.py` fails if they disagree.
2. Push with `git subtree push` (above), then tag the public repo:
   `git tag v0.1.1 && git push mcp-public v0.1.1`, or create the tag on GitHub.
3. The `Release` workflow tests, publishes to PyPI, publishes `server.json` to the MCP Registry,
   and attaches the `.mcpb` Claude Desktop extension to a GitHub release.

## Directories (one-time, manual)

- **Glama** (https://glama.ai/mcp/servers): indexes public GitHub repos. `glama.json` claims the
  listing for `sachinbkale27`.
- **Smithery** (https://smithery.ai): add the server from the public GitHub repo on the site.
- **mcp.so** (https://mcp.so/submit) and **PulseMCP** (https://www.pulsemcp.com/submit): submit
  the GitHub URL. PulseMCP also imports from the MCP Registry.

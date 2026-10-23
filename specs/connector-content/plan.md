# CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Architecture

`pyproject.toml` declares `container-manager-mcp` as the connector identity. `create_mcp_server` resolves the packaged content of `container_manager_mcp` from it.

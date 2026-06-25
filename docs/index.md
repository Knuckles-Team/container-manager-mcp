# container-manager-mcp

Manage Docker, Docker Swarm, and Podman containers through a typed **MCP server**
and an integrated **A2A agent** — local or across a fleet of hosts over SSH.

!!! info "Official documentation"
    This site is the canonical reference for `container-manager-mcp`, maintained
    alongside every release.

[![PyPI](https://img.shields.io/pypi/v/container-manager-mcp)](https://pypi.org/project/container-manager-mcp/)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
[![License](https://img.shields.io/pypi/l/container-manager-mcp)](https://github.com/Knuckles-Team/container-manager-mcp/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/Knuckles-Team/container-manager-mcp)

## Overview

`container-manager-mcp` exposes the Docker and Podman engines as a compact set of
**action-routed MCP tools** an agent or IDE can call, plus a Pydantic-AI graph agent
for conversational orchestration. It provides:

- **`DockerManager` / `PodmanManager`** — typed Python facades over the Docker and
  Podman engines covering images, containers, volumes, networks, Swarm services, and
  Compose stacks.
- **Action-routed MCP tools** — consolidated, togglable tool modules (info, image,
  container, volume, network, swarm, system, compose) that minimize LLM context
  overhead.
- **Zero-script multi-host control** — route any operation to a remote host over a
  standard SSH channel, with no Docker TCP socket exposed, driven from a unified
  `inventory.yml` (`.yaml` legacy fallback).

## Explore the documentation

<div class="grid cards" markdown>

- :material-rocket-launch: **[Installation](installation.md)** — pip, source, extras, and the prebuilt Docker image.
- :material-server-network: **[Deployment](deployment.md)** — run the MCP and agent servers, Docker Compose, Caddy + Technitium.
- :material-console: **[Usage](usage.md)** — the MCP tools, the `DockerManager` Python API, and the CLI.
- :material-sitemap: **[Overview](overview.md)** — ecosystem role, enterprise readiness, and architecture.
- :material-lan-connect: **[Multi-Host](multi_host.md)** — zero-script Docker-over-SSH control plane.
- :material-tag-multiple: **[Concepts](concepts.md)** — the `CONCEPT:CMGR-*` registry.

</div>

## Quick start

```bash
pip install "container-manager-mcp[all]"
container-manager-mcp              # stdio MCP server (default transport)
```

Run it as a network server:

```bash
container-manager-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

See **[Installation](installation.md)** and **[Deployment](deployment.md)** for the
full matrix (PyPI extras, Docker image, all transports, the agent server, reverse
proxy, and DNS).

# Installation

`container-manager-mcp` is a standard Python package and a prebuilt container image.
Pick the path that matches how you want to run it.

## Requirements

- **Python 3.11 – 3.14**.
- A reachable **Docker** or **Podman** engine on the controller host (or on the
  remote hosts you target — see [Multi-Host](multi_host.md)).

## From PyPI (recommended)

```bash
pip install container-manager-mcp
```

### Optional extras

The base install pulls in the Docker engine client. Install the extra for what you
need:

| Extra | Install | Pulls in |
|---|---|---|
| `docker` | `pip install "container-manager-mcp[docker]"` | Docker engine client (`docker`) |
| `podman` | `pip install "container-manager-mcp[podman]"` | Podman engine client (`podman`) |
| `mcp` | `pip install "container-manager-mcp[mcp]"` | FastMCP MCP-server runtime (`agent-utilities[mcp]`) |
| `agent` | `pip install "container-manager-mcp[agent]"` | Pydantic-AI agent + Logfire tracing |
| `all` | `pip install "container-manager-mcp[all]"` | Everything above |

```bash
# Typical: run the MCP server with both engines and the agent
pip install "container-manager-mcp[all]"
```

## From source

```bash
git clone https://github.com/Knuckles-Team/container-manager-mcp.git
cd container-manager-mcp
pip install -e ".[all]"          # editable install with every extra
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install -e ".[all]"
uv run container-manager-mcp
```

## Prebuilt Docker image

A multi-stage, slim image is published on every release (installs
`container-manager-mcp[all]`, entrypoint `container-manager-mcp`):

```bash
docker pull knucklessg1/container-manager-mcp:latest

docker run --rm -i \
  -v /var/run/docker.sock:/var/run/docker.sock \
  knucklessg1/container-manager-mcp:latest        # stdio transport (default)
```

For an HTTP server with a published port and the agent server, see
[Deployment](deployment.md).

## Verify the install

```bash
container-manager-mcp --help
python -c "import container_manager_mcp; print(container_manager_mcp.__version__)"
```

## Next steps

- **[Deployment](deployment.md)** — run it as a long-lived MCP / agent server behind Caddy + DNS.
- **[Usage](usage.md)** — call the tools, the `DockerManager` API, and the CLI.
- **[Configuration](deployment.md#configuration-environment)** — every environment variable.

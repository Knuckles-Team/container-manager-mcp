# Setup

A step-by-step guide to installing `container-manager-mcp`, wiring it into an MCP
client, and pointing it at remote hosts. For the deep dive on the multi-host control
plane see [Multi-Host](multi_host.md).

## 1. Install

`container-manager-mcp` is published to PyPI and exposes two console scripts —
`container-manager-mcp` (the FastMCP server) and `container-manager-agent` (the A2A
agent). Pick the install style that fits:

### uvx (recommended — zero install, run on demand)

```bash
# Docker tools only (lightweight):
uvx --from container-manager-mcp container-manager-mcp --help

# With podman + kubernetes extras:
uvx --from "container-manager-mcp[all]" container-manager-mcp
```

`uvx` resolves the package and its `tunnel-manager` dependency from PyPI into an
ephemeral environment and runs the console script — nothing is installed globally.
Because the package name equals the script name, `uvx container-manager-mcp` also works.

### pip / uv (persistent install)

```bash
uv pip install "container-manager-mcp[all]"     # or: python -m pip install "container-manager-mcp[all]"
container-manager-mcp --help
```

### Container (pre-built image)

```bash
docker run --rm -i \
  -v $HOME/.config/agent-utilities:/root/.config/agent-utilities:ro \
  -v $HOME/.ssh:/root/.ssh:ro \
  example/container-manager-mcp@sha256:<digest>
```

## 2. Configuration

The server is configured by CLI flags or environment variables:

| Env var | Purpose |
|---|---|
| `CONTAINER_MANAGER_TYPE` | Runtime: `docker` (default), `podman`, `docker-swarm`. |
| `CONTAINER_MANAGER_HOST` | Default remote host alias (else operate on the local socket). |
| `CONTAINER_MANAGER_PODMAN_BASE_URL` | Explicit Podman socket URL when using podman. |
| `COMPOSE_TOOL` | `docker-compose` (default) or `podman-compose`. |

Transport/auth flags (`-t`, `-H`, `-p`, `--auth-type`, …) mirror the rest of the MCP
fleet — see `container-manager-mcp --help`.

## 3. Wire it into an MCP client

Add a server block to your client's `mcp_config.json` (Claude Code, Cursor, Windsurf,
Antigravity, …):

```jsonc
{
  "mcpServers": {
    "container-manager-mcp": {
      "command": "uvx",
      "args": ["--from", "container-manager-mcp", "container-manager-mcp"],
      "env": {
        "CONTAINER_MANAGER_TYPE": "docker"
      }
    }
  }
}
```

## 4. Quickstart (local socket)

With no `host` argument, tools operate on the local Docker/Podman socket. Invoke a tool
such as `list_containers` (no `host`) from your MCP client and you should see the
containers on the controller. Once that works, add remote hosts (next section).

## 5. Remote hosts

Remote operation uses **Docker-over-SSH** and the **same shared inventory** as
tunnel-manager — `~/.config/agent-utilities/inventory.yml` (`.yaml` legacy fallback). Define each host once
there (see the [tunnel-manager inventory tutorial](https://knuckles-team.github.io/tunnel-manager/inventory/)),
then pass the alias as the `host` argument to any tool (or set `CONTAINER_MANAGER_HOST`).
The full lifecycle, virtual-host namespacing via the multiplexer, and SSH key
requirements are documented in [Multi-Host](multi_host.md).

```bash
# One-time: ensure passwordless SSH to the hosts in your inventory.
tunnel-manager setup-all --parallel
```

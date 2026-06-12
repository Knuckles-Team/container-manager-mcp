# Deployment

<!-- BEGIN GENERATED: deployment-options -->
## Deployment Options

`container-manager-mcp` exposes its MCP server (console script `container-manager-mcp`) four ways. Pick the row that
matches where the server runs relative to your MCP client, then copy the matching
`mcp_config.json` below. Add the service-connection environment variables documented in the **Configuration** section.

| # | Option | Transport | Where it runs | `mcp_config.json` key |
|---|--------|-----------|---------------|------------------------|
| 1 | stdio | `stdio` | client launches a subprocess | `command` |
| 2 | Streamable-HTTP (local) | `streamable-http` | a local network port | `command` or `url` |
| 3 | Local container / uv | `stdio` or `streamable-http` | Docker / Podman / uv on this host | `command` or `url` |
| 4 | Remote URL | `streamable-http` | a remote host behind Caddy | `url` |

### 1. stdio (local subprocess)

The client launches the server over stdio via `uvx` — best for local IDEs
(Cursor, Claude Desktop, VS Code):

```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "command": "uvx",
      "args": ["--from", "container-manager-mcp", "container-manager-mcp"]
    }
  }
}
```

### 2. Streamable-HTTP (local process)

Run the server as a long-lived HTTP process:

```bash
uvx --from container-manager-mcp container-manager-mcp --transport streamable-http --host 0.0.0.0 --port 8000
curl -s http://localhost:8000/health        # {"status":"OK"}
```

Then either let the client launch it:

```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "command": "uvx",
      "args": ["--from", "container-manager-mcp", "container-manager-mcp", "--transport", "streamable-http", "--port", "8000"],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000"
      }
    }
  }
}
```

…or connect to the already-running process by URL:

```json
{
  "mcpServers": {
    "container-manager-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

### 3. Local container / uv

**(a) Launch a container directly from `mcp_config.json`** (stdio over the container —
no ports to manage). Swap `docker` for `podman` for a daemonless runtime:

```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "TRANSPORT=stdio",
        "knucklessg1/container-manager-mcp:latest"
      ]
    }
  }
}
```

**(b) Run a local streamable-http container, then connect by URL:**

```bash
docker run -d --name container-manager-mcp -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e PORT=8000 \
  knucklessg1/container-manager-mcp:latest
# or, from a clone of this repo:
docker compose -f docker/mcp.compose.yml up -d
```

```json
{
  "mcpServers": {
    "container-manager-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

**(c) From a local checkout with `uv`:**

```bash
uv run container-manager-mcp --transport streamable-http --port 8000
```

### 4. Remote URL (deployed behind Caddy)

When the server is deployed remotely (e.g. as a Docker service) and published through
Caddy on the internal `*.arpa` zone, connect with the `"url"` key — no local process or
image required:

```json
{
  "mcpServers": {
    "container-manager-mcp": { "url": "http://container-manager-mcp.arpa/mcp" }
  }
}
```

Caddy reverse-proxies `http://container-manager-mcp.arpa` to the container's `:8000`
streamable-http listener; `http://container-manager-mcp.arpa/health` returns
`{"status":"OK"}` when the service is live.
<!-- END GENERATED: deployment-options -->

This page covers running `container-manager-mcp` as a long-lived server: the
transports, a Docker Compose stack, the companion **agent server**, putting it behind
a Caddy reverse proxy, and giving it a DNS name with Technitium.

> `container-manager-mcp` ships **two** console scripts: the MCP server
> (`container-manager-mcp`) and an A2A graph agent (`container-manager-agent`). The
> MCP server is a typed, deterministic tool surface; the agent server adds a
> conversational Pydantic-AI front end and Agent Web UI.

## Run the MCP server

The transport is selected with `--transport` (or the `TRANSPORT` env var):

=== "stdio (default)"

    ```bash
    container-manager-mcp
    ```
    For IDE / desktop MCP clients that launch the server as a subprocess.

=== "streamable-http"

    ```bash
    container-manager-mcp --transport streamable-http --host 0.0.0.0 --port 8000
    ```
    A network server with a `/health` endpoint and `/mcp` route.

=== "sse"

    ```bash
    container-manager-mcp --transport sse --host 0.0.0.0 --port 8000
    ```

Health check (HTTP transports):

```bash
curl -s http://localhost:8000/health        # {"status":"OK"}
```

## Configuration (environment)

`container-manager-mcp` is configured entirely from the environment. The **required**
runtime set:

| Var | Default | Meaning |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address for HTTP transports |
| `PORT` | `8000` | Listen port for HTTP transports |
| `TRANSPORT` | `stdio` | `stdio`, `streamable-http`, or `sse` |
| `ENABLE_OTEL` | `True` | Export OpenTelemetry traces / metrics |
| `EUNOMIA_TYPE` | `none` | Authorization mode: `none`, `embedded`, `remote` |

Tool modules are individually togglable (each defaults to `True`):
`INFOTOOL`, `IMAGETOOL`, `CONTAINERTOOL`, `VOLUMETOOL`, `NETWORKTOOL`, `SWARMTOOL`,
`SYSTEMTOOL`, `COMPOSETOOL`, `MISCTOOL`. The complete variable set, grouped by area,
is documented in
[`.env.example`](https://github.com/Knuckles-Team/container-manager-mcp/blob/main/.env.example).
Copy it to `.env` and adjust only what you use.

## Docker Compose

The repo ships [`docker/mcp.compose.yml`](https://github.com/Knuckles-Team/container-manager-mcp/blob/main/docker/mcp.compose.yml).
It reads a sibling `.env` and publishes the HTTP server on `:8000`:

```yaml
services:
  container-manager-mcp-mcp:
    image: knucklessg1/container-manager-mcp:latest
    container_name: container-manager-mcp-mcp
    hostname: container-manager-mcp-mcp
    restart: always
    env_file:
      - ../.env
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports:
      - "8000:8000"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
cp .env.example .env          # then edit values
docker compose -f docker/mcp.compose.yml up -d
docker compose -f docker/mcp.compose.yml logs -f
```

The server needs access to a container engine. Mount the host Docker socket
(`/var/run/docker.sock`) as shown, or point it at a remote host over SSH via the
[multi-host inventory](multi_host.md).

## Agent server

The companion agent exposes a Pydantic-AI graph agent over the Agent Control Protocol
(ACP) with an optional Agent Web UI. It is launched with the `container-manager-agent`
console script and listens on **port 9019** by default, connecting to the MCP server
through `MCP_URL`.

```bash
container-manager-agent --provider openai --model-id gpt-4o
```

The repo ships [`docker/agent.compose.yml`](https://github.com/Knuckles-Team/container-manager-mcp/blob/main/docker/agent.compose.yml),
which runs the MCP server and the agent together and wires the agent to the MCP
server by container name:

```yaml
  container-manager-mcp-agent:
    image: knucklessg1/container-manager-mcp:latest
    container_name: container-manager-mcp-agent
    restart: always
    depends_on:
      - container-manager-mcp-mcp
    env_file:
      - ../.env
    command: [ "container-manager-agent" ]
    environment:
      - HOST=0.0.0.0
      - PORT=9019
      - MCP_URL=http://container-manager-mcp-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
    ports:
      - "9019:9019"
```

```bash
docker compose -f docker/agent.compose.yml up -d
```

## Behind a Caddy reverse proxy

Expose the HTTP server on a hostname with automatic TLS. Add to your `Caddyfile`:

```caddy
# Internal (self-signed) — homelab .arpa zone
container-manager-mcp.arpa {
    tls internal
    reverse_proxy container-manager-mcp-mcp:8000
}
```

```caddy
# Public — automatic Let's Encrypt
container-manager-mcp.example.com {
    reverse_proxy container-manager-mcp-mcp:8000
}
```

Reload Caddy:

```bash
docker compose -f services/caddy/compose.yml exec caddy caddy reload --config /etc/caddy/Caddyfile
```

## DNS with Technitium

Point the hostname at the host running Caddy. Via the Technitium API:

```bash
curl -s "http://technitium.arpa:5380/api/zones/records/add" \
  --data-urlencode "token=$TECHNITIUM_DNS_TOKEN" \
  --data-urlencode "domain=container-manager-mcp.arpa" \
  --data-urlencode "zone=arpa" \
  --data-urlencode "type=A" \
  --data-urlencode "ipAddress=10.0.0.10" \
  --data-urlencode "ttl=3600"
```

…or add an **A record** `container-manager-mcp.arpa → <caddy-host-ip>` in the
Technitium web console (`http://technitium.arpa:5380`). The ecosystem
[`technitium-dns-mcp`](https://knuckles-team.github.io/technitium-dns-mcp/) automates
this as a tool.

## Register with an MCP client

Add to your client's `mcp_config.json` (multiplexer nickname `cnt`):

```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "command": "uv",
      "args": ["run", "container-manager-mcp"],
      "env": {
        "CONTAINERTOOL": "True",
        "IMAGETOOL": "True",
        "COMPOSETOOL": "True",
        "SWARMTOOL": "True"
      }
    }
  }
}
```

For a remote HTTP server, point the client at
`http://container-manager-mcp.arpa/mcp` instead.

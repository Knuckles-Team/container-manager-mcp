# Deployment

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

# Container Manager Mcp
## CLI or API | MCP | Agent

![PyPI - Version](https://img.shields.io/pypi/v/container-manager-mcp)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
![PyPI - Downloads](https://img.shields.io/pypi/dd/container-manager-mcp)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/container-manager-mcp)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/container-manager-mcp)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/container-manager-mcp)
![PyPI - License](https://img.shields.io/pypi/l/container-manager-mcp)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/container-manager-mcp)
![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/container-manager-mcp)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/container-manager-mcp)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/container-manager-mcp)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/container-manager-mcp)
![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/container-manager-mcp)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/container-manager-mcp)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/container-manager-mcp)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/container-manager-mcp)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/container-manager-mcp)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/container-manager-mcp)

*Version: 2.0.0*

> **Documentation** — Installation, deployment, usage across the API, CLI, MCP, and
> A2A agent interfaces, and the multi-host control plane are maintained in the
> [official documentation](https://knuckles-team.github.io/container-manager-mcp/).

---

## Overview

**Container Manager Mcp** is a production-grade Agent and Model Context Protocol (MCP) server designed to interface directly with Container Manager - manage Docker, Docker Swarm, and Podman containers. MCP+A2A Servers Out of the Box!.

---

## Key Features

- **Consolidated Action-Routed MCP Tools:** Minimizes token overhead and eliminates tool bloat in LLM contexts by grouping methods into optimized, togglable tool modules.
- **Enterprise-Grade Security:** Comprehensive support for Eunomia policies, OIDC token delegation, and granular execution context tracking.
- **Integrated Graph Agent:** Built-in Pydantic AI agent supporting the Agent Control Protocol (ACP) and standard Web interfaces (AG-UI).
- **Native Telemetry & Tracing:** Out-of-the-box OpenTelemetry exports and native Langfuse tracing.

---

## Multi-Host & Zero-Script Remote Docker Orchestration

`container-manager-mcp` allows a single master instance of the MCP server on your controller to route container and volume operations securely to remote hosts over SSH standard tunneling.

- **Unified Inventory**: Connection endpoints are loaded dynamically from the XDG shared inventory at `~/.config/agent-utilities/inventory.yml` (`.yml` preferred; a legacy `inventory.yaml` is still read when no `.yml` exists).
- **Zero TCP Socket Exposure**: Operations route directly over the standard SSH channel securely, removing the need to expose Docker socket TCP ports.

> **Shared inventory:** the `cm_*` host aliases you pass as `host` come from the **same**
> `inventory.yml` used by **tunnel-manager** — define your fleet once. Create and validate
> it with `tunnel-manager inventory init` / `tunnel-manager inventory doctor`. See
> tunnel-manager's [Inventory guide](https://knuckles-team.github.io/tunnel-manager/inventory/)
> for the full schema, template, and override options.

To configure and utilize the multi-host remote routing, see the detailed [Multi-Host Architecture Guide](docs/multi_host.md).

---

## CLI or API

This agent wraps the Container Manager - manage Docker, Docker Swarm, and Podman containers. MCP+A2A Servers Out of the Box! API. You can interact with it programmatically or via its integrated execution entrypoints.

Detailed instructions on how to use the underlying API wrappers, extended schema bindings, and developer SDK references are maintained in [docs/index.md](docs/index.md).

---

## MCP

This server utilizes dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools

_Auto-generated — do not edit (synced by the `mcp-readme-table` pre-commit hook)._

<!-- MCP-TOOLS-TABLE:START -->

#### Condensed action-routed tools (default — `MCP_TOOL_MODE=condensed`)

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `cm_compose_operations` | `COMPOSETOOL` | Manage docker-compose or podman-compose operations. |
| `cm_container_operations` | `CONTAINERTOOL` | Manage container operations. |
| `cm_image_operations` | `IMAGETOOL` | Manage container images. |
| `cm_info_operations` | `INFOTOOL` | Manage container manager info operations. |
| `cm_list_hosts` | `INVENTORYTOOL` | List the host aliases you can pass as ``host`` to any cm_* operation |
| `cm_network_operations` | `NETWORKTOOL` | Manage network operations. |
| `cm_swarm_operations` | `SWARMTOOL` | Manage swarm operations. |
| `cm_system_operations` | `SYSTEMTOOL` | Manage container manager system operations. |
| `cm_volume_operations` | `VOLUMETOOL` | Manage volume operations. |
| `trace_port_namespace` | `MISCTOOL` | Locate the container actively using/mapping the specified port on the target host. |

_10 action-routed tool(s) (default) · 0 verbose 1:1 tool(s). Each is enabled unless its `<DOMAIN>TOOL` toggle is set false; `MCP_TOOL_MODE` selects the surface (`condensed` default · `verbose` 1:1 · `both`). Auto-generated — do not edit._
<!-- MCP-TOOLS-TABLE:END -->

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/mcp.md](docs/mcp.md).

### Dynamic Tool Selection & Visibility

This MCP server supports dynamic toolset selection and visibility filtering at runtime. This allows you to restrict the set of exposed tools in order to prevent blowing up the LLM's context window.

You can configure tool filtering via multiple input channels:

- **CLI Arguments:** Pass `--tools` or `--toolsets` (or their disabled counterparts `--disabled-tools` and `--disabled-toolsets`) during startup.
- **Environment Variables:** Define standard environment variables:
  - `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS`
  - `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS`
- **HTTP SSE Request Headers:** Pass custom headers during transport initialization:
  - `x-mcp-enabled-tools` / `x-mcp-disabled-tools`
  - `x-mcp-enabled-tags` / `x-mcp-disabled-tags`
- **HTTP SSE Request Query Parameters:** Append query parameters directly to your transport connection URL:
  - `?tools=tool1,tool2`
  - `?tags=tag1`

When query strings or parameters are supplied, an LLM-free **Knowledge Graph resolution layer** (using `DynamicToolOrchestrator`) matches query intents against known tool tags, names, or descriptions, with safe fallback and automated 24-hour background cache refreshing.

---

### MCP Configuration Examples

> **Install the slim `[mcp]` extra.** All examples below install
> `container-manager-mcp[mcp]` — the MCP-server extra that pulls only the FastMCP /
> FastAPI tooling (`agent-utilities[mcp]`). It deliberately **excludes** the heavy
> agent runtime (the epistemic-graph engine, `pydantic-ai`, `dspy`, `llama-index`,
> `tree-sitter`), so `uvx`/container installs are dramatically smaller and faster.
> Use the full `[agent]` extra only when you need the integrated Pydantic AI agent
> (see [Installation](#installation)).

#### stdio Transport (Recommended for local IDEs e.g., Cursor, Claude Desktop)
Configure your IDE's `mcp.json` to launch the MCP server via `uvx`:

```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "container-manager-mcp[mcp]",
        "container-manager-mcp"
      ],
      "env": {
        "SYSTEM_TOOLS_ENABLE": "your_system_tools_enable_here",
        "SYSTEMS_MANAGER_ENABLE": "your_systems_manager_enable_here",
        "WEBSITE_BUILDER_ENABLE": "your_website_builder_enable_here",
        "WEB_ARTIFACTS_ENABLE": "your_web_artifacts_enable_here",
        "SECURITY_TOOLS_ENABLE": "your_security_tools_enable_here",
        "DEVELOPER_UTILITIES_ENABLE": "your_developer_utilities_enable_here",
        "BROWSER_TOOLS_ENABLE": "your_browser_tools_enable_here"
      }
    }
  }
}
```

#### Streamable-HTTP Transport (Recommended for production deployments)
Configure your client's `mcp.json` to launch the Streamable-HTTP server via `uvx` with explicit host and port definition:

```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "container-manager-mcp[mcp]",
        "container-manager-mcp"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "SYSTEM_TOOLS_ENABLE": "your_system_tools_enable_here",
        "SYSTEMS_MANAGER_ENABLE": "your_systems_manager_enable_here",
        "WEBSITE_BUILDER_ENABLE": "your_website_builder_enable_here",
        "WEB_ARTIFACTS_ENABLE": "your_web_artifacts_enable_here",
        "SECURITY_TOOLS_ENABLE": "your_security_tools_enable_here",
        "DEVELOPER_UTILITIES_ENABLE": "your_developer_utilities_enable_here",
        "BROWSER_TOOLS_ENABLE": "your_browser_tools_enable_here"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed remote or local Streamable-HTTP instance:

```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "url": "http://localhost:8000/container-manager-mcp/mcp"
    }
  }
}
```

Deploying the Streamable-HTTP server via Docker:

```bash
docker run -d \
  --name container-manager-mcp-mcp \
  -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e PORT=8000 \
  -e SYSTEM_TOOLS_ENABLE="your_value" \
  -e SYSTEMS_MANAGER_ENABLE="your_value" \
  -e WEBSITE_BUILDER_ENABLE="your_value" \
  -e WEB_ARTIFACTS_ENABLE="your_value" \
  -e SECURITY_TOOLS_ENABLE="your_value" \
  -e DEVELOPER_UTILITIES_ENABLE="your_value" \
  -e BROWSER_TOOLS_ENABLE="your_value" \
  knucklessg1/container-manager-mcp:mcp
```

> The `:mcp` tag is the **slim MCP-server image** (built from
> `docker/Dockerfile --target mcp`, installing `container-manager-mcp[mcp]`). The default
> `:latest` tag is the **full agent image** (`--target agent`, `container-manager-mcp[agent]`)
> which also bundles the Pydantic AI agent and the epistemic-graph engine — use it
> when you run `container-manager-agent` (the agent), not just the MCP server. See
> [Container images](#container-images-mcp-vs-agent).

---

<!-- BEGIN GENERATED: additional-deployment-options -->
### Additional Deployment Options

`container-manager-mcp` can also run as a **local container** (Docker / Podman / `uv`) or be
consumed from a **remote deployment**. The
[Deployment guide](https://knuckles-team.github.io/container-manager-mcp/deployment/) has full, copy-paste
`mcp_config.json` for all four transports — **stdio**, **streamable-http**,
**local container / uv**, and **remote URL**:

- **Local container / uv** — launch the server from `mcp_config.json` via `uvx`,
  `docker run`, or `podman run`, or point at a local streamable-http container by `url`.
- **Remote URL** — connect to a server deployed behind Caddy at
  `http://container-manager-mcp.arpa/mcp` using the `"url"` key.
<!-- END GENERATED: additional-deployment-options -->

---

## Environment Variables

<!-- ENV-VARS-TABLE:START -->

#### Package environment variables

| Variable | Example | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` |  |
| `PORT` | `8000` |  |
| `TRANSPORT` | `stdio` | options: stdio, streamable-http, sse |
| `ENABLE_OTEL` | `True` |  |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:8080/api/public/otel` |  |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` | `pk-...` |  |
| `OTEL_EXPORTER_OTLP_SECRET_KEY` | `sk-...` |  |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` |  |
| `EUNOMIA_TYPE` | `none` | options: none, embedded, remote |
| `EUNOMIA_POLICY_FILE` | `mcp_policies.json` |  |
| `EUNOMIA_REMOTE_URL` | `http://eunomia-server:8000` |  |
| `CONTAINER_MANAGER_TYPE` | `docker` | options: docker, podman, swarm, kubernetes |
| `CONTAINER_MANAGER_HOST` | — | remote docker daemon host (e.g. tcp://host:2375); empty = local |
| `CONTAINER_MANAGER_PODMAN_BASE_URL` | — | podman service base URL (e.g. unix:///run/podman/podman.sock) |
| `CONTAINER_MANAGER_K8S_NAMESPACE` | `default` | target namespace |
| `CONTAINER_MANAGER_KUBECONTEXT` | — | kubeconfig context name; empty = current-context |
| `KUBERNETES_SERVICE_HOST` | — | injected by the cluster when running in-pod; leave empty |
| `INVENTORYTOOL` | `True` |  |
| `INFOTOOL` | `True` |  |
| `IMAGETOOL` | `True` |  |
| `CONTAINERTOOL` | `True` |  |
| `VOLUMETOOL` | `True` |  |
| `NETWORKTOOL` | `True` |  |
| `SWARMTOOL` | `True` |  |
| `SYSTEMTOOL` | `True` |  |
| `COMPOSETOOL` | `True` |  |
| `MISCTOOL` | `True` |  |
| `SPECIALIST_DEPLOYMENTTOOL` | `True` |  |

#### Inherited agent-utilities variables (apply to every connector)

| Variable | Example | Description |
|----------|---------|-------------|
| `MCP_TOOL_MODE` | `condensed` | Tool surface: `condensed` | `verbose` | `both` |
| `MCP_ENABLED_TOOLS` | — | Comma-separated tool allow-list |
| `MCP_DISABLED_TOOLS` | — | Comma-separated tool deny-list |
| `MCP_ENABLED_TAGS` | — | Comma-separated tag allow-list |
| `MCP_DISABLED_TAGS` | — | Comma-separated tag deny-list |
| `MCP_CLIENT_AUTH` | — | Outbound MCP auth (`oidc-client-credentials` for fleet calls) |
| `OIDC_CLIENT_ID` | — | OIDC client id (service-account auth) |
| `OIDC_CLIENT_SECRET` | — | OIDC client secret (service-account auth) |
| `DEBUG` | `False` | Verbose logging |
| `PYTHONUNBUFFERED` | `1` | Unbuffered stdout (recommended in containers) |
| `MCP_URL` | `http://localhost:8000/mcp` | URL of the MCP server the agent connects to |
| `PROVIDER` | `openai` | LLM provider for the agent |
| `MODEL_ID` | `gpt-4o` | Model id for the agent |
| `ENABLE_WEB_UI` | `True` | Serve the AG-UI web interface |

_28 package + 14 inherited variable(s). Auto-generated from `.env.example` + the shared agent-utilities set — do not edit._
<!-- ENV-VARS-TABLE:END -->


Every variable the server reads, grouped by purpose.

### MCP server / transport
| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSPORT` | `stdio`, `streamable-http`, or `sse` | `stdio` |
| `HOST` | Bind host (HTTP transports) | `0.0.0.0` |
| `PORT` | Bind port (HTTP transports) | `8000` |
| `MCP_TOOL_MODE` | Tool surface: `condensed`, `verbose`, or `both` | `condensed` |
| `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS` | Comma-separated tool allow/deny list | — |
| `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS` | Comma-separated tag allow/deny list | — |
| `DEBUG` | Verbose logging | `False` |
| `PYTHONUNBUFFERED` | Unbuffered stdout (recommended in containers) | `1` |

### Multi-host control plane
| Variable | Description | Default |
|----------|-------------|---------|
| Inventory file | Remote host endpoints are loaded from the XDG shared inventory `~/.config/agent-utilities/inventory.yml` (`.yml` preferred, `.yaml` legacy fallback); managed via `tunnel-manager inventory init\|doctor` (see [Multi-Host guide](docs/multi_host.md)) | — |

### Bundled companion skill toggles
These enable optional companion tool-suites bundled with the agent (set `True` to enable).
| Variable | Description | Default |
|----------|-------------|---------|
| `SYSTEM_TOOLS_ENABLE` | Enable the `system-tools` suite | `False` |
| `SYSTEMS_MANAGER_ENABLE` | Enable the `systems-manager` suite | `False` |
| `WEBSITE_BUILDER_ENABLE` | Enable the `website-builder` suite | `False` |
| `WEB_ARTIFACTS_ENABLE` | Enable the `web-artifacts` suite | `False` |
| `SECURITY_TOOLS_ENABLE` | Enable the `security-tools` suite | `False` |
| `DEVELOPER_UTILITIES_ENABLE` | Enable the `developer-utilities` suite | `False` |
| `BROWSER_TOOLS_ENABLE` | Enable the `browser-tools` suite | `False` |

### Tool toggles
Each action-routed tool can be disabled individually via its toggle env var (set to `false`).
The full list is in the [Available MCP Tools](#available-mcp-tools) table above.
| Variable | Tool |
|----------|------|
| `INFOTOOL` | `cm_info_operations` |
| `IMAGETOOL` | `cm_image_operations` |
| `CONTAINERTOOL` | `cm_container_operations` |
| `VOLUMETOOL` | `cm_volume_operations` |
| `NETWORKTOOL` | `cm_network_operations` |
| `SWARMTOOL` | `cm_swarm_operations` |
| `SYSTEMTOOL` | `cm_system_operations` |
| `COMPOSETOOL` | `cm_compose_operations` |
| `INVENTORYTOOL` | `cm_list_hosts` |
| `MISCTOOL` | `trace_port_namespace` |

### Telemetry & governance
| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_OTEL` | Enable OpenTelemetry export | `True` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | — |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` / `OTEL_EXPORTER_OTLP_SECRET_KEY` | OTLP auth keys | — |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP protocol (e.g. `http/protobuf`) | — |
| `EUNOMIA_TYPE` | Authorization mode: `none`, `embedded`, `remote` | `none` |
| `EUNOMIA_POLICY_FILE` | Embedded policy file | `mcp_policies.json` |
| `EUNOMIA_REMOTE_URL` | Remote Eunomia server URL | — |

### Agent CLI (full `[agent]` runtime only)
| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_URL` | URL of the MCP server the agent connects to | `http://localhost:8000/mcp` |
| `PROVIDER` | LLM provider (e.g. `openai`) | `openai` |
| `MODEL_ID` | Model id (e.g. `gpt-4o`) | `gpt-4o` |
| `ENABLE_WEB_UI` | Serve the AG-UI web interface | `True` |

See [`.env.example`](.env.example) for a copy-paste starting point.

## Agent

This repository features a fully integrated Pydantic AI Graph Agent. It communicates over the **Agent Control Protocol (ACP)** and interacts seamlessly with the **Agent Web UI (AG-UI)** and Terminal interface.

### Running the Agent CLI
To start the interactive command-line agent:

```bash
# Set credentials
export SYSTEM_TOOLS_ENABLE="your_value"
export SYSTEMS_MANAGER_ENABLE="your_value"
export WEBSITE_BUILDER_ENABLE="your_value"
export WEB_ARTIFACTS_ENABLE="your_value"
export SECURITY_TOOLS_ENABLE="your_value"
export DEVELOPER_UTILITIES_ENABLE="your_value"
export BROWSER_TOOLS_ENABLE="your_value"

# Run the agent server
container-manager-agent --provider openai --model-id gpt-4o
```

### Docker Compose Orchestration
The following `docker/agent.compose.yml` configures the Agent, Web UI, and Terminal Interface together:

```yaml
version: '3.8'

services:
  container-manager-mcp-mcp:
    image: knucklessg1/container-manager-mcp:mcp
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
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  container-manager-mcp-agent:
    image: knucklessg1/container-manager-mcp:latest
    container_name: container-manager-mcp-agent
    hostname: container-manager-mcp-agent
    restart: always
    depends_on:
      - container-manager-mcp-mcp
    env_file:
      - ../.env
    command: [ "container-manager-agent" ]
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=9019
      - MCP_URL=http://container-manager-mcp-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
      - ENABLE_OTEL=True
    ports:
      - "9019:9019"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9019/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

```

Detailed graph node architecture explanations, custom skill configurations, and agentic trace guides are available in [docs/agent.md](docs/agent.md).

---

## Security & Governance

Built directly upon the enterprise-ready [`agent-utilities`](https://github.com/Knuckles-Team/agent-utilities) core, standard security parameters are fully supported:

### Access Control & Policy Enforcement
- **Eunomia Policies:** Fine-grained, policy-driven tool authorization. Supports `none`, local `embedded` (`mcp_policies.json`), or centralized `remote` modes.
- **OIDC Token Delegation:** Compliant with RFC 8693 token exchange for flowing authenticating user credentials from Web UI / ACP → Agent → MCP.
- **Scoped Credentials:** Execution context runs restricted to the specific caller identity.

### Runtime Security Grid
| Feature | Functionality | Enablement |
|---------|---------------|------------|
| **Tool Guard** | Sensitivity inspection with human-in-the-loop validation | Enabled by default |
| **Prompt Injection Defense** | Input scanning, repetition monitoring, and recursive loop blocks | Enabled by default |
| **Context Safety Guard** | Stuck-loop detectors and contextual overflow preemptive alerts | Enabled by default |

---

## Installation

Pick the extra that matches what you want to run:

| Extra | Installs | Use when |
|-------|----------|----------|
| `container-manager-mcp[mcp]` | Slim MCP server only (`agent-utilities[mcp]` — FastMCP/FastAPI) | You only run the **MCP server** (smallest install / image) |
| `container-manager-mcp[agent]` | Full agent runtime (`agent-utilities[agent,logfire]` — Pydantic AI + the epistemic-graph engine) | You run the **integrated agent** |
| `container-manager-mcp[all]` | Everything (`mcp` + `agent` + the `docker` / `podman` / `kubernetes` backends) | Development / both surfaces |

```bash
# MCP server only (recommended for tool hosting — slim deps)
uv pip install "container-manager-mcp[mcp]"

# Full agent runtime (Pydantic AI + epistemic-graph engine)
uv pip install "container-manager-mcp[agent]"

# Everything (development)
uv pip install "container-manager-mcp[all]"      # or: python -m pip install "container-manager-mcp[all]"
```

### Container images (`:mcp` vs `:agent`)

One multi-stage `docker/Dockerfile` builds two right-sized images, selected by `--target`:

| Image tag | Build target | Contents | Entrypoint |
|-----------|--------------|----------|------------|
| `knucklessg1/container-manager-mcp:mcp` | `--target mcp` | `container-manager-mcp[mcp]` — **slim**, no engine/`pydantic-ai`/`dspy`/`llama-index`/`tree-sitter` | `container-manager-mcp` |
| `knucklessg1/container-manager-mcp:latest` | `--target agent` (default) | `container-manager-mcp[agent]` — **full** agent runtime + epistemic-graph engine | `container-manager-agent` |

```bash
docker build --target mcp   -t knucklessg1/container-manager-mcp:mcp    docker/   # slim MCP server
docker build --target agent -t knucklessg1/container-manager-mcp:latest docker/   # full agent
```

`docker/mcp.compose.yml` runs the slim `:mcp` server; `docker/agent.compose.yml` runs the
agent (`:latest`) with a co-located `:mcp` sidecar.

### Knowledge-graph database (`epistemic-graph`)

The **full agent** (`[agent]` / `:latest`) embeds the **epistemic-graph** engine (pulled in
transitively via `agent-utilities[agent]`). For production — or to share one knowledge graph
across multiple agents — run **epistemic-graph as its own database container** and point the
agent at it instead of embedding it. Deployment recipes (single-node + Raft HA), connection
config, and the full database architecture (with diagrams) are documented in the
[epistemic-graph deployment guide](https://knuckles-team.github.io/epistemic-graph/deployment/).
The slim `[mcp]` server does **not** require the database.

---

## Documentation

The complete documentation is published as the
[official documentation site](https://knuckles-team.github.io/container-manager-mcp/)
and is the recommended reference for installation, deployment, and day-to-day
operation.

| Page | Contents |
|---|---|
| [Installation](https://knuckles-team.github.io/container-manager-mcp/installation/) | pip, source, extras, prebuilt Docker image |
| [Deployment](https://knuckles-team.github.io/container-manager-mcp/deployment/) | run the MCP and agent servers, Compose, Caddy + Technitium, env config |
| [Usage](https://knuckles-team.github.io/container-manager-mcp/usage/) | the MCP tools, the `DockerManager` API, the CLI |
| [Overview](https://knuckles-team.github.io/container-manager-mcp/overview/) | ecosystem role, enterprise readiness, architecture |
| [Multi-Host](https://knuckles-team.github.io/container-manager-mcp/multi_host/) | zero-script Docker-over-SSH control plane |
| [Concepts](https://knuckles-team.github.io/container-manager-mcp/concepts/) | concept registry (`CONCEPT:CMGR-*`) |

`AGENTS.md` is the canonical contributor/agent guidance.

---

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)

---

## Contribute

Contributions are welcome! Please ensure code quality by executing local checks before submitting pull requests:
- Format code using `ruff format .`
- Lint code using `ruff check .`
- Validate type-safety with `mypy .`
- Execute test suites using `pytest`


<!-- BEGIN agent-os-genesis-deploy (generated; do not edit between markers) -->

## Deploy with `agent-os-genesis`

This package can be provisioned for you — skill-guided — by the **`agent-os-genesis`**
universal skill (its *single-package deploy mode*): it picks your install method, seeds
secrets to OpenBao/Vault (or `.env`), trusts your enterprise CA, registers the MCP
server, and verifies it — the same machinery that stands up the whole Agent OS, narrowed
to just this package. Ask your agent to **"deploy `container-manager-mcp` with agent-os-genesis"**.

| Install mode | Command |
|------|---------|
| Bare-metal, prod (PyPI) | `uvx container-manager-mcp` · or `uv tool install container-manager-mcp` |
| Bare-metal, dev (editable) | `uv pip install -e ".[all]"` · or `pip install -e ".[all]"` |
| Container, prod | deploy `knucklessg1/container-manager-mcp:latest` via docker-compose / swarm / podman / podman-compose / kubernetes |
| Container, dev (editable) | deploy `docker/compose.dev.yml` (source-mounted at `/src`; edits live on restart) |

Secrets are read-existing + seeded via `vault_sync` — you are only prompted for what's missing.

<!-- END agent-os-genesis-deploy -->

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

*Version: 1.13.0*

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

## CLI or API

This agent wraps the Container Manager - manage Docker, Docker Swarm, and Podman containers. MCP+A2A Servers Out of the Box! API. You can interact with it programmatically or via its integrated execution entrypoints.

Detailed instructions on how to use the underlying API wrappers, extended schema bindings, and developer SDK references are maintained in [docs/index.md](file:///home/apps/workspace/agent-packages/agents/container-manager-mcp/docs/index.md).

---

## MCP

This server utilizes dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools
| Tool Module | Toggle Env Var | Enabled by Default | Description & Nested Methods |
|-------------|----------------|--------------------|------------------------------|
| **Info** | `INFOTOOL` | `True` | Manage container manager info operations. Action-routed methods: `get_version`, `get_info`. |
| **Image** | `IMAGETOOL` | `True` | Manage container images. Action-routed methods: `list_images`, `pull_image`, `remove_image`, `prune_images`. |
| **Container** | `CONTAINERTOOL` | `True` | Manage container operations. Action-routed methods: `list_containers`, `get_container_logs`, `stop_container`, `remove_container`, `prune_containers`, `exec_in_container`. |
| **Volume** | `VOLUMETOOL` | `True` | Manage volume operations. Action-routed methods: `list_volumes`, `create_volume`, `remove_volume`, `prune_volumes`. |
| **Network** | `NETWORKTOOL` | `True` | Manage network operations. Action-routed methods: `list_networks`, `create_network`, `remove_network`, `prune_networks`. |
| **Swarm** | `SWARMTOOL` | `True` | Manage swarm operations. Action-routed methods: `init_swarm`, `leave_swarm`, `list_nodes`, `list_services`, `create_service`, `remove_service`. |
| **System** | `SYSTEMTOOL` | `True` | Manage container manager system operations. Action-routed methods: `prune_system`, `get_info`, `get_version`. |
| **Compose** | `COMPOSETOOL` | `True` | Manage docker-compose or podman-compose operations. Action-routed methods: `up`, `down`, `ps`, `logs`. |
| **Misc** | `MISCTOOL` | `True` | Initialize and return the MCP instance. |

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/mcp.md](file:///home/apps/workspace/agent-packages/agents/container-manager-mcp/docs/mcp.md).

### MCP Configuration Examples

#### stdio Transport (Recommended for local IDEs e.g., Cursor, Claude Desktop)
Configure your IDE's `mcp.json` to launch the MCP server via `uvx`:

```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "container-manager-mcp",
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
        "container-manager-mcp",
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
  knucklessg1/container-manager-mcp:latest
```

---

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

Detailed graph node architecture explanations, custom skill configurations, and agentic trace guides are available in [docs/agent.md](file:///home/apps/workspace/agent-packages/agents/container-manager-mcp/docs/agent.md).

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

Install the Python package locally:

```bash
# Using uv (highly recommended)
uv pip install container-manager-mcp[all]

# Using standard pip
python -m pip install container-manager-mcp[all]
```

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

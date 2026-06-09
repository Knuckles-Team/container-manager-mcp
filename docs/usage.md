# Usage — API / CLI / MCP

`container-manager-mcp` exposes the same capability three ways: as **MCP tools** an
agent calls, as a **Python API** (`DockerManager` / `PodmanManager`) you import, and
as a **CLI**. The complete tool surface and ecosystem role are in
[Overview](overview.md).

## As an MCP server

Once [deployed](deployment.md), the server registers consolidated, action-routed tool
modules. Each module is individually togglable through its environment variable and is
enabled by default.

| Tool module | Toggle | Action-routed methods |
|---|---|---|
| Info | `INFOTOOL` | `get_info`, `get_version` |
| Image | `IMAGETOOL` | `list_images`, `pull_image`, `remove_image`, `prune_images` |
| Container | `CONTAINERTOOL` | `list_containers`, `stop_container`, `remove_container`, `get_container_logs`, `exec_in_container`, `prune_containers` |
| Volume | `VOLUMETOOL` | `list_volumes`, `create_volume`, `remove_volume`, `prune_volumes` |
| Network | `NETWORKTOOL` | `list_networks`, `create_network`, `remove_network`, `prune_networks` |
| Swarm | `SWARMTOOL` | `init_swarm`, `leave_swarm`, `list_nodes`, `list_services`, `create_service`, `remove_service` |
| System | `SYSTEMTOOL` | `get_info`, `get_version`, `prune_system` |
| Compose | `COMPOSETOOL` | `up`, `down`, `ps`, `logs` |

Example agent prompts that map onto these tools:

- *"List every running container on this host."* → Container module, `list_containers`
- *"Pull `nginx:latest` and start it."* → Image + Compose modules
- *"Reclaim disk space — prune dangling images and stopped containers."* → System module, `prune_system`
- *"List the services on the Swarm."* → Swarm module, `list_services`

Every tool accepts an optional `host` argument to target a remote machine over SSH —
see [Multi-Host](multi_host.md).

## As a Python API

`DockerManager` and `PodmanManager` are typed facades over the respective engines. A
bare constructor connects to the local engine; pass an inventory `host` to connect to
a remote daemon over SSH.

```python
from container_manager_mcp.container_manager import DockerManager

cm = DockerManager()                 # local Docker engine (docker.from_env())

# Reads
version = cm.get_version()           # engine version info
info = cm.get_info()                 # daemon info
images = cm.list_images()            # ImageInfo records
containers = cm.list_containers(all=True)
volumes = cm.list_volumes()
networks = cm.list_networks()

# Swarm
nodes = cm.list_nodes()
services = cm.list_services()
```

Target a remote host defined in `~/.config/agent-utilities/inventory.yaml`:

```python
cm = DockerManager(host="node-alpha")   # connects over ssh://user@host:port
remote_containers = cm.list_containers(all=True)
```

The Podman engine offers the same surface:

```python
from container_manager_mcp.container_manager import PodmanManager

cm = PodmanManager()
images = cm.list_images()
```

## As a CLI

The MCP server is itself a CLI. Run it directly to start a server, or inspect its
options:

```bash
container-manager-mcp --help
container-manager-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

Dynamic tool selection lets you restrict the exposed surface so the LLM context stays
focused:

```bash
# Expose only the container and image modules
container-manager-mcp --toolsets container,image
```

The companion agent runs as an interactive command-line / Web-UI front end:

```bash
container-manager-agent --provider openai --model-id gpt-4o
```

See [Deployment](deployment.md) for the full transport, agent-server, and
environment-variable reference.

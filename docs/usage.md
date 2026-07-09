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

### Kubernetes

Kubernetes is exposed as 8 themed, action-routed tools instead of the `host`-based inventory routing used
by Docker/Podman — see [Kubernetes](kubernetes.md) for the full tool-by-tool breakdown and
[Multi-Host](multi_host.md#kubernetes-kubeconfig-contexts) for how remote/multi-cluster targeting differs
from the Docker-host inventory model. Set `CONTAINER_MANAGER_TYPE=kubernetes` (and optionally
`CONTAINER_MANAGER_KUBECONTEXT`) to point the server at a cluster.

Example calls:

```json
{"name": "cm_k8s_workloads", "arguments": {"action": "list_pods", "namespace": "default"}}
```

```json
{"name": "cm_k8s_workloads", "arguments": {"action": "describe_pod", "namespace": "default", "name": "web-7d8f9c-abcde"}}
```

```json
{"name": "cm_k8s_config", "arguments": {"action": "create_secret", "namespace": "default", "name": "db-creds", "data": {"password": "hunter2"}}}
```

```json
{
  "name": "cm_k8s_config",
  "arguments": {
    "action": "patch_resource",
    "namespace": "default",
    "kind": "Deployment",
    "name": "web",
    "patch_type": "strategic",
    "body": {"spec": {"replicas": 3}}
  }
}
```

```json
{"name": "cm_k8s_networking", "arguments": {"action": "list_k8s_services", "namespace": "default"}}
```

```json
{"name": "cm_k8s_workloads", "arguments": {"action": "rollout_restart", "namespace": "default", "name": "web"}}
```

```json
{"name": "cm_k8s_cluster", "arguments": {"action": "cordon_node", "name": "node-3"}}
```

```json
{"name": "cm_k8s_cluster", "arguments": {"action": "drain_node", "name": "node-3"}}
```

```json
{"name": "cm_k8s_observability", "arguments": {"action": "top_pods", "namespace": "default"}}
```

Example agent prompts:

- *"List every pod in the `default` namespace."* → `cm_k8s_workloads`, `list_pods`
- *"Scale the `web` deployment to 3 replicas."* → `cm_k8s_config`, `patch_resource`
- *"Restart the rollout for the `web` deployment."* → `cm_k8s_workloads`, `rollout_restart`
- *"What's the CPU/memory usage of pods in `default`?"* → `cm_k8s_observability`, `top_pods`
- *"Cordon and drain `node-3` for maintenance."* → `cm_k8s_cluster`, `cordon_node` then `drain_node`

### Multi-context

`cm_multi_context` (toggle `MULTICONTEXTTOOL`) targets several Docker/Podman/Swarm/Kubernetes contexts in one
call — configure the pool via `K8S_CONTEXTS` / `DOCKER_CONTEXTS` / `SWARM_CONTEXTS` and their
`DEFAULT_*_CONTEXT` values:

```json
{"name": "cm_multi_context", "arguments": {"action": "list_contexts"}}
```

```json
{"name": "cm_multi_context", "arguments": {"action": "list_pods", "context": "prod-cluster", "namespace": "default"}}
```

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

Target a remote host defined in `~/.config/agent-utilities/inventory.yml` (`.yaml` legacy fallback):

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

### Environment doctor

`container-manager-doctor` diagnoses AND helps resolve the environment — the
tunnel-manager SSH inventory, kubeconfig/contexts, and docker/podman runtimes —
with real probes and concrete remediation for anything that is not OK. It exits
`0` when no check fails, `1` otherwise, so it drops into CI/health scripts.

```bash
container-manager-doctor                       # diagnose everything
container-manager-doctor --guided              # + probe every inventory host
container-manager-doctor --backend inventory --host r820
container-manager-doctor --backend kubernetes --context prod
container-manager-doctor --backend docker --host r820
container-manager-doctor --json                # machine-readable report
```

The same engine is exposed as the `cm_doctor` MCP tool (`action=run` or a focused
`check_backends` / `check_inventory` / `check_docker` / `check_podman` /
`check_kubernetes`), returning per-check
`{name, category, status: ok|warn|fail, detail, remediation}` plus a summary.

The companion agent runs as an interactive command-line / Web-UI front end:

```bash
container-manager-agent --provider openai --model-id gpt-4o
```

See [Deployment](deployment.md) for the full transport, agent-server, and
environment-variable reference.

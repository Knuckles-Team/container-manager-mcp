# Container Manager Lifecycle

Docker/Podman container and image lifecycle on local or remote hosts via the container-manager-mcp MCP server — list/inspect/stop/remove/exec containers, read logs, trace which container owns a port, pull/prune images, and run docker-compose stacks. Use when the agent must operate standalone (non-swarm) container workloads on a single engine. Do NOT use for Swarm cluster services/nodes (use container-manager-swarm) or for pushing inventory into the knowledge graph (use container-manager-kg-ingestion).

# Container Lifecycle (Docker / Podman)

Domain-typed control of single-host container workloads through the
**container-manager-mcp** server. Prefer these `cm_*` tools over raw `docker`/`podman`
shell — they carry the host-targeting, port-tracing, and destructive-confirmation
conventions and return typed records.

## When to use
- List, inspect, or read logs from containers on the LOCAL socket or a remote host.
- Stop / remove / prune containers, or `exec` a command inside one.
- Find which container is publishing a given host port.
- Pull, list, or prune images; bring a `docker-compose` file up/down.

## When NOT to use
- Swarm cluster services, replicas, or nodes → `container-manager-swarm`.
- Snapshotting inventory into the knowledge graph → `container-manager-kg-ingestion`.
- Kubernetes workloads → `container-manager-kubernetes-operations` (the full `cm_k8s_*`
  surface), not these tools.
- Podman pods, checkpoint/restore, or Kubernetes-YAML interop →
  `container-manager-podman-operations`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`container-manager-mcp`** MCP server.
A reachable Docker/Podman engine is required (local socket or a remote host over SSH).

| Variable | Required | Notes |
|----------|----------|-------|
| `CONTAINER_MANAGER_TYPE` | optional | `docker` or `podman`; auto-detected if unset |
| `CONTAINER_MANAGER_HOST` | optional | Default remote host alias (else LOCAL socket) |

Remote hosts resolve from the tunnel-manager inventory — call **`cm_list_hosts`** to see
aliases. Omit `host` to target the LOCAL Docker socket. `MCP_TOOL_MODE`
(`condensed`|`verbose`|`both`) selects the condensed `action`-based tools (used below).

## Tools & actions
| Condensed tool | Actions |
|----------------|---------|
| `cm_container_operations` | `list_containers`, `get_container_logs`, `stop_container`, `remove_container`, `prune_containers`, `exec_in_container` |
| `cm_image_operations` | `list_images`, `pull_image`, `remove_image`, `prune_images` |
| `cm_compose_operations` | `up`, `down`, `ps`, `logs` |
| `trace_port_namespace` | (single-purpose) locate the container owning a host port |
| `cm_list_hosts` | enumerate remote host aliases |

### Key parameters
- `host` — remote alias (omit for LOCAL). Same meaning across every `cm_*` tool.
- `container_id` — id or name; required for logs/stop/remove/exec.
- `command` — for `exec_in_container`.
- `all_containers` — include stopped containers in `list_containers`.
- `tail` — log-line count for `get_container_logs` (default `"50"`).
- `compose_file` — path to the compose file for `cm_compose_operations`.
- `force` — required to remove a running container / force prune.

## Recipes
List all containers (including stopped) on a remote host:
```
cm_container_operations action=list_containers all_containers=true host=<alias>
```
Tail 200 log lines from a container:
```
cm_container_operations action=get_container_logs container_id=<id|name> tail="200"
```
Find which container owns host port 8080:
```
trace_port_namespace port=8080 host=<alias>
```
Bring a compose stack up / read its services:
```
cm_compose_operations action=up compose_file=[REDACTED_POSIX_LOCAL_PATH]
cm_compose_operations action=ps compose_file=[REDACTED_POSIX_LOCAL_PATH]
```

## Gotchas
- Swarm reads (`list_nodes`/`list_services`) live on `cm_swarm_operations` and must target
  a swarm **manager**; the container tools here operate the plain engine.
- `remove_container` on a running container needs `force=true`; stop it first when you can.
- `host` targets a remote engine over SSH via the tunnel-manager inventory — an unknown
  alias fails resolution; call `cm_list_hosts` first.
- `tail` is a **string** (`"50"`), not an int.
- `prune_*` actions are destructive and unbounded — confirm intent before running.

## Related
- **`container-manager-swarm`** — cluster services / nodes / scaling.
- **`container-manager-kubernetes-operations`** — the full Kubernetes tool surface
  (workloads/config/networking/storage/RBAC/cluster/governance/observability).
- **`container-manager-podman-operations`** — Podman-specific pods, checkpoint/
  restore, and kube-yaml interop.
- **`container-manager-config-walkthrough`** — first-time setup / choosing
  `CONTAINER_MANAGER_TYPE` before using this skill.
- **`container-manager-kg-ingestion`** — push this inventory into the knowledge graph
  as typed `:Container` / `:ContainerImage` nodes.

# Container Manager Swarm

Docker Swarm cluster orchestration via the container-manager-mcp MCP server — initialize/leave a swarm, list nodes and services, and create or remove replicated services across the cluster. Use when the agent must operate a multi-host Swarm from a manager node, inventory cluster services/nodes, or deploy a service. Do NOT use for single-host container/image lifecycle (use container-manager-lifecycle), standalone docker-compose, or Kubernetes.

# Docker Swarm Orchestration

Domain-typed control of a Docker **Swarm** cluster through the **container-manager-mcp**
server. These operate the cluster control plane (services + nodes) and must be aimed at a
swarm **manager** node.

## When to use
- Inventory swarm nodes (role/availability/status) and services (image/replicas/ports).
- Create a replicated service, or remove one.
- Initialize a swarm or make a node leave it.

## When NOT to use
- Single-host container/image lifecycle or `exec`/logs → `container-manager-lifecycle`.
- Plain `docker-compose` stacks (non-swarm) → `cm_compose_operations` in the lifecycle skill.
- Kubernetes → `container-manager-kubernetes-operations`, not these tools. If the
  target is a Kubernetes cluster, `cm_swarm_operations action=list_nodes` /
  `list_services` are the **wrong tools entirely** — they only understand Docker
  Swarm's control plane, not the Kubernetes API.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`container-manager-mcp`** MCP server.
Cluster reads/writes MUST target a swarm **manager** — pass its `host` alias.

| Variable | Required | Notes |
|----------|----------|-------|
| `CONTAINER_MANAGER_TYPE` | optional | `docker` (Swarm is a Docker feature) |
| `CONTAINER_MANAGER_HOST` | optional | Default manager alias (else pass `host` explicitly) |

Call **`cm_list_hosts`** to see aliases; a swarm-manager alias is required for
`list_nodes` / `list_services` / `create_service` / `remove_service`.

## Tools & actions
| Condensed tool | Actions |
|----------------|---------|
| `cm_swarm_operations` | `init_swarm`, `leave_swarm`, `list_nodes`, `list_services`, `create_service`, `remove_service` |

### Key parameters
- `host` — the swarm **manager** alias (required for cluster reads/writes).
- `name` + `image` — required to `create_service`.
- `replicas` — desired replica count for `create_service`.
- `service_id` — required to `remove_service`.
- `advertise_addr` — for `init_swarm`.
- `force` — for `leave_swarm` on a manager.

## Recipes
Inventory the cluster (nodes then services) from a manager:
```
cm_swarm_operations action=list_nodes host=<manager-alias>
cm_swarm_operations action=list_services host=<manager-alias>
```
Deploy a replicated service:
```
cm_swarm_operations action=create_service name=web image=nginx@sha256:<digest> replicas=3 host=<manager-alias>
```
Remove a service:
```
cm_swarm_operations action=remove_service service_id=<id|name> host=<manager-alias>
```

## Gotchas
- `list_nodes` / `list_services` fail on a **worker** node — they require a manager.
  Symptom: "This node is not a swarm manager".
- Node ids are truncated in `list_nodes` output; use the returned id/hostname when
  cross-referencing tasks.
- `create_service` uses replicated mode with `replicas`; global mode and advanced
  placement/constraints are not exposed by the condensed tool.
- `init_swarm` / `leave_swarm` change cluster membership — confirm intent; `leave_swarm`
  on a manager needs `force=true`.
- `cm_swarm_operations action=list_nodes` / `list_services` return **Swarm-shaped**
  records (replicated service + Swarm node) — a different shape from the Kubernetes
  surface's `cm_k8s_workloads` (Deployments/Pods) and `cm_k8s_cluster action=list_nodes`
  (Kubernetes Node objects). If the cluster in front of you speaks the Kubernetes API,
  use `container-manager-kubernetes-operations` instead — pointing these tools at a
  Kubernetes cluster fails outright (there is no Swarm control plane to answer them).

## Related
- **`container-manager-lifecycle`** — the individual containers behind service tasks
  (logs/exec/inspect) on each host.
- **`container-manager-kubernetes-operations`** — the equivalent surface for a real
  Kubernetes cluster; pick this skill instead when the target isn't Docker Swarm.
- **`container-manager-config-walkthrough`** — first-time setup / choosing
  `CONTAINER_MANAGER_TYPE` before using this skill.
- **`container-manager-kg-ingestion`** — snapshot `:SwarmService` / `:SwarmNode`
  inventory into the knowledge graph.

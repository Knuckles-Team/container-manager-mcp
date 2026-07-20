# Container Manager Multi Context

Operate several container backends and contexts at once — Kubernetes, Docker, Podman, and Swarm — via the container-manager-mcp MCP server's cm_multi_context tool, with per-call backend/context selection and parallel fan-out across a configured pool of contexts. Use when the agent must compare, migrate between, or simultaneously act on multiple clusters/engines/hosts. Do NOT use for single-backend operations, which are cheaper via the themed skills (container-manager-lifecycle / -swarm / -kubernetes-operations / -podman-operations).

# Multi-Context Container Management

Domain-typed control across a **pool of backends and contexts** — Kubernetes clusters,
Docker hosts, Podman, and Swarm managers — through the single `cm_multi_context` tool
on the **container-manager-mcp** server. Each call selects a `backend` (and optional
`context`) so one MCP server can drive many environments without reconnecting. Best
suited to cross-cluster comparisons, staged migrations, and fan-out operations.

## When to use
- Enumerate every configured backend/context and its health in one call
  (`list_contexts`).
- Run the same container/image/volume/network/service verb against a specific
  named context without re-authenticating per call.
- Compare state across contexts (e.g. staging vs prod Kubernetes clusters, or two
  Docker hosts) before/during a migration.
- Kubernetes-specific fan-out: list pods/deployments or scale a deployment in a
  named context + namespace.

## When NOT to use
- A single, already-known backend → use the themed skill directly:
  `container-manager-lifecycle` (Docker/Podman), `container-manager-swarm`,
  `container-manager-kubernetes-operations` (full k8s surface — `cm_multi_context`'s
  Kubernetes actions are a narrow subset: containers/images/volumes/networks/
  services/pods/deployments only, not the full `cm_k8s_*` themed surface).
  `container-manager-podman-operations` (pods/checkpoint/kube-yaml — not exposed here).
- First-time setup of `K8S_CONTEXTS`/`DOCKER_CONTEXTS`/`SWARM_CONTEXTS` →
  see the Walkthrough below, then `container-manager-config-walkthrough`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`container-manager-mcp`** MCP server.
Multi-context mode is enabled by configuring one or more context-pool env vars (or
`MULTI_CONTEXT_MODE=true`); each pool var is a `;`-separated list of `name=value`
pairs.

| Variable | Required | Notes |
|----------|----------|-------|
| `MULTI_CONTEXT_MODE` | optional | Forces multi-context mode even with a single context configured |
| `K8S_CONTEXTS` | optional | `"prod=prod-ctx;staging=staging-ctx"` — kubeconfig context names |
| `DEFAULT_K8S_CONTEXT` | optional | Which `K8S_CONTEXTS` entry is used when `context` is omitted; else the first one |
| `DOCKER_CONTEXTS` | optional | `"local=;remote1=<tunnel-manager-host-alias>"` — Docker host aliases |
| `DEFAULT_DOCKER_CONTEXT` | optional | Default Docker context name |
| `SWARM_CONTEXTS` | optional | `"cluster1=<manager-host-alias>"` — Swarm manager host aliases |
| `DEFAULT_SWARM_CONTEXT` | optional | Default Swarm context name |
| `PODMAN_ENABLED` | optional | Default `true`; local Podman only (no multi-context Podman pool) |
| `MULTI_CONTEXT_MAX_WORKERS` | optional | Thread pool size for fan-out, default `8` |
| `MULTICONTEXTTOOL` | optional | Tool toggle, default `true`; set `false` to hide `cm_multi_context` |

## Walkthrough — first run
1. **Configure the pools** (example: two Kubernetes clusters + one remote Docker host):
   ```
   K8S_CONTEXTS="prod=prod-cluster;staging=staging-cluster"
   DEFAULT_K8S_CONTEXT="staging"
   DOCKER_CONTEXTS="edge1=<tunnel-manager-alias>"
   MULTI_CONTEXT_MODE=true
   ```
2. **Enumerate what's live.**
   ```
   cm_multi_context action=list_contexts
   ```
3. **Target a specific context explicitly** (omit `context` to use the backend's
   default):
   ```
   cm_multi_context action=list_pods backend=kubernetes context=prod namespace=default
   ```
4. Proceed to the recipes below; escalate to the full themed skill
   (`container-manager-kubernetes-operations`, etc.) once you've localized the
   target context and need its complete tool surface.

## Tools & actions
| Tool | Group | Actions |
|------|-------|---------|
| `cm_multi_context` | Context management | `list_contexts` |
| `cm_multi_context` | Containers | `list_containers`, `run_container`, `stop_container`, `remove_container`, `inspect_container` |
| `cm_multi_context` | Images | `list_images`, `pull_image`, `remove_image` |
| `cm_multi_context` | Volumes | `list_volumes`, `create_volume`, `remove_volume` |
| `cm_multi_context` | Networks | `list_networks`, `create_network`, `remove_network` |
| `cm_multi_context` | Services (Kubernetes/Swarm) | `list_services`, `create_service`, `remove_service` |
| `cm_multi_context` | Pods (Kubernetes only) | `list_pods`, `describe_pod` |
| `cm_multi_context` | Deployments (Kubernetes only) | `list_deployments`, `scale_deployment` |

### Key parameters
- `backend` — one of `kubernetes` (default), `docker`, `podman`, `swarm`; selects
  which pool `context` is resolved against.
- `context` — a name from that backend's pool (e.g. a `K8S_CONTEXTS` key); omit to
  use that backend's `DEFAULT_*_CONTEXT`.
- `name` — resource name for container/image/volume/network/service verbs.
- `namespace` — required alongside `name` for `describe_pod` and `scale_deployment`
  (Kubernetes only); `list_pods`/`list_deployments` default to `"default"`.
- `replicas` — required for `scale_deployment`.
- `all` — include stopped containers in `list_containers`.
- `force` — for `stop_container`/`remove_container`/`remove_image`/`remove_volume`/
  `remove_network`.

## Recipes
List every configured context and its resolved health:
```
cm_multi_context action=list_contexts
```
Compare pod counts across two Kubernetes clusters before a migration:
```
cm_multi_context action=list_pods backend=kubernetes context=staging namespace=prod
cm_multi_context action=list_pods backend=kubernetes context=prod namespace=prod
```
Scale a deployment in a named cluster context:
```
cm_multi_context action=scale_deployment backend=kubernetes context=prod name=web namespace=prod replicas=6
```
Pull the same image on two Docker hosts ahead of a rolling cutover:
```
cm_multi_context action=pull_image backend=docker context=edge1 image=myapp:2026.07.09
cm_multi_context action=pull_image backend=docker context=edge2 image=myapp:2026.07.09
```

## Gotchas
- The Kubernetes actions here (`list_pods`, `describe_pod`, `list_deployments`,
  `scale_deployment`) are a **narrow subset**: no rollouts, no config/networking/
  storage/RBAC/governance/observability themes. Once you've localized a target
  cluster context, switch to `container-manager-kubernetes-operations` (pointed at
  that same kubeconfig context via `CONTAINER_MANAGER_KUBECONTEXT`) for the full
  surface.
- Podman is **local-only** in multi-context mode — there is no `PODMAN_CONTEXTS`
  pool; `backend=podman` always targets the local engine.
- An unconfigured/unknown `context` name fails resolution — call `list_contexts`
  first to see what's actually in the pool.
- `list_services`/`create_service`/`remove_service` behave differently per backend:
  Kubernetes-shaped vs Swarm-shaped — know which `backend` you're targeting.
- Fan-out across many contexts runs on a bounded thread pool
  (`MULTI_CONTEXT_MAX_WORKERS`, default `8`); very large pools serialize beyond that.

## Related
- **`container-manager-config-walkthrough`** — how to populate `K8S_CONTEXTS` /
  `DOCKER_CONTEXTS` / `SWARM_CONTEXTS` for the first time.
- **`container-manager-kubernetes-operations`** — the full k8s tool surface once a
  context is localized.
- **`container-manager-lifecycle`** / **`container-manager-swarm`** /
  **`container-manager-podman-operations`** — the full single-backend surfaces.

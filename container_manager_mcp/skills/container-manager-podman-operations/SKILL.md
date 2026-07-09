---
name: container-manager-podman-operations
description: >-
  Rootless Podman pod/kube operations via the container-manager-mcp MCP server —
  pods (create/list/stats/top/inspect/logs/stop/rm), Kubernetes YAML interop
  (generate/play kube), checkpoint/restore, pod-scoped networks and volumes,
  health checks, and system prune. Use when the agent must drive Podman pod-level
  workloads beyond single containers. Do NOT use for single-container Docker/Podman
  lifecycle (container-manager-lifecycle) or for a real Kubernetes cluster
  (container-manager-kubernetes-operations).
license: MIT
tags: [container-manager-mcp, podman, pods, rootless, checkpoint, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Podman Pod/Kube Operations

Domain-typed control of **Podman pods** and Podman-specific features through
the single `cm_podman` tool on the **container-manager-mcp** server. Podman
groups containers into local **pods** (its rootless analogue of a Kubernetes pod) and
adds checkpoint/restore and Kubernetes-YAML interop that plain Docker doesn't have.
Prefer this `cm_*` tool over raw `podman` shell.

## When to use
- Create/list/inspect/stop/remove a Podman **pod**, and read pod-scoped stats, top,
  or logs.
- Generate a Kubernetes YAML manifest from a running pod (`podman_generate_kube_yaml`)
  or play a YAML manifest into local pods (`podman_play_kube_yaml`) — useful for
  Kubernetes-migration dry runs on a single host before targeting a real cluster.
- Checkpoint a running container to disk and restore it later (live-migration /
  fast-restart workflows unique to Podman's CRIU integration).
- Manage pod-scoped networks and volumes, run a container health check, or prune
  the Podman system.

## When NOT to use
- Single-container list/logs/stop/remove/exec, or plain images/compose →
  `container-manager-lifecycle` (works for Podman too — it auto-detects the engine).
- A real Kubernetes cluster (not just YAML interop) →
  `container-manager-kubernetes-operations`.
- Docker Swarm → `container-manager-swarm`.
- Operating Podman alongside other backends/contexts at once →
  `container-manager-multi-context`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`container-manager-mcp`** MCP server.
A reachable Podman engine is required (rootless or rootful).

| Variable | Required | Notes |
|----------|----------|-------|
| `CONTAINER_MANAGER_TYPE` | optional | Set to `podman` to force Podman (else auto-detect) |
| `PODMANTOOL` | optional | Tool toggle, default `true`; set `false` to hide `cm_podman` |

## Tools & actions
| Tool | Group | Actions |
|------|-------|---------|
| `cm_podman` | Kubernetes interop | `podman_generate_kube_yaml`, `podman_play_kube_yaml` |
| `cm_podman` | Checkpoint/restore | `podman_checkpoint`, `podman_restore` |
| `cm_podman` | Pod management | `podman_pod_create`, `podman_pod_list`, `podman_pod_stats`, `podman_pod_top`, `podman_pod_inspect`, `podman_pod_logs`, `podman_pod_stop`, `podman_pod_rm` |
| `cm_podman` | Network | `podman_network_create`, `podman_network_list`, `podman_network_inspect` |
| `cm_podman` | Volume | `podman_volume_create`, `podman_volume_list`, `podman_volume_inspect` |
| `cm_podman` | System | `podman_system_prune`, `podman_health_check` |

### Key parameters
- `pod_name` — required for every pod-scoped action (`podman_pod_*`) and for
  `podman_generate_kube_yaml`.
- `image` + `command` — required for `podman_pod_create` (the pod's infra + first
  container).
- `yaml_path` — required for `podman_play_kube_yaml`.
- `namespace` — Kubernetes namespace metadata used by `podman_generate_kube_yaml`
  (default `"default"`); Podman itself is not namespace-scoped.
- `container_id` + `checkpoint_dir` — required for both `podman_checkpoint` and
  `podman_restore`.
- `network_name` / `driver` (default `"bridge"`) / `subnet` — for
  `podman_network_create`.
- `volume_name` / `driver` (default `"local"`) — for `podman_volume_create`.
- `tail_lines` (default `100`) — for `podman_pod_logs`.
- `config` — a `dict` health-check spec, required with `container_id` for
  `podman_health_check`.

## Walkthrough — first run
1. **Confirm the engine is Podman.**
   ```
   cm_info_operations action=get_info
   ```
2. **Create a pod and inspect it.**
   ```
   cm_podman action=podman_pod_create pod_name=web-pod image=nginx:latest
   cm_podman action=podman_pod_list
   cm_podman action=podman_pod_inspect pod_name=web-pod
   ```
3. **Verify health and logs.**
   ```
   cm_podman action=podman_health_check container_id=<id> config={"test":["CMD","curl","-f","http://localhost/"]}
   cm_podman action=podman_pod_logs pod_name=web-pod tail_lines=100
   ```

## Recipes
Migration dry run — snapshot a running pod to Kubernetes YAML, then replay it:
```
cm_podman action=podman_generate_kube_yaml pod_name=web-pod namespace=prod
cm_podman action=podman_play_kube_yaml yaml_path=/srv/web-pod.yaml
```
(Once validated, hand the same manifest to a real cluster via
`container-manager-kubernetes-operations`.)

Checkpoint a container before a host maintenance window, restore after:
```
cm_podman action=podman_checkpoint container_id=<id> checkpoint_dir=/srv/checkpoints/web-2026-07-09
# ... host maintenance ...
cm_podman action=podman_restore container_id=<id> checkpoint_dir=/srv/checkpoints/web-2026-07-09
```
Create an isolated pod network with an explicit subnet:
```
cm_podman action=podman_network_create network_name=web-net driver=bridge subnet=10.89.10.0/24
```
Sustain: prune unused pods/images/volumes on a schedule:
```
cm_podman action=podman_system_prune
```

## Gotchas
- Pod-scoped actions (`podman_pod_*`) act on the **pod**, not individual containers
  inside it — use `container-manager-lifecycle`'s `cm_container_operations` for a
  single container's logs/exec once you have its id from `podman_pod_inspect`.
- `podman_checkpoint`/`podman_restore` require CRIU support on the host; unsupported
  kernels/containers fail — treat this as an availability check, not a guarantee.
- `podman_play_kube_yaml` is a **local** Podman-pod replay, not a real cluster
  apply — it doesn't create Kubernetes API objects; use
  `container-manager-kubernetes-operations` for that.
- `podman_system_prune` is destructive and unbounded — confirm intent before running.
- `config` for `podman_health_check` is a raw `dict` (not a JSON string, unlike some
  `cm_k8s_config` fields) — pass it as a native object.

## Related
- **`container-manager-lifecycle`** — single-container Docker/Podman verbs (also
  works against a Podman engine).
- **`container-manager-kubernetes-operations`** — the real Kubernetes cluster surface
  that `podman_generate_kube_yaml`/`podman_play_kube_yaml` manifests target next.
- **`container-manager-multi-context`** — operate Podman alongside other backends.

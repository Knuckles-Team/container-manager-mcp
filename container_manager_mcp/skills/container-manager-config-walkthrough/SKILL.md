---
name: container-manager-config-walkthrough
description: >-
  End-user setup guide for the container-manager-mcp MCP server — choosing
  CONTAINER_MANAGER_TYPE (docker/podman/kubernetes/multi), wiring .env /
  mcp_config toggles, connecting remote Docker/Podman hosts via the
  tunnel-manager inventory versus remote Kubernetes clusters via kubeconfig
  contexts, and a first-run verification checklist. Use when the agent is
  onboarding a new environment or troubleshooting "which skill do I even use"
  before touching any other container-manager-mcp skill. Do NOT use for actually
  operating containers/clusters once configured — use the operational skills.
license: MIT
tags: [container-manager-mcp, setup, configuration, onboarding, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Container Manager — Configuration Walkthrough

The one-time (or per-environment) setup path for **container-manager-mcp** before
using any of its operational skills. This skill is a checklist and decision guide,
not an operations surface — it drives only the read-only discovery tools
(`cm_info_operations`, `cm_list_hosts`, `cm_k8s_cluster action=list_contexts`) needed
to confirm the environment is wired correctly.

## When to use
- Standing up container-manager-mcp against a new environment for the first time.
- Deciding which backend (`docker` / `podman` / `kubernetes` / `multi`) an agent
  should target before picking an operational skill.
- Troubleshooting "no hosts found" / "no contexts found" / wrong-engine errors.
- Onboarding a **remote** Docker/Podman host or a **remote** Kubernetes cluster and
  need the split between the two connection models spelled out.

## When NOT to use
- Actually operating containers/pods/clusters once configured → the matching
  operational skill (`container-manager-lifecycle`, `container-manager-swarm`,
  `container-manager-kubernetes-operations`, `container-manager-podman-operations`,
  `container-manager-multi-context`).
- Pushing inventory into the KG → `container-manager-kg-ingestion`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`container-manager-mcp`** MCP
server. No cluster/engine access is strictly required to *read* the current
configuration, but verification steps below need a reachable target to succeed.

## The core decision: `CONTAINER_MANAGER_TYPE`
| Value | Targets | Then use |
|-------|---------|----------|
| `docker` (default if unset and Docker is detected) | A single Docker engine, local or remote | `container-manager-lifecycle`; add `container-manager-swarm` if the engine is a Swarm manager; `container-manager-podman-operations` doesn't apply |
| `podman` | A single Podman engine, local or remote | `container-manager-lifecycle` (works for Podman too) + `container-manager-podman-operations` for pods/checkpoint/kube-yaml |
| `kubernetes` | A single Kubernetes cluster/context | `container-manager-kubernetes-operations` |
| `multi` (or `MULTI_CONTEXT_MODE=true`) | A **pool** of Kubernetes contexts + Docker hosts + Swarm managers + local Podman, selected per call | `container-manager-multi-context` |

Auto-detection: if `CONTAINER_MANAGER_TYPE` is unset, the server probes for a
reachable Docker socket first. Set it explicitly rather than relying on
auto-detection whenever the target is Kubernetes or Podman, or when running in
`multi` mode.

## The critical split: remote Docker/Podman hosts vs remote Kubernetes clusters
These use **two entirely different connection models** — do not conflate them.

**Remote Docker/Podman hosts → tunnel-manager inventory (`host=` alias).**
A remote Docker or Podman engine is reached over SSH through the shared
tunnel-manager inventory. You never put a hostname/IP directly into a `cm_*` call —
you pass a **host alias** that the tunnel-manager already knows about.
```
cm_list_hosts                              # enumerate available aliases
cm_container_operations action=list_containers host=<alias>
```
`CONTAINER_MANAGER_HOST` sets the *default* alias so `host=` can be omitted per call.
An unregistered alias fails resolution — the alias must already exist in the
tunnel-manager inventory before container-manager-mcp can use it.

**Remote Kubernetes clusters → kubeconfig contexts (`K8S_CONTEXTS` / kubecontext).**
A remote Kubernetes cluster is reached the standard kubeconfig way — a `context`
entry (cluster + user + namespace) already resolvable by the Kubernetes Python
client, not a tunnel-manager alias.
```
cm_k8s_cluster action=list_contexts        # enumerate kubeconfig contexts
cm_k8s_cluster action=use_context context_name=<name>
```
Single-context mode: `CONTAINER_MANAGER_KUBECONTEXT=<name>` selects one context for
the whole server. Multi-context mode: `K8S_CONTEXTS="a=ctx-a;b=ctx-b"` +
`DEFAULT_K8S_CONTEXT` registers a pool, selected per call via
`container-manager-multi-context` (`backend=kubernetes context=<name>`).

## Per-theme k8s tool toggles
Each of the eight `cm_k8s_*` tools can be individually hidden (e.g. to shrink the
tool surface exposed to a given agent) via a `TOOL=false` env var:
`K8SWORKLOADSTOOL`, `K8SCONFIGTOOL`, `K8SNETWORKINGTOOL`, `K8SSTORAGETOOL`,
`K8SRBACTOOL`, `K8SCLUSTERTOOL`, `K8SGOVERNANCETOOL`, `K8SOBSERVABILITYTOOL` — all
default `true`. The Podman and Docker Swarm tools have their own toggles: `PODMANTOOL`,
`DOCKERSWARMTOOL`, `MULTICONTEXTTOOL` — also default `true`.

## Full environment reference
| Variable | Applies to | Notes |
|----------|-----------|-------|
| `CONTAINER_MANAGER_TYPE` | all | `docker` \| `podman` \| `kubernetes` \| `multi`; auto-detects Docker if unset |
| `CONTAINER_MANAGER_HOST` | Docker/Podman | Default tunnel-manager host alias (else LOCAL socket) |
| `CONTAINER_MANAGER_PODMAN_BASE_URL` | Podman | Explicit Podman REST base URL override |
| `CONTAINER_MANAGER_KUBECONTEXT` | Kubernetes (single-context) | kubeconfig context name; else kubeconfig's current-context |
| `K8S_CONTEXTS` / `DEFAULT_K8S_CONTEXT` | Kubernetes (multi-context) | `"name=kubecontext;..."` pool + default key |
| `DOCKER_CONTEXTS` / `DEFAULT_DOCKER_CONTEXT` | Docker (multi-context) | `"name=host-alias;..."` pool + default key |
| `SWARM_CONTEXTS` / `DEFAULT_SWARM_CONTEXT` | Swarm (multi-context) | `"name=manager-host-alias;..."` pool + default key |
| `PODMAN_ENABLED` | multi-context | Default `true`; enables the local-only Podman pool member |
| `MULTI_CONTEXT_MODE` | multi-context | Forces multi mode even with a single pool entry |
| `MULTI_CONTEXT_MAX_WORKERS` | multi-context | Fan-out thread-pool size, default `8` |
| `MCP_TOOL_MODE` | all | `condensed` \| `verbose` \| `both`; these skills assume `condensed` |
| `K8SWORKLOADSTOOL` .. `K8SOBSERVABILITYTOOL` | Kubernetes | Per-theme tool visibility toggles, default `true` |
| `PODMANTOOL` / `DOCKERSWARMTOOL` / `MULTICONTEXTTOOL` | Podman / Docker Swarm / multi-context | Tool visibility toggles, default `true` |

## Walkthrough — first-run checklist
1. **Identify what container-manager-mcp is currently pointed at.**
   ```
   cm_info_operations action=get_info
   ```
2. **If Docker/Podman:** enumerate remote host aliases.
   ```
   cm_list_hosts
   ```
   No aliases and you need a remote engine → register the host in the
   tunnel-manager inventory first (outside this skill's scope); local-socket
   operation needs no alias.
3. **If Kubernetes:** enumerate kubeconfig contexts and pick one.
   ```
   cm_k8s_cluster action=list_contexts
   cm_k8s_cluster action=use_context context_name=<name>
   cm_k8s_cluster action=get_cluster_info
   ```
4. **If multi:** enumerate the configured pool across all backends.
   ```
   cm_multi_context action=list_contexts
   ```
5. **Hand off** to the matching operational skill from the table above.

## Gotchas
- Setting `K8S_CONTEXTS` alone does **not** put the server into multi-context mode
  unless `CONTAINER_MANAGER_TYPE=multi` or `MULTI_CONTEXT_MODE=true` — a
  single-context deploy should use `CONTAINER_MANAGER_KUBECONTEXT` instead.
- A tunnel-manager host alias and a kubeconfig context name are **not
  interchangeable** — passing a k8s context name as `host=` (or vice versa) fails
  resolution with an unhelpful "unknown alias/context" error.
- `PODMAN_ENABLED` only affects the **multi-context** Podman pool member; a plain
  `CONTAINER_MANAGER_TYPE=podman` deploy ignores it entirely.
- Hiding a `cm_k8s_*` theme via its `TOOL=false` toggle removes it from the tool
  list the agent even sees — a "tool not found" symptom often traces back here,
  not to a bug.
- `MCP_TOOL_MODE=verbose` exposes a different (non-condensed) tool shape not
  covered by these skills' recipes.

## Related
- **`container-manager-kubernetes-operations`** — full k8s operational surface
  once a context is selected.
- **`container-manager-lifecycle`** / **`container-manager-swarm`** /
  **`container-manager-podman-operations`** — Docker/Podman/Swarm operational
  surfaces once a host is selected.
- **`container-manager-multi-context`** — operate the configured pool once it's
  populated.

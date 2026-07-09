---
name: container-manager-kg-ingestion
description: >-
  Snapshot a host's Docker/Podman/Swarm inventory into the epistemic-graph
  knowledge graph as typed OWL nodes via the container-manager-mcp MCP server —
  containers, images, volumes, networks, swarm services and nodes, with their
  :usesImage / :runsOn links. Use when the agent must record live container
  state into the KG for cross-source reasoning, drift detection, or provenance.
  Do NOT use for operating containers (use container-manager-lifecycle) or
  managing swarm services (use container-manager-swarm).
license: MIT
tags: [container-manager-mcp, knowledge-graph, ingestion, ontology, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Container Inventory → Knowledge Graph

Push a host's live container inventory into the ONE epistemic-graph knowledge graph as
**typed OWL nodes**, natively from the **container-manager-mcp** server. Best-effort and
engine-guarded: with no reachable KG engine the tool still lists the inventory and reports
`ingested: null` per modality.

## When to use
- Record the current containers/images/volumes/networks (and swarm services/nodes) of a
  host into the KG for later querying, drift detection, or cross-source joins.
- Record a Kubernetes cluster's live pods/deployments/namespaces/native services into
  the KG (when `CONTAINER_MANAGER_TYPE=kubernetes`).
- Refresh the KG snapshot after a deploy or before an audit.

## When NOT to use
- Operating containers (start/stop/logs/exec) → `container-manager-lifecycle`.
- Managing swarm services/nodes → `container-manager-swarm`.
- Generic KG queries → the graph-os `graph_query` / `graph_search` surface.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`container-manager-mcp`** MCP server.
A reachable Docker/Podman engine is required; a reachable epistemic-graph engine is
optional (ingestion no-ops cleanly without one).

| Variable | Required | Notes |
|----------|----------|-------|
| `CONTAINER_MANAGER_TYPE` | optional | `docker` / `podman`; auto-detected |
| `CONTAINER_MANAGER_HOST` | optional | Default host alias (else LOCAL socket) |

The typed nodes match `container_manager_mcp.ontology` (`container.ttl`), federated into
the hub under `http://knuckles.team/kg/container` (reusing the shared `:Container`,
`:ContainerImage`, `:ContainerStack`, `:Host` classes).

## Tools & actions
| Tool | Modalities |
|------|------------|
| `cm_ingest_inventory` | `all`, `containers`, `images`, `volumes`, `networks`, `services`, `nodes` |

> **Kubernetes modalities (`pods`, `deployments`, `namespaces`, `k8s_services`):**
> planned extensions of `cm_ingest_inventory` to snapshot a Kubernetes cluster
> (`:Pod` / `:Deployment` / `:Namespace` / `:K8sService` nodes, mirroring the
> containers/images pattern) alongside the existing Docker/Podman/Swarm modalities.
> As of this skill's current version the `cm_ingest_inventory` tool's `modality`
> literal does **not yet** accept these values — calling them raises a validation
> error. Until that lands, snapshot Kubernetes state by reading it live through
> `container-manager-kubernetes-operations` (`cm_k8s_workloads action=list_pods`,
> `cm_k8s_config action=list_namespaces`, `cm_k8s_networking action=list_k8s_services`)
> and hand-mapping into the KG via `graph_write` if a durable snapshot is needed now.

### Kubernetes modalities (once wired)
| Modality | Node type | Source list | Links |
|----------|-----------|-------------|-------|
| `pods` | `:Pod` | `cm_k8s_workloads action=list_pods` | `:runsOn` → `:Host`/Node, `:usesImage` → `:ContainerImage` |
| `deployments` | `:Deployment` | `cm_k8s_workloads` (StatefulSet/DaemonSet/ReplicaSet listers) | `:scheduledOnNode` fan-out via owned Pods |
| `namespaces` | `:Namespace` | `cm_k8s_config action=list_namespaces` | scoping context for the above |
| `k8s_services` | `:K8sService` | `cm_k8s_networking action=list_k8s_services` | `:selects` → `:Pod` via selector |

### Key parameters
- `modality` — which inventory to sweep (`all` covers everything; swarm modalities are
  skipped gracefully off a manager).
- `host` — remote alias (omit for LOCAL); swarm modalities need a **manager**.
- `all_containers` — include stopped containers (default `true`).

### Node model
| Modality | Node type | Id scheme | Links |
|----------|-----------|-----------|-------|
| containers | `:Container` | `container:container:<id>` | `:usesImage` → `:ContainerImage`, `:runsOn` → `:Host` |
| images | `:ContainerImage` | `container:image:<id-or-ref>` | — |
| volumes | `:ContainerVolume` | `container:volume:<name>` | — |
| networks | `:ContainerNetwork` | `container:network:<id>` | — |
| services | `:SwarmService` | `container:service:<id>` | `:usesImage` → `:ContainerImage` |
| nodes | `:SwarmNode` | `container:node:<id>` | — |

## Recipes
Full local snapshot:
```
cm_ingest_inventory modality=all
```
Only running-vs-all containers on a remote host:
```
cm_ingest_inventory modality=containers host=<alias> all_containers=true
```
Swarm services + nodes from a manager:
```
cm_ingest_inventory modality=services host=<manager-alias>
cm_ingest_inventory modality=nodes host=<manager-alias>
```
Kubernetes snapshot (once `pods`/`deployments`/`namespaces`/`k8s_services` modalities
land — see the note above; today, read live via `container-manager-kubernetes-operations`):
```
cm_ingest_inventory modality=pods
cm_ingest_inventory modality=namespaces
cm_ingest_inventory modality=k8s_services
```

## Gotchas
- Returns `{"modalities": {<name>: {"listed": n, "ingested": {...}|null}}}`; `ingested:null`
  means no KG engine was reachable — the listing still succeeded.
- Swarm modalities off a non-manager host surface a per-modality `error` but do not abort
  the rest of the sweep.
- Node ids are content-addressed from the engine (`container:<class>:<extId>`), so re-runs
  MERGE (update-in-place) rather than duplicate.
- This is a read+ingest tool (idempotent); it never mutates the containers themselves.

## Related
- **`container-manager-lifecycle`** / **`container-manager-swarm`** /
  **`container-manager-kubernetes-operations`** — produce the inventory this skill
  snapshots (or, for Kubernetes today, the live reads to hand-map until the
  `pods`/`deployments`/`namespaces`/`k8s_services` modalities are wired).
- The underlying mapper lives in `container_manager_mcp.kg_ingest`
  (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

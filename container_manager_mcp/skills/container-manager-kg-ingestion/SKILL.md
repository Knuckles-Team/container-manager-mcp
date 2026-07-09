---
name: container-manager-kg-ingestion
skill_type: skill
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

## Gotchas
- Returns `{"modalities": {<name>: {"listed": n, "ingested": {...}|null}}}`; `ingested:null`
  means no KG engine was reachable — the listing still succeeded.
- Swarm modalities off a non-manager host surface a per-modality `error` but do not abort
  the rest of the sweep.
- Node ids are content-addressed from the engine (`container:<class>:<extId>`), so re-runs
  MERGE (update-in-place) rather than duplicate.
- This is a read+ingest tool (idempotent); it never mutates the containers themselves.

## Related
- **`container-manager-lifecycle`** / **`container-manager-swarm`** — produce the inventory
  this skill snapshots.
- The underlying mapper lives in `container_manager_mcp.kg_ingest`
  (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

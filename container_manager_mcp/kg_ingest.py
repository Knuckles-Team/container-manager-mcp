"""Native epistemic-graph ingestion for container-manager records (typed graph nodes).

CONCEPT:AU-KG.ingest.enterprise-source-extractor. The container-manager-mcp package
natively pushes its Docker / Podman / Swarm inventory into the ONE epistemic-graph
knowledge graph as **typed OWL nodes** (``:Container``, ``:ContainerImage``,
``:ContainerVolume``, ``:ContainerNetwork``, ``:SwarmService``, ``:SwarmNode``) + links
(``:usesImage`` / ``:runsOn`` / ``:scheduledOnNode``), using the lightweight engine client
(``GraphComputeEngine()._client`` + ``txn``) — the same fast client the blob ``MediaStore``
uses, NOT the heavy in-process ingestion engine.

Everything is dependency-/engine-guarded: with no agent-utilities KG stack or no reachable
engine, every entry point **no-ops** (returns ``None``), so the connector keeps working with
zero KG infrastructure. It first tries the shared primitive
``agent_utilities.knowledge_graph.memory.native_ingest``; if that is not present in the
installed agent_utilities, it falls back to a self-contained txn writer. Node ids follow
``container:<class>:<extId>`` and ``type`` matches the classes federated by
``container_manager_mcp.ontology``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("container_manager_mcp.kg")

_SOURCE = "container-manager-mcp"
_DOMAIN = "container"
_DEFAULT_GRAPH = "__commons__"


def _client() -> tuple[Any | None, str]:
    """Return ``(engine_client, graph_name)`` or ``(None, "")`` when unavailable."""
    try:
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )
    except Exception as e:  # noqa: BLE001 — KG stack absent
        logger.debug("KG ingest unavailable (import): %s", e)
        return None, ""
    try:
        engine = GraphComputeEngine()
        client = getattr(engine, "_client", None)
        if client is None:
            return None, ""
        return client, (getattr(engine, "graph_name", None) or _DEFAULT_GRAPH)
    except Exception as e:  # noqa: BLE001 — engine unreachable
        logger.debug("KG ingest: engine unreachable: %s", e)
        return None, ""


def ingest_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Write typed nodes (+ edges) into epistemic-graph.

    Prefers the shared ``native_ingest.ingest_entities`` primitive; falls back to a
    self-contained txn writer when it is absent. ``entities``:
    ``[{"id":..., "type":<owl:Class>, ...props}]``; ``relationships``:
    ``[{"source":id, "target":id, "type":rel}]``. Returns ``{"nodes":n, "edges":m}``
    or ``None`` (no engine / failure; never raises).
    """
    entities = [e for e in (entities or []) if e.get("id")]
    if not entities:
        return None

    # Preferred path: the shared fleet primitive.
    if client is None:
        try:
            from agent_utilities.knowledge_graph.memory.native_ingest import (
                ingest_entities as _shared_ingest,
            )

            return _shared_ingest(
                entities,
                relationships,
                source=_SOURCE,
                domain=_DOMAIN,
            )
        except Exception as e:  # noqa: BLE001 — primitive absent, fall back
            logger.debug("KG ingest: shared primitive unavailable: %s", e)

    # Fallback / injected-client path: self-contained txn writer.
    if client is None:
        client, graph = _client()
    if client is None:
        return None
    graph = graph or _DEFAULT_GRAPH

    try:
        txn = client.txn.begin(graph=graph)
        for ent in entities:
            props = {k: v for k, v in ent.items() if k != "id" and v is not None}
            props.setdefault("source", _SOURCE)
            props.setdefault("domain", _DOMAIN)
            client.txn.add_node(txn, ent["id"], props)
        committed = client.txn.commit(txn)
    except Exception as e:  # noqa: BLE001 — engine/txn failure is non-fatal
        logger.warning("KG ingest: txn failed: %s", e)
        return None
    if not committed:
        logger.warning("KG ingest: txn not committed (conflict)")
        return None

    edges = 0
    for rel in relationships or []:
        try:
            client.edges.add(
                rel["source"], rel["target"], {"type": rel.get("type", "RELATED")}
            )
            edges += 1
        except Exception as e:  # noqa: BLE001 — pure edge link, best-effort
            logger.debug("KG ingest: edge skipped: %s", e)

    logger.info("KG ingest: wrote %d nodes, %d edges", len(entities), edges)
    return {"nodes": len(entities), "edges": edges}


def _s(value: Any) -> str | None:
    """Coerce to a non-empty string or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def ingest_containers(
    containers: list[dict[str, Any]],
    *,
    host: str | None = None,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Map container records (``ContainerInfo``) → ``:Container`` (+ ``:ContainerImage``) nodes.

    Each record carries ``id`` / ``name`` / ``image`` / ``status`` / ``ports`` / ``created``
    (see ``ContainerManagerBase.list_containers``). Emits a ``:Container`` node, an optional
    ``:ContainerImage`` node it ``:usesImage``, and an optional ``:Host`` ``:runsOn`` link.
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    host_id = f"container:host:{host}" if host else None
    seen_images: set[str] = set()
    for rec in containers or []:
        cid = _s(rec.get("id"))
        if not cid:
            continue
        node_id = f"container:container:{cid}"
        image_ref = _s(rec.get("image"))
        entities.append(
            {
                "id": node_id,
                "type": "Container",
                "name": _s(rec.get("name")),
                "image": image_ref,
                "containerStatus": _s(rec.get("status")),
                "portMappings": _s(rec.get("ports")),
                "created_at": _s(rec.get("created")),
                "runtime": _s(rec.get("runtime")) or "docker",
                "host": host,
                "externalToolId": cid,
            }
        )
        if image_ref and image_ref not in ("unknown", "none"):
            img_id = f"container:image:{image_ref}"
            if image_ref not in seen_images:
                seen_images.add(image_ref)
                entities.append(
                    {
                        "id": img_id,
                        "type": "ContainerImage",
                        "name": image_ref,
                        "externalToolId": image_ref,
                    }
                )
            relationships.append(
                {"source": node_id, "target": img_id, "type": "usesImage"}
            )
        if host_id:
            relationships.append(
                {"source": node_id, "target": host_id, "type": "runsOn"}
            )
    if host_id and entities:
        entities.append({"id": host_id, "type": "Host", "name": host})
    return ingest_entities(entities, relationships, client=client, graph=graph)


def ingest_images(
    images: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Map image records (``ImageInfo``) → ``:ContainerImage`` nodes."""
    entities: list[dict[str, Any]] = []
    for rec in images or []:
        iid = _s(rec.get("id"))
        repo = _s(rec.get("repository"))
        tag = _s(rec.get("tag"))
        ext = iid or (f"{repo}:{tag}" if repo else None)
        if not ext:
            continue
        entities.append(
            {
                "id": f"container:image:{ext}",
                "type": "ContainerImage",
                "name": f"{repo}:{tag}" if repo and tag else (repo or ext),
                "imageRepository": repo,
                "imageTag": tag,
                "imageSize": _s(rec.get("size")),
                "created_at": _s(rec.get("created")),
                "externalToolId": ext,
            }
        )
    return ingest_entities(entities, None, client=client, graph=graph)


def ingest_volumes(
    volumes: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Map volume records (``VolumeInfo``) → ``:ContainerVolume`` nodes."""
    entities: list[dict[str, Any]] = []
    for rec in volumes or []:
        name = _s(rec.get("name"))
        if not name:
            continue
        entities.append(
            {
                "id": f"container:volume:{name}",
                "type": "ContainerVolume",
                "name": name,
                "volumeDriver": _s(rec.get("driver")),
                "mountpoint": _s(rec.get("mountpoint")),
                "created_at": _s(rec.get("created")),
                "externalToolId": name,
            }
        )
    return ingest_entities(entities, None, client=client, graph=graph)


def ingest_networks(
    networks: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Map network records (``NetworkInfo``) → ``:ContainerNetwork`` nodes."""
    entities: list[dict[str, Any]] = []
    for rec in networks or []:
        nid = _s(rec.get("id")) or _s(rec.get("name"))
        if not nid:
            continue
        entities.append(
            {
                "id": f"container:network:{nid}",
                "type": "ContainerNetwork",
                "name": _s(rec.get("name")),
                "networkDriver": _s(rec.get("driver")),
                "networkScope": _s(rec.get("scope")),
                "externalToolId": nid,
            }
        )
    return ingest_entities(entities, None, client=client, graph=graph)


def ingest_services(
    services: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Map swarm service records → ``:SwarmService`` (+ ``:ContainerImage``) nodes."""
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for rec in services or []:
        sid = _s(rec.get("id")) or _s(rec.get("name"))
        if not sid:
            continue
        node_id = f"container:service:{sid}"
        image_ref = _s(rec.get("image"))
        replicas = rec.get("replicas")
        entities.append(
            {
                "id": node_id,
                "type": "SwarmService",
                "name": _s(rec.get("name")),
                "image": image_ref,
                "serviceReplicas": (
                    int(replicas)
                    if isinstance(replicas, (int, str)) and str(replicas).isdigit()
                    else None
                ),
                "portMappings": _s(rec.get("ports")),
                "created_at": _s(rec.get("created")),
                "updated_at": _s(rec.get("updated")),
                "externalToolId": sid,
            }
        )
        if image_ref and image_ref not in ("unknown", "none"):
            img_id = f"container:image:{image_ref}"
            if image_ref not in seen_images:
                seen_images.add(image_ref)
                entities.append(
                    {
                        "id": img_id,
                        "type": "ContainerImage",
                        "name": image_ref,
                        "externalToolId": image_ref,
                    }
                )
            relationships.append(
                {"source": node_id, "target": img_id, "type": "usesImage"}
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)


def ingest_nodes(
    nodes: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Map swarm node records → ``:SwarmNode`` nodes."""
    entities: list[dict[str, Any]] = []
    for rec in nodes or []:
        nid = _s(rec.get("id")) or _s(rec.get("hostname"))
        if not nid:
            continue
        entities.append(
            {
                "id": f"container:node:{nid}",
                "type": "SwarmNode",
                "name": _s(rec.get("hostname")),
                "nodeRole": _s(rec.get("role")),
                "containerStatus": _s(rec.get("status")),
                "nodeAvailability": _s(rec.get("availability")),
                "created_at": _s(rec.get("created")),
                "updated_at": _s(rec.get("updated")),
                "externalToolId": nid,
            }
        )
    return ingest_entities(entities, None, client=client, graph=graph)

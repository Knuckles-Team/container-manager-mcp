"""Native epistemic-graph ingestion for container-manager records.

CONCEPT:AU-KG.ingest.enterprise-source-extractor. Connector-specific mappers emit
canonical node_type nodes and relationship edges. The required agent-utilities
native-ingest primitive owns the transaction and raises NativeIngestError when the
authoritative engine cannot commit.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from agent_utilities.knowledge_graph.memory.native_ingest import (
    ingest_entities as _native_ingest_entities,
)

_SOURCE = "container-manager-mcp"
_DOMAIN = "container"

# OCI image-source labels, in priority order: the standard
# ``org.opencontainers.image.source`` annotation, falling back to the legacy
# ``org.label-schema.vcs-url`` convention some older images still carry.
_SOURCE_LABEL = "org.opencontainers.image.source"
_VCS_URL_LABEL = "org.label-schema.vcs-url"


def ingest_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    source: str = _SOURCE,
    domain: str = _DOMAIN,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Write canonical typed nodes and relationships through agent-utilities."""
    return _native_ingest_entities(
        entities,
        relationships,
        source=source,
        domain=domain,
        client=client,
        graph=graph,
    )


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
) -> dict[str, int]:
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
                "node_type": "Container",
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
                        "node_type": "ContainerImage",
                        "name": image_ref,
                        "externalToolId": image_ref,
                    }
                )
            relationships.append(
                {"source": node_id, "target": img_id, "relationship": "usesImage"}
            )
        if host_id:
            relationships.append(
                {"source": node_id, "target": host_id, "relationship": "runsOn"}
            )
    if host_id and entities:
        entities.append({"id": host_id, "node_type": "Host", "name": host})
    return ingest_entities(entities, relationships, client=client, graph=graph)


def _normalize_source_url(url: Any) -> tuple[str, str] | None:
    """Normalize an OCI source-label URL -> ``(clean_url, repo_node_id)``, or ``None``.

    Mirrors ``portainer_agent.kg_ingest._normalize_repo_url``'s ``git:repo:<host>/<path>``
    id convention (so a deployed ``:Stack``'s ``:Repository`` and this image's
    ``:Repository`` land on the same node id), plus the case/``www.``-insensitive
    normalization ``agent_utilities.observability.repo_crosswalk.normalize_clone_url``
    uses (so the same repo always yields the same id regardless of label casing) —
    the crosswalk then ``owl:sameAs``-unifies this URL-keyed node with the numeric-id
    node the code ingestors (github-agent/gitlab-api) create. Handles HTTP(S) and
    SCP-style (``git@host:owner/name.git``) remotes; strips embedded credentials, a
    trailing ``.git``/slash and a leading ``www.``. Returns ``None`` for an unusable
    value.
    """
    if not url or not isinstance(url, str):
        return None
    raw = url.strip()
    if not raw:
        return None
    if "://" in raw:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
        path = parts.path or ""
    elif "@" in raw and ":" in raw:
        # SCP-style remote, e.g. git@github.com:owner/name.git
        _, _, rest = raw.partition("@")
        host, _, path = rest.partition(":")
        host = host.lower()
    else:
        return None
    host = re.sub(r"^www\.", "", host)
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    path = path.lower()
    if not host or not path:
        return None
    return f"https://{host}/{path}", f"git:repo:{host}/{path}"


def ingest_images(
    images: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Map image records (``ImageInfo``) → ``:ContainerImage`` nodes.

    When an image record carries a ``labels`` dict with the OCI source label
    (``org.opencontainers.image.source``, falling back to the legacy
    ``org.label-schema.vcs-url``), also emits a ``:Repository`` node and a
    ``:builtFrom`` edge (``:ContainerImage -[:builtFrom]-> :Repository``) — the
    image->source-repo provenance hop (gap #1,
    ``reports/autonomous-sdlc-loop-design.md``). No label -> no edge (graceful).
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    seen_repos: set[str] = set()
    for rec in images or []:
        iid = _s(rec.get("id"))
        repo = _s(rec.get("repository"))
        tag = _s(rec.get("tag"))
        ext = iid or (f"{repo}:{tag}" if repo else None)
        if not ext:
            continue
        img_id = f"container:image:{ext}"
        entities.append(
            {
                "id": img_id,
                "node_type": "ContainerImage",
                "name": f"{repo}:{tag}" if repo and tag else (repo or ext),
                "imageRepository": repo,
                "imageTag": tag,
                "imageSize": _s(rec.get("size")),
                "created_at": _s(rec.get("created")),
                "externalToolId": ext,
            }
        )
        labels = rec.get("labels") or {}
        source_url = _s(labels.get(_SOURCE_LABEL)) or _s(labels.get(_VCS_URL_LABEL))
        normalized = _normalize_source_url(source_url) if source_url else None
        if normalized:
            clean_url, repo_node = normalized
            if repo_node not in seen_repos:
                seen_repos.add(repo_node)
                entities.append(
                    {
                        "id": repo_node,
                        "node_type": "Repository",
                        "url": clean_url,
                    }
                )
            relationships.append(
                {
                    "source": img_id,
                    "target": repo_node,
                    "relationship": "builtFrom",
                }
            )
    return ingest_entities(entities, relationships or None, client=client, graph=graph)


def ingest_volumes(
    volumes: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Map volume records (``VolumeInfo``) → ``:ContainerVolume`` nodes."""
    entities: list[dict[str, Any]] = []
    for rec in volumes or []:
        name = _s(rec.get("name"))
        if not name:
            continue
        entities.append(
            {
                "id": f"container:volume:{name}",
                "node_type": "ContainerVolume",
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
) -> dict[str, int]:
    """Map network records (``NetworkInfo``) → ``:ContainerNetwork`` nodes."""
    entities: list[dict[str, Any]] = []
    for rec in networks or []:
        nid = _s(rec.get("id")) or _s(rec.get("name"))
        if not nid:
            continue
        entities.append(
            {
                "id": f"container:network:{nid}",
                "node_type": "ContainerNetwork",
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
) -> dict[str, int]:
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
                "node_type": "SwarmService",
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
                        "node_type": "ContainerImage",
                        "name": image_ref,
                        "externalToolId": image_ref,
                    }
                )
            relationships.append(
                {"source": node_id, "target": img_id, "relationship": "usesImage"}
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)


def ingest_nodes(
    nodes: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Map swarm node records → ``:SwarmNode`` nodes."""
    entities: list[dict[str, Any]] = []
    for rec in nodes or []:
        nid = _s(rec.get("id")) or _s(rec.get("hostname"))
        if not nid:
            continue
        entities.append(
            {
                "id": f"container:node:{nid}",
                "node_type": "SwarmNode",
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


def ingest_pods(
    records: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Map Kubernetes pod records (``list_pods``) → ``:Pod`` nodes.

    Each record carries ``name`` / ``namespace`` / ``status`` (phase) / ``node`` /
    ``created`` / ``labels``. Emits a ``:Pod`` node with ``+podPhase`` and conditional
    ``:runsInNamespace`` (-> ``:Namespace``), ``:managedByDeployment`` (-> ``:Deployment``,
    when a ``deployment`` field is present) and ``:scheduledOnK8sNode`` (-> ``:K8sNode``) links.
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for rec in records or []:
        name = _s(rec.get("name"))
        if not name:
            continue
        ns = _s(rec.get("namespace"))
        node_id = f"container:pod:{ns}/{name}" if ns else f"container:pod:{name}"
        entities.append(
            {
                "id": node_id,
                "node_type": "Pod",
                "name": name,
                "namespace": ns,
                "podPhase": _s(rec.get("status")),
                "created_at": _s(rec.get("created")),
                "externalToolId": name,
            }
        )
        if ns:
            relationships.append(
                {
                    "source": node_id,
                    "target": f"container:namespace:{ns}",
                    "relationship": "runsInNamespace",
                }
            )
        dep = _s(rec.get("deployment"))
        if dep:
            relationships.append(
                {
                    "source": node_id,
                    "target": f"container:deployment:{dep}",
                    "relationship": "managedByDeployment",
                }
            )
        k8s_node = _s(rec.get("node"))
        if k8s_node:
            relationships.append(
                {
                    "source": node_id,
                    "target": f"container:k8snode:{k8s_node}",
                    "relationship": "scheduledOnK8sNode",
                }
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)


def ingest_deployments(
    records: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Map Kubernetes deployment records (Deployment-shaped ``list_services``) → ``:Deployment`` nodes.

    Each record carries ``id`` / ``name`` / ``namespace`` / ``image`` / ``replicas`` /
    ``ports`` / ``created``. Emits a ``:Deployment`` node with ``+deploymentReplicas``
    (and ``+deploymentReadyReplicas`` when present), an optional ``:ContainerImage`` it
    ``:usesImage``, and a conditional ``:runsInNamespace`` link.
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for rec in records or []:
        did = _s(rec.get("id")) or _s(rec.get("name"))
        if not did:
            continue
        node_id = f"container:deployment:{did}"
        ns = _s(rec.get("namespace"))
        image_ref = _s(rec.get("image"))
        replicas = rec.get("replicas")
        ready = rec.get("ready_replicas")
        entities.append(
            {
                "id": node_id,
                "node_type": "Deployment",
                "name": _s(rec.get("name")),
                "namespace": ns,
                "image": image_ref,
                "deploymentReplicas": (
                    int(replicas)
                    if isinstance(replicas, (int, str)) and str(replicas).isdigit()
                    else None
                ),
                "deploymentReadyReplicas": (
                    int(ready)
                    if isinstance(ready, (int, str)) and str(ready).isdigit()
                    else None
                ),
                "portMappings": _s(rec.get("ports")),
                "created_at": _s(rec.get("created")),
                "updated_at": _s(rec.get("updated")),
                "externalToolId": did,
            }
        )
        if image_ref and image_ref not in ("unknown", "none"):
            img_id = f"container:image:{image_ref}"
            if image_ref not in seen_images:
                seen_images.add(image_ref)
                entities.append(
                    {
                        "id": img_id,
                        "node_type": "ContainerImage",
                        "name": image_ref,
                        "externalToolId": image_ref,
                    }
                )
            relationships.append(
                {"source": node_id, "target": img_id, "relationship": "usesImage"}
            )
        if ns:
            relationships.append(
                {
                    "source": node_id,
                    "target": f"container:namespace:{ns}",
                    "relationship": "runsInNamespace",
                }
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)


def ingest_namespaces(
    records: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Map Kubernetes namespace records (``list_namespaces``) → ``:Namespace`` nodes."""
    entities: list[dict[str, Any]] = []
    for rec in records or []:
        name = _s(rec.get("name"))
        if not name:
            continue
        entities.append(
            {
                "id": f"container:namespace:{name}",
                "node_type": "Namespace",
                "name": name,
                "namespaceStatus": _s(rec.get("status")),
                "created_at": _s(rec.get("created")),
                "externalToolId": name,
            }
        )
    return ingest_entities(entities, None, client=client, graph=graph)


def ingest_k8s_services(
    records: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Map native Kubernetes service records (``list_native_services``) → ``:K8sService`` nodes.

    Each record carries ``name`` / ``namespace`` / ``type`` / ``cluster_ip`` / ``ports`` /
    ``created``. Emits a ``:K8sService`` node with ``+serviceType`` and a conditional
    ``:runsInNamespace`` (-> ``:Namespace``) link.
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for rec in records or []:
        name = _s(rec.get("name"))
        if not name:
            continue
        ns = _s(rec.get("namespace"))
        node_id = (
            f"container:k8sservice:{ns}/{name}"
            if ns
            else f"container:k8sservice:{name}"
        )
        entities.append(
            {
                "id": node_id,
                "node_type": "K8sService",
                "name": name,
                "namespace": ns,
                "serviceType": _s(rec.get("type")),
                "created_at": _s(rec.get("created")),
                "externalToolId": name,
            }
        )
        if ns:
            relationships.append(
                {
                    "source": node_id,
                    "target": f"container:namespace:{ns}",
                    "relationship": "runsInNamespace",
                }
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)

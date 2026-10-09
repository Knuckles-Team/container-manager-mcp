"""Native epistemic-graph typed-node ingestion — Wire-First coverage.

Exercises the real ``ingest_entities`` + per-modality mappers with a fake
transport one level below ``agent_connector_sdk.ingest.KnowledgeIngest`` (no
engine required), so the SDK's own request-building/validation/privacy-guard
contract runs unfaked, asserting the container-manager record → typed-node
mapping.
CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from agent_connector_sdk.ingest import IngestError, KnowledgeIngest

from container_manager_mcp.kg_ingest import (
    ingest_containers,
    ingest_deployments,
    ingest_entities,
    ingest_images,
    ingest_k8s_services,
    ingest_namespaces,
    ingest_networks,
    ingest_nodes,
    ingest_pods,
    ingest_services,
    ingest_volumes,
)


class _FakeTransport:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def source_status(self, connector: str, stream: str):
        return SimpleNamespace(accepted_checkpoint=None)

    async def submit(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            affected_count=len(request.records),
            relationship_count=len(request.relationships),
        )

    async def store_blob(self, data: bytes) -> str:
        raise AssertionError("this connector's node/edge ingestion carries no media")


@pytest.fixture
def ingest():
    transport = _FakeTransport()
    return KnowledgeIngest(transport, loop=None), transport


def _by_id(request):
    return {record.record_id: record for record in request.records}


def _edge_types(request):
    return {
        (
            rel.source.record_id,
            rel.target.record_id,
            rel.relation_reference.rsplit("/relations/", 1)[-1],
        )
        for rel in request.relationships
    }


def _is_type(record, type_name: str) -> bool:
    return record.mapping_reference.endswith(f"/{type_name}")


async def test_ingest_entities_writes_nodes_and_edges(ingest):
    service, transport = ingest
    res = await ingest_entities(
        [
            {"id": "a", "node_type": "Container", "name": "web"},
            {"id": "b", "node_type": "ContainerImage"},
        ],
        [{"source": "a", "target": "b", "relationship": "usesImage"}],
        ingest=service,
    )
    assert res == {"nodes": 2, "edges": 1}
    request = transport.requests[0]
    assert set(_by_id(request)) == {"a", "b"}
    assert _edge_types(request) == {("a", "b", "usesImage")}


async def test_ingest_containers_maps_container_image_and_host(ingest):
    service, transport = ingest
    res = await ingest_containers(
        [
            {
                "id": "abc123",
                "name": "web",
                "image": "nginx:latest",
                "status": "running",
                "ports": "0.0.0.0:8080->80/tcp",
                "created": "2026-07-04T00:00:00Z",
            }
        ],
        host="test-node-1",
        ingest=service,
    )
    # container + image + host = 3 nodes; usesImage + runsOn = 2 edges
    assert res == {"nodes": 3, "edges": 2}
    request = transport.requests[0]
    by_id = _by_id(request)
    cont = by_id["container:container:abc123"]
    assert _is_type(cont, "Container")
    assert cont.payload["containerStatus"] == "running"
    # the SDK's persistence privacy guard redacts literal IPv4 addresses.
    assert cont.payload["portMappings"] == "[REDACTED_IPV4]:8080->80/tcp"
    assert cont.payload["externalToolId"] == "abc123"
    assert _is_type(by_id["container:image:nginx:latest"], "ContainerImage")
    assert _is_type(by_id["container:host:test-node-1"], "Host")
    edges = _edge_types(request)
    assert ("container:container:abc123", "container:image:nginx:latest", "usesImage") in edges
    assert ("container:container:abc123", "container:host:test-node-1", "runsOn") in edges


async def test_ingest_images_maps_repo_tag_size(ingest):
    service, transport = ingest
    res = await ingest_images(
        [
            {
                "id": "f00dcafe",
                "repository": "nginx",
                "tag": "latest",
                "size": "142MB",
                "created": "2026-07-01T00:00:00Z",
            }
        ],
        ingest=service,
    )
    assert res == {"nodes": 1, "edges": 0}
    img = _by_id(transport.requests[0])["container:image:f00dcafe"]
    assert _is_type(img, "ContainerImage")
    assert img.payload["imageRepository"] == "nginx"
    assert img.payload["imageTag"] == "latest"
    assert img.payload["imageSize"] == "142MB"


async def test_ingest_images_with_source_label_emits_repository_and_builtfrom(ingest):
    service, transport = ingest
    res = await ingest_images(
        [
            {
                "id": "f00dcafe",
                "repository": "nginx",
                "tag": "latest",
                "size": "142MB",
                "labels": {
                    "org.opencontainers.image.source": "https://github.com/Org/Repo.git"
                },
            }
        ],
        ingest=service,
    )
    assert res == {"nodes": 2, "edges": 1}
    request = transport.requests[0]
    repo = _by_id(request)["git:repo:github.com/org/repo"]
    assert _is_type(repo, "Repository")
    # the SDK's persistence privacy guard unconditionally redacts fields it
    # recognizes as location fields by name (e.g. "url") before durable storage.
    assert repo.payload["url"] == "[REDACTED_LOCATION]"
    assert (
        "container:image:f00dcafe",
        "git:repo:github.com/org/repo",
        "builtFrom",
    ) in _edge_types(request)


async def test_ingest_images_with_vcs_url_label_fallback(ingest):
    service, transport = ingest
    res = await ingest_images(
        [
            {
                "id": "cafef00d",
                "repository": "myapp",
                "tag": "1.0",
                "labels": {"org.label-schema.vcs-url": "git@gitlab.com:team/myapp.git"},
            }
        ],
        ingest=service,
    )
    assert res == {"nodes": 2, "edges": 1}
    request = transport.requests[0]
    assert "git:repo:gitlab.com/team/myapp" in _by_id(request)
    assert (
        "container:image:cafef00d",
        "git:repo:gitlab.com/team/myapp",
        "builtFrom",
    ) in _edge_types(request)


async def test_ingest_images_without_source_label_emits_no_builtfrom_edge(ingest):
    service, transport = ingest
    res = await ingest_images(
        [
            {
                "id": "deadbeef",
                "repository": "redis",
                "tag": "7",
                "labels": {"maintainer": "nobody"},
            }
        ],
        ingest=service,
    )
    assert res == {"nodes": 1, "edges": 0}
    assert not transport.requests[0].relationships


async def test_ingest_images_no_labels_at_all_is_graceful_noop_edge(ingest):
    service, transport = ingest
    res = await ingest_images(
        [{"id": "abc12345", "repository": "alpine", "tag": "latest"}], ingest=service
    )
    assert res == {"nodes": 1, "edges": 0}
    assert not transport.requests[0].relationships


async def test_ingest_volumes_and_networks(ingest):
    service, transport = ingest
    res = await ingest_volumes(
        [{"name": "pgdata", "driver": "local", "mountpoint": "/var/lib/x"}],
        ingest=service,
    )
    assert res == {"nodes": 1, "edges": 0}
    assert (
        _by_id(transport.requests[0])["container:volume:pgdata"].payload["volumeDriver"]
        == "local"
    )

    transport2 = _FakeTransport()
    service2 = KnowledgeIngest(transport2, loop=None)
    res2 = await ingest_networks(
        [{"id": "net1", "name": "backend", "driver": "overlay", "scope": "swarm"}],
        ingest=service2,
    )
    assert res2 == {"nodes": 1, "edges": 0}
    n = _by_id(transport2.requests[0])["container:network:net1"]
    assert _is_type(n, "ContainerNetwork")
    assert n.payload["networkDriver"] == "overlay"
    assert n.payload["networkScope"] == "swarm"


async def test_ingest_services_maps_replicas_and_image_edge(ingest):
    service, transport = ingest
    res = await ingest_services(
        [
            {
                "id": "svc1",
                "name": "web",
                "image": "nginx:latest",
                "replicas": 3,
                "ports": "8080->80/tcp",
            }
        ],
        ingest=service,
    )
    assert res == {"nodes": 2, "edges": 1}
    request = transport.requests[0]
    svc = _by_id(request)["container:service:svc1"]
    assert _is_type(svc, "SwarmService")
    assert svc.payload["serviceReplicas"] == 3
    assert _edge_types(request) == {
        ("container:service:svc1", "container:image:nginx:latest", "usesImage")
    }


async def test_ingest_nodes_maps_role_and_availability(ingest):
    service, transport = ingest
    res = await ingest_nodes(
        [
            {
                "id": "node1",
                "hostname": "rw710",
                "role": "manager",
                "status": "ready",
                "availability": "active",
            }
        ],
        ingest=service,
    )
    assert res == {"nodes": 1, "edges": 0}
    n = _by_id(transport.requests[0])["container:node:node1"]
    assert _is_type(n, "SwarmNode")
    assert n.payload["nodeRole"] == "manager"
    assert n.payload["nodeAvailability"] == "active"


async def test_ingest_pods_maps_phase_namespace_and_node(ingest):
    service, transport = ingest
    res = await ingest_pods(
        [
            {
                "name": "web-abc123",
                "namespace": "default",
                "status": "Running",
                "node": "node-1",
                "deployment": "web",
                "created": "2026-07-08T00:00:00Z",
            }
        ],
        ingest=service,
    )
    # pod node + runsInNamespace + managedByDeployment + scheduledOnK8sNode
    assert res == {"nodes": 1, "edges": 3}
    request = transport.requests[0]
    pod = _by_id(request)["container:pod:default/web-abc123"]
    assert _is_type(pod, "Pod")
    assert pod.payload["podPhase"] == "Running"
    assert pod.payload["externalToolId"] == "web-abc123"
    edges = _edge_types(request)
    assert ("container:pod:default/web-abc123", "container:namespace:default", "runsInNamespace") in edges
    assert ("container:pod:default/web-abc123", "container:deployment:web", "managedByDeployment") in edges
    assert ("container:pod:default/web-abc123", "container:k8snode:node-1", "scheduledOnK8sNode") in edges


async def test_ingest_deployments_maps_replicas_image_and_namespace(ingest):
    service, transport = ingest
    res = await ingest_deployments(
        [
            {
                "id": "dep123",
                "name": "web",
                "namespace": "default",
                "image": "nginx:latest",
                "replicas": 3,
                "ports": "80",
            }
        ],
        ingest=service,
    )
    # deployment + image = 2 nodes; usesImage + runsInNamespace = 2 edges
    assert res == {"nodes": 2, "edges": 2}
    request = transport.requests[0]
    dep = _by_id(request)["container:deployment:dep123"]
    assert _is_type(dep, "Deployment")
    assert dep.payload["deploymentReplicas"] == 3
    edges = _edge_types(request)
    assert ("container:deployment:dep123", "container:image:nginx:latest", "usesImage") in edges
    assert ("container:deployment:dep123", "container:namespace:default", "runsInNamespace") in edges


async def test_ingest_namespaces_maps_status(ingest):
    service, transport = ingest
    res = await ingest_namespaces(
        [{"name": "kube-system", "status": "Active"}], ingest=service
    )
    assert res == {"nodes": 1, "edges": 0}
    ns = _by_id(transport.requests[0])["container:namespace:kube-system"]
    assert _is_type(ns, "Namespace")
    assert ns.payload["namespaceStatus"] == "Active"


async def test_ingest_k8s_services_maps_type_and_namespace(ingest):
    service, transport = ingest
    res = await ingest_k8s_services(
        [{"name": "web", "namespace": "default", "type": "ClusterIP"}], ingest=service
    )
    # service node + runsInNamespace edge
    assert res == {"nodes": 1, "edges": 1}
    request = transport.requests[0]
    svc = _by_id(request)["container:k8sservice:default/web"]
    assert _is_type(svc, "K8sService")
    assert svc.payload["serviceType"] == "ClusterIP"
    assert (
        "container:k8sservice:default/web",
        "container:namespace:default",
        "runsInNamespace",
    ) in _edge_types(request)


async def test_retired_node_type_alias_is_rejected(ingest):
    service, _ = ingest
    with pytest.raises(IngestError, match="node_type"):
        await ingest_entities(
            [{"id": "retired", "type": "RetiredAlias"}],
            ingest=service,
        )


async def test_empty_native_ingest_is_rejected(ingest):
    service, _ = ingest
    with pytest.raises(IngestError, match="at least one entity"):
        await ingest_entities([], ingest=service)

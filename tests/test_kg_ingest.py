"""Native epistemic-graph typed-node ingestion — Wire-First coverage.

Exercises the real ``ingest_entities`` + per-modality mappers with a fake
ChangeEnvelope-capable engine client (no engine required), asserting the
applied node/edge writes and the container-manager record → typed-node
mapping. The fake client and governed-session fixture mirror agent-utilities'
own ``tests/knowledge_graph/test_native_ingest.py`` reference fake — the shape
``_change_envelope_authority`` actually requires (``changes``/``nodes``/``rdf``/
``supports``; the retired raw ``txn``-only fake is rejected).
CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from typing import Any

import msgpack
import pytest
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.memory.native_ingest import NativeIngestError
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

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


@pytest.fixture(autouse=True)
def _governed_session():
    """Ambient actor + GraphSession required by native_ingest's injected-client path."""
    actor = ActorContext(
        actor_id="subject:opaque:synthetic",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=(),
        tenant_id="tenant:opaque:synthetic",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:write"}),
        graph="__commons__",
        policy_version="policy:opaque:synthetic",
        audience="epistemic-graph",
    )
    with use_actor(actor), use_session(session):
        yield


class _FakeNodes:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {}

    def properties(self, node_id: str) -> dict[str, Any] | None:
        return self.values.get(node_id)

    def list(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self.values.items())


class _FakeChanges:
    def __init__(self, nodes: _FakeNodes) -> None:
        self.nodes = nodes
        self.edges: list[tuple[str, str, dict[str, Any]]] = []
        self.applied: list[dict[str, Any]] = []
        self.records: dict[str, dict[str, Any]] = {}
        self.versions: dict[str, dict[str, Any]] = {}

    def get(self, envelope_id: str) -> dict[str, Any] | None:
        return self.records.get(envelope_id)

    def content_version(self, object_id: str) -> dict[str, Any] | None:
        return self.versions.get(object_id)

    def cursor(self, _source: str, _partition: str = "") -> None:
        return None

    def apply(self, envelope: dict[str, Any]) -> dict[str, Any]:
        self.applied.append(envelope)
        mutation = envelope["mutation"]
        for operation in mutation["operations"]:
            method = operation["method"]
            params = method["params"]
            properties = msgpack.unpackb(params["properties_msgpack"], raw=False)
            if method["method"] == "AddNode":
                self.nodes.values[params["node_id"]] = properties
            elif method["method"] == "AddEdge":
                self.edges.append(
                    (params["source_id"], params["target_id"], properties)
                )
        version = envelope["content_version"]
        self.versions[version["object_id"]] = version
        self.records[envelope["envelope_id"]] = envelope
        return {
            "batch_id": mutation["batch_id"],
            "replayed": False,
            "projection_pending": False,
        }


class _FakeRdf:
    def validate_shacl(self, _shapes: str, _data_graph: str) -> dict[str, Any]:
        return {"conforms": True, "results": []}


class _FakeClient:
    def __init__(self) -> None:
        self.nodes = _FakeNodes()
        self.changes = _FakeChanges(self.nodes)
        self.rdf = _FakeRdf()

    @staticmethod
    def supports(operation: str) -> bool:
        return operation == "ApplyChangeEnvelope"


def test_ingest_entities_writes_nodes_and_edges():
    c = _FakeClient()
    res = ingest_entities(
        [
            {"id": "a", "node_type": "Container", "name": "web"},
            {"id": "b", "node_type": "ContainerImage"},
        ],
        [{"source": "a", "target": "b", "relationship": "usesImage"}],
        client=c,
        graph="__commons__",
    )
    assert res == {"nodes": 2, "edges": 1}
    assert len(c.changes.applied) == 1
    assert set(c.nodes.values) == {"a", "b"}
    # provenance is stamped
    assert c.nodes.values["a"]["source"] == "container-manager-mcp"
    assert c.nodes.values["a"]["domain"] == "container"
    assert c.changes.edges == [("a", "b", {"relationship": "usesImage"})]


def test_ingest_containers_maps_container_image_and_host():
    c = _FakeClient()
    res = ingest_containers(
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
        client=c,
        graph="__commons__",
    )
    # container + image + host = 3 nodes; usesImage + runsOn = 2 edges
    assert res == {"nodes": 3, "edges": 2}
    cont = c.nodes.values["container:container:abc123"]
    assert cont["node_type"] == "Container"
    assert cont["containerStatus"] == "running"
    # the persistence privacy guard redacts literal IPv4 addresses in payload
    # fields before they reach durable storage.
    assert cont["portMappings"] == "[REDACTED_IPV4]:8080->80/tcp"
    assert cont["externalToolId"] == "abc123"
    assert c.nodes.values["container:image:nginx:latest"]["node_type"] == "ContainerImage"
    assert c.nodes.values["container:host:test-node-1"]["node_type"] == "Host"
    assert (
        "container:container:abc123",
        "container:image:nginx:latest",
        {"relationship": "usesImage"},
    ) in c.changes.edges
    assert (
        "container:container:abc123",
        "container:host:test-node-1",
        {"relationship": "runsOn"},
    ) in c.changes.edges


def test_ingest_images_maps_repo_tag_size():
    c = _FakeClient()
    res = ingest_images(
        [
            {
                "id": "f00dcafe",
                "repository": "nginx",
                "tag": "latest",
                "size": "142MB",
                "created": "2026-07-01T00:00:00Z",
            }
        ],
        client=c,
    )
    assert res == {"nodes": 1, "edges": 0}
    img = c.nodes.values["container:image:f00dcafe"]
    assert img["node_type"] == "ContainerImage"
    assert img["imageRepository"] == "nginx"
    assert img["imageTag"] == "latest"
    assert img["imageSize"] == "142MB"


def test_ingest_images_with_source_label_emits_repository_and_builtfrom():
    c = _FakeClient()
    res = ingest_images(
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
        client=c,
    )
    assert res == {"nodes": 2, "edges": 1}
    repo = c.nodes.values["git:repo:github.com/org/repo"]
    assert repo["node_type"] == "Repository"
    # the persistence privacy guard unconditionally redacts fields it
    # recognizes as location fields by name (e.g. "url") before durable storage.
    assert repo["url"] == "[REDACTED_LOCATION]"
    assert (
        "container:image:f00dcafe",
        "git:repo:github.com/org/repo",
        {"relationship": "builtFrom"},
    ) in c.changes.edges


def test_ingest_images_with_vcs_url_label_fallback():
    c = _FakeClient()
    res = ingest_images(
        [
            {
                "id": "cafef00d",
                "repository": "myapp",
                "tag": "1.0",
                "labels": {"org.label-schema.vcs-url": "git@gitlab.com:team/myapp.git"},
            }
        ],
        client=c,
    )
    assert res == {"nodes": 2, "edges": 1}
    assert "git:repo:gitlab.com/team/myapp" in c.nodes.values
    assert (
        "container:image:cafef00d",
        "git:repo:gitlab.com/team/myapp",
        {"relationship": "builtFrom"},
    ) in c.changes.edges


def test_ingest_images_without_source_label_emits_no_builtfrom_edge():
    c = _FakeClient()
    res = ingest_images(
        [
            {
                "id": "deadbeef",
                "repository": "redis",
                "tag": "7",
                "labels": {"maintainer": "nobody"},
            }
        ],
        client=c,
    )
    assert res == {"nodes": 1, "edges": 0}
    assert c.changes.edges == []


def test_ingest_images_no_labels_at_all_is_graceful_noop_edge():
    c = _FakeClient()
    res = ingest_images(
        [{"id": "abc12345", "repository": "alpine", "tag": "latest"}], client=c
    )
    assert res == {"nodes": 1, "edges": 0}
    assert c.changes.edges == []


def test_ingest_volumes_and_networks():
    c = _FakeClient()
    assert ingest_volumes(
        [{"name": "pgdata", "driver": "local", "mountpoint": "/var/lib/x"}], client=c
    ) == {"nodes": 1, "edges": 0}
    assert c.nodes.values["container:volume:pgdata"]["volumeDriver"] == "local"

    c2 = _FakeClient()
    assert ingest_networks(
        [{"id": "net1", "name": "backend", "driver": "overlay", "scope": "swarm"}],
        client=c2,
    ) == {"nodes": 1, "edges": 0}
    n = c2.nodes.values["container:network:net1"]
    assert n["node_type"] == "ContainerNetwork"
    assert n["networkDriver"] == "overlay"
    assert n["networkScope"] == "swarm"


def test_ingest_services_maps_replicas_and_image_edge():
    c = _FakeClient()
    res = ingest_services(
        [
            {
                "id": "svc1",
                "name": "web",
                "image": "nginx:latest",
                "replicas": 3,
                "ports": "8080->80/tcp",
            }
        ],
        client=c,
    )
    assert res == {"nodes": 2, "edges": 1}
    svc = c.nodes.values["container:service:svc1"]
    assert svc["node_type"] == "SwarmService"
    assert svc["serviceReplicas"] == 3
    assert c.changes.edges == [
        (
            "container:service:svc1",
            "container:image:nginx:latest",
            {"relationship": "usesImage"},
        )
    ]


def test_ingest_nodes_maps_role_and_availability():
    c = _FakeClient()
    res = ingest_nodes(
        [
            {
                "id": "node1",
                "hostname": "rw710",
                "role": "manager",
                "status": "ready",
                "availability": "active",
            }
        ],
        client=c,
    )
    assert res == {"nodes": 1, "edges": 0}
    n = c.nodes.values["container:node:node1"]
    assert n["node_type"] == "SwarmNode"
    assert n["nodeRole"] == "manager"
    assert n["nodeAvailability"] == "active"


def test_ingest_pods_maps_phase_namespace_and_node():
    c = _FakeClient()
    res = ingest_pods(
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
        client=c,
    )
    # pod node + runsInNamespace + managedByDeployment + scheduledOnK8sNode
    assert res == {"nodes": 1, "edges": 3}
    pod = c.nodes.values["container:pod:default/web-abc123"]
    assert pod["node_type"] == "Pod"
    assert pod["podPhase"] == "Running"
    assert pod["externalToolId"] == "web-abc123"
    assert (
        "container:pod:default/web-abc123",
        "container:namespace:default",
        {"relationship": "runsInNamespace"},
    ) in c.changes.edges
    assert (
        "container:pod:default/web-abc123",
        "container:deployment:web",
        {"relationship": "managedByDeployment"},
    ) in c.changes.edges
    assert (
        "container:pod:default/web-abc123",
        "container:k8snode:node-1",
        {"relationship": "scheduledOnK8sNode"},
    ) in c.changes.edges


def test_ingest_deployments_maps_replicas_image_and_namespace():
    c = _FakeClient()
    res = ingest_deployments(
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
        client=c,
    )
    # deployment + image = 2 nodes; usesImage + runsInNamespace = 2 edges
    assert res == {"nodes": 2, "edges": 2}
    dep = c.nodes.values["container:deployment:dep123"]
    assert dep["node_type"] == "Deployment"
    assert dep["deploymentReplicas"] == 3
    assert (
        "container:deployment:dep123",
        "container:image:nginx:latest",
        {"relationship": "usesImage"},
    ) in c.changes.edges
    assert (
        "container:deployment:dep123",
        "container:namespace:default",
        {"relationship": "runsInNamespace"},
    ) in c.changes.edges


def test_ingest_namespaces_maps_status():
    c = _FakeClient()
    res = ingest_namespaces([{"name": "kube-system", "status": "Active"}], client=c)
    assert res == {"nodes": 1, "edges": 0}
    ns = c.nodes.values["container:namespace:kube-system"]
    assert ns["node_type"] == "Namespace"
    assert ns["namespaceStatus"] == "Active"


def test_ingest_k8s_services_maps_type_and_namespace():
    c = _FakeClient()
    res = ingest_k8s_services(
        [{"name": "web", "namespace": "default", "type": "ClusterIP"}], client=c
    )
    # service node + runsInNamespace edge
    assert res == {"nodes": 1, "edges": 1}
    svc = c.nodes.values["container:k8sservice:default/web"]
    assert svc["node_type"] == "K8sService"
    assert svc["serviceType"] == "ClusterIP"
    assert (
        "container:k8sservice:default/web",
        "container:namespace:default",
        {"relationship": "runsInNamespace"},
    ) in c.changes.edges


def test_retired_node_type_alias_is_rejected():
    with pytest.raises(NativeIngestError, match="canonical node_type"):
        ingest_entities(
            [{"id": "retired", "type": "RetiredAlias"}],
            client=_FakeClient(),
        )


def test_empty_native_ingest_is_rejected():
    with pytest.raises(NativeIngestError, match="at least one entity"):
        ingest_entities([], client=_FakeClient())

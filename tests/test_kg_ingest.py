"""Native epistemic-graph typed-node ingestion — Wire-First coverage.

Exercises the real ``ingest_entities`` + per-modality mappers with a fake engine client
(no engine required), asserting the txn add_node/commit + edge calls and the
container-manager record → typed-node mapping. CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

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


class _FakeTxn:
    def __init__(self):
        self.nodes = {}
        self.committed = False

    def begin(self, graph=None):
        self.graph = graph
        return "txn-1"

    def add_node(self, txn, node_id, props):
        self.nodes[node_id] = props

    def commit(self, txn):
        self.committed = True
        return True


class _FakeEdges:
    def __init__(self):
        self.edges = []

    def add(self, src, dst, props):
        self.edges.append((src, dst, props))


class _FakeClient:
    def __init__(self):
        self.txn = _FakeTxn()
        self.edges = _FakeEdges()


def test_ingest_entities_writes_nodes_and_edges():
    c = _FakeClient()
    res = ingest_entities(
        [
            {"id": "a", "type": "Container", "name": "web"},
            {"id": "b", "type": "ContainerImage"},
        ],
        [{"source": "a", "target": "b", "type": "usesImage"}],
        client=c,
        graph="__commons__",
    )
    assert res == {"nodes": 2, "edges": 1}
    assert c.txn.committed is True
    assert set(c.txn.nodes) == {"a", "b"}
    # provenance is stamped
    assert c.txn.nodes["a"]["source"] == "container-manager-mcp"
    assert c.txn.nodes["a"]["domain"] == "container"
    assert c.edges.edges == [("a", "b", {"type": "usesImage"})]


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
        host="rw710",
        client=c,
        graph="__commons__",
    )
    # container + image + host = 3 nodes; usesImage + runsOn = 2 edges
    assert res == {"nodes": 3, "edges": 2}
    cont = c.txn.nodes["container:container:abc123"]
    assert cont["type"] == "Container"
    assert cont["containerStatus"] == "running"
    assert cont["portMappings"] == "0.0.0.0:8080->80/tcp"
    assert cont["externalToolId"] == "abc123"
    assert c.txn.nodes["container:image:nginx:latest"]["type"] == "ContainerImage"
    assert c.txn.nodes["container:host:rw710"]["type"] == "Host"
    assert (
        "container:container:abc123",
        "container:image:nginx:latest",
        {"type": "usesImage"},
    ) in c.edges.edges
    assert (
        "container:container:abc123",
        "container:host:rw710",
        {"type": "runsOn"},
    ) in c.edges.edges


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
    img = c.txn.nodes["container:image:f00dcafe"]
    assert img["type"] == "ContainerImage"
    assert img["imageRepository"] == "nginx"
    assert img["imageTag"] == "latest"
    assert img["imageSize"] == "142MB"


def test_ingest_volumes_and_networks():
    c = _FakeClient()
    assert ingest_volumes(
        [{"name": "pgdata", "driver": "local", "mountpoint": "/var/lib/x"}], client=c
    ) == {"nodes": 1, "edges": 0}
    assert c.txn.nodes["container:volume:pgdata"]["volumeDriver"] == "local"

    c2 = _FakeClient()
    assert ingest_networks(
        [{"id": "net1", "name": "backend", "driver": "overlay", "scope": "swarm"}],
        client=c2,
    ) == {"nodes": 1, "edges": 0}
    n = c2.txn.nodes["container:network:net1"]
    assert n["type"] == "ContainerNetwork"
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
    svc = c.txn.nodes["container:service:svc1"]
    assert svc["type"] == "SwarmService"
    assert svc["serviceReplicas"] == 3
    assert c.edges.edges == [
        (
            "container:service:svc1",
            "container:image:nginx:latest",
            {"type": "usesImage"},
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
    n = c.txn.nodes["container:node:node1"]
    assert n["type"] == "SwarmNode"
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
    pod = c.txn.nodes["container:pod:default/web-abc123"]
    assert pod["type"] == "Pod"
    assert pod["podPhase"] == "Running"
    assert pod["externalToolId"] == "web-abc123"
    assert (
        "container:pod:default/web-abc123",
        "container:namespace:default",
        {"type": "runsInNamespace"},
    ) in c.edges.edges
    assert (
        "container:pod:default/web-abc123",
        "container:deployment:web",
        {"type": "managedByDeployment"},
    ) in c.edges.edges
    assert (
        "container:pod:default/web-abc123",
        "container:k8snode:node-1",
        {"type": "scheduledOnK8sNode"},
    ) in c.edges.edges


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
    dep = c.txn.nodes["container:deployment:dep123"]
    assert dep["type"] == "Deployment"
    assert dep["deploymentReplicas"] == 3
    assert (
        "container:deployment:dep123",
        "container:image:nginx:latest",
        {"type": "usesImage"},
    ) in c.edges.edges
    assert (
        "container:deployment:dep123",
        "container:namespace:default",
        {"type": "runsInNamespace"},
    ) in c.edges.edges


def test_ingest_namespaces_maps_status():
    c = _FakeClient()
    res = ingest_namespaces([{"name": "kube-system", "status": "Active"}], client=c)
    assert res == {"nodes": 1, "edges": 0}
    ns = c.txn.nodes["container:namespace:kube-system"]
    assert ns["type"] == "Namespace"
    assert ns["namespaceStatus"] == "Active"


def test_ingest_k8s_services_maps_type_and_namespace():
    c = _FakeClient()
    res = ingest_k8s_services(
        [{"name": "web", "namespace": "default", "type": "ClusterIP"}], client=c
    )
    # service node + runsInNamespace edge
    assert res == {"nodes": 1, "edges": 1}
    svc = c.txn.nodes["container:k8sservice:default/web"]
    assert svc["type"] == "K8sService"
    assert svc["serviceType"] == "ClusterIP"
    assert (
        "container:k8sservice:default/web",
        "container:namespace:default",
        {"type": "runsInNamespace"},
    ) in c.edges.edges


def test_ingest_noops_without_engine():
    # No injected client + no reachable engine -> clean no-op.
    assert ingest_entities([{"id": "a", "type": "Container"}]) is None


def test_ingest_empty_is_noop():
    assert ingest_entities([], client=_FakeClient()) is None
    assert ingest_containers([], client=_FakeClient()) is None
    assert ingest_images([], client=_FakeClient()) is None
    assert ingest_services([], client=_FakeClient()) is None

"""Tests for the Kubernetes backend (KubernetesManager).

The ``kubernetes`` client is mocked at the module level so these run without a
cluster or the library installed — mirroring how ``test_swarm_ops.py`` patches
``docker``.
"""

import types
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import container_manager_mcp.k8s_manager as k8s_mod
from container_manager_mcp.k8s_manager import KubernetesManager


class _FakeApiException(Exception):
    pass


@pytest.fixture(autouse=True)
def _mock_k8s():
    """Keep the k8s client mocked for the whole test (methods use it at call time)."""
    with (
        patch.object(k8s_mod, "k8s_client", MagicMock()),
        patch.object(k8s_mod, "k8s_config", MagicMock()),
        patch.object(k8s_mod, "ApiException", _FakeApiException),
    ):
        yield


def _make_manager() -> KubernetesManager:
    """Construct a KubernetesManager with the k8s client fully mocked."""
    manager = KubernetesManager(namespace="default")
    manager.core = MagicMock()
    manager.apps = MagicMock()
    return manager


def _fake_node(name="R820", uid="abc123def456789", role="worker", unschedulable=False):
    labels = {"name": name}
    if role == "manager":
        labels["node-role.kubernetes.io/control-plane"] = ""
    return types.SimpleNamespace(
        metadata=types.SimpleNamespace(
            name=name, uid=uid, labels=labels, creation_timestamp=datetime(2026, 6, 1)
        ),
        spec=types.SimpleNamespace(unschedulable=unschedulable),
        status=types.SimpleNamespace(
            conditions=[types.SimpleNamespace(type="Ready", status="True")],
            node_info=types.SimpleNamespace(
                kubelet_version="v1.31.0", os_image="Ubuntu", architecture="amd64"
            ),
            addresses=[types.SimpleNamespace(type="InternalIP", address="10.0.0.13")],
        ),
    )


def _fake_deployment(name="caddy", uid="dep111222333", replicas=1, image="caddy:2"):
    container = types.SimpleNamespace(
        image=image, ports=[types.SimpleNamespace(container_port=80)]
    )
    return types.SimpleNamespace(
        metadata=types.SimpleNamespace(
            name=name,
            uid=uid,
            namespace="default",
            creation_timestamp=datetime(2026, 6, 1),
        ),
        spec=types.SimpleNamespace(
            replicas=replicas,
            template=types.SimpleNamespace(
                spec=types.SimpleNamespace(containers=[container])
            ),
        ),
    )


class TestNodeOps:
    def test_list_nodes_maps_role_and_state(self):
        manager = _make_manager()
        manager.core.list_node.return_value = types.SimpleNamespace(
            items=[_fake_node(role="manager"), _fake_node(name="R710", role="worker")]
        )
        result = manager.list_nodes()
        assert result[0]["hostname"] == "R820"
        assert result[0]["role"] == "manager"
        assert result[0]["status"] == "ready"
        assert result[0]["availability"] == "active"
        assert result[1]["role"] == "worker"

    def test_update_node_merges_labels_and_cordons_on_drain(self):
        manager = _make_manager()
        manager.core.read_node.return_value = _fake_node()
        manager.core.list_pod_for_all_namespaces.return_value = types.SimpleNamespace(
            items=[]
        )
        manager.update_node("R820", labels={"gpu": "true"}, availability="drain")
        # patch_node called with merged labels and unschedulable=True
        body = manager.core.patch_node.call_args[0][1]
        assert body["metadata"]["labels"]["name"] == "R820"
        assert body["metadata"]["labels"]["gpu"] == "true"
        assert body["spec"]["unschedulable"] is True

    def test_inspect_node_summary(self):
        manager = _make_manager()
        manager.core.read_node.return_value = _fake_node(role="manager")
        summary = manager.inspect_node("R820")
        assert summary["hostname"] == "R820"
        assert summary["manager"] is True
        assert summary["engine_version"] == "v1.31.0"
        assert summary["addr"] == "10.0.0.13"


class TestServiceOps:
    def test_create_service_builds_deployment_and_service(self):
        manager = _make_manager()
        manager.apps.create_namespaced_deployment.return_value = _fake_deployment()
        result = manager.create_service(
            "caddy", "caddy:2", replicas=2, ports={"80": "8080"}
        )
        manager.apps.create_namespaced_deployment.assert_called_once()
        manager.core.create_namespaced_service.assert_called_once()
        assert result["name"] == "caddy"
        assert result["image"] == "caddy:2"

    def test_create_service_without_ports_skips_service(self):
        manager = _make_manager()
        manager.apps.create_namespaced_deployment.return_value = _fake_deployment()
        manager.create_service("worker", "worker:1")
        manager.core.create_namespaced_service.assert_not_called()

    def test_scale_service_patches_scale(self):
        manager = _make_manager()
        result = manager.scale_service("caddy", 5)
        manager.apps.patch_namespaced_deployment_scale.assert_called_once_with(
            "caddy", "default", {"spec": {"replicas": 5}}
        )
        assert result["replicas"] == 5
        assert result["scaled"] is True

    def test_update_service_translates_constraints_to_node_selector(self):
        manager = _make_manager()
        manager.update_service(
            "caddy", image="caddy:3", constraints=["node.labels.name == R820"]
        )
        body = manager.apps.patch_namespaced_deployment.call_args[0][2]
        assert body["spec"]["template"]["spec"]["nodeSelector"] == {"name": "R820"}
        assert body["spec"]["template"]["spec"]["containers"][0]["image"] == "caddy:3"

    def test_service_ps_maps_pods(self):
        manager = _make_manager()
        pod = types.SimpleNamespace(
            metadata=types.SimpleNamespace(
                name="caddy-abc", creation_timestamp=datetime(2026, 6, 1)
            ),
            spec=types.SimpleNamespace(node_name="R820"),
            status=types.SimpleNamespace(phase="Running", container_statuses=[]),
        )
        manager.core.list_namespaced_pod.return_value = types.SimpleNamespace(
            items=[pod]
        )
        result = manager.service_ps("caddy")
        assert result[0]["id"] == "caddy-abc"
        assert result[0]["node"] == "R820"
        assert result[0]["state"] == "Running"

    def test_list_services_summarizes_deployments(self):
        manager = _make_manager()
        manager.apps.list_deployment_for_all_namespaces.return_value = (
            types.SimpleNamespace(items=[_fake_deployment()])
        )
        result = manager.list_services()
        assert result[0]["name"] == "caddy"
        assert result[0]["replicas"] == 1


class TestLifecycleNoOps:
    def test_init_swarm_is_graceful_noop(self):
        manager = _make_manager()
        result = manager.init_swarm()
        assert result["status"] == "kubernetes"
        assert "RKE2" in result["note"]

    def test_leave_swarm_is_graceful_noop(self):
        manager = _make_manager()
        result = manager.leave_swarm()
        assert result["status"] == "kubernetes"


class TestUnsupportedOps:
    @pytest.mark.parametrize(
        "op",
        [
            "list_images",
            "run_container",
            "create_network",
            "list_volumes",
            "prune_system",
        ],
    )
    def test_node_local_ops_raise(self, op):
        manager = _make_manager()
        method = getattr(manager, op)
        with pytest.raises(
            RuntimeError, match="not available on the Kubernetes backend"
        ):
            if op == "run_container":
                method("nginx")
            elif op == "create_network":
                method("netname")
            else:
                method()


class TestHelpers:
    def test_constraints_to_node_selector(self):
        sel = KubernetesManager._constraints_to_node_selector(
            ["node.labels.name == R820", "node.hostname==R710"]
        )
        assert sel == {"name": "R820", "kubernetes.io/hostname": "R710"}

    def test_node_role_detection(self):
        assert (
            KubernetesManager._node_role({"node-role.kubernetes.io/control-plane": ""})
            == "manager"
        )
        assert KubernetesManager._node_role({"name": "R710"}) == "worker"


class TestFactoryRouting:
    def test_create_manager_routes_to_kubernetes(self):
        from container_manager_mcp.container_manager import create_manager

        with (
            patch.object(k8s_mod, "k8s_client", MagicMock()),
            patch.object(k8s_mod, "k8s_config", MagicMock()),
            patch.object(k8s_mod, "ApiException", _FakeApiException),
        ):
            manager = create_manager("kubernetes")
        assert isinstance(manager, KubernetesManager)

    def test_create_manager_routes_rke2_alias(self):
        from container_manager_mcp.container_manager import create_manager

        with (
            patch.object(k8s_mod, "k8s_client", MagicMock()),
            patch.object(k8s_mod, "k8s_config", MagicMock()),
            patch.object(k8s_mod, "ApiException", _FakeApiException),
        ):
            manager = create_manager("rke2")
        assert isinstance(manager, KubernetesManager)

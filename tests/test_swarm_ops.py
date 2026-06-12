"""Tests for the swarm node/service operations on DockerManager."""

from unittest.mock import MagicMock, patch

import pytest

from container_manager_mcp.container_manager import DockerManager


def _make_manager(mock_docker) -> DockerManager:
    mock_client = MagicMock()
    mock_docker.from_env.return_value = mock_client
    manager = DockerManager()
    manager.client = mock_client
    return manager


def _fake_node(node_id="abc123def456", hostname="R820", labels=None, role="worker"):
    node = MagicMock()
    node.id = node_id
    node.attrs = {
        "ID": node_id,
        "Spec": {
            "Role": role,
            "Availability": "active",
            "Labels": labels or {},
        },
        "Description": {
            "Hostname": hostname,
            "Engine": {"EngineVersion": "29.5.2"},
            "Platform": {"Architecture": "x86_64", "OS": "linux"},
        },
        "Status": {"State": "ready", "Addr": "10.0.0.13"},
        "ManagerStatus": {"Leader": True},
    }
    return node


class TestNodeOps:
    @patch("container_manager_mcp.container_manager.docker")
    def test_update_node_merges_labels(self, mock_docker):
        manager = _make_manager(mock_docker)
        node = _fake_node(labels={"name": "R820"})
        manager.client.nodes.get.return_value = node

        result = manager.update_node("R820", labels={"poweredge": "true"})

        # node.update called once with merged labels + preserved role/availability
        node.update.assert_called_once()
        sent_spec = node.update.call_args[0][0]
        assert sent_spec["Labels"] == {"name": "R820", "poweredge": "true"}
        assert sent_spec["Role"] == "worker"
        assert sent_spec["Availability"] == "active"
        assert result["hostname"] == "R820"

    @patch("container_manager_mcp.container_manager.docker")
    def test_update_node_replace_labels(self, mock_docker):
        manager = _make_manager(mock_docker)
        node = _fake_node(labels={"old": "x"})
        manager.client.nodes.get.return_value = node

        manager.update_node("R820", labels={"poweredge": "true"}, replace_labels=True)
        sent_spec = node.update.call_args[0][0]
        assert sent_spec["Labels"] == {"poweredge": "true"}

    @patch("container_manager_mcp.container_manager.docker")
    def test_update_node_availability_drain(self, mock_docker):
        manager = _make_manager(mock_docker)
        node = _fake_node()
        manager.client.nodes.get.return_value = node

        manager.update_node("R820", availability="drain")
        sent_spec = node.update.call_args[0][0]
        assert sent_spec["Availability"] == "drain"

    @patch("container_manager_mcp.container_manager.docker")
    def test_resolve_node_by_hostname_fallback(self, mock_docker):
        manager = _make_manager(mock_docker)
        node = _fake_node(hostname="RW710")
        manager.client.nodes.get.side_effect = Exception("not found by id")
        manager.client.nodes.list.return_value = [node]

        resolved = manager._resolve_node("RW710")
        assert resolved is node

    @patch("container_manager_mcp.container_manager.docker")
    def test_inspect_node(self, mock_docker):
        manager = _make_manager(mock_docker)
        manager.client.nodes.get.return_value = _fake_node(labels={"poweredge": "true"})

        result = manager.inspect_node("R820")
        assert result["labels"] == {"poweredge": "true"}
        assert result["role"] == "worker"
        assert result["engine_version"] == "29.5.2"

    @patch("container_manager_mcp.container_manager.docker")
    def test_remove_node(self, mock_docker):
        manager = _make_manager(mock_docker)
        manager.client.nodes.get.return_value = _fake_node(node_id="nid123")

        result = manager.remove_node("nid123", force=True)
        manager.client.api.remove_node.assert_called_once_with("nid123", force=True)
        assert result["removed"] == "nid123"


def _fake_service(service_id="svc123", image="img:1", env=None, replicas=2):
    service = MagicMock()
    service.id = service_id
    service.attrs = {
        "ID": service_id,
        "Version": {"Index": 42},
        "Spec": {
            "Name": "searxng_searxng",
            "Labels": {"com.docker.stack.namespace": "searxng"},
            "Mode": {"Replicated": {"Replicas": replicas}},
            "EndpointSpec": {"Mode": "vip"},
            "TaskTemplate": {
                "ContainerSpec": {
                    "Image": image,
                    "Env": env or ["A=1"],
                },
                "Placement": {"Constraints": ["node.labels.poweredge==true"]},
                "Networks": [{"Target": "netid"}],
            },
        },
    }
    return service


class TestServiceOps:
    @patch("container_manager_mcp.container_manager.docker")
    def test_scale_service(self, mock_docker):
        manager = _make_manager(mock_docker)
        service = _fake_service()
        manager.client.services.get.return_value = service

        result = manager.scale_service("searxng_searxng", 3)
        service.scale.assert_called_once_with(3)
        assert result["replicas"] == 3

    @patch("container_manager_mcp.container_manager.docker")
    def test_update_service_preserves_spec_and_sets_image(self, mock_docker):
        manager = _make_manager(mock_docker)
        service = _fake_service(image="old:1", env=["KEEP=yes"])
        manager.client.services.get.return_value = service

        manager.update_service("searxng_searxng", image="new:2")

        manager.client.api.update_service.assert_called_once()
        _, kwargs = manager.client.api.update_service.call_args
        tt = kwargs["task_template"]
        # image updated, but env + placement + networks preserved (no reset footgun)
        assert tt["ContainerSpec"]["Image"] == "new:2"
        assert tt["ContainerSpec"]["Env"] == ["KEEP=yes"]
        assert tt["Placement"]["Constraints"] == ["node.labels.poweredge==true"]
        assert tt["Networks"] == [{"Target": "netid"}]

    @patch("container_manager_mcp.container_manager.docker")
    def test_update_service_force_increments(self, mock_docker):
        manager = _make_manager(mock_docker)
        service = _fake_service()
        manager.client.services.get.return_value = service

        manager.update_service("searxng_searxng", force=True)
        _, kwargs = manager.client.api.update_service.call_args
        assert kwargs["task_template"]["ForceUpdate"] == 1

    @patch("container_manager_mcp.container_manager.docker")
    def test_service_ps(self, mock_docker):
        manager = _make_manager(mock_docker)
        service = _fake_service()
        service.tasks.return_value = [
            {
                "ID": "task1234567890",
                "NodeID": "node1234567890",
                "DesiredState": "running",
                "Status": {"State": "running", "Timestamp": "t"},
            }
        ]
        manager.client.services.get.return_value = service

        result = manager.service_ps("searxng_searxng")
        assert result[0]["state"] == "running"
        assert result[0]["id"] == "task12345678"

    @patch("container_manager_mcp.container_manager.docker")
    def test_service_logs_joins_stream(self, mock_docker):
        manager = _make_manager(mock_docker)
        service = _fake_service()
        service.logs.return_value = [b"line1\n", b"line2\n"]
        manager.client.services.get.return_value = service

        result = manager.service_logs("searxng_searxng", tail=10)
        assert "line1" in result["logs"] and "line2" in result["logs"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

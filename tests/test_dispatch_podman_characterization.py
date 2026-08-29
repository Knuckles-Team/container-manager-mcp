"""Characterization tests for cm_podman's execute_operation (CCN 45), before
decomposition.

cm_podman is a 20-action flat dispatcher directly against a PodmanManager
where the manager method name always equals the action name. This pins:
which manager.<method> each action calls and with exactly which args
(including the namespace/tail_lines/driver default-fallback quirks), each
guard's exact ValueError with the manager left uncalled, and the
unknown-action ValueError.
"""

import asyncio
from unittest.mock import MagicMock

import pytest


def _capture_tool(register_fn):
    captured = {}

    def tool_decorator(*args, **kwargs):
        def wrapper(fn):
            captured["fn"] = fn
            return fn

        return wrapper

    fake_mcp = MagicMock()
    fake_mcp.tool = tool_decorator
    register_fn(fake_mcp)
    return captured["fn"]


@pytest.fixture
def tool_and_manager(monkeypatch):
    from container_manager_mcp.mcp import mcp_podman

    fake_manager = MagicMock()
    monkeypatch.setattr(mcp_podman, "create_manager", lambda backend: fake_manager)
    tool = _capture_tool(mcp_podman.register_podman_tools)
    return tool, fake_manager


def _run(tool, **kwargs):
    return asyncio.run(tool(**kwargs))


def test_podman_generate_kube_yaml_dispatches_with_namespace_default(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_generate_kube_yaml", pod_name="pod1", namespace=None)
    manager.podman_generate_kube_yaml.assert_called_once_with("pod1", "default")


def test_podman_generate_kube_yaml_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="pod_name is required for podman_generate_kube_yaml"):
        _run(tool, action="podman_generate_kube_yaml", pod_name=None)
    manager.podman_generate_kube_yaml.assert_not_called()


def test_podman_play_kube_yaml_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_play_kube_yaml", yaml_path="/tmp/x.yaml")
    manager.podman_play_kube_yaml.assert_called_once_with("/tmp/x.yaml")


def test_podman_play_kube_yaml_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="yaml_path is required for podman_play_kube_yaml"):
        _run(tool, action="podman_play_kube_yaml", yaml_path=None)


def test_podman_checkpoint_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_checkpoint", container_id="c1", checkpoint_dir="/cp")
    manager.podman_checkpoint.assert_called_once_with("c1", "/cp")


def test_podman_checkpoint_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="container_id and checkpoint_dir are required for podman_checkpoint"):
        _run(tool, action="podman_checkpoint", container_id=None, checkpoint_dir=None)


def test_podman_restore_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_restore", container_id="c1", checkpoint_dir="/cp")
    manager.podman_restore.assert_called_once_with("c1", "/cp")


def test_podman_restore_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="container_id and checkpoint_dir are required for podman_restore"):
        _run(tool, action="podman_restore", container_id=None, checkpoint_dir=None)


def test_podman_pod_create_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_pod_create", pod_name="pod1", image="nginx", command="run")
    manager.podman_pod_create.assert_called_once_with("pod1", "nginx", "run")


def test_podman_pod_create_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="pod_name and image are required for podman_pod_create"):
        _run(tool, action="podman_pod_create", pod_name=None, image=None)


def test_podman_pod_list_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_pod_list")
    manager.podman_pod_list.assert_called_once_with()


def test_podman_pod_stats_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_pod_stats", pod_name="pod1")
    manager.podman_pod_stats.assert_called_once_with("pod1")


def test_podman_pod_stats_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="pod_name is required for podman_pod_stats"):
        _run(tool, action="podman_pod_stats", pod_name=None)


def test_podman_pod_top_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_pod_top", pod_name="pod1")
    manager.podman_pod_top.assert_called_once_with("pod1")


def test_podman_pod_top_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="pod_name is required for podman_pod_top"):
        _run(tool, action="podman_pod_top", pod_name=None)


def test_podman_pod_inspect_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_pod_inspect", pod_name="pod1")
    manager.podman_pod_inspect.assert_called_once_with("pod1")


def test_podman_pod_inspect_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="pod_name is required for podman_pod_inspect"):
        _run(tool, action="podman_pod_inspect", pod_name=None)


def test_podman_pod_logs_dispatches_with_tail_default(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_pod_logs", pod_name="pod1", tail_lines=None)
    manager.podman_pod_logs.assert_called_once_with("pod1", 100)


def test_podman_pod_logs_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="pod_name is required for podman_pod_logs"):
        _run(tool, action="podman_pod_logs", pod_name=None)


def test_podman_pod_stop_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_pod_stop", pod_name="pod1")
    manager.podman_pod_stop.assert_called_once_with("pod1")


def test_podman_pod_stop_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="pod_name is required for podman_pod_stop"):
        _run(tool, action="podman_pod_stop", pod_name=None)


def test_podman_pod_rm_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_pod_rm", pod_name="pod1")
    manager.podman_pod_rm.assert_called_once_with("pod1")


def test_podman_pod_rm_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="pod_name is required for podman_pod_rm"):
        _run(tool, action="podman_pod_rm", pod_name=None)


def test_podman_network_create_dispatches_with_driver_default(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_network_create", network_name="net1", driver=None, subnet="10.0.0.0/24")
    manager.podman_network_create.assert_called_once_with("net1", "bridge", "10.0.0.0/24")


def test_podman_network_create_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="network_name is required for podman_network_create"):
        _run(tool, action="podman_network_create", network_name=None)


def test_podman_network_list_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_network_list")
    manager.podman_network_list.assert_called_once_with()


def test_podman_network_inspect_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_network_inspect", network_name="net1")
    manager.podman_network_inspect.assert_called_once_with("net1")


def test_podman_network_inspect_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="network_name is required for podman_network_inspect"):
        _run(tool, action="podman_network_inspect", network_name=None)


def test_podman_volume_create_dispatches_with_driver_default(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_volume_create", volume_name="vol1", driver=None)
    manager.podman_volume_create.assert_called_once_with("vol1", "local")


def test_podman_volume_create_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="volume_name is required for podman_volume_create"):
        _run(tool, action="podman_volume_create", volume_name=None)


def test_podman_volume_list_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_volume_list")
    manager.podman_volume_list.assert_called_once_with()


def test_podman_volume_inspect_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_volume_inspect", volume_name="vol1")
    manager.podman_volume_inspect.assert_called_once_with("vol1")


def test_podman_volume_inspect_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="volume_name is required for podman_volume_inspect"):
        _run(tool, action="podman_volume_inspect", volume_name=None)


def test_podman_system_prune_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_system_prune")
    manager.podman_system_prune.assert_called_once_with()


def test_podman_health_check_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="podman_health_check", container_id="c1", config={"a": 1})
    manager.podman_health_check.assert_called_once_with("c1", {"a": 1})


def test_podman_health_check_guard(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="container_id and config are required for podman_health_check"):
        _run(tool, action="podman_health_check", container_id=None, config=None)


def test_unknown_action_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="Unknown action: bogus_action"):
        _run(tool, action="bogus_action")

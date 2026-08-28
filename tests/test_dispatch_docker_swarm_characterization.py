"""Characterization tests for cm_docker_swarm's execute_operation (CCN 43),
before decomposition.

cm_docker_swarm is a 19-action flat dispatcher directly against a
``DockerManager`` (no per-domain sub-dispatch layer, unlike the k8s
dispatchers in this repo). For every action this pins: which
``manager.<method>`` is called and with exactly which positional args
(including the ``worker or True`` / ``force or False`` / ``replicas or 1``
/ ``tail_lines or 100`` default-fallback quirks), the required-param guard's
exact ``ValueError`` message with the manager left uncalled, and the
unknown-action ``ValueError``.

Harness pattern (`_capture_tool`) matches the repo's own precedent in
tests/test_docker_swarm_ops.py.
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
    from container_manager_mcp.mcp import mcp_docker_swarm

    fake_manager = MagicMock()
    monkeypatch.setattr(
        mcp_docker_swarm, "create_manager", lambda backend: fake_manager
    )
    tool = _capture_tool(mcp_docker_swarm.register_dockerswarm_tools)
    return tool, fake_manager


def _run(tool, **kwargs):
    return asyncio.run(tool(**kwargs))


# --- Swarm operations -------------------------------------------------
def test_docker_swarm_init_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    manager.docker_swarm_init.return_value = {"ok": True}
    result = _run(tool, action="docker_swarm_init", advertise_addr="10.0.0.1", listen_addr=None)
    manager.docker_swarm_init.assert_called_once_with("10.0.0.1", None)
    assert result == {"ok": True}


def test_docker_swarm_init_guard_raises_without_calling_manager(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="advertise_addr is required for docker_swarm_init"):
        _run(tool, action="docker_swarm_init", advertise_addr=None)
    manager.docker_swarm_init.assert_not_called()


def test_docker_swarm_join_dispatches_with_worker_default(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_swarm_join", remote_addr="10.0.0.1:2377", token="tok", worker=None)
    manager.docker_swarm_join.assert_called_once_with("10.0.0.1:2377", "tok", True)


def test_docker_swarm_join_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="remote_addr and token are required for docker_swarm_join"):
        _run(tool, action="docker_swarm_join", remote_addr=None, token=None)
    manager.docker_swarm_join.assert_not_called()


def test_docker_swarm_leave_dispatches_with_force_default(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_swarm_leave", force=None)
    manager.docker_swarm_leave.assert_called_once_with(False)


def test_docker_swarm_leave_force_true(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_swarm_leave", force=True)
    manager.docker_swarm_leave.assert_called_once_with(True)


# --- Service operations -------------------------------------------------
def test_docker_service_create_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(
        tool,
        action="docker_service_create",
        service_name="svc",
        image="nginx",
        replicas=None,
        ports=[80],
    )
    manager.docker_service_create.assert_called_once_with("svc", "nginx", 1, [80])


def test_docker_service_create_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="service_name and image are required for docker_service_create"):
        _run(tool, action="docker_service_create", service_name=None, image=None)
    manager.docker_service_create.assert_not_called()


def test_docker_service_list_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    manager.docker_service_list.return_value = []
    result = _run(tool, action="docker_service_list")
    manager.docker_service_list.assert_called_once_with()
    assert result == []


def test_docker_service_update_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_service_update", service_name="svc", image="nginx:2", replicas=3)
    manager.docker_service_update.assert_called_once_with("svc", "nginx:2", 3)


def test_docker_service_update_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="service_name is required for docker_service_update"):
        _run(tool, action="docker_service_update", service_name=None)
    manager.docker_service_update.assert_not_called()


def test_docker_service_rm_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_service_rm", service_name="svc")
    manager.docker_service_rm.assert_called_once_with("svc")


def test_docker_service_rm_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="service_name is required for docker_service_rm"):
        _run(tool, action="docker_service_rm", service_name=None)


def test_docker_service_logs_dispatches_with_tail_default(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_service_logs", service_name="svc", tail_lines=None)
    manager.docker_service_logs.assert_called_once_with("svc", 100)


def test_docker_service_logs_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="service_name is required for docker_service_logs"):
        _run(tool, action="docker_service_logs", service_name=None)


def test_docker_service_ps_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_service_ps")
    manager.docker_service_ps.assert_called_once_with()


# --- Stack operations -------------------------------------------------
def test_docker_stack_deploy_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_stack_deploy", stack_name="stk", compose_file="docker-compose.yml")
    manager.docker_stack_deploy.assert_called_once_with("stk", "docker-compose.yml")


def test_docker_stack_deploy_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="stack_name and compose_file are required for docker_stack_deploy"):
        _run(tool, action="docker_stack_deploy", stack_name=None, compose_file=None)


def test_docker_stack_services_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_stack_services", stack_name="stk")
    manager.docker_stack_services.assert_called_once_with("stk")


def test_docker_stack_services_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="stack_name is required for docker_stack_services"):
        _run(tool, action="docker_stack_services", stack_name=None)


def test_docker_stack_rm_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_stack_rm", stack_name="stk")
    manager.docker_stack_rm.assert_called_once_with("stk")


def test_docker_stack_rm_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="stack_name is required for docker_stack_rm"):
        _run(tool, action="docker_stack_rm", stack_name=None)


# --- Config operations -------------------------------------------------
def test_docker_config_create_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_config_create", config_name="cfg", data="payload")
    manager.docker_config_create.assert_called_once_with("cfg", "payload")


def test_docker_config_create_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="config_name and data are required for docker_config_create"):
        _run(tool, action="docker_config_create", config_name=None, data=None)


def test_docker_config_list_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_config_list")
    manager.docker_config_list.assert_called_once_with()


# --- Secret operations -------------------------------------------------
def test_docker_secret_create_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_secret_create", secret_name="sec", data="payload")
    manager.docker_secret_create.assert_called_once_with("sec", "payload")


def test_docker_secret_create_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="secret_name and data are required for docker_secret_create"):
        _run(tool, action="docker_secret_create", secret_name=None, data=None)


def test_docker_secret_list_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_secret_list")
    manager.docker_secret_list.assert_called_once_with()


# --- Node operations -------------------------------------------------
def test_docker_node_ls_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_node_ls")
    manager.docker_node_ls.assert_called_once_with()


def test_docker_node_update_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_node_update", node_id="n1", availability="drain")
    manager.docker_node_update.assert_called_once_with("n1", "drain")


def test_docker_node_update_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="node_id and availability are required for docker_node_update"):
        _run(tool, action="docker_node_update", node_id=None, availability=None)


def test_docker_node_inspect_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="docker_node_inspect", node_id="n1")
    manager.docker_node_inspect.assert_called_once_with("n1")


def test_docker_node_inspect_guard_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="node_id is required for docker_node_inspect"):
        _run(tool, action="docker_node_inspect", node_id=None)


def test_unknown_action_raises(tool_and_manager):
    tool, manager = tool_and_manager
    with pytest.raises(ValueError, match="Unknown action: bogus_action"):
        _run(tool, action="bogus_action")

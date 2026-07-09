"""Unit tests for the real DockerManager swarm/service/stack/config/secret/node ops.

These prove the ``cm_docker_swarm`` surface invokes the real docker SDK call (or
the ``docker stack`` CLI where the SDK has no equivalent) and shapes the result —
without needing a live daemon. The SDK client and ``subprocess.run`` are mocked.
"""

from __future__ import annotations

import asyncio
import logging
from unittest import mock
from unittest.mock import MagicMock

from container_manager_mcp.container_manager import DockerManager


def _mgr() -> DockerManager:
    m = DockerManager.__new__(DockerManager)
    m.client = mock.MagicMock()
    m.logger = logging.getLogger("test-docker")
    return m


def test_ports_list_to_map():
    assert DockerManager._ports_list_to_map(["8080:80"]) == {"80/tcp": "8080"}
    assert DockerManager._ports_list_to_map(["53:53/udp"]) == {"53/udp": "53"}
    assert DockerManager._ports_list_to_map(None) is None


def test_swarm_join_calls_sdk():
    m = _mgr()
    m.client.swarm.join.return_value = True
    out = m.docker_swarm_join("10.0.0.1:2377", "tok", worker=True)
    m.client.swarm.join.assert_called_once_with(
        remote_addrs=["10.0.0.1:2377"], join_token="tok"
    )
    assert out["joined"] is True and out["remote_addr"] == "10.0.0.1:2377"


def test_swarm_leave_calls_sdk():
    m = _mgr()
    m.docker_swarm_leave(force=True)
    m.client.swarm.leave.assert_called_once_with(force=True)


def test_service_list_uses_services_list():
    m = _mgr()
    m.client.services.list.return_value = []
    assert m.docker_service_list() == []
    m.client.services.list.assert_called_once()


def test_service_ps_iterates_tasks():
    m = _mgr()
    svc = mock.MagicMock()
    svc.id = "svc123456789"
    svc.attrs = {"Spec": {"Name": "web"}}
    svc.tasks.return_value = [
        {
            "ID": "task123456789",
            "NodeID": "node123456789",
            "DesiredState": "running",
            "Status": {"State": "running"},
        }
    ]
    m.client.services.list.return_value = [svc]
    out = m.docker_service_ps()
    svc.tasks.assert_called_once()
    assert out[0]["service_name"] == "web"
    assert out[0]["current_state"] == "running"


def test_config_create_calls_sdk():
    m = _mgr()
    created = mock.MagicMock()
    created.id = "cfg1"
    m.client.configs.create.return_value = created
    out = m.docker_config_create("mycfg", "hello")
    m.client.configs.create.assert_called_once_with(name="mycfg", data=b"hello")
    assert out["id"] == "cfg1" and out["status"] == "created"


def test_secret_list_calls_sdk():
    m = _mgr()
    s = mock.MagicMock()
    s.id = "sec1"
    s.attrs = {"Spec": {"Name": "db-pass"}, "CreatedAt": "t"}
    m.client.secrets.list.return_value = [s]
    out = m.docker_secret_list()
    assert out == [{"id": "sec1", "name": "db-pass", "created": "t"}]


def test_stack_deploy_shells_out():
    m = _mgr()
    fake = mock.MagicMock(returncode=0, stdout="deployed ok", stderr="")
    with mock.patch(
        "container_manager_mcp.container_manager.subprocess.run", return_value=fake
    ) as run:
        out = m.docker_stack_deploy("mystack", "compose.yml")
    cmd = run.call_args[0][0]
    assert cmd == ["docker", "stack", "deploy", "-c", "compose.yml", "mystack"]
    assert out["status"] == "deployed"


def test_stack_services_parses_rows():
    m = _mgr()
    fake = mock.MagicMock(returncode=0, stdout="web\t3/3\tnginx\n", stderr="")
    with mock.patch(
        "container_manager_mcp.container_manager.subprocess.run", return_value=fake
    ):
        out = m.docker_stack_services("mystack")
    assert out == [
        {"name": "web", "replicas": "3/3", "image": "nginx", "stack_name": "mystack"}
    ]


def test_stack_rm_raises_on_cli_failure():
    m = _mgr()
    fake = mock.MagicMock(returncode=1, stdout="", stderr="no such stack")
    with mock.patch(
        "container_manager_mcp.container_manager.subprocess.run", return_value=fake
    ):
        try:
            m.docker_stack_rm("nope")
        except RuntimeError as e:
            assert "no such stack" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected RuntimeError")


# ---------------------------------------------------------------------------
# MCP tool (regression: ctx_log wrong-arg-shape + run_blocking misused as a
# decorator both raised TypeError on every real cm_docker_swarm invocation)
# ---------------------------------------------------------------------------
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


def test_cm_docker_swarm_tool_with_ctx_does_not_raise(monkeypatch):
    """Regression: ``cm_docker_swarm`` called ``ctx_log("Docker Swarm operations",
    action=..., service_name=...)`` — a leading string plus kwargs the shim
    doesn't accept — raising ``TypeError`` on every real invocation with a ctx.
    This test drives a truthy ``ctx`` (MagicMock) through the tool and asserts
    no TypeError.
    """
    from container_manager_mcp.mcp import mcp_docker_swarm

    fake_manager = MagicMock()
    fake_manager.docker_service_list.return_value = []
    monkeypatch.setattr(
        mcp_docker_swarm, "create_manager", lambda backend: fake_manager
    )
    tool = _capture_tool(mcp_docker_swarm.register_dockerswarm_tools)

    fake_ctx = MagicMock()
    result = asyncio.run(tool(action="docker_service_list", ctx=fake_ctx))

    assert result == []
    assert fake_ctx.info.called

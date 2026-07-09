"""Unit tests for the real PodmanManager pod/network/volume/kube/checkpoint ops.

These prove the ``cm_podman`` surface invokes the real podman-py SDK call (or the
``podman`` CLI where podman-py has no equivalent) and shapes the result — without
a live Podman socket. The SDK client and ``subprocess.run`` are mocked.
"""

from __future__ import annotations

import asyncio
import logging
from unittest import mock
from unittest.mock import MagicMock

from container_manager_mcp.container_manager import PodmanManager


def _mgr() -> PodmanManager:
    m = PodmanManager.__new__(PodmanManager)
    m.client = mock.MagicMock()
    m.logger = logging.getLogger("test-podman")
    return m


def test_pod_create_calls_sdk():
    m = _mgr()
    pod = mock.MagicMock()
    pod.attrs = {"Id": "podabc"}
    m.client.pods.create.return_value = pod
    out = m.podman_pod_create("mypod", "nginx", None)
    m.client.pods.create.assert_called_once_with(name="mypod")
    assert out["id"] == "podabc" and out["status"] == "created"


def test_pod_list_calls_sdk():
    m = _mgr()
    pod = mock.MagicMock()
    pod.attrs = {"Name": "p1", "Id": "id1", "Status": "Running"}
    m.client.pods.list.return_value = [pod]
    out = m.podman_pod_list()
    m.client.pods.list.assert_called_once()
    assert out[0]["name"] == "p1" and out[0]["status"] == "Running"


def test_pod_stop_calls_sdk():
    m = _mgr()
    pod = mock.MagicMock()
    m.client.pods.get.return_value = pod
    m.podman_pod_stop("mypod")
    m.client.pods.get.assert_called_once_with("mypod")
    pod.stop.assert_called_once()


def test_pod_rm_calls_sdk():
    m = _mgr()
    pod = mock.MagicMock()
    m.client.pods.get.return_value = pod
    m.podman_pod_rm("mypod")
    pod.remove.assert_called_once()


def test_network_create_calls_sdk_with_subnet():
    m = _mgr()
    net = mock.MagicMock()
    net.attrs = {"Id": "netid"}
    m.client.networks.create.return_value = net
    out = m.podman_network_create("mynet", "bridge", "10.1.0.0/24")
    m.client.networks.create.assert_called_once_with(
        "mynet", driver="bridge", subnet="10.1.0.0/24"
    )
    assert out["id"] == "netid"


def test_volume_list_calls_sdk():
    m = _mgr()
    vol = mock.MagicMock()
    vol.attrs = {"Name": "v1", "Driver": "local", "Mountpoint": "/x"}
    m.client.volumes.list.return_value = [vol]
    out = m.podman_volume_list()
    assert out[0]["name"] == "v1" and out[0]["mountpoint"] == "/x"


def test_pod_stats_shells_out():
    m = _mgr()
    fake = mock.MagicMock(returncode=0, stdout='[{"cpu":"1%"}]', stderr="")
    with mock.patch(
        "container_manager_mcp.container_manager.subprocess.run", return_value=fake
    ) as run:
        out = m.podman_pod_stats("mypod")
    cmd = run.call_args[0][0]
    assert cmd[:3] == ["podman", "pod", "stats"] and "mypod" in cmd
    assert out["stats"] == [{"cpu": "1%"}]


def test_generate_kube_yaml_shells_out():
    m = _mgr()
    fake = mock.MagicMock(returncode=0, stdout="apiVersion: v1\nkind: Pod\n", stderr="")
    with mock.patch(
        "container_manager_mcp.container_manager.subprocess.run", return_value=fake
    ) as run:
        out = m.podman_generate_kube_yaml("mypod")
    assert run.call_args[0][0] == ["podman", "generate", "kube", "mypod"]
    assert out["yaml"].startswith("apiVersion: v1")


def test_system_prune_shells_out():
    m = _mgr()
    fake = mock.MagicMock(returncode=0, stdout="reclaimed", stderr="")
    with mock.patch(
        "container_manager_mcp.container_manager.subprocess.run", return_value=fake
    ) as run:
        out = m.podman_system_prune()
    assert run.call_args[0][0] == ["podman", "system", "prune", "-f"]
    assert out["status"] == "pruned"


def test_health_check_shells_out():
    m = _mgr()
    fake = mock.MagicMock(returncode=0, stdout="healthy", stderr="")
    with mock.patch(
        "container_manager_mcp.container_manager.subprocess.run", return_value=fake
    ) as run:
        out = m.podman_health_check("cid", {})
    assert run.call_args[0][0] == ["podman", "healthcheck", "run", "cid"]
    assert out["status"] == "healthy"


def test_checkpoint_shells_out_with_export():
    m = _mgr()
    fake = mock.MagicMock(returncode=0, stdout="", stderr="")
    with mock.patch(
        "container_manager_mcp.container_manager.subprocess.run", return_value=fake
    ) as run:
        m.podman_checkpoint("cid", "/tmp/ckpt")
    assert run.call_args[0][0] == [
        "podman",
        "container",
        "checkpoint",
        "--export",
        "/tmp/ckpt",
        "cid",
    ]


def test_cli_failure_raises():
    m = _mgr()
    fake = mock.MagicMock(returncode=1, stdout="", stderr="boom")
    with mock.patch(
        "container_manager_mcp.container_manager.subprocess.run", return_value=fake
    ):
        try:
            m.podman_system_prune()
        except RuntimeError as e:
            assert "boom" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected RuntimeError")


# ---------------------------------------------------------------------------
# MCP tool (regression: ctx_log wrong-arg-shape + run_blocking misused as a
# decorator both raised TypeError on every real cm_podman invocation)
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


def test_cm_podman_tool_with_ctx_does_not_raise(monkeypatch):
    """Regression: ``cm_podman`` called ``ctx_log("Podman operations", action=...,
    pod_name=...)`` — a leading string plus kwargs the shim doesn't accept —
    raising ``TypeError`` on every real invocation with a ctx. This test drives
    a truthy ``ctx`` (MagicMock) through the tool and asserts no TypeError.
    """
    from container_manager_mcp.mcp import mcp_podman

    fake_manager = MagicMock()
    fake_manager.podman_pod_list.return_value = []
    monkeypatch.setattr(mcp_podman, "create_manager", lambda backend: fake_manager)
    tool = _capture_tool(mcp_podman.register_podman_tools)

    fake_ctx = MagicMock()
    result = asyncio.run(tool(action="podman_pod_list", ctx=fake_ctx))

    assert result == []
    assert fake_ctx.info.called

"""Unit tests for the real ``kubectl cp`` pod-copy implementation.

Proves copy_to_pod / copy_from_pod / cp_pod invoke ``kubectl cp`` with the right
source/destination ordering and raise on failure — without a live cluster.
``subprocess.run`` is mocked.
"""

from __future__ import annotations

import logging
import tempfile
from unittest import mock

from container_manager_mcp.k8s.workloads import WorkloadsMixin


class _Stub(WorkloadsMixin):
    def __init__(self, context=None):
        self.namespace = "default"
        self.logger = logging.getLogger("test-copy")
        if context is not None:
            self.context = context

    def log_action(self, *a, **k):
        pass


def _ok():
    return mock.MagicMock(returncode=0, stdout=b"", stderr=b"")


def test_copy_to_pod_invokes_kubectl_cp():
    stub = _Stub()
    with tempfile.NamedTemporaryFile() as f:
        with mock.patch(
            "container_manager_mcp.k8s.workloads.subprocess.run", return_value=_ok()
        ) as run:
            out = stub.copy_to_pod("mypod", "ns1", f.name, "/etc/app.conf")
    assert run.call_args[0][0] == [
        "kubectl",
        "cp",
        f.name,
        "ns1/mypod:/etc/app.conf",
    ]
    assert out["direction"] == "local_to_pod" and out["status"] == "copied"


def test_copy_to_pod_missing_source_raises():
    stub = _Stub()
    with mock.patch(
        "container_manager_mcp.k8s.workloads.subprocess.run", return_value=_ok()
    ):
        try:
            stub.copy_to_pod("mypod", "ns1", "/no/such/file", "/dst")
        except RuntimeError as e:
            assert "not found" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected RuntimeError")


def test_copy_from_pod_invokes_kubectl_cp():
    stub = _Stub()
    with mock.patch(
        "container_manager_mcp.k8s.workloads.subprocess.run", return_value=_ok()
    ) as run:
        out = stub.copy_from_pod("mypod", "ns1", "/var/log/app.log", "/tmp/app.log")
    assert run.call_args[0][0] == [
        "kubectl",
        "cp",
        "ns1/mypod:/var/log/app.log",
        "/tmp/app.log",
    ]
    assert out["direction"] == "pod_to_local"


def test_copy_passes_context_when_present():
    stub = _Stub(context="prod")
    with mock.patch(
        "container_manager_mcp.k8s.workloads.subprocess.run", return_value=_ok()
    ) as run:
        stub.copy_from_pod("mypod", "ns1", "/a", "/b")
    assert run.call_args[0][0][-2:] == ["--context", "prod"]


def test_cp_pod_delegates_pod_to_local():
    stub = _Stub()
    with mock.patch.object(stub, "copy_from_pod", return_value={"ok": 1}) as cf:
        out = stub.cp_pod("mypod", "ns1", "/abs/pod/path", "relative/local")
    cf.assert_called_once_with("mypod", "ns1", "/abs/pod/path", "relative/local")
    assert out == {"ok": 1}


def test_cp_pod_delegates_local_to_pod():
    stub = _Stub()
    with mock.patch.object(stub, "copy_to_pod", return_value={"ok": 2}) as ct:
        out = stub.cp_pod("mypod", "ns1", "relative/local", "/abs/pod/path")
    ct.assert_called_once_with("mypod", "ns1", "relative/local", "/abs/pod/path")
    assert out == {"ok": 2}


def test_kubectl_cp_failure_raises():
    stub = _Stub()
    fail = mock.MagicMock(returncode=1, stdout=b"", stderr=b"pod not found")
    with mock.patch(
        "container_manager_mcp.k8s.workloads.subprocess.run", return_value=fail
    ):
        try:
            stub.copy_from_pod("mypod", "ns1", "/a", "/b")
        except RuntimeError as e:
            assert "pod not found" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected RuntimeError")

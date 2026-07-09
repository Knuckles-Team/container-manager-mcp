"""Tests for the container-manager environment doctor (engine + CLI + MCP tool).

Every external surface is mocked — HostManager, KubernetesManager/create_manager,
is_app_installed, module availability, and the TCP probe — so no real daemon,
cluster, or SSH host is required.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from container_manager_mcp import doctor as doc


class _FakeManager:
    """A create_manager() result that is fully reachable."""

    def get_version(self):
        return {"version": "1.2.3"}

    def validate_kubeconfig(self):
        return {"status": "valid"}

    def list_nodes(self):
        return [{"id": "n1"}]


class _FakeHostManager:
    def __init__(self, hosts):
        self._hosts = hosts

    def list_hosts(self):
        return dict(self._hosts)

    def get_host(self, alias):
        return self._hosts.get(alias)


def _all_ok(monkeypatch, tmp_path):
    """Wire every probe to succeed."""
    monkeypatch.setattr(doc, "_module_available", lambda name: True)
    monkeypatch.setattr(doc, "is_app_installed", lambda *a, **k: True)
    monkeypatch.setattr(doc, "create_manager", lambda *a, **k: _FakeManager())
    monkeypatch.setattr(
        doc, "_probe_tcp", lambda host, port, timeout=5.0: (True, "reachable")
    )

    inv = tmp_path / "inventory.yml"
    inv.write_text("all: {}\n")
    hosts = {"h1": SimpleNamespace(hostname="10.0.0.1", port=22)}
    monkeypatch.setattr(doc, "HostManager", lambda path=None: _FakeHostManager(hosts))
    monkeypatch.setattr(doc, "default_inventory_path", lambda: str(inv))

    # kubeconfig "present"
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("CONTAINER_MANAGER_TYPE", "docker")
    monkeypatch.delenv("K8S_CONTEXTS", raising=False)
    return inv


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def test_all_ok_path(monkeypatch, tmp_path):
    inv = _all_ok(monkeypatch, tmp_path)
    report = doc.run_doctor(backend="all", inventory=str(inv))
    assert report["summary"]["fail"] == 0
    assert report["summary"]["status"] in ("ok", "warn")
    assert report["summary"]["total"] == len(report["checks"])
    # every check has the required shape
    for c in report["checks"]:
        assert set(c) == {"name", "category", "status", "detail", "remediation"}
        assert c["status"] in ("ok", "warn", "fail")


def test_backends_missing_warns_with_remediation(monkeypatch):
    monkeypatch.setattr(doc, "_module_available", lambda name: False)
    monkeypatch.setattr(doc, "is_app_installed", lambda *a, **k: False)
    checks = doc._check_backends()
    missing = [c for c in checks if c["status"] == "warn"]
    assert missing, "expected warnings for missing libs/CLIs"
    assert all(c["remediation"] for c in missing)


def test_docker_missing_focused_fails_with_remediation(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("Cannot connect to the Docker daemon")

    monkeypatch.setattr(doc, "create_manager", _boom)
    report = doc.run_doctor(backend="docker")
    docker_checks = [c for c in report["checks"] if c["category"] == "docker"]
    assert docker_checks and docker_checks[0]["status"] == "fail"
    assert docker_checks[0]["remediation"]
    assert report["summary"]["fail"] >= 1


def test_podman_missing_focused_fails_with_remediation(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("podman socket not found")

    monkeypatch.setattr(doc, "create_manager", _boom)
    report = doc.run_doctor(backend="podman")
    podman_checks = [c for c in report["checks"] if c["category"] == "podman"]
    assert podman_checks and podman_checks[0]["status"] == "fail"
    assert "podman.socket" in podman_checks[0]["remediation"]


def test_inventory_missing_fails(monkeypatch, tmp_path):
    missing = tmp_path / "nope.yml"
    monkeypatch.setattr(doc, "HostManager", lambda path=None: _FakeHostManager({}))
    report = doc.run_doctor(backend="inventory", inventory=str(missing))
    inv_checks = [c for c in report["checks"] if c["category"] == "inventory"]
    assert inv_checks and inv_checks[0]["status"] == "fail"
    assert "inventory" in inv_checks[0]["remediation"].lower()
    assert report["summary"]["fail"] >= 1


def test_kube_context_unreachable_fails(monkeypatch):
    class _Unreachable:
        def validate_kubeconfig(self):
            return {"status": "valid"}

        def get_version(self):
            raise RuntimeError("Connection refused to https://10.0.0.1:6443")

    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")  # kubeconfig "present"
    monkeypatch.setenv("K8S_CONTEXTS", "prod=prod-cluster")
    monkeypatch.delenv("DEFAULT_K8S_CONTEXT", raising=False)
    monkeypatch.setattr(doc, "create_manager", lambda *a, **k: _Unreachable())

    report = doc.run_doctor(backend="kubernetes", context="prod")
    kube_fail = [
        c
        for c in report["checks"]
        if c["category"] == "kubernetes" and c["status"] == "fail"
    ]
    assert kube_fail, "expected a failing kubernetes context check"
    assert kube_fail[0]["remediation"]
    assert report["summary"]["fail"] >= 1


def test_inventory_host_probe_ok(monkeypatch, tmp_path):
    inv = tmp_path / "inventory.yml"
    inv.write_text("all: {}\n")
    hosts = {"h1": SimpleNamespace(hostname="10.0.0.1", port=22)}
    monkeypatch.setattr(doc, "HostManager", lambda path=None: _FakeHostManager(hosts))
    monkeypatch.setattr(
        doc, "_probe_tcp", lambda host, port, timeout=5.0: (True, "TCP ok")
    )
    report = doc.run_doctor(backend="inventory", host="h1", inventory=str(inv))
    host_checks = [c for c in report["checks"] if c["name"] == "host 'h1'"]
    assert host_checks and host_checks[0]["status"] == "ok"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_exit_zero_when_no_fail(monkeypatch, tmp_path):
    inv = _all_ok(monkeypatch, tmp_path)
    rc = doc.doctor(["--backend", "inventory", "--host", "h1", "--inventory", str(inv)])
    assert rc == 0


def test_cli_exit_one_on_failure(monkeypatch, tmp_path):
    missing = tmp_path / "nope.yml"
    monkeypatch.setattr(doc, "HostManager", lambda path=None: _FakeHostManager({}))
    rc = doc.doctor(["--backend", "inventory", "--inventory", str(missing)])
    assert rc == 1


def test_cli_json_output_shape(monkeypatch, tmp_path, capsys):
    inv = _all_ok(monkeypatch, tmp_path)
    doc.doctor(["--backend", "inventory", "--inventory", str(inv), "--json"])
    import json

    out = json.loads(capsys.readouterr().out)
    assert set(out) >= {"backend", "checks", "summary", "host", "context"}
    assert set(out["summary"]) == {"total", "ok", "warn", "fail", "status"}


# ---------------------------------------------------------------------------
# MCP tool
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


def test_cm_doctor_tool_returns_dict(monkeypatch, tmp_path):
    from container_manager_mcp.mcp import mcp_doctor

    sentinel = {"checks": [], "summary": {"status": "ok", "fail": 0}}
    monkeypatch.setattr(mcp_doctor, "run_doctor", lambda **k: sentinel)
    tool = _capture_tool(mcp_doctor.register_doctor_tools)
    result = asyncio.run(tool(action="run"))
    assert result == sentinel


def test_cm_doctor_tool_maps_actions_to_backend(monkeypatch):
    from container_manager_mcp.mcp import mcp_doctor

    seen = {}

    def _fake_run(**kwargs):
        seen.update(kwargs)
        return {"summary": {"status": "ok"}}

    monkeypatch.setattr(mcp_doctor, "run_doctor", _fake_run)
    tool = _capture_tool(mcp_doctor.register_doctor_tools)
    asyncio.run(tool(action="check_kubernetes", context="prod"))
    assert seen["backend"] == "kubernetes"
    assert seen["context"] == "prod"


def test_cm_doctor_tool_with_ctx_does_not_raise(monkeypatch):
    """Regression: ``ctx_log`` previously required a ``server_logger`` argument
    the doctor tool never passed, so any call with a real (truthy) ``ctx``
    raised ``TypeError: ctx_log() missing 1 required positional argument:
    'message'``. The other ``cm_doctor`` tests above only exercise
    ``ctx=None`` (the ``if ctx:`` guard short-circuits), so they never
    caught it — this test passes a truthy ctx to hit the log call.
    """
    from container_manager_mcp.mcp import mcp_doctor

    sentinel = {"checks": [], "summary": {"status": "ok", "fail": 0}}
    monkeypatch.setattr(mcp_doctor, "run_doctor", lambda **k: sentinel)
    tool = _capture_tool(mcp_doctor.register_doctor_tools)

    fake_ctx = MagicMock()
    result = asyncio.run(tool(action="check_backends", ctx=fake_ctx))

    assert result == sentinel
    assert fake_ctx.info.called


def test_cm_doctor_tool_with_ctx_error_path_does_not_raise(monkeypatch):
    """Same regression, but through the ``except`` branch's ``ctx_log`` call."""
    from container_manager_mcp.mcp import mcp_doctor

    def _boom(**k):
        raise RuntimeError("boom")

    monkeypatch.setattr(mcp_doctor, "run_doctor", _boom)
    tool = _capture_tool(mcp_doctor.register_doctor_tools)

    fake_ctx = MagicMock()
    result = asyncio.run(tool(action="check_backends", ctx=fake_ctx))

    assert result == {"error": "boom", "action": "check_backends"}
    assert fake_ctx.error.called


def test_register_doctor_tools_gated_off(monkeypatch):
    from container_manager_mcp.mcp_server import register_doctor_tools

    monkeypatch.setenv("DOCTORTOOL", "False")
    fake_mcp = MagicMock()

    def _boom(*a, **k):
        raise AssertionError("should not register when DOCTORTOOL is off")

    fake_mcp.tool = _boom
    # When toggled off, the wrapper returns before importing/registering anything.
    assert register_doctor_tools(fake_mcp) is None

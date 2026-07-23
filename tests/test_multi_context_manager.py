#!/usr/bin/env python
"""Tests for the MultiContextManager (K8S/Docker/Podman/Swarm context pooling)."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.multi_context_manager import MultiContextManager


def _make_k8s_manager_mock(version_result=None, get_version_side_effect=None):
    """Build a MagicMock standing in for a KubernetesManager instance."""
    manager = MagicMock()
    if get_version_side_effect is not None:
        manager.get_version.side_effect = get_version_side_effect
    else:
        manager.get_version.return_value = version_result or {"version": "1.0"}
    return manager


@pytest.fixture
def patched_backends():
    """Patch out the real Docker/Podman/Kubernetes manager classes.

    KubernetesManager is imported lazily inside _add_k8s_context (`from
    container_manager_mcp.k8s_manager import KubernetesManager`), so it must be
    patched at its defining module. DockerManager/PodmanManager are imported at
    module scope in multi_context_manager, so they're patched there.
    """
    with (
        patch(
            "container_manager_mcp.multi_context_manager.DockerManager"
        ) as mock_docker_cls,
        patch(
            "container_manager_mcp.multi_context_manager.PodmanManager"
        ) as mock_podman_cls,
        patch("container_manager_mcp.k8s_manager.KubernetesManager") as mock_k8s_cls,
    ):
        mock_docker_cls.side_effect = lambda **kwargs: MagicMock(
            get_version=MagicMock(return_value={"version": "docker"})
        )
        mock_podman_cls.side_effect = lambda **kwargs: MagicMock(
            get_version=MagicMock(return_value={"version": "podman"})
        )
        mock_k8s_cls.side_effect = lambda **kwargs: MagicMock(
            get_version=MagicMock(return_value={"version": "k8s"})
        )
        yield {
            "docker": mock_docker_cls,
            "podman": mock_podman_cls,
            "k8s": mock_k8s_cls,
        }


class TestConstruction:
    """(a) Construction succeeds with K8S_CONTEXTS + DOCKER_CONTEXTS configured."""

    def test_constructs_with_k8s_and_docker_contexts(
        self, monkeypatch, patched_backends
    ):
        monkeypatch.setenv("K8S_CONTEXTS", "a=ctxA;b=ctxB")
        monkeypatch.setenv("DOCKER_CONTEXTS", "local=;remote=remote-host")
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        try:
            assert mgr.silent is True
            assert set(mgr.k8s_managers.keys()) == {"a", "b"}
            assert set(mgr.docker_managers.keys()) == {"local", "remote"}
            assert patched_backends["k8s"].call_count == 2
            assert patched_backends["docker"].call_count == 2
        finally:
            mgr.shutdown()

    def test_construction_sets_silent_and_log_file_before_init(
        self, monkeypatch, patched_backends
    ):
        """Regression test for the historical super().__init__ bug: silent/log_file
        must be set (and readable by _add_*_context) before managers are built."""
        monkeypatch.setenv("K8S_CONTEXTS", "only=ctxOnly")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True, log_file="/tmp/mcm-test.log")
        try:
            assert mgr.silent is True
            assert mgr.log_file == "/tmp/mcm-test.log"
            assert "only" in mgr.k8s_managers
        finally:
            mgr.shutdown()


class TestDefaultContextSelection:
    """(b) Default-context auto-selection."""

    def test_default_context_auto_selected_when_not_configured(
        self, monkeypatch, patched_backends
    ):
        monkeypatch.setenv("K8S_CONTEXTS", "first=ctx1;second=ctx2")
        monkeypatch.delenv("DEFAULT_K8S_CONTEXT", raising=False)
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        try:
            assert mgr.default_k8s_context == "first"
        finally:
            mgr.shutdown()

    def test_default_context_honored_when_configured(
        self, monkeypatch, patched_backends
    ):
        monkeypatch.setenv("K8S_CONTEXTS", "first=ctx1;second=ctx2")
        monkeypatch.setenv("DEFAULT_K8S_CONTEXT", "second")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        try:
            assert mgr.default_k8s_context == "second"
            assert mgr.get_k8s_manager() is mgr.k8s_managers["second"]
        finally:
            mgr.shutdown()


class TestFanOut:
    """(c) fan_out runs across contexts and (d) isolates per-context errors."""

    def test_fan_out_runs_across_two_contexts(self, monkeypatch, patched_backends):
        monkeypatch.setenv("K8S_CONTEXTS", "a=ctxA;b=ctxB")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        try:
            mgr.k8s_managers["a"].list_containers.return_value = ["a-container"]
            mgr.k8s_managers["b"].list_containers.return_value = ["b-container"]

            results = mgr.fan_out("kubernetes", "list_containers")

            assert results == {"a": ["a-container"], "b": ["b-container"]}
        finally:
            mgr.shutdown()

    def test_fan_out_isolates_errors_per_context(self, monkeypatch, patched_backends):
        monkeypatch.setenv("K8S_CONTEXTS", "a=ctxA;b=ctxB")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        try:
            mgr.k8s_managers["a"].list_containers.side_effect = RuntimeError("boom")
            mgr.k8s_managers["b"].list_containers.return_value = ["b-container"]

            results = mgr.fan_out("kubernetes", "list_containers")

            assert results["b"] == ["b-container"]
            # per the security-hardening line (3ad52b5), fan_out reports a generic
            # message and logs the real error server-side rather than leaking the
            # raw exception text into the result payload.
            assert results["a"] == {"error": "Operation failed"}
        finally:
            mgr.shutdown()

    def test_fan_out_with_explicit_contexts_subset(self, monkeypatch, patched_backends):
        monkeypatch.setenv("K8S_CONTEXTS", "a=ctxA;b=ctxB;c=ctxC")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        try:
            mgr.k8s_managers["a"].list_images.return_value = ["img-a"]
            mgr.k8s_managers["c"].list_images.return_value = ["img-c"]

            results = mgr.fan_out("kubernetes", "list_images", contexts=["a", "c"])

            assert set(results.keys()) == {"a", "c"}
            assert results["a"] == ["img-a"]
            assert results["c"] == ["img-c"]
        finally:
            mgr.shutdown()

    def test_fan_out_all_covers_every_backend(self, monkeypatch, patched_backends):
        monkeypatch.setenv("K8S_CONTEXTS", "a=ctxA")
        monkeypatch.setenv("DOCKER_CONTEXTS", "local=")
        monkeypatch.setenv("PODMAN_ENABLED", "true")

        mgr = MultiContextManager(silent=True)
        try:
            mgr.k8s_managers["a"].get_version.return_value = {"version": "k8s"}
            mgr.docker_managers["local"].get_version.return_value = {
                "version": "docker"
            }
            mgr.podman_manager.get_version.return_value = {"version": "podman"}

            results = mgr.fan_out_all("get_version")

            assert results["kubernetes"]["a"] == {"version": "k8s"}
            assert results["docker"]["local"] == {"version": "docker"}
            assert results["podman"]["local"] == {"version": "podman"}
            assert results["swarm"] == {}
        finally:
            mgr.shutdown()


class TestHealthAndReconnect:
    """(e) Health-check + lazy reconnect."""

    def test_unhealthy_manager_triggers_lazy_reconnect(
        self, monkeypatch, patched_backends
    ):
        monkeypatch.setenv("K8S_CONTEXTS", "a=ctxA")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")
        monkeypatch.setenv("HEALTH_CHECK_TTL_SECONDS", "30")

        mgr = MultiContextManager(silent=True)
        try:
            failing_manager = mgr.k8s_managers["a"]
            failing_manager.get_version.side_effect = RuntimeError("connection dropped")

            # A second call to KubernetesManager(...) (the reconnect) should
            # produce a healthy manager.
            reconnected_manager = MagicMock()
            reconnected_manager.get_version.return_value = {
                "version": "k8s-reconnected"
            }
            patched_backends["k8s"].side_effect = lambda **kwargs: reconnected_manager

            resolved = mgr.get_k8s_manager("a")

            assert resolved is reconnected_manager
            assert resolved is not failing_manager
            # constructed once at init, once on reconnect
            assert patched_backends["k8s"].call_count == 2
        finally:
            mgr.shutdown()

    def test_healthy_manager_is_not_reconnected(self, monkeypatch, patched_backends):
        monkeypatch.setenv("K8S_CONTEXTS", "a=ctxA")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        try:
            healthy_manager = mgr.k8s_managers["a"]
            healthy_manager.get_version.return_value = {"version": "fine"}

            resolved = mgr.get_k8s_manager("a")

            assert resolved is healthy_manager
            assert patched_backends["k8s"].call_count == 1
        finally:
            mgr.shutdown()

    def test_is_healthy_caches_within_ttl(self, monkeypatch, patched_backends):
        monkeypatch.setenv("K8S_CONTEXTS", "a=ctxA")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        try:
            manager = mgr.k8s_managers["a"]
            manager.get_version.return_value = {"version": "fine"}

            assert mgr._is_healthy(manager, "kubernetes", "a") is True
            assert mgr._is_healthy(manager, "kubernetes", "a") is True
            # Cached: get_version should only actually be invoked once.
            assert manager.get_version.call_count == 1
        finally:
            mgr.shutdown()


class TestBackwardsCompatibleSurface:
    """Existing public methods and env-driven config surface remain intact."""

    def test_list_available_contexts_shape(self, monkeypatch, patched_backends):
        monkeypatch.setenv("K8S_CONTEXTS", "a=ctxA")
        monkeypatch.setenv("DOCKER_CONTEXTS", "local=")
        monkeypatch.setenv("PODMAN_ENABLED", "true")

        mgr = MultiContextManager(silent=True)
        try:
            contexts = mgr.list_available_contexts()
            assert contexts["kubernetes"]["contexts"] == ["a"]
            assert contexts["kubernetes"]["default"] == "a"
            assert contexts["docker"]["contexts"] == ["local"]
            assert contexts["podman"]["enabled"] is True
        finally:
            mgr.shutdown()

    def test_list_containers_delegates_to_backend_manager(
        self, monkeypatch, patched_backends
    ):
        monkeypatch.setenv("K8S_CONTEXTS", "a=ctxA")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        try:
            mgr.k8s_managers["a"].list_containers.return_value = ["c1"]
            result = mgr.list_containers(backend="kubernetes", context="a")
            assert result == ["c1"]
        finally:
            mgr.shutdown()

    def test_get_manager_unsupported_backend_raises(
        self, monkeypatch, patched_backends
    ):
        monkeypatch.delenv("K8S_CONTEXTS", raising=False)
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        try:
            with pytest.raises(ValueError, match="Unsupported backend"):
                mgr.get_manager(backend="bogus")
        finally:
            mgr.shutdown()

    def test_get_context_not_found_raises(self, monkeypatch, patched_backends):
        monkeypatch.setenv("K8S_CONTEXTS", "a=ctxA")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        try:
            with pytest.raises(ValueError, match="not found"):
                mgr.get_k8s_manager("does-not-exist")
        finally:
            mgr.shutdown()

    def test_shutdown_closes_executor(self, monkeypatch, patched_backends):
        monkeypatch.delenv("K8S_CONTEXTS", raising=False)
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        mgr = MultiContextManager(silent=True)
        mgr.shutdown()
        assert mgr._executor._shutdown is True


# ---------------------------------------------------------------------------
# MCP tool (regression: ctx_log wrong-arg-shape + run_blocking misused as a
# decorator both raised TypeError on every real cm_multi_context invocation)
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


def test_cm_multi_context_tool_with_ctx_does_not_raise(monkeypatch):
    """Regression: ``cm_multi_context`` called ``ctx_log("Multi-context operations",
    action=..., backend=..., context=...)`` — a leading string plus kwargs the
    shim doesn't accept — raising ``TypeError`` on every real invocation with a
    ctx. This test drives a truthy ``ctx`` (MagicMock) through the tool and
    asserts no TypeError.
    """
    from container_manager_mcp.mcp import mcp_multi_context

    fake_manager = MagicMock()
    fake_manager.list_available_contexts.return_value = ["default"]
    monkeypatch.setattr(
        mcp_multi_context, "create_manager", lambda manager_type: fake_manager
    )
    tool = _capture_tool(mcp_multi_context.register_multicontext_tools)

    fake_ctx = MagicMock()
    result = asyncio.run(tool(action="list_contexts", ctx=fake_ctx))

    assert result == ["default"]
    assert fake_ctx.info.called


# ---------------------------------------------------------------------------
# Regression: create_manager(<backend>) under MULTI_CONTEXT_MODE must resolve
# to the concrete per-backend manager (KubernetesManager/DockerManager/
# PodmanManager), not the MultiContextManager pool itself. Every themed MCP
# tool (cm_k8s_cluster, cm_k8s_workloads, ..., cm_docker_swarm, cm_podman)
# calls create_manager("kubernetes"/"docker"/"podman") and then invokes
# backend-specific verbs (e.g. manager.get_cluster_info(), manager.list_nodes())
# directly on the result. Before this fix, multi-context mode always returned
# the bare MultiContextManager -- which only defines pool-management methods
# plus a handful of generic list_* delegators -- so every backend-specific
# verb raised AttributeError.
# ---------------------------------------------------------------------------
class TestCreateManagerMultiContextDelegation:
    def test_create_manager_kubernetes_resolves_to_default_k8s_manager(
        self, monkeypatch, patched_backends
    ):
        monkeypatch.setenv("MULTI_CONTEXT_MODE", "true")
        monkeypatch.setenv("K8S_CONTEXTS", "only=ctxOnly")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        manager = create_manager("kubernetes")

        # Not the pooling wrapper -- the concrete default-context manager.
        assert not isinstance(manager, MultiContextManager)

        # A representative k8s-only verb (absent from MultiContextManager)
        # must be reachable and callable without AttributeError.
        manager.get_cluster_info.return_value = {"version": "v1.31"}
        manager.list_namespaces.return_value = ["default", "kube-system"]

        assert manager.get_cluster_info() == {"version": "v1.31"}
        assert manager.list_namespaces() == ["default", "kube-system"]

    def test_create_manager_multi_still_returns_pool(
        self, monkeypatch, patched_backends
    ):
        monkeypatch.setenv("MULTI_CONTEXT_MODE", "true")
        monkeypatch.setenv("K8S_CONTEXTS", "only=ctxOnly")
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        manager = create_manager("multi")

        assert isinstance(manager, MultiContextManager)

    def test_create_manager_kubernetes_no_context_configured_raises_clear_error(
        self, monkeypatch, patched_backends
    ):
        """No K8S_CONTEXTS configured: resolution must fail with a clear,
        catchable error -- not crash or silently return an unusable object."""
        monkeypatch.setenv("MULTI_CONTEXT_MODE", "true")
        monkeypatch.delenv("K8S_CONTEXTS", raising=False)
        monkeypatch.delenv("DOCKER_CONTEXTS", raising=False)
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        with pytest.raises(ValueError, match="No k8s contexts are available"):
            create_manager("kubernetes")

    def test_create_manager_docker_resolves_to_default_docker_manager(
        self, monkeypatch, patched_backends
    ):
        monkeypatch.setenv("MULTI_CONTEXT_MODE", "true")
        monkeypatch.delenv("K8S_CONTEXTS", raising=False)
        monkeypatch.setenv("DOCKER_CONTEXTS", "local=")
        monkeypatch.setenv("PODMAN_ENABLED", "false")

        manager = create_manager("docker")

        assert not isinstance(manager, MultiContextManager)
        manager.docker_service_list.return_value = []
        assert manager.docker_service_list() == []

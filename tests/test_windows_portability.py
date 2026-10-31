"""Windows-portability: the Windows branches, exercised via monkeypatching."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from container_manager_mcp import platform_compat as pc
from container_manager_mcp.container_manager import PodmanManager
from container_manager_mcp.k8s.cluster_nodes import ClusterNodesMixin
from container_manager_mcp.k8s.kubeconfig_import import (
    _write_kubeconfig,
    default_kubeconfig_path,
)


def _windows(monkeypatch):
    monkeypatch.setattr(pc, "is_windows", lambda: True)


def test_docker_default_endpoint_is_named_pipe_on_windows(monkeypatch):
    _windows(monkeypatch)
    assert pc.default_docker_endpoint() == "npipe:////./pipe/docker_engine"
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    assert pc.docker_environment()["DOCKER_HOST"] == "npipe:////./pipe/docker_engine"


def test_docker_default_endpoint_is_unix_socket_on_posix(monkeypatch):
    monkeypatch.setattr(pc, "is_windows", lambda: False)
    assert pc.default_docker_endpoint() == "unix:///var/run/docker.sock"


def test_docker_host_env_is_never_overridden(monkeypatch):
    _windows(monkeypatch)
    env = pc.docker_environment({"DOCKER_HOST": "tcp://h:2375"})
    assert env["DOCKER_HOST"] == "tcp://h:2375"


def test_current_uid_is_none_without_getuid(monkeypatch):
    monkeypatch.delattr(os, "getuid", raising=False)
    assert pc.current_uid() is None


def _podman_stub():
    mgr = object.__new__(PodmanManager)
    mgr._get_podman_cli_sockets = lambda: []
    return mgr


def test_podman_candidates_on_windows_use_pipes_and_never_getuid(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.delattr(os, "getuid", raising=False)
    mgr = _podman_stub()
    mgr._is_wsl = lambda: False
    urls = mgr._podman_socket_candidates()
    assert "npipe:////./pipe/podman-machine-default" in urls
    assert not any("/run/user/" in u for u in urls)


def test_podman_candidates_on_linux_without_uid_skip_user_socket(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.delattr(os, "getuid", raising=False)
    mgr = _podman_stub()
    mgr._is_wsl = lambda: False
    urls = mgr._podman_socket_candidates()
    assert "unix:///run/podman/podman.sock" in urls
    assert not any("/run/user/" in u for u in urls)


def test_default_kubeconfig_follows_userprofile(monkeypatch, tmp_path):
    monkeypatch.delenv("KUBECONFIG", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert Path(default_kubeconfig_path()) == tmp_path / ".kube" / "config"


def test_kubeconfig_first_entry_uses_platform_pathsep(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "pathsep", ";")
    monkeypatch.setenv("KUBECONFIG", f"{tmp_path / 'a'};{tmp_path / 'b'}")
    assert Path(default_kubeconfig_path()) == tmp_path / "a"


def test_restrict_to_owner_skips_chmod_on_windows(monkeypatch, tmp_path):
    _windows(monkeypatch)
    calls = []
    monkeypatch.setattr(os, "chmod", lambda *a: calls.append(a))
    pc.restrict_to_owner(tmp_path, 0o700)
    assert calls == []


def test_write_private_text_without_fchmod(monkeypatch, tmp_path):
    monkeypatch.delattr(os, "fchmod", raising=False)
    target = tmp_path / "f.json"
    pc.write_private_text(target, "{}")
    assert target.read_text(encoding="utf-8") == "{}"


def test_write_kubeconfig_on_windows_branch(monkeypatch, tmp_path):
    _windows(monkeypatch)
    monkeypatch.delattr(os, "fchmod", raising=False)
    target = tmp_path / "kube" / "config"
    _write_kubeconfig(str(target), {"apiVersion": "v1"})
    assert "apiVersion" in target.read_text(encoding="utf-8")


def test_cluster_dump_on_windows_branch(monkeypatch):
    _windows(monkeypatch)
    monkeypatch.delattr(os, "fchmod", raising=False)
    mgr = object.__new__(ClusterNodesMixin)
    empty = lambda: SimpleNamespace(items=[])  # noqa: E731
    mgr.core = SimpleNamespace(
        list_node=empty, list_namespace=empty, list_service_for_all_namespaces=empty
    )
    mgr.log_action = lambda *a, **k: None
    result = mgr.cluster_info_dump()
    assert json.loads(Path(result["file"]).read_text(encoding="utf-8"))["nodes"] == []
    import shutil

    shutil.rmtree(result["output_dir"])

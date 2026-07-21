from __future__ import annotations

import os
import shutil
import stat
from types import SimpleNamespace

import pytest

from container_manager_mcp.k8s.cluster_nodes import ClusterNodesMixin


def _manager() -> ClusterNodesMixin:
    manager = object.__new__(ClusterNodesMixin)
    manager.core = SimpleNamespace(
        list_node=lambda: SimpleNamespace(items=[]),
        list_namespace=lambda: SimpleNamespace(items=[]),
        list_service_for_all_namespaces=lambda: SimpleNamespace(items=[]),
    )
    manager.log_action = lambda *_args, **_kwargs: None
    return manager


def test_cluster_dump_defaults_to_private_unique_storage() -> None:
    result = _manager().cluster_info_dump()
    try:
        directory_mode = stat.S_IMODE(os.stat(result["output_dir"]).st_mode)
        file_mode = stat.S_IMODE(os.stat(result["file"]).st_mode)
        assert directory_mode == 0o700
        assert file_mode == 0o600
    finally:
        shutil.rmtree(result["output_dir"])


def test_cluster_dump_rejects_symlink_output_directory(tmp_path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Failed to dump cluster info"):
        _manager().cluster_info_dump(str(alias))

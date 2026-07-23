"""Identity-scoped context auto-load (CONCEPT:AU-OS.identity.identity-scoped-resource-autoload).

The caller's entitled contexts auto-load; a non-entitled named context is denied;
the default falls to the first entitled when the configured default is off-limits.
Tests the resolution/enforcement logic with the entitlement source mocked (the
resolver itself is tested in agent-utilities).
"""

import pytest

from container_manager_mcp.multi_context_manager import MultiContextManager


def _mgr(entitled):
    m = object.__new__(MultiContextManager)  # bypass env-reading __init__
    m._entitled = lambda namespace, names: [n for n in names if n in entitled]
    return m


def test_auto_selects_entitled_default():
    m = _mgr({"prod"})
    pool = {"prod": 1, "dev": 2}
    assert m._resolve_context("k8s", pool, "prod", None) == "prod"


def test_default_not_entitled_falls_to_first_entitled():
    m = _mgr({"dev"})
    pool = {"prod": 1, "dev": 2}
    assert m._resolve_context("k8s", pool, "prod", None) == "dev"


def test_named_context_not_entitled_is_denied():
    m = _mgr({"prod"})
    pool = {"prod": 1, "dev": 2}
    with pytest.raises(PermissionError):
        m._resolve_context("k8s", pool, "prod", "dev")


def test_named_entitled_context_allowed():
    m = _mgr({"prod", "dev"})
    pool = {"prod": 1, "dev": 2}
    assert m._resolve_context("k8s", pool, "prod", "dev") == "dev"


def test_no_entitled_contexts_raises():
    m = _mgr(set())
    with pytest.raises(ValueError, match="available to your identity"):
        m._resolve_context("k8s", {"prod": 1}, "prod", None)


def test_unknown_context_name_raises():
    m = _mgr({"prod"})
    with pytest.raises(ValueError, match="not found"):
        m._resolve_context("k8s", {"prod": 1}, "prod", "ghost")

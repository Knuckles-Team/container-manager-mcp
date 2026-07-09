#!/usr/bin/env python
"""Kubernetes backend compatibility module.

Historically this file held the whole ``KubernetesManager``; it has been split
into the :mod:`container_manager_mcp.k8s` subpackage. This module remains the
canonical import path AND the single patch point for the guarded ``kubernetes``
client symbols: tests and mixins read ``k8s_client`` / ``k8s_config`` /
``ApiException`` through this module object, so patching
``container_manager_mcp.k8s_manager.k8s_client`` governs the whole backend.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Bound only for type-checkers / ``__all__`` resolution; the runtime import
    # happens lazily in ``__getattr__`` below to avoid an import cycle.
    from container_manager_mcp.k8s.manager import KubernetesManager

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
    from kubernetes.client.rest import ApiException
except ImportError:
    k8s_client = None  # type: ignore
    k8s_config = None  # type: ignore
    ApiException = Exception  # type: ignore

__all__ = ["KubernetesManager", "k8s_client", "k8s_config", "ApiException"]


def __getattr__(name):
    # Lazy import to avoid an import cycle (the subpackage's mixins import this
    # module for the guarded symbols above).
    if name == "KubernetesManager":
        from container_manager_mcp.k8s.manager import KubernetesManager

        return KubernetesManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""``container_manager_mcp.k8s`` — the Kubernetes backend package.

The guarded ``kubernetes`` client symbols (``k8s_client``/``k8s_config``/
``ApiException``) live in :mod:`container_manager_mcp.k8s_manager`, which is the
single patch point every mixin reads through.
"""

from container_manager_mcp.k8s.manager import KubernetesManager

__all__ = ["KubernetesManager"]

"""MCP tool registration modules for container-manager-mcp.

The base container/image/volume/network/swarm/system/compose/info/misc tools are
defined inline in ``container_manager_mcp.mcp_server`` and auto-discovered by
``register_tool_surface``. This subpackage holds the themed Kubernetes dispatchers
(8 ``cm_k8s_*`` tools) plus the advanced Docker/Podman and multi-context tools; the
thin wrappers in ``mcp_server`` lazy-import each ``register_*_tools`` below.
"""

from container_manager_mcp.mcp.mcp_docker_advanced import register_docker_advanced_tools
from container_manager_mcp.mcp.mcp_k8s_cluster import register_k8s_cluster_tools
from container_manager_mcp.mcp.mcp_k8s_config import register_k8s_config_tools
from container_manager_mcp.mcp.mcp_k8s_governance import register_k8s_governance_tools
from container_manager_mcp.mcp.mcp_k8s_networking import register_k8s_networking_tools
from container_manager_mcp.mcp.mcp_k8s_observability import (
    register_k8s_observability_tools,
)
from container_manager_mcp.mcp.mcp_k8s_rbac import register_k8s_rbac_tools
from container_manager_mcp.mcp.mcp_k8s_storage import register_k8s_storage_tools
from container_manager_mcp.mcp.mcp_k8s_workloads import register_k8s_workloads_tools
from container_manager_mcp.mcp.mcp_multi_context import register_multi_context_tools
from container_manager_mcp.mcp.mcp_podman_advanced import register_podman_advanced_tools

__all__ = [
    "register_k8s_workloads_tools",
    "register_k8s_config_tools",
    "register_k8s_networking_tools",
    "register_k8s_storage_tools",
    "register_k8s_rbac_tools",
    "register_k8s_cluster_tools",
    "register_k8s_governance_tools",
    "register_k8s_observability_tools",
    "register_docker_advanced_tools",
    "register_podman_advanced_tools",
    "register_multi_context_tools",
]

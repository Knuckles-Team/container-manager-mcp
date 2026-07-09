"""MCP tool registration modules for container-manager-mcp.

The base container/image/volume/network/swarm/system/compose/info/misc tools are
defined inline in ``container_manager_mcp.mcp_server`` and auto-discovered by
``register_tool_surface``. This subpackage holds the themed Kubernetes dispatchers
(8 ``cm_k8s_*`` tools) plus the advanced Docker/Podman and multi-context tools; the
thin wrappers in ``mcp_server`` lazy-import each ``register_*_tools`` below.
"""

from container_manager_mcp.mcp.mcp_docker_advanced import register_dockeradvanced_tools
from container_manager_mcp.mcp.mcp_k8s_cluster import register_k8scluster_tools
from container_manager_mcp.mcp.mcp_k8s_config import register_k8sconfig_tools
from container_manager_mcp.mcp.mcp_k8s_governance import register_k8sgovernance_tools
from container_manager_mcp.mcp.mcp_k8s_networking import register_k8snetworking_tools
from container_manager_mcp.mcp.mcp_k8s_observability import (
    register_k8sobservability_tools,
)
from container_manager_mcp.mcp.mcp_k8s_rbac import register_k8srbac_tools
from container_manager_mcp.mcp.mcp_k8s_storage import register_k8sstorage_tools
from container_manager_mcp.mcp.mcp_k8s_workloads import register_k8sworkloads_tools
from container_manager_mcp.mcp.mcp_multi_context import register_multicontext_tools
from container_manager_mcp.mcp.mcp_podman_advanced import register_podmanadvanced_tools

__all__ = [
    "register_k8sworkloads_tools",
    "register_k8sconfig_tools",
    "register_k8snetworking_tools",
    "register_k8sstorage_tools",
    "register_k8srbac_tools",
    "register_k8scluster_tools",
    "register_k8sgovernance_tools",
    "register_k8sobservability_tools",
    "register_dockeradvanced_tools",
    "register_podmanadvanced_tools",
    "register_multicontext_tools",
]

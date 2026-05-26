"""MCP tool registration modules for container-manager-mcp.

Auto-generated during ecosystem standardization.
Each domain has its own module with a register_*_tools function.
"""

from container_manager_mcp.mcp.mcp_compose import register_compose_tools
from container_manager_mcp.mcp.mcp_container import register_container_tools
from container_manager_mcp.mcp.mcp_image import register_image_tools
from container_manager_mcp.mcp.mcp_info import register_info_tools
from container_manager_mcp.mcp.mcp_misc import register_misc_tools
from container_manager_mcp.mcp.mcp_network import register_network_tools
from container_manager_mcp.mcp.mcp_swarm import register_swarm_tools
from container_manager_mcp.mcp.mcp_system import register_system_tools
from container_manager_mcp.mcp.mcp_volume import register_volume_tools

__all__ = [
    "register_compose_tools",
    "register_container_tools",
    "register_image_tools",
    "register_info_tools",
    "register_misc_tools",
    "register_network_tools",
    "register_swarm_tools",
    "register_system_tools",
    "register_volume_tools",
]

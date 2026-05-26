"""MCP tools for swarm operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

import logging
import os
from typing import Literal

from agent_utilities.mcp_utilities import (
    ctx_confirm_destructive,
    ctx_log,
)
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_swarm_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Swarm Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"swarm"},
    )
    async def cm_swarm_operations(
        action: Literal[
            "init_swarm",
            "leave_swarm",
            "list_nodes",
            "list_services",
            "create_service",
            "remove_service",
        ] = Field(
            description="Action to perform. Must be one of: 'init_swarm', 'leave_swarm', 'list_nodes', 'list_services', 'create_service', 'remove_service'"
        ),
        service_id: str | None = Field(default=None, description="Service ID or name"),
        advertise_addr: str | None = Field(
            default=None, description="Advertise address for init_swarm"
        ),
        image: str | None = Field(default=None, description="Image to use for service"),
        name: str | None = Field(default=None, description="Name for the service"),
        ports: str | None = Field(
            default=None, description="Port mappings as JSON string"
        ),
        mounts: str | None = Field(
            default=None, description="Mounts mappings as JSON string list"
        ),
        replicas: int = Field(default=1, description="Number of replicas"),
        force: bool = Field(default=False, description="Force operation"),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str | list:
        """
        Manage swarm operations.
        """
        import json

        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_swarm_operations: {action}")

        try:
            if action == "init_swarm":
                if ctx and not await ctx_confirm_destructive(ctx, "init swarm"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                return manager.init_swarm(advertise_addr)
            elif action == "leave_swarm":
                if ctx and not await ctx_confirm_destructive(ctx, "leave swarm"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                return manager.leave_swarm(force=force)
            elif action == "list_nodes":
                return manager.list_nodes()
            elif action == "list_services":
                return manager.list_services()
            elif action == "create_service":
                if not name or not image:
                    return "Error: 'name' and 'image' are required for create_service"
                p_ports = json.loads(ports) if ports else None
                p_mounts = json.loads(mounts) if mounts else None
                return manager.create_service(
                    name=name,
                    image=image,
                    ports=p_ports,
                    mounts=p_mounts,
                    replicas=replicas,
                )
            elif action == "remove_service":
                if not service_id:
                    return "Error: 'service_id' is required"
                if ctx and not await ctx_confirm_destructive(ctx, "remove service"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                return manager.remove_service(service_id)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

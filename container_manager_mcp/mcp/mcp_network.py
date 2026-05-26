"""MCP tools for network operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

import logging
import os
from typing import Any, Literal

from agent_utilities.mcp_utilities import (
    ctx_confirm_destructive,
    ctx_log,
    ctx_progress,
)
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_network_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Network Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"network"},
    )
    async def cm_network_operations(
        action: Literal[
            "list_networks", "create_network", "remove_network", "prune_networks"
        ] = Field(
            description="Action to perform. Must be one of: 'list_networks', 'create_network', 'remove_network', 'prune_networks'"
        ),
        network_id: str | None = Field(default=None, description="Network ID or name"),
        driver: str = Field(default="bridge", description="Network driver"),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager",
        ),
        ctx: Context | None = None,
    ) -> Any:
        """
        Manage network operations.
        """
        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_network_operations: {action}")

        try:
            if action == "list_networks":
                return manager.list_networks()
            elif action == "create_network":
                if not network_id:
                    return "Error: 'network_id' is required for create_network"
                return manager.create_network(network_id, driver=driver)
            elif action == "remove_network":
                if not network_id:
                    return "Error: 'network_id' is required"
                if ctx and not await ctx_confirm_destructive(ctx, "remove network"):
                    return {"status": "cancelled"}
                return manager.remove_network(network_id)
            elif action == "prune_networks":
                if ctx and not await ctx_confirm_destructive(ctx, "prune networks"):
                    return {"status": "cancelled"}
                if ctx:
                    await ctx_progress(ctx, 0, 100)
                return manager.prune_networks()
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

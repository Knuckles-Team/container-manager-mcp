"""MCP tools for volume operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

import logging
import os
from typing import Any, Literal

from agent_utilities.mcp_utilities import (
    ctx_confirm_destructive,
    ctx_log,
    ctx_progress,
    run_blocking,
)
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_volume_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Volume Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"volume"},
    )
    async def cm_volume_operations(
        action: Literal[
            "list_volumes", "create_volume", "remove_volume", "prune_volumes"
        ] = Field(
            description="Action to perform. Must be one of: 'list_volumes', 'create_volume', 'remove_volume', 'prune_volumes'"
        ),
        name: str | None = Field(default=None, description="Volume name"),
        force: bool = Field(default=False, description="Force operation"),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager",
        ),
        ctx: Context | None = None,
    ) -> Any:
        """
        Manage volume operations.
        """
        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_volume_operations: {action}")

        try:
            if action == "list_volumes":
                return await run_blocking(manager.list_volumes)
            elif action == "create_volume":
                if not name:
                    return "Error: 'name' is required for create_volume"
                return await run_blocking(manager.create_volume, name)
            elif action == "remove_volume":
                if not name:
                    return "Error: 'name' is required"
                if ctx and not await ctx_confirm_destructive(ctx, "remove volume"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                return await run_blocking(manager.remove_volume, name, force=force)
            elif action == "prune_volumes":
                if ctx and not await ctx_confirm_destructive(ctx, "prune volumes"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                if ctx:
                    await ctx_progress(ctx, 0, 100)
                return await run_blocking(manager.prune_volumes)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

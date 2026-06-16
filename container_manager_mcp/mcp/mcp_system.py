"""MCP tools for system operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

import logging
import os
from typing import Literal

from agent_utilities.mcp_utilities import (
    ctx_confirm_destructive,
    ctx_log,
    ctx_progress,
    run_blocking,
)
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_system_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager System Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"system"},
    )
    async def cm_system_operations(
        action: Literal["prune_system", "get_info", "get_version"] = Field(
            description="Action to perform. Must be one of: 'prune_system', 'get_info', 'get_version'"
        ),
        force: bool = Field(default=False, description="Force prune system"),
        all: bool = Field(default=False, description="Prune all resources"),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str:
        """
        Manage container manager system operations.
        """
        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_system_operations: {action}")

        try:
            if action == "prune_system":
                if ctx:
                    if not await ctx_confirm_destructive(ctx, "prune system"):
                        return {
                            "status": "cancelled",
                            "message": "Operation cancelled by user",
                        }
                    await ctx_progress(ctx, 0, 100)
                return await run_blocking(manager.prune_system, force=force, all=all)
            elif action == "get_info":
                return await run_blocking(manager.get_info)
            elif action == "get_version":
                return await run_blocking(manager.get_version)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

"""MCP tools for info operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

import logging
import os

from agent_utilities.mcp_utilities import (
    ctx_log,
)
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_info_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Info Operations",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"info"},
    )
    async def cm_info_operations(
        action: str = Field(
            description="Action to perform. Must be one of: 'get_version', 'get_info'"
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str:
        """
        Manage container manager info operations.
        """
        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_info_operations: {action}")

        try:
            if action == "get_version":
                return manager.get_version()
            elif action == "get_info":
                return manager.get_info()
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            return f"Error executing {action}: {e}"

"""MCP tools for compose operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

import logging
import os

from agent_utilities.mcp_utilities import (
    ctx_log,
    run_blocking,
)
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_compose_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Compose Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"compose"},
    )
    async def cm_compose_operations(
        action: str = Field(
            description="Action to perform. Must be one of: 'up', 'down', 'ps', 'logs'"
        ),
        compose_file: str = Field(description="Path to compose file"),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str | list:
        """
        Manage docker-compose or podman-compose operations.
        """
        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_compose_operations: {action}")

        try:
            if action == "up":
                return await run_blocking(manager.compose_up, compose_file)
            elif action == "down":
                return await run_blocking(manager.compose_down, compose_file)
            elif action == "ps":
                return await run_blocking(manager.compose_ps, compose_file)
            elif action == "logs":
                return await run_blocking(manager.compose_logs, compose_file)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            return f"Error executing {action}: {e}"

"""MCP tools for container operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

import logging
import os
from typing import Literal

from agent_utilities.mcp_utilities import (
    ctx_confirm_destructive,
    ctx_log,
    ctx_progress,
)
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_container_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Container Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"container"},
    )
    async def cm_container_operations(
        action: Literal[
            "list_containers",
            "get_container_logs",
            "stop_container",
            "remove_container",
            "prune_containers",
            "exec_in_container",
        ] = Field(
            description="Action to perform. Must be one of: 'list_containers', 'get_container_logs', 'stop_container', 'remove_container', 'prune_containers', 'exec_in_container'"
        ),
        container_id: str | None = Field(
            default=None, description="Container ID or name"
        ),
        command: str | None = Field(default=None, description="Command to execute"),
        all_containers: bool = Field(default=False, description="Show all containers"),
        force: bool = Field(default=False, description="Force operation"),
        tail: str = Field(default="50", description="Number of log lines to tail"),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str | list:
        """
        Manage container operations.
        """
        import shlex

        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_container_operations: {action}")

        try:
            if action == "list_containers":
                return manager.list_containers(all=all_containers)
            elif action == "get_container_logs":
                if not container_id:
                    return "Error: 'container_id' is required"
                return manager.get_container_logs(container_id, tail=tail)
            elif action == "stop_container":
                if not container_id:
                    return "Error: 'container_id' is required"
                return manager.stop_container(container_id)
            elif action == "remove_container":
                if not container_id:
                    return "Error: 'container_id' is required"
                if ctx and not await ctx_confirm_destructive(ctx, "remove container"):
                    return {"status": "cancelled"}
                return manager.remove_container(container_id, force=force)
            elif action == "prune_containers":
                if ctx and not await ctx_confirm_destructive(ctx, "prune containers"):
                    return {"status": "cancelled"}
                if ctx:
                    await ctx_progress(ctx, 0, 100)
                return manager.prune_containers()
            elif action == "exec_in_container":
                if not container_id or not command:
                    return "Error: 'container_id' and 'command' required"
                return manager.exec_in_container(container_id, shlex.split(command))
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

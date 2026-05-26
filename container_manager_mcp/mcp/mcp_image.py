"""MCP tools for image operations.

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


def register_image_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Image Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"image"},
    )
    async def cm_image_operations(
        action: Literal[
            "list_images", "pull_image", "remove_image", "prune_images"
        ] = Field(
            description="Action to perform. Must be one of: 'list_images', 'pull_image', 'remove_image', 'prune_images'"
        ),
        image: str | None = Field(default=None, description="Image name"),
        tag: str = Field(default="latest", description="Image tag"),
        platform: str | None = Field(
            default=None, description="Platform (e.g., linux/amd64)"
        ),
        force: bool = Field(default=False, description="Force operation"),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """
        Manage container images.
        """
        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_image_operations: {action}")

        try:
            if action == "list_images":
                return manager.list_images()
            elif action == "pull_image":
                if not image:
                    return "Error: 'image' is required for pull_image"
                if ctx:
                    await ctx_progress(ctx, 0, 100)
                return manager.pull_image(image, tag=tag, platform=platform)
            elif action == "remove_image":
                if not image:
                    return "Error: 'image' is required for remove_image"
                if ctx and not await ctx_confirm_destructive(ctx, "remove image"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                return manager.remove_image(image, force=force)
            elif action == "prune_images":
                if ctx and not await ctx_confirm_destructive(ctx, "prune images"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                if ctx:
                    await ctx_progress(ctx, 0, 100)
                return manager.prune_images()
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

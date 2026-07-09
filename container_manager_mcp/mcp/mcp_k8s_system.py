"""MCP tools for Kubernetes system and context management.

This module provides system operations including namespace management and kubeconfig context management.
"""

import logging
import os
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_system_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes System and Context Management",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "system"},
    )
    async def cm_k8s_system(
        action: Literal[
            "create_namespace",
            "delete_namespace",
            "list_contexts",
            "use_context",
            "get_config",
            "rename_context",
        ] = Field(
            description="Action to perform. Must be one of: 'create_namespace', 'delete_namespace', 'list_contexts', 'use_context', 'get_config', 'rename_context'"
        ),
        namespace_name: str | None = Field(default=None, description="Namespace name"),
        context_name: str | None = Field(default=None, description="Context name"),
        new_context_name: str | None = Field(default=None, description="New context name for rename"),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list:
        """
        Manage Kubernetes system resources and kubeconfig contexts.
        
        This tool provides namespace management and kubeconfig context operations
        for system-level Kubernetes management.
        """
        if manager_type is None:
            manager_type = "kubernetes"

        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_system: {action}")

        try:
            if action == "create_namespace":
                if not namespace_name:
                    return "Error: 'namespace_name' is required for create_namespace"
                return await run_blocking(manager.create_namespace, name=namespace_name)
            elif action == "delete_namespace":
                if not namespace_name:
                    return "Error: 'namespace_name' is required for delete_namespace"
                return await run_blocking(manager.delete_namespace, name=namespace_name)
            elif action == "list_contexts":
                return await run_blocking(manager.list_contexts)
            elif action == "use_context":
                if not context_name:
                    return "Error: 'context_name' is required for use_context"
                return await run_blocking(manager.use_context, context_name=context_name)
            elif action == "get_config":
                return await run_blocking(manager.get_config)
            elif action == "rename_context":
                if not context_name or not new_context_name:
                    return "Error: 'context_name' and 'new_context_name' are required for rename_context"
                return await run_blocking(
                    manager.rename_context, current_name=context_name, new_name=new_context_name
                )
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

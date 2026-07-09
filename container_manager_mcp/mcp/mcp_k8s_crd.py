"""MCP tools for Kubernetes CRD and custom resource operations.

This module provides comprehensive CRD management and custom resource operations.
"""

import json
import logging
import os
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_crd_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes CRD and Custom Resource Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "crd", "custom-resources"},
    )
    async def cm_k8s_crd(
        action: Literal[
            "list_crds",
            "describe_crd",
            "list_custom_resources",
        ] = Field(
            description="Action to perform. Must be one of: 'list_crds', 'describe_crd', 'list_custom_resources'"
        ),
        crd_name: str | None = Field(default=None, description="CRD name for describe_crd"),
        crd_group: str | None = Field(default=None, description="CRD group for custom resources"),
        crd_version: str | None = Field(default=None, description="CRD version for custom resources"),
        crd_plural: str | None = Field(default=None, description="CRD plural name for custom resources"),
        namespace: str | None = Field(default=None, description="Target namespace for namespaced custom resources"),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list:
        """
        Manage Kubernetes CRDs and custom resources.
        
        This tool provides comprehensive CRD management and custom resource operations
        for extending Kubernetes with custom resources.
        """
        if manager_type is None:
            manager_type = "kubernetes"

        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_crd: {action}")

        try:
            if action == "list_crds":
                return await run_blocking(manager.list_crds)
            elif action == "describe_crd":
                if not crd_name:
                    return "Error: 'crd_name' is required for describe_crd"
                return await run_blocking(manager.describe_crd, crd_name=crd_name)
            elif action == "list_custom_resources":
                if not crd_group or not crd_version or not crd_plural:
                    return "Error: 'crd_group', 'crd_version', and 'crd_plural' are required for list_custom_resources"
                return await run_blocking(
                    manager.list_custom_resources,
                    group=crd_group,
                    version=crd_version,
                    plural=crd_plural,
                    namespace=namespace,
                )
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

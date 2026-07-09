"""MCP tools for Kubernetes monitoring and metrics operations.

This module provides monitoring operations including resource usage metrics and cluster health.
"""

import logging
import os
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_monitoring_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Monitoring and Metrics",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        tags={"kubernetes", "monitoring", "metrics"},
    )
    async def cm_k8s_monitoring(
        action: Literal[
            "top_pods",
            "top_nodes",
        ] = Field(
            description="Action to perform. Must be one of: 'top_pods', 'top_nodes'"
        ),
        namespace: str | None = Field(default=None, description="Target namespace for pod metrics"),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list:
        """
        Monitor Kubernetes resources (pod metrics, node metrics).
        
        This tool provides monitoring and metrics operations including resource usage
        for pods and nodes. Note: actual metrics require the Kubernetes metrics server
        to be installed in the cluster.
        """
        if manager_type is None:
            manager_type = "kubernetes"

        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_monitoring: {action}")

        try:
            if action == "top_pods":
                return await run_blocking(manager.top_pods, namespace=namespace)
            elif action == "top_nodes":
                return await run_blocking(manager.top_nodes)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

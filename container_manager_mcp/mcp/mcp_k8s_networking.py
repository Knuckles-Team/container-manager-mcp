"""MCP tools for Kubernetes networking operations.

This module provides comprehensive networking management including ingress, network policies, and endpoints.
"""

import json
import logging
import os
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_networking_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Networking Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "networking"},
    )
    async def cm_k8s_networking(
        action: Literal[
            "list_ingress",
            "create_ingress",
            "delete_ingress",
            "list_networkpolicies",
            "create_networkpolicy",
            "delete_networkpolicy",
            "list_endpoints",
            "list_endpointslices",
        ] = Field(
            description="Action to perform. Must be one of: 'list_ingress', 'create_ingress', 'delete_ingress', 'list_networkpolicies', 'create_networkpolicy', 'delete_networkpolicy', 'list_endpoints', 'list_endpointslices'"
        ),
        ingress_name: str | None = Field(default=None, description="Ingress name"),
        namespace: str | None = Field(default=None, description="Target namespace"),
        ingress_spec: str | None = Field(default=None, description="Ingress spec as JSON string"),
        netpol_name: str | None = Field(default=None, description="NetworkPolicy name"),
        netpol_spec: str | None = Field(default=None, description="NetworkPolicy spec as JSON string"),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list:
        """
        Manage Kubernetes networking resources (ingress, network policies, endpoints).
        
        This tool provides comprehensive networking management including ingress controllers,
        network policies for traffic control, and endpoint management.
        """
        if manager_type is None:
            manager_type = "kubernetes"

        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_networking: {action}")

        try:
            if action == "list_ingress":
                return await run_blocking(manager.list_ingress, namespace=namespace)
            elif action == "create_ingress":
                if not ingress_name:
                    return "Error: 'ingress_name' is required for create_ingress"
                spec = json.loads(ingress_spec) if ingress_spec else None
                return await run_blocking(
                    manager.create_ingress, name=ingress_name, namespace=namespace, spec=spec
                )
            elif action == "delete_ingress":
                if not ingress_name:
                    return "Error: 'ingress_name' is required for delete_ingress"
                return await run_blocking(
                    manager.delete_ingress, name=ingress_name, namespace=namespace
                )
            elif action == "list_networkpolicies":
                return await run_blocking(manager.list_networkpolicies, namespace=namespace)
            elif action == "create_networkpolicy":
                if not netpol_name:
                    return "Error: 'netpol_name' is required for create_networkpolicy"
                spec = json.loads(netpol_spec) if netpol_spec else None
                return await run_blocking(
                    manager.create_networkpolicy, name=netpol_name, namespace=namespace, spec=spec
                )
            elif action == "delete_networkpolicy":
                if not netpol_name:
                    return "Error: 'netpol_name' is required for delete_networkpolicy"
                return await run_blocking(
                    manager.delete_networkpolicy, name=netpol_name, namespace=namespace
                )
            elif action == "list_endpoints":
                return await run_blocking(manager.list_endpoints, namespace=namespace)
            elif action == "list_endpointslices":
                return await run_blocking(manager.list_endpointslices, namespace=namespace)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

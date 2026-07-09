"""MCP tools for Kubernetes storage operations.

This module provides comprehensive storage management including PV, PVC, StatefulSets, DaemonSets, and StorageClasses.
"""

import json
import logging
import os
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_storage_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Storage Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "storage"},
    )
    async def cm_k8s_storage(
        action: Literal[
            "list_persistent_volumes",
            "list_persistent_volume_claims",
            "create_persistent_volume_claim",
            "delete_persistent_volume_claim",
            "list_storage_classes",
            "list_statefulsets",
            "scale_statefulset",
            "list_daemonsets",
            "list_volume_snapshots",
            "expand_pvc",
        ] = Field(
            description="Action to perform. Must be one of: 'list_persistent_volumes', 'list_persistent_volume_claims', 'create_persistent_volume_claim', 'delete_persistent_volume_claim', 'list_storage_classes', 'list_statefulsets', 'scale_statefulset', 'list_daemonsets', 'list_volume_snapshots', 'expand_pvc'"
        ),
        pvc_name: str | None = Field(default=None, description="PVC name"),
        namespace: str | None = Field(default=None, description="Target namespace"),
        pvc_spec: str | None = Field(default=None, description="PVC spec as JSON string"),
        pvc_size: str | None = Field(default=None, description="New PVC size for expansion"),
        statefulset_name: str | None = Field(default=None, description="StatefulSet name"),
        replicas: int = Field(default=1, description="Replica count for scaling"),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list:
        """
        Manage Kubernetes storage resources (PV, PVC, StatefulSets, DaemonSets, StorageClasses).
        
        This tool provides comprehensive storage management including persistent volumes,
        persistent volume claims, StatefulSets, DaemonSets, and storage classes.
        """
        if manager_type is None:
            manager_type = "kubernetes"

        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_storage: {action}")

        try:
            if action == "list_persistent_volumes":
                return await run_blocking(manager.list_persistent_volumes)
            elif action == "list_persistent_volume_claims":
                return await run_blocking(manager.list_persistent_volume_claims, namespace=namespace)
            elif action == "create_persistent_volume_claim":
                if not pvc_name:
                    return "Error: 'pvc_name' is required for create_persistent_volume_claim"
                spec = json.loads(pvc_spec) if pvc_spec else None
                return await run_blocking(
                    manager.create_persistent_volume_claim, name=pvc_name, namespace=namespace, spec=spec
                )
            elif action == "delete_persistent_volume_claim":
                if not pvc_name:
                    return "Error: 'pvc_name' is required for delete_persistent_volume_claim"
                return await run_blocking(
                    manager.delete_persistent_volume_claim, name=pvc_name, namespace=namespace
                )
            elif action == "list_storage_classes":
                return await run_blocking(manager.list_storage_classes)
            elif action == "list_statefulsets":
                return await run_blocking(manager.list_statefulsets, namespace=namespace)
            elif action == "scale_statefulset":
                if not statefulset_name:
                    return "Error: 'statefulset_name' is required for scale_statefulset"
                return await run_blocking(
                    manager.scale_statefulset, name=statefulset_name, namespace=namespace, replicas=replicas
                )
            elif action == "list_daemonsets":
                return await run_blocking(manager.list_daemonsets, namespace=namespace)
            elif action == "list_volume_snapshots":
                return await run_blocking(manager.list_volume_snapshots)
            elif action == "expand_pvc":
                if not pvc_name or not pvc_size:
                    return "Error: 'pvc_name' and 'pvc_size' are required for expand_pvc"
                return await run_blocking(
                    manager.expand_pvc, name=pvc_name, namespace=namespace, size=pvc_size
                )
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

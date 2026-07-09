"""MCP tools for advanced Kubernetes storage operations.

This module provides advanced storage operations including CSI driver management,
storage class management, and volume snapshot operations.
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_advanced_storage_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Advanced Storage Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "advanced-storage"},
    )
    async def cm_k8s_advanced_storage(
        action: Literal[
            # CSI Driver Operations
            "list_csi_drivers",
            "describe_csi_driver",
            "get_csi_driver_capacity",
            # Storage Class Management
            "set_default_storage_class",
            "get_storage_class_provisioner",
            "expand_persistent_volume",
            # Volume Snapshot Operations
            "list_volume_snapshots",
            "create_volume_snapshot",
        ] = Field(
            description="Action to perform. Advanced storage operations."
        ),
        # Common parameters
        name: str | None = Field(default=None, description="Resource name for operations"),
        namespace: str | None = Field(default=None, description="Target namespace (default: from config)"),
        spec: dict | None = Field(default=None, description="Specification for operations"),
        size: str | None = Field(default=None, description="Size for volume expansion"),
        driver_name: str | None = Field(default=None, description="CSI driver name"),
    ) -> dict | list:
        """Manage advanced Kubernetes storage operations (CSI drivers, storage classes, volume snapshots)."""
        
        ctx_log("Advanced storage operations", action=action, name=name, namespace=namespace)
        
        @run_blocking
        def execute_operation():
            manager = create_manager("kubernetes")
            k8s_manager = getattr(manager, "k8s_manager", manager)
            
            # CSI Driver Operations
            if action == "list_csi_drivers":
                return k8s_manager.list_csi_drivers()
            elif action == "describe_csi_driver":
                if not name:
                    raise ValueError("name is required for describe_csi_driver")
                return k8s_manager.describe_csi_driver(name)
            elif action == "get_csi_driver_capacity":
                if not driver_name:
                    raise ValueError("driver_name is required for get_csi_driver_capacity")
                return k8s_manager.get_csi_driver_capacity(driver_name)
            
            # Storage Class Management
            elif action == "set_default_storage_class":
                if not name:
                    raise ValueError("name is required for set_default_storage_class")
                return k8s_manager.set_default_storage_class(name)
            elif action == "get_storage_class_provisioner":
                if not name:
                    raise ValueError("name is required for get_storage_class_provisioner")
                return k8s_manager.get_storage_class_provisioner(name)
            elif action == "expand_persistent_volume":
                if not name or not namespace or not size:
                    raise ValueError("name, namespace, and size are required for expand_persistent_volume")
                return k8s_manager.expand_persistent_volume(name, namespace, size)
            
            # Volume Snapshot Operations
            elif action == "list_volume_snapshots":
                return k8s_manager.list_volume_snapshots(namespace=namespace)
            elif action == "create_volume_snapshot":
                if not name or not namespace or not spec:
                    raise ValueError("name, namespace, and spec are required for create_volume_snapshot")
                return k8s_manager.create_volume_snapshot(name, namespace, spec)
            
            else:
                raise ValueError(f"Unknown action: {action}")
        
        return execute_operation()

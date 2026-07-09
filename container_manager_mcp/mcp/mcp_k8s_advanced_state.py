"""MCP tools for advanced Kubernetes state operations.

This module provides advanced state operations including ConfigMap/Secret state management
and resource version tracking.
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_advanced_state_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Advanced State Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "advanced-state"},
    )
    async def cm_k8s_advanced_state(
        action: Literal[
            # ConfigMap/Secret State Management
            "compare_configmap_state",
            "sync_configmap_from_file",
            "get_secret_state_hash",
            # Resource Version Tracking
            "track_resource_version",
            "wait_for_resource_version",
        ] = Field(
            description="Action to perform. Advanced state operations."
        ),
        # Common parameters
        name: str | None = Field(default=None, description="Resource name for operations"),
        namespace: str | None = Field(default=None, description="Target namespace (default: from config)"),
        expected_data: dict | None = Field(default=None, description="Expected data for comparison"),
        file_path: str | None = Field(default=None, description="File path for sync operations"),
        resource_type: str | None = Field(default=None, description="Resource type for version tracking"),
        target_version: str | None = Field(default=None, description="Target resource version"),
        timeout: int | None = Field(default=None, description="Timeout for wait operations"),
    ) -> dict:
        """Manage advanced Kubernetes state operations (ConfigMap/Secret state, resource version tracking)."""
        
        ctx_log("Advanced state operations", action=action, name=name, namespace=namespace)
        
        @run_blocking
        def execute_operation():
            manager = create_manager("kubernetes")
            k8s_manager = getattr(manager, "k8s_manager", manager)
            
            # ConfigMap/Secret State Management
            if action == "compare_configmap_state":
                if not name or not namespace or not expected_data:
                    raise ValueError("name, namespace, and expected_data are required for compare_configmap_state")
                return k8s_manager.compare_configmap_state(name, namespace, expected_data)
            elif action == "sync_configmap_from_file":
                if not name or not namespace or not file_path:
                    raise ValueError("name, namespace, and file_path are required for sync_configmap_from_file")
                return k8s_manager.sync_configmap_from_file(name, namespace, file_path)
            elif action == "get_secret_state_hash":
                if not name or not namespace:
                    raise ValueError("name and namespace are required for get_secret_state_hash")
                return k8s_manager.get_secret_state_hash(name, namespace)
            
            # Resource Version Tracking
            elif action == "track_resource_version":
                if not resource_type or not name:
                    raise ValueError("resource_type and name are required for track_resource_version")
                return k8s_manager.track_resource_version(resource_type, name, namespace)
            elif action == "wait_for_resource_version":
                if not resource_type or not name or not namespace or not target_version:
                    raise ValueError("resource_type, name, namespace, and target_version are required for wait_for_resource_version")
                return k8s_manager.wait_for_resource_version(resource_type, name, namespace, target_version, timeout or 60)
            
            else:
                raise ValueError(f"Unknown action: {action}")
        
        return execute_operation()

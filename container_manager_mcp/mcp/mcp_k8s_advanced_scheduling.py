"""MCP tools for advanced Kubernetes scheduling operations.

This module provides advanced scheduling operations including taints and tolerations,
node affinity, and pod anti-affinity.
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_advanced_scheduling_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Advanced Scheduling Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "advanced-scheduling"},
    )
    async def cm_k8s_advanced_scheduling(
        action: Literal[
            # Taints and Tolerations
            "taint_node",
            "untaint_node",
            "list_node_taints",
            # Node Affinity
            "set_node_affinity",
            "get_node_affinity",
            # Pod Anti-Affinity
            "set_pod_anti_affinity",
        ] = Field(
            description="Action to perform. Advanced scheduling operations."
        ),
        # Common parameters
        node_name: str | None = Field(default=None, description="Node name for operations"),
        pod_name: str | None = Field(default=None, description="Pod name for operations"),
        namespace: str | None = Field(default=None, description="Target namespace (default: from config)"),
        taints: list | None = Field(default=None, description="Taints for node operations"),
        taint_key: str | None = Field(default=None, description="Taint key for removal"),
        affinity: dict | None = Field(default=None, description="Affinity configuration"),
        anti_affinity: dict | None = Field(default=None, description="Anti-affinity configuration"),
    ) -> dict | list:
        """Manage advanced Kubernetes scheduling operations (taints/tolerations, node affinity, pod anti-affinity)."""
        
        ctx_log("Advanced scheduling operations", action=action, node_name=node_name, pod_name=pod_name, namespace=namespace)
        
        @run_blocking
        def execute_operation():
            manager = create_manager("kubernetes")
            k8s_manager = getattr(manager, "k8s_manager", manager)
            
            # Taints and Tolerations
            if action == "taint_node":
                if not node_name or not taints:
                    raise ValueError("node_name and taints are required for taint_node")
                return k8s_manager.taint_node(node_name, taints)
            elif action == "untaint_node":
                if not node_name or not taint_key:
                    raise ValueError("node_name and taint_key are required for untaint_node")
                return k8s_manager.untaint_node(node_name, taint_key)
            elif action == "list_node_taints":
                return k8s_manager.list_node_taints()
            
            # Node Affinity
            elif action == "set_node_affinity":
                if not pod_name or not namespace or not affinity:
                    raise ValueError("pod_name, namespace, and affinity are required for set_node_affinity")
                return k8s_manager.set_node_affinity(pod_name, namespace, affinity)
            elif action == "get_node_affinity":
                if not pod_name or not namespace:
                    raise ValueError("pod_name and namespace are required for get_node_affinity")
                return k8s_manager.get_node_affinity(pod_name, namespace)
            
            # Pod Anti-Affinity
            elif action == "set_pod_anti_affinity":
                if not pod_name or not namespace or not anti_affinity:
                    raise ValueError("pod_name, namespace, and anti_affinity are required for set_pod_anti_affinity")
                return k8s_manager.set_pod_anti_affinity(pod_name, namespace, anti_affinity)
            
            else:
                raise ValueError(f"Unknown action: {action}")
        
        return execute_operation()

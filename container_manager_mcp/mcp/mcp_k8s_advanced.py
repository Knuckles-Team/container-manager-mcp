"""MCP tools for Kubernetes advanced operations.

This module provides advanced operations including rollouts, tainting, labeling, and annotation.
"""

import json
import logging
import os
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_advanced_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Advanced Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "advanced"},
    )
    async def cm_k8s_advanced(
        action: Literal[
            "rollout_status",
            "rollout_history",
            "rollout_restart",
            "rollout_undo",
            "rollout_pause",
            "rollout_resume",
            "taint_node",
            "label_resource",
            "annotate_resource",
        ] = Field(
            description="Action to perform. Must be one of: 'rollout_status', 'rollout_history', 'rollout_restart', 'rollout_undo', 'rollout_pause', 'rollout_resume', 'taint_node', 'label_resource', 'annotate_resource'"
        ),
        resource_type: str | None = Field(default=None, description="Resource type for operations"),
        resource_name: str | None = Field(default=None, description="Resource name"),
        namespace: str | None = Field(default=None, description="Target namespace"),
        rollout_revision: int | None = Field(default=None, description="Revision number for rollout undo"),
        node_name: str | None = Field(default=None, description="Node name for tainting"),
        taints: str | None = Field(default=None, description="Taints as JSON string"),
        labels: str | None = Field(default=None, description="Labels as JSON string"),
        annotations: str | None = Field(default=None, description="Annotations as JSON string"),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list:
        """
        Manage Kubernetes advanced operations (rollouts, taints, labels, annotations).
        
        This tool provides advanced operations including rollout management, node tainting,
        resource labeling, and annotation management.
        """
        if manager_type is None:
            manager_type = "kubernetes"

        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_advanced: {action}")

        try:
            if action == "rollout_status":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_status"
                return await run_blocking(
                    manager.rollout_status, resource_type=resource_type, name=resource_name, namespace=namespace
                )
            elif action == "rollout_history":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_history"
                return await run_blocking(
                    manager.rollout_history, resource_type=resource_type, name=resource_name, namespace=namespace
                )
            elif action == "rollout_restart":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_restart"
                return await run_blocking(
                    manager.rollout_restart, resource_type=resource_type, name=resource_name, namespace=namespace
                )
            elif action == "rollout_undo":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_undo"
                return await run_blocking(
                    manager.rollout_undo, resource_type=resource_type, name=resource_name, namespace=namespace, revision=rollout_revision
                )
            elif action == "rollout_pause":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_pause"
                return await run_blocking(
                    manager.rollout_pause, resource_type=resource_type, name=resource_name, namespace=namespace
                )
            elif action == "rollout_resume":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_resume"
                return await run_blocking(
                    manager.rollout_resume, resource_type=resource_type, name=resource_name, namespace=namespace
                )
            elif action == "taint_node":
                if not node_name:
                    return "Error: 'node_name' is required for taint_node"
                taints_list = json.loads(taints) if taints else []
                return await run_blocking(manager.taint_node, node_name=node_name, taints=taints_list)
            elif action == "label_resource":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for label_resource"
                labels_dict = json.loads(labels) if labels else None
                return await run_blocking(
                    manager.label_resource,
                    resource_type=resource_type,
                    name=resource_name,
                    namespace=namespace,
                    labels=labels_dict,
                )
            elif action == "annotate_resource":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for annotate_resource"
                annotations_dict = json.loads(annotations) if annotations else None
                return await run_blocking(
                    manager.annotate_resource,
                    resource_type=resource_type,
                    name=resource_name,
                    namespace=namespace,
                    annotations=annotations_dict,
                )
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

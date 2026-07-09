"""MCP tools for enhanced Kubernetes resource operations.

This module provides comprehensive Kubernetes resource management beyond the basic
Swarm-mapped operations, including pod management, configuration, and system operations.
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_resources_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Resource Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "resources"},
    )
    async def cm_k8s_resources(
        action: Literal[
            "list_pods",
            "describe_pod",
            "exec_pod",
            "port_forward_pod",
            "attach_pod",
            "cp_pod",
            "list_configmaps",
            "create_configmap",
            "list_secrets",
            "create_secret",
            "list_namespaces",
            "list_events",
        ] = Field(
            description="Action to perform. Must be one of: 'list_pods', 'describe_pod', 'exec_pod', 'port_forward_pod', 'attach_pod', 'cp_pod', 'list_configmaps', 'create_configmap', 'list_secrets', 'create_secret', 'list_namespaces', 'list_events'"
        ),
        pod_name: str | None = Field(default=None, description="Pod name for describe_pod"),
        namespace: str | None = Field(default=None, description="Target namespace (default: from config)"),
        label_selector: str | None = Field(default=None, description="Label selector for filtering pods"),
        exec_command: str | None = Field(default=None, description="Command to execute in pod (space-separated)"),
        exec_container: str | None = Field(default=None, description="Container name for exec"),
        local_port: int | None = Field(default=None, description="Local port for port-forward"),
        remote_port: int | None = Field(default=None, description="Remote port for port-forward"),
        attach_container: str | None = Field(default=None, description="Container name for attach"),
        cp_source: str | None = Field(default=None, description="Source path for cp"),
        cp_destination: str | None = Field(default=None, description="Destination path for cp"),
        configmap_name: str | None = Field(default=None, description="ConfigMap name"),
        configmap_data: str | None = Field(default=None, description="ConfigMap data as JSON string"),
        configmap_from_file: str | None = Field(default=None, description="Path to file for ConfigMap data"),
        secret_name: str | None = Field(default=None, description="Secret name"),
        secret_type: str = Field(default="Opaque", description="Secret type (Opaque, kubernetes.io/docker-configjson, etc.)"),
        secret_data: str | None = Field(default=None, description="Secret data as JSON string (base64-encoded values)"),
        field_selector: str | None = Field(default=None, description="Field selector for events"),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list:
        """
        Manage Kubernetes resources (pods, configmaps, secrets, namespaces, events).
        
        This tool provides comprehensive Kubernetes resource operations beyond basic
        deployment management, including pod inspection, configuration management,
        and cluster event monitoring.
        """
        import json

        # Determine manager type
        if manager_type is None:
            manager_type = "kubernetes"

        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_resources: {action}")

        try:
            if action == "list_pods":
                return await run_blocking(
                    manager.list_pods, namespace=namespace, label_selector=label_selector
                )
            elif action == "describe_pod":
                if not pod_name:
                    return "Error: 'pod_name' is required for describe_pod"
                return await run_blocking(
                    manager.describe_pod, pod_name=pod_name, namespace=namespace
                )
            elif action == "exec_pod":
                if not pod_name:
                    return "Error: 'pod_name' is required for exec_pod"
                command = exec_command.split() if exec_command else None
                return await run_blocking(
                    manager.exec_pod, pod_name=pod_name, namespace=namespace, command=command, container=exec_container
                )
            elif action == "port_forward_pod":
                if not pod_name or not local_port or not remote_port:
                    return "Error: 'pod_name', 'local_port', and 'remote_port' are required for port_forward_pod"
                return await run_blocking(
                    manager.port_forward_pod, pod_name=pod_name, namespace=namespace, local_port=local_port, remote_port=remote_port
                )
            elif action == "attach_pod":
                if not pod_name:
                    return "Error: 'pod_name' is required for attach_pod"
                return await run_blocking(
                    manager.attach_pod, pod_name=pod_name, namespace=namespace, container=attach_container
                )
            elif action == "cp_pod":
                if not pod_name or not cp_source or not cp_destination:
                    return "Error: 'pod_name', 'cp_source', and 'cp_destination' are required for cp_pod"
                return await run_blocking(
                    manager.cp_pod, pod_name=pod_name, namespace=namespace, source=cp_source, destination=cp_destination
                )
            elif action == "list_configmaps":
                return await run_blocking(manager.list_configmaps, namespace=namespace)
            elif action == "create_configmap":
                if not configmap_name:
                    return "Error: 'configmap_name' is required for create_configmap"
                cm_data = json.loads(configmap_data) if configmap_data else None
                return await run_blocking(
                    manager.create_configmap,
                    name=configmap_name,
                    namespace=namespace,
                    data=cm_data,
                    from_file=configmap_from_file,
                )
            elif action == "list_secrets":
                return await run_blocking(manager.list_secrets, namespace=namespace)
            elif action == "create_secret":
                if not secret_name:
                    return "Error: 'secret_name' is required for create_secret"
                secret_data_dict = json.loads(secret_data) if secret_data else None
                return await run_blocking(
                    manager.create_secret,
                    name=secret_name,
                    namespace=namespace,
                    secret_type=secret_type,
                    data=secret_data_dict,
                )
            elif action == "list_namespaces":
                return await run_blocking(manager.list_namespaces)
            elif action == "list_events":
                return await run_blocking(
                    manager.list_events, namespace=namespace, field_selector=field_selector
                )
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

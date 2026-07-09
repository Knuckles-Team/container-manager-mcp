"""MCP tools for remaining Kubernetes gap operations.

This module provides the remaining Kubernetes operations needed for closer to 100% coverage
including pod operations (port-forward, exec, attach, copy), ingress, storage classes,
persistent volumes/claims, stateful sets, and daemon sets.
"""

from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_remaining_gaps_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Remaining Gap Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "remaining-gaps"},
    )
    async def cm_k8s_remaining_gaps(
        action: Literal[
            # Pod Operations
            "port_forward_pod",
            "exec_pod",
            "attach_pod",
            "copy_to_pod",
            "copy_from_pod",
            # Ingress Operations
            "create_ingress",
            "list_ingresses",
            # Storage Class Operations
            "create_storage_class",
            "list_storage_classes",
            # Persistent Volume Operations
            "create_persistent_volume",
            "list_persistent_volumes",
            # Persistent Volume Claim Operations
            "create_persistent_volume_claim",
            "list_persistent_volume_claims",
            # StatefulSet Operations
            "create_stateful_set",
            "list_stateful_sets",
            # DaemonSet Operations
            "create_daemon_set",
            "list_daemon_sets",
        ] = Field(
            description="Action to perform. Kubernetes remaining gap operations."
        ),
        # Common parameters
        pod_name: str | None = Field(default=None, description="Pod name for operations"),
        namespace: str | None = Field(default=None, description="Target namespace (default: from config)"),
        local_port: int | None = Field(default=None, description="Local port for port-forward"),
        remote_port: int | None = Field(default=None, description="Remote port for port-forward"),
        command: list | None = Field(default=None, description="Command to execute"),
        source: str | None = Field(default=None, description="Source file path for copy operations"),
        destination: str | None = Field(default=None, description="Destination file path for copy operations"),
        name: str | None = Field(default=None, description="Resource name"),
        spec: dict | None = Field(default=None, description="Resource specification"),
        provisioner: str | None = Field(default=None, description="Storage provisioner"),
        parameters: dict | None = Field(default=None, description="Storage parameters"),
    ) -> dict | list:
        """Manage Kubernetes remaining gap operations (pod ops, ingress, storage, statefulsets, daemonsets)."""
        
        ctx_log("Kubernetes remaining gap operations", action=action, name=name, namespace=namespace)
        
        @run_blocking
        def execute_operation():
            manager = create_manager("kubernetes")
            k8s_manager = getattr(manager, "k8s_manager", manager)
            
            # Pod Operations
            if action == "port_forward_pod":
                if not pod_name or not namespace or local_port is None or remote_port is None:
                    raise ValueError("pod_name, namespace, local_port, and remote_port are required for port_forward_pod")
                return k8s_manager.port_forward_pod(pod_name, namespace, local_port, remote_port)
            elif action == "exec_pod":
                if not pod_name or not namespace or not command:
                    raise ValueError("pod_name, namespace, and command are required for exec_pod")
                return k8s_manager.exec_pod(pod_name, namespace, command)
            elif action == "attach_pod":
                if not pod_name or not namespace:
                    raise ValueError("pod_name and namespace are required for attach_pod")
                return k8s_manager.attach_pod(pod_name, namespace)
            elif action == "copy_to_pod":
                if not pod_name or not namespace or not source or not destination:
                    raise ValueError("pod_name, namespace, source, and destination are required for copy_to_pod")
                return k8s_manager.copy_to_pod(pod_name, namespace, source, destination)
            elif action == "copy_from_pod":
                if not pod_name or not namespace or not source or not destination:
                    raise ValueError("pod_name, namespace, source, and destination are required for copy_from_pod")
                return k8s_manager.copy_from_pod(pod_name, namespace, source, destination)
            
            # Ingress Operations
            elif action == "create_ingress":
                if not name or not namespace or not spec:
                    raise ValueError("name, namespace, and spec are required for create_ingress")
                return k8s_manager.create_ingress(name, namespace, spec)
            elif action == "list_ingresses":
                return k8s_manager.list_ingresses(namespace)
            
            # Storage Class Operations
            elif action == "create_storage_class":
                if not name or not provisioner:
                    raise ValueError("name and provisioner are required for create_storage_class")
                return k8s_manager.create_storage_class(name, provisioner, parameters)
            elif action == "list_storage_classes":
                return k8s_manager.list_storage_classes()
            
            # Persistent Volume Operations
            elif action == "create_persistent_volume":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_persistent_volume")
                return k8s_manager.create_persistent_volume(name, spec)
            elif action == "list_persistent_volumes":
                return k8s_manager.list_persistent_volumes()
            
            # Persistent Volume Claim Operations
            elif action == "create_persistent_volume_claim":
                if not name or not namespace or not spec:
                    raise ValueError("name, namespace, and spec are required for create_persistent_volume_claim")
                return k8s_manager.create_persistent_volume_claim(name, namespace, spec)
            elif action == "list_persistent_volume_claims":
                return k8s_manager.list_persistent_volume_claims(namespace)
            
            # StatefulSet Operations
            elif action == "create_stateful_set":
                if not name or not namespace or not spec:
                    raise ValueError("name, namespace, and spec are required for create_stateful_set")
                return k8s_manager.create_stateful_set(name, namespace, spec)
            elif action == "list_stateful_sets":
                return k8s_manager.list_stateful_sets(namespace)
            
            # DaemonSet Operations
            elif action == "create_daemon_set":
                if not name or not namespace or not spec:
                    raise ValueError("name, namespace, and spec are required for create_daemon_set")
                return k8s_manager.create_daemon_set(name, namespace, spec)
            elif action == "list_daemon_sets":
                return k8s_manager.list_daemon_sets(namespace)
            
            else:
                raise ValueError(f"Unknown action: {action}")
        
        return execute_operation()

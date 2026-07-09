"""MCP tools for advanced Podman operations.

This module provides advanced Podman operations including pod management,
network management, volume management, checkpoint/restore, and system operations.
"""

from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_podmanadvanced_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Advanced Podman Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"podman", "advanced"},
    )
    async def cm_podman_advanced(
        action: Literal[
            # Kubernetes Integration
            "podman_generate_kube_yaml",
            "podman_play_kube_yaml",
            # Checkpoint/Restore
            "podman_checkpoint",
            "podman_restore",
            # Pod Management
            "podman_pod_create",
            "podman_pod_list",
            "podman_pod_stats",
            "podman_pod_top",
            "podman_pod_inspect",
            "podman_pod_logs",
            "podman_pod_stop",
            "podman_pod_rm",
            # Network Management
            "podman_network_create",
            "podman_network_list",
            "podman_network_inspect",
            # Volume Management
            "podman_volume_create",
            "podman_volume_list",
            "podman_volume_inspect",
            # System Operations
            "podman_system_prune",
            "podman_health_check",
        ] = Field(
            description="Action to perform. Advanced Podman operations."
        ),
        # Common parameters
        pod_name: str | None = Field(default=None, description="Pod name for operations"),
        namespace: str | None = Field(default="default", description="Kubernetes namespace"),
        yaml_path: str | None = Field(default=None, description="YAML file path"),
        container_id: str | None = Field(default=None, description="Container ID"),
        checkpoint_dir: str | None = Field(default=None, description="Checkpoint directory"),
        image: str | None = Field(default=None, description="Container image"),
        command: str | None = Field(default=None, description="Container command"),
        tail_lines: int | None = Field(default=100, description="Tail lines for logs"),
        network_name: str | None = Field(default=None, description="Network name"),
        driver: str | None = Field(default="bridge", description="Network driver"),
        subnet: str | None = Field(default=None, description="Network subnet"),
        volume_name: str | None = Field(default=None, description="Volume name"),
        config: dict | None = Field(default=None, description="Health check config"),
    ) -> dict | list:
        """Manage advanced Podman operations (pods, networks, volumes, checkpoint/restore, system)."""
        
        ctx_log("Advanced Podman operations", action=action, pod_name=pod_name)
        
        @run_blocking
        def execute_operation():
            manager = create_manager("podman")
            k8s_manager = getattr(manager, "k8s_manager", manager)
            
            # Kubernetes Integration
            if action == "podman_generate_kube_yaml":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_generate_kube_yaml")
                return k8s_manager.podman_generate_kube_yaml(pod_name, namespace or "default")
            elif action == "podman_play_kube_yaml":
                if not yaml_path:
                    raise ValueError("yaml_path is required for podman_play_kube_yaml")
                return k8s_manager.podman_play_kube_yaml(yaml_path)
            
            # Checkpoint/Restore
            elif action == "podman_checkpoint":
                if not container_id or not checkpoint_dir:
                    raise ValueError("container_id and checkpoint_dir are required for podman_checkpoint")
                return k8s_manager.podman_checkpoint(container_id, checkpoint_dir)
            elif action == "podman_restore":
                if not container_id or not checkpoint_dir:
                    raise ValueError("container_id and checkpoint_dir are required for podman_restore")
                return k8s_manager.podman_restore(container_id, checkpoint_dir)
            
            # Pod Management
            elif action == "podman_pod_create":
                if not pod_name or not image:
                    raise ValueError("pod_name and image are required for podman_pod_create")
                return k8s_manager.podman_pod_create(pod_name, image, command)
            elif action == "podman_pod_list":
                return k8s_manager.podman_pod_list()
            elif action == "podman_pod_stats":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_stats")
                return k8s_manager.podman_pod_stats(pod_name)
            elif action == "podman_pod_top":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_top")
                return k8s_manager.podman_pod_top(pod_name)
            elif action == "podman_pod_inspect":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_inspect")
                return k8s_manager.podman_pod_inspect(pod_name)
            elif action == "podman_pod_logs":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_logs")
                return k8s_manager.podman_pod_logs(pod_name, tail_lines or 100)
            elif action == "podman_pod_stop":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_stop")
                return k8s_manager.podman_pod_stop(pod_name)
            elif action == "podman_pod_rm":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_rm")
                return k8s_manager.podman_pod_rm(pod_name)
            
            # Network Management
            elif action == "podman_network_create":
                if not network_name:
                    raise ValueError("network_name is required for podman_network_create")
                return k8s_manager.podman_network_create(network_name, driver or "bridge", subnet)
            elif action == "podman_network_list":
                return k8s_manager.podman_network_list()
            elif action == "podman_network_inspect":
                if not network_name:
                    raise ValueError("network_name is required for podman_network_inspect")
                return k8s_manager.podman_network_inspect(network_name)
            
            # Volume Management
            elif action == "podman_volume_create":
                if not volume_name:
                    raise ValueError("volume_name is required for podman_volume_create")
                return k8s_manager.podman_volume_create(volume_name, driver or "local")
            elif action == "podman_volume_list":
                return k8s_manager.podman_volume_list()
            elif action == "podman_volume_inspect":
                if not volume_name:
                    raise ValueError("volume_name is required for podman_volume_inspect")
                return k8s_manager.podman_volume_inspect(volume_name)
            
            # System Operations
            elif action == "podman_system_prune":
                return k8s_manager.podman_system_prune()
            elif action == "podman_health_check":
                if not container_id or not config:
                    raise ValueError("container_id and config are required for podman_health_check")
                return k8s_manager.podman_health_check(container_id, config)
            
            else:
                raise ValueError(f"Unknown action: {action}")
        
        return execute_operation()

"""MCP tools for function-based Podman operations.

This module provides Podman operations including pod management,
network management, volume management, checkpoint/restore, Kubernetes YAML
interop, and system operations — dispatched directly onto the real
``PodmanManager`` (podman-py + ``podman`` CLI).
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log


def register_podman_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Podman Pod/Network/Volume Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"podman"},
    )
    async def cm_podman(
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
            description="Action to perform. Podman pod/network/volume operations."
        ),
        # Common parameters
        pod_name: str | None = Field(
            default=None, description="Pod name for operations"
        ),
        namespace: str | None = Field(
            default="default", description="Kubernetes namespace"
        ),
        yaml_path: str | None = Field(default=None, description="YAML file path"),
        container_id: str | None = Field(default=None, description="Container ID"),
        checkpoint_dir: str | None = Field(
            default=None, description="Checkpoint directory"
        ),
        image: str | None = Field(default=None, description="Container image"),
        command: str | None = Field(default=None, description="Container command"),
        tail_lines: int | None = Field(default=100, description="Tail lines for logs"),
        network_name: str | None = Field(default=None, description="Network name"),
        driver: str | None = Field(default="bridge", description="Network driver"),
        subnet: str | None = Field(default=None, description="Network subnet"),
        volume_name: str | None = Field(default=None, description="Volume name"),
        config: dict | None = Field(default=None, description="Health check config"),
        ctx: Context | None = None,
    ) -> dict | list:
        """Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system)."""

        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_podman: {action}")

        def execute_operation():
            manager = create_manager("podman")

            # Kubernetes Integration
            if action == "podman_generate_kube_yaml":
                if not pod_name:
                    raise ValueError(
                        "pod_name is required for podman_generate_kube_yaml"
                    )
                return manager.podman_generate_kube_yaml(
                    pod_name, namespace or "default"
                )
            elif action == "podman_play_kube_yaml":
                if not yaml_path:
                    raise ValueError("yaml_path is required for podman_play_kube_yaml")
                return manager.podman_play_kube_yaml(yaml_path)

            # Checkpoint/Restore
            elif action == "podman_checkpoint":
                if not container_id or not checkpoint_dir:
                    raise ValueError(
                        "container_id and checkpoint_dir are required for podman_checkpoint"
                    )
                return manager.podman_checkpoint(container_id, checkpoint_dir)
            elif action == "podman_restore":
                if not container_id or not checkpoint_dir:
                    raise ValueError(
                        "container_id and checkpoint_dir are required for podman_restore"
                    )
                return manager.podman_restore(container_id, checkpoint_dir)

            # Pod Management
            elif action == "podman_pod_create":
                if not pod_name or not image:
                    raise ValueError(
                        "pod_name and image are required for podman_pod_create"
                    )
                return manager.podman_pod_create(pod_name, image, command)
            elif action == "podman_pod_list":
                return manager.podman_pod_list()
            elif action == "podman_pod_stats":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_stats")
                return manager.podman_pod_stats(pod_name)
            elif action == "podman_pod_top":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_top")
                return manager.podman_pod_top(pod_name)
            elif action == "podman_pod_inspect":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_inspect")
                return manager.podman_pod_inspect(pod_name)
            elif action == "podman_pod_logs":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_logs")
                return manager.podman_pod_logs(pod_name, tail_lines or 100)
            elif action == "podman_pod_stop":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_stop")
                return manager.podman_pod_stop(pod_name)
            elif action == "podman_pod_rm":
                if not pod_name:
                    raise ValueError("pod_name is required for podman_pod_rm")
                return manager.podman_pod_rm(pod_name)

            # Network Management
            elif action == "podman_network_create":
                if not network_name:
                    raise ValueError(
                        "network_name is required for podman_network_create"
                    )
                return manager.podman_network_create(
                    network_name, driver or "bridge", subnet
                )
            elif action == "podman_network_list":
                return manager.podman_network_list()
            elif action == "podman_network_inspect":
                if not network_name:
                    raise ValueError(
                        "network_name is required for podman_network_inspect"
                    )
                return manager.podman_network_inspect(network_name)

            # Volume Management
            elif action == "podman_volume_create":
                if not volume_name:
                    raise ValueError("volume_name is required for podman_volume_create")
                return manager.podman_volume_create(volume_name, driver or "local")
            elif action == "podman_volume_list":
                return manager.podman_volume_list()
            elif action == "podman_volume_inspect":
                if not volume_name:
                    raise ValueError(
                        "volume_name is required for podman_volume_inspect"
                    )
                return manager.podman_volume_inspect(volume_name)

            # System Operations
            elif action == "podman_system_prune":
                return manager.podman_system_prune()
            elif action == "podman_health_check":
                if not container_id or not config:
                    raise ValueError(
                        "container_id and config are required for podman_health_check"
                    )
                return manager.podman_health_check(container_id, config)

            else:
                raise ValueError(f"Unknown action: {action}")

        return await run_blocking(execute_operation)

"""MCP tools for function-based Podman operations.

This module provides Podman operations including pod management,
network management, volume management, checkpoint/restore, Kubernetes YAML
interop, and system operations — dispatched directly onto the real
``PodmanManager`` (podman-py + ``podman`` CLI).
"""

import logging
from collections.abc import Callable
from typing import Any, Literal

from agent_connector_sdk.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log

# Every action's manager method name equals the action name; only the
# required fields, guard message, and positional call args differ. The third
# tuple element builds the manager method's positional args from the request
# `values` dict (see `_dispatch_podman_action` below).
_PODMAN_ACTIONS: dict[
    str, tuple[tuple[str, ...], str | None, Callable[[dict[str, Any]], tuple]]
] = {
    "podman_generate_kube_yaml": (
        ("pod_name",),
        "pod_name is required for podman_generate_kube_yaml",
        lambda v: (v["pod_name"], v["namespace"] or "default"),
    ),
    "podman_play_kube_yaml": (
        ("yaml_path",),
        "yaml_path is required for podman_play_kube_yaml",
        lambda v: (v["yaml_path"],),
    ),
    "podman_checkpoint": (
        ("container_id", "checkpoint_dir"),
        "container_id and checkpoint_dir are required for podman_checkpoint",
        lambda v: (v["container_id"], v["checkpoint_dir"]),
    ),
    "podman_restore": (
        ("container_id", "checkpoint_dir"),
        "container_id and checkpoint_dir are required for podman_restore",
        lambda v: (v["container_id"], v["checkpoint_dir"]),
    ),
    "podman_pod_create": (
        ("pod_name", "image"),
        "pod_name and image are required for podman_pod_create",
        lambda v: (v["pod_name"], v["image"], v["command"]),
    ),
    "podman_pod_list": ((), None, lambda v: ()),
    "podman_pod_stats": (
        ("pod_name",),
        "pod_name is required for podman_pod_stats",
        lambda v: (v["pod_name"],),
    ),
    "podman_pod_top": (
        ("pod_name",),
        "pod_name is required for podman_pod_top",
        lambda v: (v["pod_name"],),
    ),
    "podman_pod_inspect": (
        ("pod_name",),
        "pod_name is required for podman_pod_inspect",
        lambda v: (v["pod_name"],),
    ),
    "podman_pod_logs": (
        ("pod_name",),
        "pod_name is required for podman_pod_logs",
        lambda v: (v["pod_name"], v["tail_lines"] or 100),
    ),
    "podman_pod_stop": (
        ("pod_name",),
        "pod_name is required for podman_pod_stop",
        lambda v: (v["pod_name"],),
    ),
    "podman_pod_rm": (
        ("pod_name",),
        "pod_name is required for podman_pod_rm",
        lambda v: (v["pod_name"],),
    ),
    "podman_network_create": (
        ("network_name",),
        "network_name is required for podman_network_create",
        lambda v: (v["network_name"], v["driver"] or "bridge", v["subnet"]),
    ),
    "podman_network_list": ((), None, lambda v: ()),
    "podman_network_inspect": (
        ("network_name",),
        "network_name is required for podman_network_inspect",
        lambda v: (v["network_name"],),
    ),
    "podman_volume_create": (
        ("volume_name",),
        "volume_name is required for podman_volume_create",
        lambda v: (v["volume_name"], v["driver"] or "local"),
    ),
    "podman_volume_list": ((), None, lambda v: ()),
    "podman_volume_inspect": (
        ("volume_name",),
        "volume_name is required for podman_volume_inspect",
        lambda v: (v["volume_name"],),
    ),
    "podman_system_prune": ((), None, lambda v: ()),
    "podman_health_check": (
        ("container_id", "config"),
        "container_id and config are required for podman_health_check",
        lambda v: (v["container_id"], v["config"]),
    ),
}


def _dispatch_podman_action(action, manager, values):
    """Look up and invoke the manager method for `action` via `_PODMAN_ACTIONS`."""
    spec = _PODMAN_ACTIONS.get(action)
    if spec is None:
        raise ValueError(f"Unknown action: {action}")
    required, error_message, build_args = spec
    if required and not all(values[field] for field in required):
        raise ValueError(error_message)
    return getattr(manager, action)(*build_args(values))


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
            values = {
                "pod_name": pod_name,
                "namespace": namespace,
                "yaml_path": yaml_path,
                "container_id": container_id,
                "checkpoint_dir": checkpoint_dir,
                "image": image,
                "command": command,
                "tail_lines": tail_lines,
                "network_name": network_name,
                "driver": driver,
                "subnet": subnet,
                "volume_name": volume_name,
                "config": config,
            }
            return _dispatch_podman_action(action, manager, values)

        return await run_blocking(execute_operation)

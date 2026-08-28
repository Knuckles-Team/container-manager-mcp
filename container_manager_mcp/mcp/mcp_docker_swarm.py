"""MCP tools for Docker Swarm operations.

This module provides Docker Swarm operations including Swarm lifecycle, services,
stacks, configs, secrets, and node management — dispatched directly onto the real
``DockerManager`` (docker SDK + ``docker stack`` CLI).
"""

import logging
from typing import Literal

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log

# Every action's manager method name equals the action name; only the
# required fields, guard message, and positional call args differ.
_SWARM_ACTIONS: dict[str, tuple[tuple[str, ...], str | None, "callable"]] = {
    "docker_swarm_init": (
        ("advertise_addr",),
        "advertise_addr is required for docker_swarm_init",
        lambda v: (v["advertise_addr"], v["listen_addr"]),
    ),
    "docker_swarm_join": (
        ("remote_addr", "token"),
        "remote_addr and token are required for docker_swarm_join",
        lambda v: (v["remote_addr"], v["token"], v["worker"] or True),
    ),
    "docker_swarm_leave": ((), None, lambda v: (v["force"] or False,)),
    "docker_service_create": (
        ("service_name", "image"),
        "service_name and image are required for docker_service_create",
        lambda v: (v["service_name"], v["image"], v["replicas"] or 1, v["ports"]),
    ),
    "docker_service_list": ((), None, lambda v: ()),
    "docker_service_update": (
        ("service_name",),
        "service_name is required for docker_service_update",
        lambda v: (v["service_name"], v["image"], v["replicas"]),
    ),
    "docker_service_rm": (
        ("service_name",),
        "service_name is required for docker_service_rm",
        lambda v: (v["service_name"],),
    ),
    "docker_service_logs": (
        ("service_name",),
        "service_name is required for docker_service_logs",
        lambda v: (v["service_name"], v["tail_lines"] or 100),
    ),
    "docker_service_ps": ((), None, lambda v: ()),
    "docker_stack_deploy": (
        ("stack_name", "compose_file"),
        "stack_name and compose_file are required for docker_stack_deploy",
        lambda v: (v["stack_name"], v["compose_file"]),
    ),
    "docker_stack_services": (
        ("stack_name",),
        "stack_name is required for docker_stack_services",
        lambda v: (v["stack_name"],),
    ),
    "docker_stack_rm": (
        ("stack_name",),
        "stack_name is required for docker_stack_rm",
        lambda v: (v["stack_name"],),
    ),
    "docker_config_create": (
        ("config_name", "data"),
        "config_name and data are required for docker_config_create",
        lambda v: (v["config_name"], v["data"]),
    ),
    "docker_config_list": ((), None, lambda v: ()),
    "docker_secret_create": (
        ("secret_name", "data"),
        "secret_name and data are required for docker_secret_create",
        lambda v: (v["secret_name"], v["data"]),
    ),
    "docker_secret_list": ((), None, lambda v: ()),
    "docker_node_ls": ((), None, lambda v: ()),
    "docker_node_update": (
        ("node_id", "availability"),
        "node_id and availability are required for docker_node_update",
        lambda v: (v["node_id"], v["availability"]),
    ),
    "docker_node_inspect": (
        ("node_id",),
        "node_id is required for docker_node_inspect",
        lambda v: (v["node_id"],),
    ),
}


def _dispatch_docker_swarm_action(action, manager, values):
    """Look up and invoke the manager method for `action` via `_SWARM_ACTIONS`."""
    spec = _SWARM_ACTIONS.get(action)
    if spec is None:
        raise ValueError(f"Unknown action: {action}")
    required, error_message, build_args = spec
    if required and not all(values[field] for field in required):
        raise ValueError(error_message)
    return getattr(manager, action)(*build_args(values))


def register_dockerswarm_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Docker Swarm Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"docker", "swarm"},
    )
    async def cm_docker_swarm(
        action: Literal[
            # Swarm Operations
            "docker_swarm_init",
            "docker_swarm_join",
            "docker_swarm_leave",
            # Service Operations
            "docker_service_create",
            "docker_service_list",
            "docker_service_update",
            "docker_service_rm",
            "docker_service_logs",
            "docker_service_ps",
            # Stack Operations
            "docker_stack_deploy",
            "docker_stack_services",
            "docker_stack_rm",
            # Config Operations
            "docker_config_create",
            "docker_config_list",
            # Secret Operations
            "docker_secret_create",
            "docker_secret_list",
            # Node Operations
            "docker_node_ls",
            "docker_node_update",
            "docker_node_inspect",
        ] = Field(
            description="Action to perform. Docker Swarm/service/stack operations."
        ),
        # Common parameters
        advertise_addr: str | None = Field(
            default=None, description="Swarm advertise address"
        ),
        listen_addr: str | None = Field(
            default=None, description="Swarm listen address"
        ),
        remote_addr: str | None = Field(
            default=None, description="Remote swarm address"
        ),
        token: str | None = Field(default=None, description="Swarm join token"),
        worker: bool | None = Field(default=True, description="Join as worker"),
        force: bool | None = Field(default=False, description="Force operation"),
        service_name: str | None = Field(default=None, description="Service name"),
        image: str | None = Field(default=None, description="Container image"),
        replicas: int | None = Field(default=None, description="Number of replicas"),
        ports: list | None = Field(default=None, description="Service ports"),
        tail_lines: int | None = Field(default=100, description="Tail lines for logs"),
        stack_name: str | None = Field(default=None, description="Stack name"),
        compose_file: str | None = Field(default=None, description="Compose file path"),
        config_name: str | None = Field(default=None, description="Config name"),
        secret_name: str | None = Field(default=None, description="Secret name"),
        data: str | None = Field(default=None, description="Config/secret data"),
        node_id: str | None = Field(default=None, description="Node ID"),
        availability: str | None = Field(default=None, description="Node availability"),
        ctx: Context | None = None,
    ) -> dict | list:
        """Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes)."""

        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_docker_swarm: {action}")

        def execute_operation():
            manager = create_manager("docker")
            values = {
                "advertise_addr": advertise_addr,
                "listen_addr": listen_addr,
                "remote_addr": remote_addr,
                "token": token,
                "worker": worker,
                "force": force,
                "service_name": service_name,
                "image": image,
                "replicas": replicas,
                "ports": ports,
                "tail_lines": tail_lines,
                "stack_name": stack_name,
                "compose_file": compose_file,
                "config_name": config_name,
                "secret_name": secret_name,
                "data": data,
                "node_id": node_id,
                "availability": availability,
            }
            return _dispatch_docker_swarm_action(action, manager, values)

        return await run_blocking(execute_operation)

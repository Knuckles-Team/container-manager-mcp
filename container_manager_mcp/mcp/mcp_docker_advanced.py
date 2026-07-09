"""MCP tools for advanced Docker operations.

This module provides advanced Docker operations including Swarm, services, stacks,
configs, secrets, and node management.
"""

from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_dockeradvanced_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Advanced Docker Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"docker", "advanced"},
    )
    async def cm_docker_advanced(
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
        ] = Field(description="Action to perform. Advanced Docker operations."),
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
    ) -> dict | list:
        """Manage advanced Docker operations (Swarm, services, stacks, configs, secrets, nodes)."""

        ctx_log("Advanced Docker operations", action=action, service_name=service_name)

        @run_blocking
        def execute_operation():
            manager = create_manager("docker")
            k8s_manager = getattr(manager, "k8s_manager", manager)

            # Swarm Operations
            if action == "docker_swarm_init":
                if not advertise_addr:
                    raise ValueError("advertise_addr is required for docker_swarm_init")
                return k8s_manager.docker_swarm_init(advertise_addr, listen_addr)
            elif action == "docker_swarm_join":
                if not remote_addr or not token:
                    raise ValueError(
                        "remote_addr and token are required for docker_swarm_join"
                    )
                return k8s_manager.docker_swarm_join(remote_addr, token, worker or True)
            elif action == "docker_swarm_leave":
                return k8s_manager.docker_swarm_leave(force or False)

            # Service Operations
            elif action == "docker_service_create":
                if not service_name or not image:
                    raise ValueError(
                        "service_name and image are required for docker_service_create"
                    )
                return k8s_manager.docker_service_create(
                    service_name, image, replicas or 1, ports
                )
            elif action == "docker_service_list":
                return k8s_manager.docker_service_list()
            elif action == "docker_service_update":
                if not service_name:
                    raise ValueError(
                        "service_name is required for docker_service_update"
                    )
                return k8s_manager.docker_service_update(service_name, image, replicas)
            elif action == "docker_service_rm":
                if not service_name:
                    raise ValueError("service_name is required for docker_service_rm")
                return k8s_manager.docker_service_rm(service_name)
            elif action == "docker_service_logs":
                if not service_name:
                    raise ValueError("service_name is required for docker_service_logs")
                return k8s_manager.docker_service_logs(service_name, tail_lines or 100)
            elif action == "docker_service_ps":
                return k8s_manager.docker_service_ps()

            # Stack Operations
            elif action == "docker_stack_deploy":
                if not stack_name or not compose_file:
                    raise ValueError(
                        "stack_name and compose_file are required for docker_stack_deploy"
                    )
                return k8s_manager.docker_stack_deploy(stack_name, compose_file)
            elif action == "docker_stack_services":
                if not stack_name:
                    raise ValueError("stack_name is required for docker_stack_services")
                return k8s_manager.docker_stack_services(stack_name)
            elif action == "docker_stack_rm":
                if not stack_name:
                    raise ValueError("stack_name is required for docker_stack_rm")
                return k8s_manager.docker_stack_rm(stack_name)

            # Config Operations
            elif action == "docker_config_create":
                if not config_name or not data:
                    raise ValueError(
                        "config_name and data are required for docker_config_create"
                    )
                return k8s_manager.docker_config_create(config_name, data)
            elif action == "docker_config_list":
                return k8s_manager.docker_config_list()

            # Secret Operations
            elif action == "docker_secret_create":
                if not secret_name or not data:
                    raise ValueError(
                        "secret_name and data are required for docker_secret_create"
                    )
                return k8s_manager.docker_secret_create(secret_name, data)
            elif action == "docker_secret_list":
                return k8s_manager.docker_secret_list()

            # Node Operations
            elif action == "docker_node_ls":
                return k8s_manager.docker_node_ls()
            elif action == "docker_node_update":
                if not node_id or not availability:
                    raise ValueError(
                        "node_id and availability are required for docker_node_update"
                    )
                return k8s_manager.docker_node_update(node_id, availability)
            elif action == "docker_node_inspect":
                if not node_id:
                    raise ValueError("node_id is required for docker_node_inspect")
                return k8s_manager.docker_node_inspect(node_id)

            else:
                raise ValueError(f"Unknown action: {action}")

        return execute_operation()

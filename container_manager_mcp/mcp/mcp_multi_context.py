"""MCP tools for multi-context container management.

This module provides unified tools that can manage Kubernetes, Docker, Podman, and Swarm
simultaneously with context selection.
"""

from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_multicontext_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Multi-Context Container Management",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"multi-context", "kubernetes", "docker", "podman", "swarm"},
    )
    async def cm_multi_context(
        action: Literal[
            # Context Management
            "list_contexts",
            # Container Operations
            "list_containers",
            "run_container",
            "stop_container",
            "remove_container",
            "inspect_container",
            # Image Operations
            "list_images",
            "pull_image",
            "remove_image",
            # Volume Operations
            "list_volumes",
            "create_volume",
            "remove_volume",
            # Network Operations
            "list_networks",
            "create_network",
            "remove_network",
            # Service Operations (Kubernetes/Swarm)
            "list_services",
            "create_service",
            "remove_service",
            # Pod Operations (Kubernetes)
            "list_pods",
            "describe_pod",
            # Deployment Operations (Kubernetes)
            "list_deployments",
            "scale_deployment",
        ] = Field(description="Action to perform. Multi-context container management."),
        # Backend and Context Selection
        backend: Literal["kubernetes", "docker", "podman", "swarm"] = Field(
            default="kubernetes", description="Container backend to use"
        ),
        context: str | None = Field(
            default=None, description="Context name (uses default if None)"
        ),
        # Common parameters
        name: str | None = Field(default=None, description="Resource name"),
        namespace: str | None = Field(default=None, description="Kubernetes namespace"),
        image: str | None = Field(default=None, description="Container image"),
        command: str | None = Field(default=None, description="Container command"),
        all: bool = Field(default=False, description="List all resources"),
        force: bool = Field(default=False, description="Force operation"),
        spec: dict | None = Field(default=None, description="Resource specification"),
        replicas: int | None = Field(default=None, description="Number of replicas"),
        driver: str | None = Field(default=None, description="Network/volume driver"),
        ports: list | None = Field(default=None, description="Port mappings"),
        volumes: dict | None = Field(default=None, description="Volume mappings"),
        environment: dict | None = Field(
            default=None, description="Environment variables"
        ),
    ) -> dict | list:
        """Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection."""

        ctx_log(
            "Multi-context operations", action=action, backend=backend, context=context
        )

        @run_blocking
        def execute_operation():
            manager = create_manager(manager_type="multi")

            # Context Management
            if action == "list_contexts":
                return manager.list_available_contexts()

            # Get the appropriate manager for the backend
            target_manager = manager.get_manager(backend, context)

            # Container Operations
            if action == "list_containers":
                return target_manager.list_containers(all=all)
            elif action == "run_container":
                if not image:
                    raise ValueError("image is required for run_container")
                return target_manager.run_container(
                    image=image,
                    name=name,
                    command=command,
                    ports=ports,
                    volumes=volumes,
                    environment=environment,
                )
            elif action == "stop_container":
                if not name:
                    raise ValueError("name is required for stop_container")
                return target_manager.stop_container(name, force=force)
            elif action == "remove_container":
                if not name:
                    raise ValueError("name is required for remove_container")
                return target_manager.remove_container(name, force=force)
            elif action == "inspect_container":
                if not name:
                    raise ValueError("name is required for inspect_container")
                return target_manager.inspect_container(name)

            # Image Operations
            elif action == "list_images":
                return target_manager.list_images()
            elif action == "pull_image":
                if not image:
                    raise ValueError("image is required for pull_image")
                return target_manager.pull_image(image)
            elif action == "remove_image":
                if not name:
                    raise ValueError("name is required for remove_image")
                return target_manager.remove_image(name, force=force)

            # Volume Operations
            elif action == "list_volumes":
                return target_manager.list_volumes()
            elif action == "create_volume":
                if not name:
                    raise ValueError("name is required for create_volume")
                return target_manager.create_volume(name, driver)
            elif action == "remove_volume":
                if not name:
                    raise ValueError("name is required for remove_volume")
                return target_manager.remove_volume(name, force=force)

            # Network Operations
            elif action == "list_networks":
                return target_manager.list_networks()
            elif action == "create_network":
                if not name:
                    raise ValueError("name is required for create_network")
                return target_manager.create_network(name, driver)
            elif action == "remove_network":
                if not name:
                    raise ValueError("name is required for remove_network")
                return target_manager.remove_network(name, force=force)

            # Service Operations
            elif action == "list_services":
                return target_manager.list_services()
            elif action == "create_service":
                if not name or not image:
                    raise ValueError("name and image are required for create_service")
                return target_manager.create_service(name, image, ports)
            elif action == "remove_service":
                if not name:
                    raise ValueError("name is required for remove_service")
                return target_manager.remove_service(name)

            # Kubernetes-Specific Operations
            elif action == "list_pods":
                if backend == "kubernetes":
                    return target_manager.list_pods(namespace=namespace or "default")
                else:
                    raise ValueError(
                        f"list_pods is only available for Kubernetes, not {backend}"
                    )

            elif action == "describe_pod":
                if backend == "kubernetes":
                    if not name or not namespace:
                        raise ValueError(
                            "name and namespace are required for describe_pod"
                        )
                    return target_manager.describe_pod(name, namespace or "default")
                else:
                    raise ValueError(
                        f"describe_pod is only available for Kubernetes, not {backend}"
                    )

            elif action == "list_deployments":
                if backend == "kubernetes":
                    return target_manager.list_deployments(
                        namespace=namespace or "default"
                    )
                else:
                    raise ValueError(
                        f"list_deployments is only available for Kubernetes, not {backend}"
                    )

            elif action == "scale_deployment":
                if backend == "kubernetes":
                    if not name or not namespace or replicas is None:
                        raise ValueError(
                            "name, namespace, and replicas are required for scale_deployment"
                        )
                    return target_manager.scale_service(
                        name, replicas, namespace=namespace or "default"
                    )
                else:
                    raise ValueError(
                        f"scale_deployment is only available for Kubernetes, not {backend}"
                    )

            else:
                raise ValueError(f"Unknown action: {action}")

        return execute_operation()

"""MCP tools for multi-context container management.

This module provides unified tools that can manage Kubernetes, Docker, Podman, and Swarm
simultaneously with context selection.
"""

import logging
from typing import Literal

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log


def _dispatch_container_action(
    action, target_manager, all_, command, environment, force, image, name, ports, volumes
):
    if action == "list_containers":
        return target_manager.list_containers(all=all_)
    if action == "run_container":
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
    if action == "stop_container":
        if not name:
            raise ValueError("name is required for stop_container")
        return target_manager.stop_container(name, force=force)
    if action == "remove_container":
        if not name:
            raise ValueError("name is required for remove_container")
        return target_manager.remove_container(name, force=force)
    # action == "inspect_container"
    if not name:
        raise ValueError("name is required for inspect_container")
    return target_manager.inspect_container(name)


def _dispatch_image_action(action, target_manager, force, image, name):
    if action == "list_images":
        return target_manager.list_images()
    if action == "pull_image":
        if not image:
            raise ValueError("image is required for pull_image")
        return target_manager.pull_image(image)
    # action == "remove_image"
    if not name:
        raise ValueError("name is required for remove_image")
    return target_manager.remove_image(name, force=force)


def _dispatch_volume_action(action, target_manager, driver, force, name):
    if action == "list_volumes":
        return target_manager.list_volumes()
    if action == "create_volume":
        if not name:
            raise ValueError("name is required for create_volume")
        return target_manager.create_volume(name, driver)
    # action == "remove_volume"
    if not name:
        raise ValueError("name is required for remove_volume")
    return target_manager.remove_volume(name, force=force)


def _dispatch_network_action(action, target_manager, driver, force, name):
    if action == "list_networks":
        return target_manager.list_networks()
    if action == "create_network":
        if not name:
            raise ValueError("name is required for create_network")
        return target_manager.create_network(name, driver)
    # action == "remove_network"
    if not name:
        raise ValueError("name is required for remove_network")
    return target_manager.remove_network(name, force=force)


def _dispatch_service_action(action, target_manager, image, name, ports):
    if action == "list_services":
        return target_manager.list_services()
    if action == "create_service":
        if not name or not image:
            raise ValueError("name and image are required for create_service")
        return target_manager.create_service(name, image, ports)
    # action == "remove_service"
    if not name:
        raise ValueError("name is required for remove_service")
    return target_manager.remove_service(name)


def _describe_pod_kubernetes_action(target_manager, name, namespace, ns):
    if not name or not namespace:
        raise ValueError("name and namespace are required for describe_pod")
    return target_manager.describe_pod(name, ns)


def _scale_deployment_kubernetes_action(target_manager, name, namespace, replicas, ns):
    if not name or not namespace or replicas is None:
        raise ValueError("name, namespace, and replicas are required for scale_deployment")
    return target_manager.scale_service(name, replicas, namespace=ns)


def _dispatch_kubernetes_only_action(action, target_manager, backend, name, namespace, replicas):
    if backend != "kubernetes":
        raise ValueError(f"{action} is only available for Kubernetes, not {backend}")
    ns = namespace or "default"
    if action == "list_pods":
        return target_manager.list_pods(namespace=ns)
    if action == "describe_pod":
        return _describe_pod_kubernetes_action(target_manager, name, namespace, ns)
    if action == "list_deployments":
        return target_manager.list_deployments(namespace=ns)
    # action == "scale_deployment"
    return _scale_deployment_kubernetes_action(target_manager, name, namespace, replicas, ns)


_ACTION_GROUPS: dict[str, str] = {
    "list_containers": "container",
    "run_container": "container",
    "stop_container": "container",
    "remove_container": "container",
    "inspect_container": "container",
    "list_images": "image",
    "pull_image": "image",
    "remove_image": "image",
    "list_volumes": "volume",
    "create_volume": "volume",
    "remove_volume": "volume",
    "list_networks": "network",
    "create_network": "network",
    "remove_network": "network",
    "list_services": "service",
    "create_service": "service",
    "remove_service": "service",
    "list_pods": "kubernetes_only",
    "describe_pod": "kubernetes_only",
    "list_deployments": "kubernetes_only",
    "scale_deployment": "kubernetes_only",
}

_GROUP_FUNCS = {
    "container": _dispatch_container_action,
    "image": _dispatch_image_action,
    "volume": _dispatch_volume_action,
    "network": _dispatch_network_action,
    "service": _dispatch_service_action,
    "kubernetes_only": _dispatch_kubernetes_only_action,
}

_GROUP_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "container": ("all_", "command", "environment", "force", "image", "name", "ports", "volumes"),
    "image": ("force", "image", "name"),
    "volume": ("driver", "force", "name"),
    "network": ("driver", "force", "name"),
    "service": ("image", "name", "ports"),
    "kubernetes_only": ("backend", "name", "namespace", "replicas"),
}


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
        ctx: Context | None = None,
    ) -> dict | list:
        """Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection."""

        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_multi_context: {action}")

        def execute_operation():
            manager = create_manager(manager_type="multi")

            # Context Management
            if action == "list_contexts":
                return manager.list_available_contexts()

            # Get the appropriate manager for the backend
            target_manager = manager.get_manager(backend, context)

            group = _ACTION_GROUPS.get(action)
            if group is None:
                raise ValueError(f"Unknown action: {action}")
            all_values = {
                "all_": all,
                "backend": backend,
                "command": command,
                "driver": driver,
                "environment": environment,
                "force": force,
                "image": image,
                "name": name,
                "namespace": namespace,
                "ports": ports,
                "replicas": replicas,
                "volumes": volumes,
            }
            group_kwargs = {n: all_values[n] for n in _GROUP_PARAM_NAMES[group]}
            return _GROUP_FUNCS[group](action, target_manager, **group_kwargs)

        return await run_blocking(execute_operation)

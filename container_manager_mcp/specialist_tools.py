#!/usr/bin/python
"""Agent OS Specialist Deployment Tools for container-manager-mcp.

Provides containerized specialist lifecycle management as MCP tools.
These tools are the container layer of the Agent OS Triad and are
discoverable through the Knowledge Graph in agent-utilities.

Requires ``agent-utilities >= 0.3.0`` for ``ContainerConfig`` model.
Falls back gracefully when not available.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from pydantic import Field

logger = logging.getLogger(__name__)

# Graceful import — becomes no-op without agent-utilities
try:
    from agent_utilities.core.registry_cli import ContainerConfig  # noqa: F401

    _HAS_AGENT_UTILITIES = True
except ImportError:
    _HAS_AGENT_UTILITIES = False
    logger.debug(
        "[Agent OS] agent-utilities >= 0.3.0 not found. "
        "Specialist deployment tools will use inline config."
    )


def register_specialist_deployment_tools(mcp: Any) -> None:
    """Register specialist container lifecycle tools."""

    @mcp.tool(
        annotations={
            "title": "Deploy Specialist Container",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"agent_os", "specialist"},
    )
    async def deploy_specialist_container(
        image: str = Field(
            description="Container image (e.g. knucklessg1/salesforce-agent:latest)"
        ),
        name: str = Field(description="Container name", default=""),
        ports: str = Field(
            description='Port mappings as JSON object (e.g. \'{"8004": "8004"}\')',
            default="{}",
        ),
        env: str = Field(
            description="Environment variables as JSON object",
            default="{}",
        ),
        labels: str = Field(
            description="Container labels as JSON object",
            default='{"managed-by": "agent-os"}',
        ),
        health_check: str = Field(
            description="Health check command (e.g. 'curl -f http://localhost:8004/health')",
            default="",
        ),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
    ) -> dict:
        """Deploys a specialist agent as a container using the Agent OS ContainerConfig schema.

        Pulls the image, creates the container with port mappings, env vars,
        labels, and optional health check, then starts it.
        """
        from container_manager_mcp.container_manager import create_manager

        _ = health_check  # Silence vulture
        try:
            port_map = json.loads(ports) if isinstance(ports, str) else ports
            env_map = json.loads(env) if isinstance(env, str) else env
            label_map = json.loads(labels) if isinstance(labels, str) else labels
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON parameter: {e}"}

        try:
            manager = create_manager(manager_type, silent=False, log_file=None)

            # Pull image
            manager.pull_image(image)

            # Build run arguments
            container_name = name or f"specialist-{image.split('/')[-1].split(':')[0]}"
            env_list = [f"{k}={v}" for k, v in env_map.items()]
            port_bindings = {v: k for k, v in port_map.items()}

            # Run container
            result = manager.run_container(
                image=image,
                name=container_name,
                detach=True,
                ports=port_bindings if port_bindings else None,
                environment=env_list if env_list else None,
                labels=label_map,
            )

            return {
                "success": True,
                "container_id": result.get("id", result.get("Id", "")),
                "name": container_name,
                "image": image,
                "ports": port_map,
                "labels": label_map,
                "status": "running",
            }
        except Exception as e:
            logger.error(f"[Agent OS] Failed to deploy specialist container: {e}")
            return {"success": False, "error": str(e)}

    @mcp.tool(
        annotations={
            "title": "Stop Specialist Container",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"agent_os", "specialist"},
    )
    async def stop_specialist_container(
        container_id: str = Field(description="Container ID or name"),
        remove: bool = Field(
            description="Remove container after stopping", default=False
        ),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
    ) -> dict:
        """Stops a running specialist container, optionally removing it."""
        from container_manager_mcp.container_manager import create_manager

        try:
            manager = create_manager(manager_type, silent=False, log_file=None)
            manager.stop_container(container_id)
            result = {
                "success": True,
                "container_id": container_id,
                "status": "stopped",
            }

            if remove:
                manager.remove_container(container_id, force=True)
                result["status"] = "removed"

            return result
        except Exception as e:
            logger.error(f"[Agent OS] Failed to stop specialist container: {e}")
            return {"success": False, "error": str(e)}

    @mcp.tool(
        annotations={
            "title": "Get Specialist Container Status",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"agent_os", "specialist"},
    )
    async def get_specialist_status(
        container_id: str = Field(description="Container ID or name"),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
    ) -> dict:
        """Gets the status and health of a specialist container."""
        from container_manager_mcp.container_manager import create_manager

        try:
            manager = create_manager(manager_type, silent=False, log_file=None)
            info = manager.inspect_container(container_id)
            state = info.get("State", {})
            return {
                "container_id": container_id,
                "status": state.get("Status", "unknown"),
                "running": state.get("Running", False),
                "health": state.get("Health", {}).get("Status", "none"),
                "started_at": state.get("StartedAt", ""),
                "image": info.get("Config", {}).get("Image", ""),
                "labels": info.get("Config", {}).get("Labels", {}),
            }
        except Exception as e:
            logger.error(f"[Agent OS] Failed to get specialist status: {e}")
            return {"success": False, "error": str(e)}

    @mcp.tool(
        annotations={
            "title": "List Specialist Containers",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"agent_os", "specialist"},
    )
    async def list_specialist_containers(
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
    ) -> dict:
        """Lists all containers managed by Agent OS (filtered by managed-by=agent-os label)."""
        from container_manager_mcp.container_manager import create_manager

        try:
            manager = create_manager(manager_type, silent=False, log_file=None)
            all_containers = manager.list_containers(all=True)

            specialists = []
            for c in all_containers:
                labels = c.get("labels", c.get("Labels", {}))
                if isinstance(labels, dict) and labels.get("managed-by") == "agent-os":
                    specialists.append(
                        {
                            "id": c.get("id", c.get("Id", ""))[:12],
                            "name": c.get(
                                "name",
                                (
                                    c.get("Names", [""])[0]
                                    if isinstance(c.get("Names"), list)
                                    else ""
                                ),
                            ),
                            "image": c.get("image", c.get("Image", "")),
                            "status": c.get("status", c.get("Status", "")),
                            "labels": labels,
                        }
                    )

            return {"specialists": specialists, "count": len(specialists)}
        except Exception as e:
            logger.error(f"[Agent OS] Failed to list specialist containers: {e}")
            return {"success": False, "error": str(e)}

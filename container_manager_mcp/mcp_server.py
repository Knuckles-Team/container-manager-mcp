#!/usr/bin/python
import warnings

# Filter RequestsDependencyWarning early to prevent log spam
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from requests.exceptions import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except ImportError:
        pass

# General urllib3/chardet mismatch warnings
warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")

import logging
import os
import sys
from typing import Any

from agent_utilities.base_utilities import to_boolean
from agent_utilities.mcp_utilities import (
    create_mcp_server,
    ctx_confirm_destructive,
    ctx_log,
    ctx_progress,
)
from dotenv import find_dotenv, load_dotenv
from fastmcp import Context, FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from container_manager_mcp.container_manager import create_manager

__version__ = "1.11.0"

logger = get_logger(name="TokenMiddleware")
logger.setLevel(logging.DEBUG)


def parse_image_string(image: str, default_tag: str = "latest") -> tuple[str, str]:
    """
    Parse a container image string into image and tag components.

    Args:
        image: Input image string (e.g., 'registry.arpa/ubuntu/ubuntu:latest' or 'nginx')
        default_tag: Fallback tag if none is specified (default: 'latest')

    Returns:
        Tuple of (image, tag) where image includes registry/repository, tag is the tag or default_tag
    """
    if ":" in image:
        parts = image.rsplit(":", 1)
        image_name, tag = parts[0], parts[1]
        if "/" in tag or not tag:
            return image, default_tag
        return image_name, tag
    return image, default_tag


def register_misc_tools(mcp: FastMCP):
    pass
    pass


def register_info_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Get Version",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"info"},
    )
    async def get_version(
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Retrieves the version information of the container manager (Docker or Podman).
        Returns: A dictionary with keys like 'version', 'api_version', etc., detailing the manager's version.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Getting version for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.get_version()
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to get version: {str(e)}")
            raise RuntimeError(f"Failed to get version: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Get Info",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"info"},
    )
    async def get_info(
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Retrieves detailed information about the container manager system.
        Returns: A dictionary containing system info such as OS, architecture, storage driver, and more.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Getting info for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.get_info()
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to get info: {str(e)}")
            raise RuntimeError(f"Failed to get info: {str(e)}") from e


def register_image_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "List Images",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"image"},
    )
    async def list_images(
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[dict]:
        """
        Lists all container images available on the system.
        Returns: A list of dictionaries, each with image details like 'id', 'tags', 'created', 'size'.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Listing images for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.list_images()
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to list images: {str(e)}")
            raise RuntimeError(f"Failed to list images: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Pull Image",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"image"},
    )
    async def pull_image(
        image: str = Field(
            description="Image name to pull (e.g., nginx, registry.arpa/ubuntu/ubuntu:latest)."
        ),
        tag: str = Field(
            description="Image tag (overridden if tag is included in image string)",
            default="latest",
        ),
        platform: str | None = Field(
            description="Platform (e.g., linux/amd64)", default=None
        ),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Pulls a container image from a registry.
        Returns: A dictionary with the pull status, including 'id' of the pulled image and any error messages.
        """
        logger = logging.getLogger("ContainerManager")
        parsed_image, parsed_tag = parse_image_string(image, tag)
        logger.debug(
            f"Pulling image {parsed_image}:{parsed_tag} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.pull_image(parsed_image, parsed_tag, platform)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to pull image: {str(e)}")
            raise RuntimeError(f"Failed to pull image: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Remove Image",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"image"},
    )
    async def remove_image(
        image: str = Field(description="Image name or ID to remove"),
        force: bool = Field(description="Force removal", default=False),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Removes a specified container image.
        Returns: A dictionary indicating success or failure, with details like removed image ID.
        """
        if not await ctx_confirm_destructive(_ctx, "remove image"):
            return {"status": "cancelled", "message": "Operation cancelled by user"}
        await ctx_progress(_ctx, 0, 100)
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Removing image {image} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.remove_image(image, force)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to remove image: {str(e)}")
            raise RuntimeError(f"Failed to remove image: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Prune Images",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"image"},
    )
    async def prune_images(
        all: bool = Field(description="Prune all unused images", default=False),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Prunes unused container images.
        Returns: A dictionary with prune results, including space reclaimed and list of deleted images.
        """
        if not await ctx_confirm_destructive(_ctx, "prune images"):
            return {"status": "cancelled", "message": "Operation cancelled by user"}
        await ctx_progress(_ctx, 0, 100)
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Pruning images for {manager_type}, all: {all}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.prune_images(all=all)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to prune images: {str(e)}")
            raise RuntimeError(f"Failed to prune images: {str(e)}") from e


def register_container_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "List Containers",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"container"},
    )
    async def list_containers(
        all: bool = Field(
            description="Show all containers (default running only)", default=False
        ),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[dict]:
        """
        Lists containers on the system.
        Returns: A list of dictionaries, each with container details like 'id', 'name', 'status', 'image'.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Listing containers for {manager_type}, all: {all}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.list_containers(all)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to list containers: {str(e)}")
            raise RuntimeError(f"Failed to list containers: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Run Container",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"container"},
    )
    async def run_container(
        image: str = Field(description="Image to run"),
        name: str | None = Field(description="Container name", default=None),
        command: str | None = Field(
            description="Command to run in container", default=None
        ),
        detach: bool = Field(description="Run in detached mode", default=False),
        ports: dict[str, str] | None = Field(
            description="Port mappings {container_port: host_port}", default=None
        ),
        volumes: dict[str, dict] | None = Field(
            description="Volume mappings {/host/path: {bind: /container/path, mode: rw}}",
            default=None,
        ),
        environment: dict[str, str] | None = Field(
            description="Environment variables", default=None
        ),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Runs a new container from the specified image.
        Returns: A dictionary with the container's ID and status after starting.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Running container from {image} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.run_container(
                image, name, command, detach, ports, volumes, environment
            )
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to run container: {str(e)}")
            raise RuntimeError(f"Failed to run container: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Stop Container",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"container"},
    )
    async def stop_container(
        container_id: str = Field(description="Container ID or name"),
        timeout: int = Field(description="Timeout in seconds", default=10),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Stops a running container.
        Returns: A dictionary confirming the stop action, with container ID and any errors.
        """
        if not await ctx_confirm_destructive(_ctx, "stop container"):
            return {"status": "cancelled", "message": "Operation cancelled by user"}
        await ctx_progress(_ctx, 0, 100)
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Stopping container {container_id} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.stop_container(container_id, timeout)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to stop container: {str(e)}")
            raise RuntimeError(f"Failed to stop container: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Remove Container",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"container"},
    )
    async def remove_container(
        container_id: str = Field(description="Container ID or name"),
        force: bool = Field(description="Force removal", default=False),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Removes a container.
        Returns: A dictionary with removal status, including deleted container ID.
        """
        if not await ctx_confirm_destructive(_ctx, "remove container"):
            return {"status": "cancelled", "message": "Operation cancelled by user"}
        await ctx_progress(_ctx, 0, 100)
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Removing container {container_id} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.remove_container(container_id, force)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to remove container: {str(e)}")
            raise RuntimeError(f"Failed to remove container: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Prune Containers",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"container"},
    )
    async def prune_containers(
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Prunes stopped containers.
        Returns: A dictionary with prune results, including space reclaimed and deleted containers.
        """
        if not await ctx_confirm_destructive(_ctx, "prune containers"):
            return {"status": "cancelled", "message": "Operation cancelled by user"}
        await ctx_progress(_ctx, 0, 100)
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Pruning containers for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.prune_containers()
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to prune containers: {str(e)}")
            raise RuntimeError(f"Failed to prune containers: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Exec in Container",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"container"},
    )
    async def exec_in_container(
        container_id: str = Field(description="Container ID or name"),
        command: list[str] = Field(description="Command to execute"),
        detach: bool = Field(description="Detach execution", default=False),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Executes a command inside a running container.
        Returns: A dictionary with execution results, including 'exit_code' and 'output' as string.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Executing {command} in container {container_id} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.exec_in_container(container_id, command, detach)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to exec in container: {str(e)}")
            raise RuntimeError(f"Failed to exec in container: {str(e)}") from e


def register_log_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Get Container Logs",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"log", "debug", "container"},
    )
    async def get_container_logs(
        container_id: str = Field(description="Container ID or name"),
        tail: str = Field(
            description="Number of lines to show from the end (or 'all')", default="all"
        ),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """
        Retrieves logs from a container.
        Returns: A string containing the log output, parse as plain text lines.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Getting logs for container {container_id} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.get_container_logs(container_id, tail)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to get container logs: {str(e)}")
            raise RuntimeError(f"Failed to get container logs: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Compose Logs",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"log", "compose"},
    )
    async def compose_logs(
        compose_file: str = Field(description="Path to compose file"),
        service: str | None = Field(description="Specific service", default=None),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """
        Retrieves logs for services in a Docker Compose project.
        Returns: A string containing combined log output, prefixed by service names; parse as text lines.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Compose logs {compose_file} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.compose_logs(compose_file, service)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to compose logs: {str(e)}")
            raise RuntimeError(f"Failed to compose logs: {str(e)}") from e


def register_volume_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "List Volumes",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"volume"},
    )
    async def list_volumes(
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Lists all volumes.
        Returns: A dictionary with 'volumes' as a list of dicts containing name, driver, mountpoint, etc.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Listing volumes for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.list_volumes()
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to list volumes: {str(e)}")
            raise RuntimeError(f"Failed to list volumes: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Create Volume",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"volume"},
    )
    async def create_volume(
        name: str = Field(description="Volume name"),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Creates a new volume.
        Returns: A dictionary with details of the created volume, like 'name' and 'mountpoint'.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Creating volume {name} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.create_volume(name)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to create volume: {str(e)}")
            raise RuntimeError(f"Failed to create volume: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Remove Volume",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"volume"},
    )
    async def remove_volume(
        name: str = Field(description="Volume name"),
        force: bool = Field(description="Force removal", default=False),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Removes a volume.
        Returns: A dictionary confirming removal, with deleted volume name.
        """
        if not await ctx_confirm_destructive(_ctx, "remove volume"):
            return {"status": "cancelled", "message": "Operation cancelled by user"}
        await ctx_progress(_ctx, 0, 100)
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Removing volume {name} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.remove_volume(name, force)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to remove volume: {str(e)}")
            raise RuntimeError(f"Failed to remove volume: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Prune Volumes",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"volume"},
    )
    async def prune_volumes(
        all: bool = Field(description="Remove all volumes (dangerous)", default=False),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Prunes unused volumes.
        Returns: A dictionary with prune results, including space reclaimed and deleted volumes.
        """
        if not await ctx_confirm_destructive(_ctx, "prune volumes"):
            return {"status": "cancelled", "message": "Operation cancelled by user"}
        await ctx_progress(_ctx, 0, 100)
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Pruning volumes for {manager_type}, all: {all}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.prune_volumes(all=all)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to prune volumes: {str(e)}")
            raise RuntimeError(f"Failed to prune volumes: {str(e)}") from e


def register_network_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "List Networks",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"network"},
    )
    async def list_networks(
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[dict]:
        """
        Lists all networks.
        Returns: A list of dictionaries, each with network details like 'id', 'name', 'driver', 'scope'.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Listing networks for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.list_networks()
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to list networks: {str(e)}")
            raise RuntimeError(f"Failed to list networks: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Create Network",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"network"},
    )
    async def create_network(
        name: str = Field(description="Network name"),
        driver: str = Field(
            description="Network driver (e.g., bridge)", default="bridge"
        ),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Creates a new network.
        Returns: A dictionary with the created network's ID and details.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Creating network {name} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.create_network(name, driver)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to create network: {str(e)}")
            raise RuntimeError(f"Failed to create network: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Remove Network",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"network"},
    )
    async def remove_network(
        network_id: str = Field(description="Network ID or name"),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Removes a network.
        Returns: A dictionary confirming removal, with deleted network ID.
        """
        if not await ctx_confirm_destructive(_ctx, "remove network"):
            return {"status": "cancelled", "message": "Operation cancelled by user"}
        await ctx_progress(_ctx, 0, 100)
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Removing network {network_id} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.remove_network(network_id)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to remove network: {str(e)}")
            raise RuntimeError(f"Failed to remove network: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Prune Networks",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"network"},
    )
    async def prune_networks(
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Prunes unused networks.
        Returns: A dictionary with prune results, including deleted networks.
        """
        if not await ctx_confirm_destructive(_ctx, "prune networks"):
            return {"status": "cancelled", "message": "Operation cancelled by user"}
        await ctx_progress(_ctx, 0, 100)
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Pruning networks for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.prune_networks()
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to prune networks: {str(e)}")
            raise RuntimeError(f"Failed to prune networks: {str(e)}") from e


def register_system_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Prune System",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"system"},
    )
    async def prune_system(
        force: bool = Field(description="Force prune", default=False),
        all: bool = Field(description="Prune all unused resources", default=False),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Prunes all unused system resources (containers, images, volumes, networks).
        Returns: A dictionary summarizing the prune operation across resources.
        """
        if not await ctx_confirm_destructive(_ctx, "prune system"):
            return {"status": "cancelled", "message": "Operation cancelled by user"}
        await ctx_progress(_ctx, 0, 100)
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Pruning system for {manager_type}, force: {force}, all: {all}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.prune_system(force, all)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to prune system: {str(e)}")
            raise RuntimeError(f"Failed to prune system: {str(e)}") from e


def register_swarm_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Init Swarm",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"swarm"},
    )
    async def init_swarm(
        advertise_addr: str | None = Field(
            description="Advertise address", default=None
        ),
        manager_type: str | None = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Initializes a Docker Swarm cluster.
        Returns: A dictionary with swarm info, including join tokens for manager and worker.
        """
        if manager_type and manager_type != "docker":
            raise ValueError("Swarm operations are only supported on Docker")
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Initializing swarm for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.init_swarm(advertise_addr)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to init swarm: {str(e)}")
            raise RuntimeError(f"Failed to init swarm: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Leave Swarm",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"swarm"},
    )
    async def leave_swarm(
        force: bool = Field(description="Force leave", default=False),
        manager_type: str | None = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Leaves the Docker Swarm cluster.
        Returns: A dictionary confirming the leave action.
        """
        if manager_type and manager_type != "docker":
            raise ValueError("Swarm operations are only supported on Docker")
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Leaving swarm for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.leave_swarm(force)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to leave swarm: {str(e)}")
            raise RuntimeError(f"Failed to leave swarm: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "List Nodes",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"swarm"},
    )
    async def list_nodes(
        manager_type: str | None = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[dict]:
        """
        Lists nodes in the Docker Swarm cluster.
        Returns: A list of dictionaries, each with node details like 'id', 'hostname', 'status', 'role'.
        """
        if manager_type and manager_type != "docker":
            raise ValueError("Swarm operations are only supported on Docker")
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Listing nodes for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.list_nodes()
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to list nodes: {str(e)}")
            raise RuntimeError(f"Failed to list nodes: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "List Services",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"swarm"},
    )
    async def list_services(
        manager_type: str | None = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[dict]:
        """
        Lists services in the Docker Swarm.
        Returns: A list of dictionaries, each with service details like 'id', 'name', 'replicas', 'image'.
        """
        if manager_type and manager_type != "docker":
            raise ValueError("Swarm operations are only supported on Docker")
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Listing services for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.list_services()
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to list services: {str(e)}")
            raise RuntimeError(f"Failed to list services: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Create Service",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"swarm"},
    )
    async def create_service(
        name: str = Field(description="Service name"),
        image: str = Field(description="Image for the service"),
        replicas: int = Field(description="Number of replicas", default=1),
        ports: dict[str, str] | None = Field(
            description="Port mappings {target: published}", default=None
        ),
        mounts: list[str] | None = Field(
            description="Mounts [source:target:mode]", default=None
        ),
        manager_type: str | None = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Creates a new service in Docker Swarm.
        Returns: A dictionary with the created service's ID and details.
        """
        if manager_type and manager_type != "docker":
            raise ValueError("Swarm operations are only supported on Docker")
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Creating service {name} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.create_service(name, image, replicas, ports, mounts)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to create service: {str(e)}")
            raise RuntimeError(f"Failed to create service: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Remove Service",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"swarm"},
    )
    async def remove_service(
        service_id: str = Field(description="Service ID or name"),
        manager_type: str | None = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """
        Removes a service from Docker Swarm.
        Returns: A dictionary confirming the removal.
        """
        if not await ctx_confirm_destructive(_ctx, "remove service"):
            return {"status": "cancelled", "message": "Operation cancelled by user"}
        await ctx_progress(_ctx, 0, 100)
        if manager_type and manager_type != "docker":
            raise ValueError("Swarm operations are only supported on Docker")
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Removing service {service_id} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.remove_service(service_id)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to remove service: {str(e)}")
            raise RuntimeError(f"Failed to remove service: {str(e)}") from e


def register_compose_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Compose Up",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"compose"},
    )
    async def compose_up(
        compose_file: str = Field(description="Path to compose file"),
        detach: bool = Field(description="Detach mode", default=True),
        build: bool = Field(description="Build images", default=False),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """
        Starts services defined in a Docker Compose file.
        Returns: A string with the output of the compose up command, parse for status messages.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Compose up {compose_file} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.compose_up(compose_file, detach, build)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to compose up: {str(e)}")
            raise RuntimeError(f"Failed to compose up: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Compose Down",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"compose"},
    )
    async def compose_down(
        compose_file: str = Field(description="Path to compose file"),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """
        Stops and removes services from a Docker Compose file.
        Returns: A string with the output of the compose down command, parse for status messages.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Compose down {compose_file} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.compose_down(compose_file)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to compose down: {str(e)}")
            raise RuntimeError(f"Failed to compose down: {str(e)}") from e

    @mcp.tool(
        annotations={
            "title": "Compose Ps",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"compose"},
    )
    async def compose_ps(
        compose_file: str = Field(description="Path to compose file"),
        manager_type: str | None = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: bool | None = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: str | None = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        _ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """
        Lists containers for a Docker Compose project.
        Returns: A string in table format listing name, command, state, ports; parse as text table.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Compose ps {compose_file} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(
                manager_type, silent if silent is not None else False, log_file
            )
            return manager.compose_ps(compose_file)
        except Exception as e:
            ctx_log(_ctx, logger, "error", f"Failed to compose ps: {str(e)}")
            raise RuntimeError(f"Failed to compose ps: {str(e)}") from e


def register_prompts(mcp: FastMCP):
    print(f"mcp_server v{__version__}", file=sys.stderr)

    @mcp.prompt
    def get_logs(
        container: str,
    ) -> str:
        """
        Generates a prompt for getting the logs of a running container
        """
        return f"Get the logs for the following service: {container}"


def get_mcp_instance() -> tuple[Any, Any, Any, Any]:
    """Initialize and return the MCP instance, args, and middlewares."""
    load_dotenv(find_dotenv())

    args, mcp, middlewares = create_mcp_server(
        name="ContainerManager",
        version=__version__,
        instructions="Container Manager MCP Server - Manage Docker and Podman containers, images, volumes, networks, and swarm.",
    )

    DEFAULT_MISCTOOL = to_boolean(os.getenv("MISCTOOL", "True"))
    if DEFAULT_MISCTOOL:
        register_misc_tools(mcp)
    DEFAULT_INFOTOOL = to_boolean(os.getenv("INFOTOOL", "True"))
    if DEFAULT_INFOTOOL:
        register_info_tools(mcp)
    DEFAULT_IMAGETOOL = to_boolean(os.getenv("IMAGETOOL", "True"))
    if DEFAULT_IMAGETOOL:
        register_image_tools(mcp)
    DEFAULT_CONTAINERTOOL = to_boolean(os.getenv("CONTAINERTOOL", "True"))
    if DEFAULT_CONTAINERTOOL:
        register_container_tools(mcp)
    DEFAULT_LOGTOOL = to_boolean(os.getenv("LOGTOOL", "True"))
    if DEFAULT_LOGTOOL:
        register_log_tools(mcp)
    DEFAULT_VOLUMETOOL = to_boolean(os.getenv("VOLUMETOOL", "True"))
    if DEFAULT_VOLUMETOOL:
        register_volume_tools(mcp)
    DEFAULT_NETWORKTOOL = to_boolean(os.getenv("NETWORKTOOL", "True"))
    if DEFAULT_NETWORKTOOL:
        register_network_tools(mcp)
    DEFAULT_SYSTEMTOOL = to_boolean(os.getenv("SYSTEMTOOL", "True"))
    if DEFAULT_SYSTEMTOOL:
        register_system_tools(mcp)
    DEFAULT_SWARMTOOL = to_boolean(os.getenv("SWARMTOOL", "True"))
    if DEFAULT_SWARMTOOL:
        register_swarm_tools(mcp)
    DEFAULT_COMPOSETOOL = to_boolean(os.getenv("COMPOSETOOL", "True"))
    if DEFAULT_COMPOSETOOL:
        register_compose_tools(mcp)

    # ── Agent OS Specialist Deployment Tools ────────────────────────
    DEFAULT_SPECIALIST_TOOL = to_boolean(os.getenv("SPECIALIST_TOOL", "True"))
    if DEFAULT_SPECIALIST_TOOL:
        try:
            from container_manager_mcp.specialist_tools import (
                register_specialist_deployment_tools,
            )

            register_specialist_deployment_tools(mcp)
        except Exception as e:
            logger.warning("Specialist deployment tools not available: %s", e)

    register_prompts(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)

    registered_tags = []
    # FastMCP Typically stores tools in .get_tools() or ._tools
    tools_dict = (
        mcp._tools
        if hasattr(mcp, "_tools")
        else mcp.get_tools() if hasattr(mcp, "get_tools") else {}
    )
    for tool in tools_dict.values():
        if hasattr(tool, "tags") and tool.tags:
            registered_tags.extend(list(tool.tags))

    print(f"{'container-manager-mcp'} MCP v{__version__}", file=sys.stderr)
    print("\nStarting MCP Server", file=sys.stderr)
    print(f"  Transport: {args.transport.upper()}", file=sys.stderr)
    print(f"  Auth: {args.auth_type}", file=sys.stderr)
    print(f"  Dynamic Tags Loaded: {len(set(registered_tags))}", file=sys.stderr)

    return mcp, args, middlewares, registered_tags


def mcp_server() -> None:
    mcp, args, middlewares, registered_tags = get_mcp_instance()
    print(f"{'container-manager-mcp'} MCP v{__version__}", file=sys.stderr)
    print("\nStarting MCP Server", file=sys.stderr)
    print(f"  Transport: {args.transport.upper()}", file=sys.stderr)
    print(f"  Auth: {args.auth_type}", file=sys.stderr)
    print(f"  Dynamic Tags Loaded: {len(set(registered_tags))}", file=sys.stderr)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error("Invalid transport", extra={"transport": args.transport})
        sys.exit(1)


if __name__ == "__main__":
    mcp_server()

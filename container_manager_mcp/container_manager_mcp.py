#!/usr/bin/python
# coding: utf-8

import os
import argparse
import sys
import logging
from typing import Optional, List, Dict, Union

import requests
from pydantic import Field
from eunomia_mcp.middleware import EunomiaMcpMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastmcp import FastMCP, Context
from fastmcp.server.auth.oidc_proxy import OIDCProxy
from fastmcp.server.auth import OAuthProxy, RemoteAuthProvider
from fastmcp.server.auth.providers.jwt import JWTVerifier, StaticTokenVerifier
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.utilities.logging import get_logger
from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.utils import to_boolean, to_integer
from container_manager_mcp.middlewares import (
    UserTokenMiddleware,
    JWTClaimsLoggingMiddleware,
)

__version__ = "1.2.11"

logger = get_logger(name="TokenMiddleware")
logger.setLevel(logging.DEBUG)

config = {
    "enable_delegation": to_boolean(os.environ.get("ENABLE_DELEGATION", "False")),
    "audience": os.environ.get("AUDIENCE", None),
    "delegated_scopes": os.environ.get("DELEGATED_SCOPES", "api"),
    "token_endpoint": None,  # Will be fetched dynamically from OIDC config
    "oidc_client_id": os.environ.get("OIDC_CLIENT_ID", None),
    "oidc_client_secret": os.environ.get("OIDC_CLIENT_SECRET", None),
    "oidc_config_url": os.environ.get("OIDC_CONFIG_URL", None),
    "jwt_jwks_uri": os.getenv("FASTMCP_SERVER_AUTH_JWT_JWKS_URI", None),
    "jwt_issuer": os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER", None),
    "jwt_audience": os.getenv("FASTMCP_SERVER_AUTH_JWT_AUDIENCE", None),
    "jwt_algorithm": os.getenv("FASTMCP_SERVER_AUTH_JWT_ALGORITHM", None),
    "jwt_secret": os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY", None),
    "jwt_required_scopes": os.getenv("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES", None),
}

DEFAULT_TRANSPORT = os.getenv("TRANSPORT", "stdio")
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = to_integer(string=os.getenv("PORT", "8000"))


def parse_image_string(image: str, default_tag: str = "latest") -> tuple[str, str]:
    """
    Parse a container image string into image and tag components.

    Args:
        image: Input image string (e.g., 'registry.arpa/ubuntu/ubuntu:latest' or 'nginx')
        default_tag: Fallback tag if none is specified (default: 'latest')

    Returns:
        Tuple of (image, tag) where image includes registry/repository, tag is the tag or default_tag
    """
    # Split on the last ':' to separate image and tag
    if ":" in image:
        parts = image.rsplit(":", 1)
        image_name, tag = parts[0], parts[1]
        # Ensure tag is valid (not a port or malformed)
        if "/" in tag or not tag:
            # If tag contains '/' or is empty, assume no tag was provided
            return image, default_tag
        return image_name, tag
    return image, default_tag


def register_tools(mcp: FastMCP):
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})

    @mcp.tool(
        annotations={
            "title": "Get Version",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"container_manager_info"},
    )
    async def get_version(
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Retrieves the version information of the container manager (Docker or Podman).
        Returns: A dictionary with keys like 'version', 'api_version', etc., detailing the manager's version.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Getting version for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.get_version()
        except Exception as e:
            logger.error(f"Failed to get version: {str(e)}")
            raise RuntimeError(f"Failed to get version: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Get Info",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"container_manager_info"},
    )
    async def get_info(
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Retrieves detailed information about the container manager system.
        Returns: A dictionary containing system info such as OS, architecture, storage driver, and more.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Getting info for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.get_info()
        except Exception as e:
            logger.error(f"Failed to get info: {str(e)}")
            raise RuntimeError(f"Failed to get info: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "List Images",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"image_management"},
    )
    async def list_images(
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> List[Dict]:
        """
        Lists all container images available on the system.
        Returns: A list of dictionaries, each with image details like 'id', 'tags', 'created', 'size'.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Listing images for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.list_images()
        except Exception as e:
            logger.error(f"Failed to list images: {str(e)}")
            raise RuntimeError(f"Failed to list images: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Pull Image",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"image_management"},
    )
    async def pull_image(
        image: str = Field(
            description="Image name to pull (e.g., nginx, registry.arpa/ubuntu/ubuntu:latest)."
        ),
        tag: str = Field(
            description="Image tag (overridden if tag is included in image string)",
            default="latest",
        ),
        platform: Optional[str] = Field(
            description="Platform (e.g., linux/amd64)", default=None
        ),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Pulls a container image from a registry.
        Returns: A dictionary with the pull status, including 'id' of the pulled image and any error messages.
        """
        logger = logging.getLogger("ContainerManager")
        # Parse image string to separate image and tag
        parsed_image, parsed_tag = parse_image_string(image, tag)
        logger.debug(
            f"Pulling image {parsed_image}:{parsed_tag} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.pull_image(parsed_image, parsed_tag, platform)
        except Exception as e:
            logger.error(f"Failed to pull image: {str(e)}")
            raise RuntimeError(f"Failed to pull image: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Remove Image",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"image_management"},
    )
    async def remove_image(
        image: str = Field(description="Image name or ID to remove"),
        force: bool = Field(description="Force removal", default=False),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Removes a specified container image.
        Returns: A dictionary indicating success or failure, with details like removed image ID.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Removing image {image} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.remove_image(image, force)
        except Exception as e:
            logger.error(f"Failed to remove image: {str(e)}")
            raise RuntimeError(f"Failed to remove image: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Prune Images",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"image_management"},
    )
    async def prune_images(
        all: bool = Field(description="Prune all unused images", default=False),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Prunes unused container images.
        Returns: A dictionary with prune results, including space reclaimed and list of deleted images.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Pruning images for {manager_type}, all: {all}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.prune_images(all=all)
        except Exception as e:
            logger.error(f"Failed to prune images: {str(e)}")
            raise RuntimeError(f"Failed to prune images: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "List Containers",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"container_management"},
    )
    async def list_containers(
        all: bool = Field(
            description="Show all containers (default running only)", default=False
        ),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> List[Dict]:
        """
        Lists containers on the system.
        Returns: A list of dictionaries, each with container details like 'id', 'name', 'status', 'image'.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Listing containers for {manager_type}, all: {all}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.list_containers(all)
        except Exception as e:
            logger.error(f"Failed to list containers: {str(e)}")
            raise RuntimeError(f"Failed to list containers: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Run Container",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"container_management"},
    )
    async def run_container(
        image: str = Field(description="Image to run"),
        name: Optional[str] = Field(description="Container name", default=None),
        command: Optional[str] = Field(
            description="Command to run in container", default=None
        ),
        detach: bool = Field(description="Run in detached mode", default=False),
        ports: Optional[Dict[str, str]] = Field(
            description="Port mappings {container_port: host_port}", default=None
        ),
        volumes: Optional[Dict[str, Dict]] = Field(
            description="Volume mappings {/host/path: {bind: /container/path, mode: rw}}",
            default=None,
        ),
        environment: Optional[Dict[str, str]] = Field(
            description="Environment variables", default=None
        ),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Runs a new container from the specified image.
        Returns: A dictionary with the container's ID and status after starting.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Running container from {image} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.run_container(
                image, name, command, detach, ports, volumes, environment
            )
        except Exception as e:
            logger.error(f"Failed to run container: {str(e)}")
            raise RuntimeError(f"Failed to run container: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Stop Container",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"container_management"},
    )
    async def stop_container(
        container_id: str = Field(description="Container ID or name"),
        timeout: int = Field(description="Timeout in seconds", default=10),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Stops a running container.
        Returns: A dictionary confirming the stop action, with container ID and any errors.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Stopping container {container_id} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.stop_container(container_id, timeout)
        except Exception as e:
            logger.error(f"Failed to stop container: {str(e)}")
            raise RuntimeError(f"Failed to stop container: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Remove Container",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"container_management"},
    )
    async def remove_container(
        container_id: str = Field(description="Container ID or name"),
        force: bool = Field(description="Force removal", default=False),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Removes a container.
        Returns: A dictionary with removal status, including deleted container ID.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Removing container {container_id} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.remove_container(container_id, force)
        except Exception as e:
            logger.error(f"Failed to remove container: {str(e)}")
            raise RuntimeError(f"Failed to remove container: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Prune Containers",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"container_management"},
    )
    async def prune_containers(
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Prunes stopped containers.
        Returns: A dictionary with prune results, including space reclaimed and deleted containers.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Pruning containers for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.prune_containers()
        except Exception as e:
            logger.error(f"Failed to prune containers: {str(e)}")
            raise RuntimeError(f"Failed to prune containers: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Get Container Logs",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"log_management", "debug", "container_management"},
    )
    async def get_container_logs(
        container_id: str = Field(description="Container ID or name"),
        tail: str = Field(
            description="Number of lines to show from the end (or 'all')", default="all"
        ),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
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
            manager = create_manager(manager_type, silent, log_file)
            return manager.get_container_logs(container_id, tail)
        except Exception as e:
            logger.error(f"Failed to get container logs: {str(e)}")
            raise RuntimeError(f"Failed to get container logs: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Exec in Container",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"container_management"},
    )
    async def exec_in_container(
        container_id: str = Field(description="Container ID or name"),
        command: List[str] = Field(description="Command to execute"),
        detach: bool = Field(description="Detach execution", default=False),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Executes a command inside a running container.
        Returns: A dictionary with execution results, including 'exit_code' and 'output' as string.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Executing {command} in container {container_id} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.exec_in_container(container_id, command, detach)
        except Exception as e:
            logger.error(f"Failed to exec in container: {str(e)}")
            raise RuntimeError(f"Failed to exec in container: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "List Volumes",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"volume_management"},
    )
    async def list_volumes(
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Lists all volumes.
        Returns: A dictionary with 'volumes' as a list of dicts containing name, driver, mountpoint, etc.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Listing volumes for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.list_volumes()
        except Exception as e:
            logger.error(f"Failed to list volumes: {str(e)}")
            raise RuntimeError(f"Failed to list volumes: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Create Volume",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"volume_management"},
    )
    async def create_volume(
        name: str = Field(description="Volume name"),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Creates a new volume.
        Returns: A dictionary with details of the created volume, like 'name' and 'mountpoint'.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Creating volume {name} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.create_volume(name)
        except Exception as e:
            logger.error(f"Failed to create volume: {str(e)}")
            raise RuntimeError(f"Failed to create volume: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Remove Volume",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"volume_management"},
    )
    async def remove_volume(
        name: str = Field(description="Volume name"),
        force: bool = Field(description="Force removal", default=False),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Removes a volume.
        Returns: A dictionary confirming removal, with deleted volume name.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Removing volume {name} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.remove_volume(name, force)
        except Exception as e:
            logger.error(f"Failed to remove volume: {str(e)}")
            raise RuntimeError(f"Failed to remove volume: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Prune Volumes",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"volume_management"},
    )
    async def prune_volumes(
        all: bool = Field(description="Remove all volumes (dangerous)", default=False),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Prunes unused volumes.
        Returns: A dictionary with prune results, including space reclaimed and deleted volumes.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Pruning volumes for {manager_type}, all: {all}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.prune_volumes(all=all)
        except Exception as e:
            logger.error(f"Failed to prune volumes: {str(e)}")
            raise RuntimeError(f"Failed to prune volumes: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "List Networks",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"network_management"},
    )
    async def list_networks(
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> List[Dict]:
        """
        Lists all networks.
        Returns: A list of dictionaries, each with network details like 'id', 'name', 'driver', 'scope'.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Listing networks for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.list_networks()
        except Exception as e:
            logger.error(f"Failed to list networks: {str(e)}")
            raise RuntimeError(f"Failed to list networks: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Create Network",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"network_management"},
    )
    async def create_network(
        name: str = Field(description="Network name"),
        driver: str = Field(
            description="Network driver (e.g., bridge)", default="bridge"
        ),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Creates a new network.
        Returns: A dictionary with the created network's ID and details.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Creating network {name} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.create_network(name, driver)
        except Exception as e:
            logger.error(f"Failed to create network: {str(e)}")
            raise RuntimeError(f"Failed to create network: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Remove Network",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"network_management"},
    )
    async def remove_network(
        network_id: str = Field(description="Network ID or name"),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Removes a network.
        Returns: A dictionary confirming removal, with deleted network ID.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Removing network {network_id} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.remove_network(network_id)
        except Exception as e:
            logger.error(f"Failed to remove network: {str(e)}")
            raise RuntimeError(f"Failed to remove network: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Prune Networks",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"network_management"},
    )
    async def prune_networks(
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Prunes unused networks.
        Returns: A dictionary with prune results, including deleted networks.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Pruning networks for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.prune_networks()
        except Exception as e:
            logger.error(f"Failed to prune networks: {str(e)}")
            raise RuntimeError(f"Failed to prune networks: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Prune System",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"system_management"},
    )
    async def prune_system(
        force: bool = Field(description="Force prune", default=False),
        all: bool = Field(description="Prune all unused resources", default=False),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Prunes all unused system resources (containers, images, volumes, networks).
        Returns: A dictionary summarizing the prune operation across resources.
        """
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Pruning system for {manager_type}, force: {force}, all: {all}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.prune_system(force, all)
        except Exception as e:
            logger.error(f"Failed to prune system: {str(e)}")
            raise RuntimeError(f"Failed to prune system: {str(e)}")

    # Swarm-specific tools

    @mcp.tool(
        annotations={
            "title": "Init Swarm",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"swarm_management", "swarm"},
    )
    async def init_swarm(
        advertise_addr: Optional[str] = Field(
            description="Advertise address", default=None
        ),
        manager_type: Optional[str] = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
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
            manager = create_manager(manager_type, silent, log_file)
            return manager.init_swarm(advertise_addr)
        except Exception as e:
            logger.error(f"Failed to init swarm: {str(e)}")
            raise RuntimeError(f"Failed to init swarm: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Leave Swarm",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"swarm_management", "swarm"},
    )
    async def leave_swarm(
        force: bool = Field(description="Force leave", default=False),
        manager_type: Optional[str] = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
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
            manager = create_manager(manager_type, silent, log_file)
            return manager.leave_swarm(force)
        except Exception as e:
            logger.error(f"Failed to leave swarm: {str(e)}")
            raise RuntimeError(f"Failed to leave swarm: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "List Nodes",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"swarm_management", "swarm"},
    )
    async def list_nodes(
        manager_type: Optional[str] = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> List[Dict]:
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
            manager = create_manager(manager_type, silent, log_file)
            return manager.list_nodes()
        except Exception as e:
            logger.error(f"Failed to list nodes: {str(e)}")
            raise RuntimeError(f"Failed to list nodes: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "List Services",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"swarm_management", "swarm"},
    )
    async def list_services(
        manager_type: Optional[str] = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> List[Dict]:
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
            manager = create_manager(manager_type, silent, log_file)
            return manager.list_services()
        except Exception as e:
            logger.error(f"Failed to list services: {str(e)}")
            raise RuntimeError(f"Failed to list services: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Create Service",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"swarm_management", "swarm"},
    )
    async def create_service(
        name: str = Field(description="Service name"),
        image: str = Field(description="Image for the service"),
        replicas: int = Field(description="Number of replicas", default=1),
        ports: Optional[Dict[str, str]] = Field(
            description="Port mappings {target: published}", default=None
        ),
        mounts: Optional[List[str]] = Field(
            description="Mounts [source:target:mode]", default=None
        ),
        manager_type: Optional[str] = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
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
            manager = create_manager(manager_type, silent, log_file)
            return manager.create_service(name, image, replicas, ports, mounts)
        except Exception as e:
            logger.error(f"Failed to create service: {str(e)}")
            raise RuntimeError(f"Failed to create service: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Remove Service",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"swarm_management", "swarm"},
    )
    async def remove_service(
        service_id: str = Field(description="Service ID or name"),
        manager_type: Optional[str] = Field(
            description="Container manager: must be docker for swarm (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """
        Removes a service from Docker Swarm.
        Returns: A dictionary confirming the removal.
        """
        if manager_type and manager_type != "docker":
            raise ValueError("Swarm operations are only supported on Docker")
        logger = logging.getLogger("ContainerManager")
        logger.debug(
            f"Removing service {service_id} for {manager_type}, silent: {silent}, log_file: {log_file}"
        )
        try:
            manager = create_manager(manager_type, silent, log_file)
            return manager.remove_service(service_id)
        except Exception as e:
            logger.error(f"Failed to remove service: {str(e)}")
            raise RuntimeError(f"Failed to remove service: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Compose Up",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"compose_management", "compose"},
    )
    async def compose_up(
        compose_file: str = Field(description="Path to compose file"),
        detach: bool = Field(description="Detach mode", default=True),
        build: bool = Field(description="Build images", default=False),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
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
            manager = create_manager(manager_type, silent, log_file)
            return manager.compose_up(compose_file, detach, build)
        except Exception as e:
            logger.error(f"Failed to compose up: {str(e)}")
            raise RuntimeError(f"Failed to compose up: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Compose Down",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"compose_management", "compose"},
    )
    async def compose_down(
        compose_file: str = Field(description="Path to compose file"),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
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
            manager = create_manager(manager_type, silent, log_file)
            return manager.compose_down(compose_file)
        except Exception as e:
            logger.error(f"Failed to compose down: {str(e)}")
            raise RuntimeError(f"Failed to compose down: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Compose Ps",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"compose_management", "compose"},
    )
    async def compose_ps(
        compose_file: str = Field(description="Path to compose file"),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
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
            manager = create_manager(manager_type, silent, log_file)
            return manager.compose_ps(compose_file)
        except Exception as e:
            logger.error(f"Failed to compose ps: {str(e)}")
            raise RuntimeError(f"Failed to compose ps: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Compose Logs",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"log_management", "compose", "compose_management"},
    )
    async def compose_logs(
        compose_file: str = Field(description="Path to compose file"),
        service: Optional[str] = Field(description="Specific service", default=None),
        manager_type: Optional[str] = Field(
            description="Container manager: docker, podman (default: auto-detect)",
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
        ),
        silent: Optional[bool] = Field(
            description="Suppress output",
            default=to_boolean(os.environ.get("CONTAINER_MANAGER_SILENT", False)),
        ),
        log_file: Optional[str] = Field(
            description="Path to log file",
            default=os.environ.get("CONTAINER_MANAGER_LOG_FILE", None),
        ),
        ctx: Context = Field(
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
            manager = create_manager(manager_type, silent, log_file)
            return manager.compose_logs(compose_file, service)
        except Exception as e:
            logger.error(f"Failed to compose logs: {str(e)}")
            raise RuntimeError(f"Failed to compose logs: {str(e)}")


def register_prompts(mcp: FastMCP):
    print(f"container_manager_mcp v{__version__}")

    # Prompts
    @mcp.prompt
    def get_logs(
        container: str,
    ) -> str:
        """
        Generates a prompt for getting the logs of a running container
        """
        return f"Get the logs for the following service: {container}"


def container_manager_mcp():
    parser = argparse.ArgumentParser(description="Container Manager MCP Server")
    parser.add_argument(
        "-t",
        "--transport",
        default=DEFAULT_TRANSPORT,
        choices=["stdio", "streamable-http", "sse"],
        help="Transport method: 'stdio', 'streamable-http', or 'sse' [legacy] (default: stdio)",
    )
    parser.add_argument(
        "-s",
        "--host",
        default=DEFAULT_HOST,
        help="Host address for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port number for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--auth-type",
        default="none",
        choices=["none", "static", "jwt", "oauth-proxy", "oidc-proxy", "remote-oauth"],
        help="Authentication type for MCP server: 'none' (disabled), 'static' (internal), 'jwt' (external token verification), 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (external) (default: none)",
    )
    # JWT/Token params
    parser.add_argument(
        "--token-jwks-uri", default=None, help="JWKS URI for JWT verification"
    )
    parser.add_argument(
        "--token-issuer", default=None, help="Issuer for JWT verification"
    )
    parser.add_argument(
        "--token-audience", default=None, help="Audience for JWT verification"
    )
    parser.add_argument(
        "--token-algorithm",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_ALGORITHM"),
        choices=[
            "HS256",
            "HS384",
            "HS512",
            "RS256",
            "RS384",
            "RS512",
            "ES256",
            "ES384",
            "ES512",
        ],
        help="JWT signing algorithm (required for HMAC or static key). Auto-detected for JWKS.",
    )
    parser.add_argument(
        "--token-secret",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Shared secret for HMAC (HS*) or PEM public key for static asymmetric verification.",
    )
    parser.add_argument(
        "--token-public-key",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Path to PEM public key file or inline PEM string (for static asymmetric keys).",
    )
    parser.add_argument(
        "--required-scopes",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES"),
        help="Comma-separated list of required scopes (e.g., containermanager.read,containermanager.write).",
    )
    # OAuth Proxy params
    parser.add_argument(
        "--oauth-upstream-auth-endpoint",
        default=None,
        help="Upstream authorization endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-token-endpoint",
        default=None,
        help="Upstream token endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-id",
        default=None,
        help="Upstream client ID for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-secret",
        default=None,
        help="Upstream client secret for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-base-url", default=None, help="Base URL for OAuth Proxy"
    )
    # OIDC Proxy params
    parser.add_argument(
        "--oidc-config-url", default=None, help="OIDC configuration URL"
    )
    parser.add_argument("--oidc-client-id", default=None, help="OIDC client ID")
    parser.add_argument("--oidc-client-secret", default=None, help="OIDC client secret")
    parser.add_argument("--oidc-base-url", default=None, help="Base URL for OIDC Proxy")
    # Remote OAuth params
    parser.add_argument(
        "--remote-auth-servers",
        default=None,
        help="Comma-separated list of authorization servers for Remote OAuth",
    )
    parser.add_argument(
        "--remote-base-url", default=None, help="Base URL for Remote OAuth"
    )
    # Common
    parser.add_argument(
        "--allowed-client-redirect-uris",
        default=None,
        help="Comma-separated list of allowed client redirect URIs",
    )
    # Eunomia params
    parser.add_argument(
        "--eunomia-type",
        default="none",
        choices=["none", "embedded", "remote"],
        help="Eunomia authorization type: 'none' (disabled), 'embedded' (built-in), 'remote' (external) (default: none)",
    )
    parser.add_argument(
        "--eunomia-policy-file",
        default="mcp_policies.json",
        help="Policy file for embedded Eunomia (default: mcp_policies.json)",
    )
    parser.add_argument(
        "--eunomia-remote-url", default=None, help="URL for remote Eunomia server"
    )
    # Delegation params
    parser.add_argument(
        "--enable-delegation",
        action="store_true",
        default=to_boolean(os.environ.get("ENABLE_DELEGATION", "False")),
        help="Enable OIDC token delegation",
    )
    parser.add_argument(
        "--audience",
        default=os.environ.get("AUDIENCE", None),
        help="Audience for the delegated token",
    )
    parser.add_argument(
        "--delegated-scopes",
        default=os.environ.get("DELEGATED_SCOPES", "api"),
        help="Scopes for the delegated token (space-separated)",
    )
    parser.add_argument(
        "--openapi-file",
        default=None,
        help="Path to the OpenAPI JSON file to import additional tools from",
    )
    parser.add_argument(
        "--openapi-base-url",
        default=None,
        help="Base URL for the OpenAPI client (overrides instance URL)",
    )
    parser.add_argument(
        "--openapi-use-token",
        action="store_true",
        help="Use the incoming Bearer token (from MCP request) to authenticate OpenAPI import",
    )

    parser.add_argument(
        "--openapi-username",
        default=os.getenv("OPENAPI_USERNAME"),
        help="Username for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-password",
        default=os.getenv("OPENAPI_PASSWORD"),
        help="Password for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-id",
        default=os.getenv("OPENAPI_CLIENT_ID"),
        help="OAuth client ID for OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-secret",
        default=os.getenv("OPENAPI_CLIENT_SECRET"),
        help="OAuth client secret for OpenAPI import",
    )

    args = parser.parse_args()

    if args.port < 0 or args.port > 65535:
        print(f"Error: Port {args.port} is out of valid range (0-65535).")
        sys.exit(1)

    # Update config with CLI arguments
    config["enable_delegation"] = args.enable_delegation
    config["audience"] = args.audience or config["audience"]
    config["delegated_scopes"] = args.delegated_scopes or config["delegated_scopes"]
    config["oidc_config_url"] = args.oidc_config_url or config["oidc_config_url"]
    config["oidc_client_id"] = args.oidc_client_id or config["oidc_client_id"]
    config["oidc_client_secret"] = (
        args.oidc_client_secret or config["oidc_client_secret"]
    )

    # Configure delegation if enabled
    if config["enable_delegation"]:
        if args.auth_type != "oidc-proxy":
            logger.error("Token delegation requires auth-type=oidc-proxy")
            sys.exit(1)
        if not config["audience"]:
            logger.error("audience is required for delegation")
            sys.exit(1)
        if not all(
            [
                config["oidc_config_url"],
                config["oidc_client_id"],
                config["oidc_client_secret"],
            ]
        ):
            logger.error(
                "Delegation requires complete OIDC configuration (oidc-config-url, oidc-client-id, oidc-client-secret)"
            )
            sys.exit(1)

        # Fetch OIDC configuration to get token_endpoint
        try:
            logger.info(
                "Fetching OIDC configuration",
                extra={"oidc_config_url": config["oidc_config_url"]},
            )
            oidc_config_resp = requests.get(config["oidc_config_url"])
            oidc_config_resp.raise_for_status()
            oidc_config = oidc_config_resp.json()
            config["token_endpoint"] = oidc_config.get("token_endpoint")
            if not config["token_endpoint"]:
                logger.error("No token_endpoint found in OIDC configuration")
                raise ValueError("No token_endpoint found in OIDC configuration")
            logger.info(
                "OIDC configuration fetched successfully",
                extra={"token_endpoint": config["token_endpoint"]},
            )
        except Exception as e:
            print(f"Failed to fetch OIDC configuration: {e}")
            logger.error(
                "Failed to fetch OIDC configuration",
                extra={"error_type": type(e).__name__, "error_message": str(e)},
            )
            sys.exit(1)

    # Set auth based on type
    auth = None
    allowed_uris = (
        args.allowed_client_redirect_uris.split(",")
        if args.allowed_client_redirect_uris
        else None
    )

    if args.auth_type == "none":
        auth = None
    elif args.auth_type == "static":
        auth = StaticTokenVerifier(
            tokens={
                "test-token": {"client_id": "test-user", "scopes": ["read", "write"]},
                "admin-token": {"client_id": "admin", "scopes": ["admin"]},
            }
        )
    elif args.auth_type == "jwt":
        # Fallback to env vars if not provided via CLI
        jwks_uri = args.token_jwks_uri or os.getenv("FASTMCP_SERVER_AUTH_JWT_JWKS_URI")
        issuer = args.token_issuer or os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER")
        audience = args.token_audience or os.getenv("FASTMCP_SERVER_AUTH_JWT_AUDIENCE")
        algorithm = args.token_algorithm
        secret_or_key = args.token_secret or args.token_public_key
        public_key_pem = None

        if not (jwks_uri or secret_or_key):
            logger.error(
                "JWT auth requires either --token-jwks-uri or --token-secret/--token-public-key"
            )
            sys.exit(1)
        if not (issuer and audience):
            logger.error("JWT requires --token-issuer and --token-audience")
            sys.exit(1)

        # Load static public key from file if path is given
        if args.token_public_key and os.path.isfile(args.token_public_key):
            try:
                with open(args.token_public_key, "r") as f:
                    public_key_pem = f.read()
                logger.info(f"Loaded static public key from {args.token_public_key}")
            except Exception as e:
                print(f"Failed to read public key file: {e}")
                logger.error(f"Failed to read public key file: {e}")
                sys.exit(1)
        elif args.token_public_key:
            public_key_pem = args.token_public_key  # Inline PEM

        # Validation: Conflicting options
        if jwks_uri and (algorithm or secret_or_key):
            logger.warning(
                "JWKS mode ignores --token-algorithm and --token-secret/--token-public-key"
            )

        # HMAC mode
        if algorithm and algorithm.startswith("HS"):
            if not secret_or_key:
                logger.error(f"HMAC algorithm {algorithm} requires --token-secret")
                sys.exit(1)
            if jwks_uri:
                logger.error("Cannot use --token-jwks-uri with HMAC")
                sys.exit(1)
            public_key = secret_or_key
        else:
            public_key = public_key_pem

        # Required scopes
        required_scopes = None
        if args.required_scopes:
            required_scopes = [
                s.strip() for s in args.required_scopes.split(",") if s.strip()
            ]

        try:
            auth = JWTVerifier(
                jwks_uri=jwks_uri,
                public_key=public_key,
                issuer=issuer,
                audience=audience,
                algorithm=(
                    algorithm if algorithm and algorithm.startswith("HS") else None
                ),
                required_scopes=required_scopes,
            )
            logger.info(
                "JWTVerifier configured",
                extra={
                    "mode": (
                        "JWKS"
                        if jwks_uri
                        else (
                            "HMAC"
                            if algorithm and algorithm.startswith("HS")
                            else "Static Key"
                        )
                    ),
                    "algorithm": algorithm,
                    "required_scopes": required_scopes,
                },
            )
        except Exception as e:
            print(f"Failed to initialize JWTVerifier: {e}")
            logger.error(f"Failed to initialize JWTVerifier: {e}")
            sys.exit(1)
    elif args.auth_type == "oauth-proxy":
        if not (
            args.oauth_upstream_auth_endpoint
            and args.oauth_upstream_token_endpoint
            and args.oauth_upstream_client_id
            and args.oauth_upstream_client_secret
            and args.oauth_base_url
            and args.token_jwks_uri
            and args.token_issuer
            and args.token_audience
        ):
            print(
                "oauth-proxy requires oauth-upstream-auth-endpoint, oauth-upstream-token-endpoint, "
                "oauth-upstream-client-id, oauth-upstream-client-secret, oauth-base-url, token-jwks-uri, "
                "token-issuer, token-audience"
            )
            logger.error(
                "oauth-proxy requires oauth-upstream-auth-endpoint, oauth-upstream-token-endpoint, "
                "oauth-upstream-client-id, oauth-upstream-client-secret, oauth-base-url, token-jwks-uri, "
                "token-issuer, token-audience",
                extra={
                    "auth_endpoint": args.oauth_upstream_auth_endpoint,
                    "token_endpoint": args.oauth_upstream_token_endpoint,
                    "client_id": args.oauth_upstream_client_id,
                    "base_url": args.oauth_base_url,
                    "jwks_uri": args.token_jwks_uri,
                    "issuer": args.token_issuer,
                    "audience": args.token_audience,
                },
            )
            sys.exit(1)
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        auth = OAuthProxy(
            upstream_authorization_endpoint=args.oauth_upstream_auth_endpoint,
            upstream_token_endpoint=args.oauth_upstream_token_endpoint,
            upstream_client_id=args.oauth_upstream_client_id,
            upstream_client_secret=args.oauth_upstream_client_secret,
            token_verifier=token_verifier,
            base_url=args.oauth_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "oidc-proxy":
        if not (
            args.oidc_config_url
            and args.oidc_client_id
            and args.oidc_client_secret
            and args.oidc_base_url
        ):
            logger.error(
                "oidc-proxy requires oidc-config-url, oidc-client-id, oidc-client-secret, oidc-base-url",
                extra={
                    "config_url": args.oidc_config_url,
                    "client_id": args.oidc_client_id,
                    "base_url": args.oidc_base_url,
                },
            )
            sys.exit(1)
        auth = OIDCProxy(
            config_url=args.oidc_config_url,
            client_id=args.oidc_client_id,
            client_secret=args.oidc_client_secret,
            base_url=args.oidc_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "remote-oauth":
        if not (
            args.remote_auth_servers
            and args.remote_base_url
            and args.token_jwks_uri
            and args.token_issuer
            and args.token_audience
        ):
            logger.error(
                "remote-oauth requires remote-auth-servers, remote-base-url, token-jwks-uri, token-issuer, token-audience",
                extra={
                    "auth_servers": args.remote_auth_servers,
                    "base_url": args.remote_base_url,
                    "jwks_uri": args.token_jwks_uri,
                    "issuer": args.token_issuer,
                    "audience": args.token_audience,
                },
            )
            sys.exit(1)
        auth_servers = [url.strip() for url in args.remote_auth_servers.split(",")]
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        auth = RemoteAuthProvider(
            token_verifier=token_verifier,
            authorization_servers=auth_servers,
            base_url=args.remote_base_url,
        )

    # === 2. Build Middleware List ===
    middlewares: List[
        Union[
            UserTokenMiddleware,
            ErrorHandlingMiddleware,
            RateLimitingMiddleware,
            TimingMiddleware,
            LoggingMiddleware,
            JWTClaimsLoggingMiddleware,
            EunomiaMcpMiddleware,
        ]
    ] = [
        ErrorHandlingMiddleware(include_traceback=True, transform_errors=True),
        RateLimitingMiddleware(max_requests_per_second=10.0, burst_capacity=20),
        TimingMiddleware(),
        LoggingMiddleware(),
        JWTClaimsLoggingMiddleware(),
    ]
    if config["enable_delegation"] or args.auth_type == "jwt":
        middlewares.insert(0, UserTokenMiddleware(config=config))  # Must be first

    if args.eunomia_type in ["embedded", "remote"]:
        try:
            from eunomia_mcp import create_eunomia_middleware

            policy_file = args.eunomia_policy_file or "mcp_policies.json"
            eunomia_endpoint = (
                args.eunomia_remote_url if args.eunomia_type == "remote" else None
            )
            eunomia_mw = create_eunomia_middleware(
                policy_file=policy_file, eunomia_endpoint=eunomia_endpoint
            )
            middlewares.append(eunomia_mw)
            logger.info(f"Eunomia middleware enabled ({args.eunomia_type})")
        except Exception as e:
            print(f"Failed to load Eunomia middleware: {e}")
            logger.error("Failed to load Eunomia middleware", extra={"error": str(e)})
            sys.exit(1)

    mcp = FastMCP("ContainerManager", auth=auth)
    register_tools(mcp)
    register_prompts(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)

    print(f"Container Manager MCP v{__version__}")
    print("\nStarting Container Manager MCP Server")
    print(f"  Transport: {args.transport.upper()}")
    print(f"  Auth: {args.auth_type}")
    print(f"  Delegation: {'ON' if config['enable_delegation'] else 'OFF'}")
    print(f"  Eunomia: {args.eunomia_type}")

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
    container_manager_mcp()

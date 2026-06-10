#!/usr/bin/python
import warnings

from fastmcp import Context, FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field

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

# Filter AuthlibDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="authlib.*")

import logging
import os
from typing import Any, Literal

from agent_utilities.base_utilities import to_boolean
from agent_utilities.mcp_utilities import create_mcp_server

# Resilient context helpers to handle environment-specific import issues
try:
    from agent_utilities.mcp.context_helpers import ctx_progress as _ctx_progress

    async def ctx_progress(ctx: Any, progress: int, total: int = 100) -> None:
        if ctx:
            try:
                await _ctx_progress(ctx, progress, total)
            except Exception:
                if hasattr(ctx, "report_progress"):
                    await ctx.report_progress(progress=progress, total=total)

except ImportError:

    async def ctx_progress(ctx: Any, progress: int, total: int = 100) -> None:
        if ctx and hasattr(ctx, "report_progress"):
            try:
                await ctx.report_progress(progress=progress, total=total)
            except Exception:
                pass


try:
    from agent_utilities.mcp.context_helpers import (
        ctx_confirm_destructive as _ctx_confirm,
    )

    async def ctx_confirm_destructive(ctx: Any, action_description: str) -> bool:
        try:
            return await _ctx_confirm(ctx, action_description)
        except Exception:
            return True

except ImportError:

    async def ctx_confirm_destructive(ctx: Any, action_description: str) -> bool:
        if not ctx:
            return True
        try:
            result = await ctx.elicit(
                f"⚠️ Are you sure you want to {action_description}?",
                response_type=bool,
            )
            return result.action == "accept" and bool(result.data)
        except Exception:
            return True


def ctx_log(ctx: Any, *args, **kwargs) -> None:
    try:
        from agent_utilities.mcp.context_helpers import ctx_log as _real_ctx_log
    except ImportError:
        _real_ctx_log = None

    if len(args) == 2:
        level, message = args
        if isinstance(level, int):
            level_map = {
                logging.DEBUG: "debug",
                logging.INFO: "info",
                logging.WARNING: "warning",
                logging.ERROR: "error",
                logging.CRITICAL: "error",
            }
            level_str = level_map.get(level, "info")
        else:
            level_str = str(level).lower()

        log_fn = getattr(logger, level_str, None) or getattr(logger, "info", None)
        if log_fn:
            log_fn(message)
        if ctx:
            client_fn = getattr(ctx, level_str, None) or getattr(ctx, "info", None)
            if client_fn:
                try:
                    import asyncio
                    import inspect

                    res = client_fn(message)
                    if inspect.iscoroutine(res):
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(res)
                        except RuntimeError:
                            pass
                except Exception:
                    pass
    elif len(args) == 3:
        server_logger, level, message = args
        level_str = str(level).lower()
        log_fn = getattr(server_logger, level_str, None) or getattr(
            server_logger, "info", None
        )
        if log_fn:
            log_fn(message)
        if ctx:
            client_fn = getattr(ctx, level_str, None) or getattr(ctx, "info", None)
            if client_fn:
                try:
                    import asyncio
                    import inspect

                    res = client_fn(message)
                    if inspect.iscoroutine(res):
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(res)
                        except RuntimeError:
                            pass
                except Exception:
                    pass
    else:
        if _real_ctx_log:
            try:
                _real_ctx_log(ctx, *args, **kwargs)
            except Exception:
                pass


from dotenv import find_dotenv, load_dotenv

from container_manager_mcp.container_manager import create_manager

__version__ = "1.41.0"

logger = get_logger(name="ContainerManagerServer")
logger.setLevel(logging.DEBUG)


def register_info_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Info Operations",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"info"},
    )
    async def cm_info_operations(
        action: str = Field(
            description="Action to perform. Must be one of: 'get_version', 'get_info'"
        ),
        host: str | None = Field(
            default=None,
            description="Host alias defined in inventory.yaml (default: local host)",
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str:
        """
        Manage container manager info operations.
        """
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_info_operations: {action}")

        try:
            if action == "get_version":
                return manager.get_version()
            elif action == "get_info":
                return manager.get_info()
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            return f"Error executing {action}: {e}"


def register_image_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Image Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"image"},
    )
    async def cm_image_operations(
        action: Literal[
            "list_images", "pull_image", "remove_image", "prune_images"
        ] = Field(
            description="Action to perform. Must be one of: 'list_images', 'pull_image', 'remove_image', 'prune_images'"
        ),
        image: str | None = Field(default=None, description="Image name"),
        tag: str = Field(default="latest", description="Image tag"),
        platform: str | None = Field(
            default=None, description="Platform (e.g., linux/amd64)"
        ),
        force: bool = Field(default=False, description="Force operation"),
        host: str | None = Field(
            default=None,
            description="Host alias defined in inventory.yaml (default: local host)",
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """
        Manage container images.
        """
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_image_operations: {action}")

        try:
            if action == "list_images":
                return manager.list_images()
            elif action == "pull_image":
                if not image:
                    return "Error: 'image' is required for pull_image"
                if ctx:
                    await ctx_progress(ctx, 0, 100)
                return manager.pull_image(image, tag=tag, platform=platform)
            elif action == "remove_image":
                if not image:
                    return "Error: 'image' is required for remove_image"
                if ctx and not await ctx_confirm_destructive(ctx, "remove image"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                return manager.remove_image(image, force=force)
            elif action == "prune_images":
                if ctx and not await ctx_confirm_destructive(ctx, "prune images"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                if ctx:
                    await ctx_progress(ctx, 0, 100)
                return manager.prune_images()
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"


def register_container_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Container Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"container"},
    )
    async def cm_container_operations(
        action: Literal[
            "list_containers",
            "get_container_logs",
            "stop_container",
            "remove_container",
            "prune_containers",
            "exec_in_container",
        ] = Field(
            description="Action to perform. Must be one of: 'list_containers', 'get_container_logs', 'stop_container', 'remove_container', 'prune_containers', 'exec_in_container'"
        ),
        container_id: str | None = Field(
            default=None, description="Container ID or name"
        ),
        command: str | None = Field(default=None, description="Command to execute"),
        all_containers: bool = Field(default=False, description="Show all containers"),
        force: bool = Field(default=False, description="Force operation"),
        tail: str = Field(default="50", description="Number of log lines to tail"),
        host: str | None = Field(
            default=None,
            description="Host alias defined in inventory.yaml (default: local host)",
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str | list:
        """
        Manage container operations.
        """
        import shlex

        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_container_operations: {action}")

        try:
            if action == "list_containers":
                return manager.list_containers(all=all_containers)
            elif action == "get_container_logs":
                if not container_id:
                    return "Error: 'container_id' is required"
                return manager.get_container_logs(container_id, tail=tail)
            elif action == "stop_container":
                if not container_id:
                    return "Error: 'container_id' is required"
                return manager.stop_container(container_id)
            elif action == "remove_container":
                if not container_id:
                    return "Error: 'container_id' is required"
                if ctx and not await ctx_confirm_destructive(ctx, "remove container"):
                    return {"status": "cancelled"}
                return manager.remove_container(container_id, force=force)
            elif action == "prune_containers":
                if ctx and not await ctx_confirm_destructive(ctx, "prune containers"):
                    return {"status": "cancelled"}
                if ctx:
                    await ctx_progress(ctx, 0, 100)
                return manager.prune_containers()
            elif action == "exec_in_container":
                if not container_id or not command:
                    return "Error: 'container_id' and 'command' required"
                return manager.exec_in_container(container_id, shlex.split(command))
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"


def register_volume_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Volume Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"volume"},
    )
    async def cm_volume_operations(
        action: Literal[
            "list_volumes", "create_volume", "remove_volume", "prune_volumes"
        ] = Field(
            description="Action to perform. Must be one of: 'list_volumes', 'create_volume', 'remove_volume', 'prune_volumes'"
        ),
        name: str | None = Field(default=None, description="Volume name"),
        force: bool = Field(default=False, description="Force operation"),
        host: str | None = Field(
            default=None,
            description="Host alias defined in inventory.yaml (default: local host)",
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager",
        ),
        ctx: Context | None = None,
    ) -> Any:
        """
        Manage volume operations.
        """
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_volume_operations: {action}")

        try:
            if action == "list_volumes":
                return manager.list_volumes()
            elif action == "create_volume":
                if not name:
                    return "Error: 'name' is required for create_volume"
                return manager.create_volume(name)
            elif action == "remove_volume":
                if not name:
                    return "Error: 'name' is required"
                if ctx and not await ctx_confirm_destructive(ctx, "remove volume"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                return manager.remove_volume(name, force=force)
            elif action == "prune_volumes":
                if ctx and not await ctx_confirm_destructive(ctx, "prune volumes"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                if ctx:
                    await ctx_progress(ctx, 0, 100)
                return manager.prune_volumes()
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"


def register_network_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Network Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"network"},
    )
    async def cm_network_operations(
        action: Literal[
            "list_networks", "create_network", "remove_network", "prune_networks"
        ] = Field(
            description="Action to perform. Must be one of: 'list_networks', 'create_network', 'remove_network', 'prune_networks'"
        ),
        network_id: str | None = Field(default=None, description="Network ID or name"),
        driver: str = Field(default="bridge", description="Network driver"),
        host: str | None = Field(
            default=None,
            description="Host alias defined in inventory.yaml (default: local host)",
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager",
        ),
        ctx: Context | None = None,
    ) -> Any:
        """
        Manage network operations.
        """
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_network_operations: {action}")

        try:
            if action == "list_networks":
                return manager.list_networks()
            elif action == "create_network":
                if not network_id:
                    return "Error: 'network_id' is required for create_network"
                return manager.create_network(network_id, driver=driver)
            elif action == "remove_network":
                if not network_id:
                    return "Error: 'network_id' is required"
                if ctx and not await ctx_confirm_destructive(ctx, "remove network"):
                    return {"status": "cancelled"}
                return manager.remove_network(network_id)
            elif action == "prune_networks":
                if ctx and not await ctx_confirm_destructive(ctx, "prune networks"):
                    return {"status": "cancelled"}
                if ctx:
                    await ctx_progress(ctx, 0, 100)
                return manager.prune_networks()
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"


def register_swarm_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Swarm Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"swarm"},
    )
    async def cm_swarm_operations(
        action: Literal[
            "init_swarm",
            "leave_swarm",
            "list_nodes",
            "list_services",
            "create_service",
            "remove_service",
        ] = Field(
            description="Action to perform. Must be one of: 'init_swarm', 'leave_swarm', 'list_nodes', 'list_services', 'create_service', 'remove_service'"
        ),
        service_id: str | None = Field(default=None, description="Service ID or name"),
        advertise_addr: str | None = Field(
            default=None, description="Advertise address for init_swarm"
        ),
        image: str | None = Field(default=None, description="Image to use for service"),
        name: str | None = Field(default=None, description="Name for the service"),
        ports: str | None = Field(
            default=None, description="Port mappings as JSON string"
        ),
        mounts: str | None = Field(
            default=None, description="Mounts mappings as JSON string list"
        ),
        replicas: int = Field(default=1, description="Number of replicas"),
        force: bool = Field(default=False, description="Force operation"),
        host: str | None = Field(
            default=None,
            description="Host alias defined in inventory.yaml (default: local host)",
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str | list:
        """
        Manage swarm operations.
        """
        import json

        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_swarm_operations: {action}")

        try:
            if action == "init_swarm":
                if ctx and not await ctx_confirm_destructive(ctx, "init swarm"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                return manager.init_swarm(advertise_addr)
            elif action == "leave_swarm":
                if ctx and not await ctx_confirm_destructive(ctx, "leave swarm"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                return manager.leave_swarm(force=force)
            elif action == "list_nodes":
                return manager.list_nodes()
            elif action == "list_services":
                return manager.list_services()
            elif action == "create_service":
                if not name or not image:
                    return "Error: 'name' and 'image' are required for create_service"
                p_ports = json.loads(ports) if ports else None
                p_mounts = json.loads(mounts) if mounts else None
                return manager.create_service(
                    name=name,
                    image=image,
                    ports=p_ports,
                    mounts=p_mounts,
                    replicas=replicas,
                )
            elif action == "remove_service":
                if not service_id:
                    return "Error: 'service_id' is required"
                if ctx and not await ctx_confirm_destructive(ctx, "remove service"):
                    return {
                        "status": "cancelled",
                        "message": "Operation cancelled by user",
                    }
                return manager.remove_service(service_id)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"


def register_system_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager System Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"system"},
    )
    async def cm_system_operations(
        action: Literal["prune_system", "get_info", "get_version"] = Field(
            description="Action to perform. Must be one of: 'prune_system', 'get_info', 'get_version'"
        ),
        force: bool = Field(default=False, description="Force prune system"),
        all: bool = Field(default=False, description="Prune all resources"),
        host: str | None = Field(
            default=None,
            description="Host alias defined in inventory.yaml (default: local host)",
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str:
        """
        Manage container manager system operations.
        """
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_system_operations: {action}")

        try:
            if action == "prune_system":
                if ctx:
                    if not await ctx_confirm_destructive(ctx, "prune system"):
                        return {
                            "status": "cancelled",
                            "message": "Operation cancelled by user",
                        }
                    await ctx_progress(ctx, 0, 100)
                return manager.prune_system(force=force, all=all)
            elif action == "get_info":
                return manager.get_info()
            elif action == "get_version":
                return manager.get_version()
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"


def register_compose_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Compose Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"compose"},
    )
    async def cm_compose_operations(
        action: str = Field(
            description="Action to perform. Must be one of: 'up', 'down', 'ps', 'logs'"
        ),
        compose_file: str = Field(description="Path to compose file"),
        host: str | None = Field(
            default=None,
            description="Host alias defined in inventory.yaml (default: local host)",
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str | list:
        """
        Manage docker-compose or podman-compose operations.
        """
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_compose_operations: {action}")

        try:
            if action == "up":
                return manager.compose_up(compose_file)
            elif action == "down":
                return manager.compose_down(compose_file)
            elif action == "ps":
                return manager.compose_ps(compose_file)
            elif action == "logs":
                return manager.compose_logs(compose_file)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            return f"Error executing {action}: {e}"


def register_misc_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Trace Port Namespace",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        tags={"misc"},
    )
    async def trace_port_namespace(
        port: int = Field(description="Port number to trace"),
        host: str | None = Field(
            default=None,
            description="Host alias defined in inventory.yaml (default: local host)",
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> list[dict] | str:
        """
        Locate the container actively using/mapping the specified port on the target host.
        """
        if ctx:
            ctx_log(ctx, logging.INFO, f"Tracing port {port} on host {host}")

        try:
            manager = create_manager(manager_type, host=host)
            containers = manager.list_containers(all=True)
            matching_containers = []
            for c in containers:
                if not c.ports or c.ports == "none":
                    continue
                mappings = [m.strip() for m in c.ports.split(",")]
                for m in mappings:
                    if "->" in m:
                        host_part, container_part = m.split("->", 1)
                        if ":" in host_part:
                            ip, host_port = host_part.rsplit(":", 1)
                            if host_port == str(port):
                                matching_containers.append(c.model_dump())
                                break
            return matching_containers
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error tracing port {port}: {e}")
            return f"Error tracing port {port}: {e}"


def get_mcp_instance() -> tuple[Any, ...]:
    """Initialize and return the MCP instance."""
    load_dotenv(find_dotenv())
    args, mcp, middlewares = create_mcp_server(
        name="container-manager-mcp",
        version=__version__,
        instructions="Container Manager MCP Server - Manage Docker and Podman containers, images, volumes, networks, and swarm.",
    )

    # We maintain these default configurations for backwards compatibility
    DEFAULT_INFOTOOL = to_boolean(os.getenv("INFOTOOL", "True"))
    if DEFAULT_INFOTOOL:
        register_info_tools(mcp)

    DEFAULT_IMAGETOOL = to_boolean(os.getenv("IMAGETOOL", "True"))
    if DEFAULT_IMAGETOOL:
        register_image_tools(mcp)

    DEFAULT_CONTAINERTOOL = to_boolean(os.getenv("CONTAINERTOOL", "True"))
    if DEFAULT_CONTAINERTOOL:
        register_container_tools(mcp)

    DEFAULT_VOLUMETOOL = to_boolean(os.getenv("VOLUMETOOL", "True"))
    if DEFAULT_VOLUMETOOL:
        register_volume_tools(mcp)

    DEFAULT_NETWORKTOOL = to_boolean(os.getenv("NETWORKTOOL", "True"))
    if DEFAULT_NETWORKTOOL:
        register_network_tools(mcp)

    DEFAULT_SWARMTOOL = to_boolean(os.getenv("SWARMTOOL", "True"))
    if DEFAULT_SWARMTOOL:
        register_swarm_tools(mcp)

    DEFAULT_SYSTEMTOOL = to_boolean(os.getenv("SYSTEMTOOL", "True"))
    if DEFAULT_SYSTEMTOOL:
        register_system_tools(mcp)

    DEFAULT_COMPOSETOOL = to_boolean(os.getenv("COMPOSETOOL", "True"))
    if DEFAULT_COMPOSETOOL:
        register_compose_tools(mcp)

    DEFAULT_MISCTOOL = to_boolean(os.getenv("MISCTOOL", "True"))
    if DEFAULT_MISCTOOL:
        register_misc_tools(mcp)

    return args, mcp, middlewares


def mcp_server() -> None:
    """Main entry point for the MCP server."""
    import sys

    args, mcp, middlewares = get_mcp_instance()
    print(f"container-manager-mcp MCP v{__version__}", file=sys.stderr)
    print("\nStarting MCP Server", file=sys.stderr)
    print(f"  Transport: {args.transport.upper()}", file=sys.stderr)
    print(f"  Auth: {args.auth_type}", file=sys.stderr)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error("Invalid transport", extra={"transport": args.transport})
        sys.exit(1)


def main():
    """Main entry point for the MCP server."""
    mcp_server()


if __name__ == "__main__":
    main()

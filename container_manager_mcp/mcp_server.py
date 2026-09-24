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

import json
import logging
import os
import sys
from typing import Any, Literal

from agent_connector_sdk.config import load_config
from agent_connector_sdk.mcp.action_dispatch import resolve_action
from agent_connector_sdk.mcp.concurrency import run_blocking
from agent_connector_sdk.mcp.server import create_mcp_server
from agent_connector_sdk.mcp.tool_surface import register_tool_surface

_SERVICE = "container-manager-mcp"

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


_CTX_LOG_LEVEL_NAMES = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}


def _ctx_log_notify_client(ctx: Any, level_str: str, message: Any) -> None:
    """Best-effort mirror of a log line onto the MCP client's ctx, if present."""
    if not ctx:
        return
    client_fn = getattr(ctx, level_str, None) or getattr(ctx, "info", None)
    if not client_fn:
        return
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


def _ctx_log_two_arg(ctx: Any, level: Any, message: Any) -> None:
    if isinstance(level, int):
        level_str = _CTX_LOG_LEVEL_NAMES.get(level, "info")
    else:
        level_str = str(level).lower()

    log_fn = getattr(logger, level_str, None) or getattr(logger, "info", None)
    if log_fn:
        log_fn(message)
    _ctx_log_notify_client(ctx, level_str, message)


def _ctx_log_three_arg(ctx: Any, server_logger: Any, level: Any, message: Any) -> None:
    level_str = str(level).lower()
    log_fn = getattr(server_logger, level_str, None) or getattr(
        server_logger, "info", None
    )
    if log_fn:
        log_fn(message)
    _ctx_log_notify_client(ctx, level_str, message)


def _ctx_log_fallback(ctx: Any, args: tuple, kwargs: dict) -> None:
    try:
        from agent_utilities.mcp.context_helpers import ctx_log as _real_ctx_log
    except ImportError:
        return
    try:
        _real_ctx_log(ctx, *args, **kwargs)
    except Exception:
        pass


def ctx_log(ctx: Any, *args, **kwargs) -> None:
    if len(args) == 2:
        _ctx_log_two_arg(ctx, *args)
    elif len(args) == 3:
        _ctx_log_three_arg(ctx, *args)
    else:
        _ctx_log_fallback(ctx, args, kwargs)


from container_manager_mcp.container_manager import (
    ContainerManagerBase,
    create_manager,
    list_inventory_hosts,
)

__version__ = "3.1.0"

logger = get_logger(name="ContainerManagerServer")
logger.setLevel(logging.DEBUG)


def register_inventory_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "List Container Hosts",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"inventory"},
    )
    async def cm_list_hosts(ctx: Context | None = None) -> dict:
        """List the host aliases you can pass as ``host`` to any cm_* operation
        to manage a REMOTE machine's Docker (via Docker-over-SSH).

        Every cm_* tool accepts ``host``: omit it to use the local Docker
        socket, or pass an alias here to target another machine — e.g. a swarm
        MANAGER node for cluster ops — without deploying a container-manager on
        each box. Aliases come from the tunnel-manager inventory
        (``~/.config/agent-utilities/inventory.yaml``) and connect as
        ``ssh://<user>@<hostname>:<port>``.
        """
        try:
            return list_inventory_hosts()
        except Exception:  # missing/empty inventory, etc.
            return {
                "error": "Operation failed",
                "hosts": {},
                "hint": (
                    "No inventory configured. Add hosts to "
                    "~/.config/agent-utilities/inventory.yaml (tunnel-manager "
                    "format) and ensure SSH keys are available."
                ),
            }


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
            description=(
                "Remote host alias to target that machine's Docker over SSH "
                "(resolved from the tunnel-manager inventory). Omit for the "
                "LOCAL Docker socket. Call cm_list_hosts to see the available "
                "aliases. Note: swarm-cluster actions (list_nodes/services) "
                "must target a swarm MANAGER node."
            ),
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman, kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str:
        """
        Manage container manager info operations.
        """
        resolved = resolve_action(action, ["get_version", "get_info"], service=_SERVICE)
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_info_operations: {action}")

        try:
            if action == "get_version":
                return await run_blocking(manager.get_version)
            elif action == "get_info":
                return await run_blocking(manager.get_info)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            return f"Error executing {action}: {type(e).__name__}"


async def _list_images_action(manager):
    return await run_blocking(manager.list_images)


async def _pull_image_action(manager, ctx, image, tag, platform):
    if not image:
        return "Error: 'image' is required for pull_image"
    if ctx:
        await ctx_progress(ctx, 0, 100)
    return await run_blocking(manager.pull_image, image, tag=tag, platform=platform)


async def _remove_image_action(manager, ctx, image, force):
    if not image:
        return "Error: 'image' is required for remove_image"
    if ctx and not await ctx_confirm_destructive(ctx, "remove image"):
        return {"status": "cancelled", "message": "Operation cancelled by user"}
    return await run_blocking(manager.remove_image, image, force=force)


async def _prune_images_action(manager, ctx):
    if ctx and not await ctx_confirm_destructive(ctx, "prune images"):
        return {"status": "cancelled", "message": "Operation cancelled by user"}
    if ctx:
        await ctx_progress(ctx, 0, 100)
    return await run_blocking(manager.prune_images)


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
        meta={
            "eg.annotations": {"modalities_in": ["text"], "modalities_out": ["text"]}
        },
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
            description=(
                "Remote host alias to target that machine's Docker over SSH "
                "(resolved from the tunnel-manager inventory). Omit for the "
                "LOCAL Docker socket. Call cm_list_hosts to see the available "
                "aliases. Note: swarm-cluster actions (list_nodes/services) "
                "must target a swarm MANAGER node."
            ),
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman, kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """
        Manage container images.
        """
        resolved = resolve_action(
            action,
            ["list_images", "pull_image", "remove_image", "prune_images"],
            service=_SERVICE,
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_image_operations: {action}")

        try:
            if action == "list_images":
                return await _list_images_action(manager)
            if action == "pull_image":
                return await _pull_image_action(manager, ctx, image, tag, platform)
            if action == "remove_image":
                return await _remove_image_action(manager, ctx, image, force)
            if action == "prune_images":
                return await _prune_images_action(manager, ctx)
            return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"


async def _list_containers_action(manager, all_containers):
    return await run_blocking(manager.list_containers, all=all_containers)


async def _get_container_logs_action(manager, container_id, tail):
    if not container_id:
        return "Error: 'container_id' is required"
    return await run_blocking(manager.get_container_logs, container_id, tail=tail)


async def _stop_container_action(manager, container_id):
    if not container_id:
        return "Error: 'container_id' is required"
    return await run_blocking(manager.stop_container, container_id)


async def _remove_container_action(manager, ctx, container_id, force):
    if not container_id:
        return "Error: 'container_id' is required"
    if ctx and not await ctx_confirm_destructive(ctx, "remove container"):
        return {"status": "cancelled"}
    return await run_blocking(manager.remove_container, container_id, force=force)


async def _prune_containers_action(manager, ctx):
    if ctx and not await ctx_confirm_destructive(ctx, "prune containers"):
        return {"status": "cancelled"}
    if ctx:
        await ctx_progress(ctx, 0, 100)
    return await run_blocking(manager.prune_containers)


async def _exec_in_container_action(manager, container_id, command, binary):
    if not container_id or not command:
        return "Error: 'container_id' and 'command' required"
    import shlex

    return await run_blocking(
        manager.exec_in_container, container_id, shlex.split(command), binary=binary
    )


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
        meta={
            "eg.annotations": {
                "modalities_in": ["text"],
                "modalities_out": ["text"],
                "provides": ["eg:capability/action/process-exec"],
            }
        },
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
        binary: bool = Field(
            default=False,
            description=(
                "For exec_in_container: return base64'd stdout bytes ('output_b64') "
                "instead of UTF-8 text, so binary output (e.g. a screenshot PNG) "
                "survives uncorrupted."
            ),
        ),
        all_containers: bool = Field(default=False, description="Show all containers"),
        force: bool = Field(default=False, description="Force operation"),
        tail: str = Field(default="50", description="Number of log lines to tail"),
        host: str | None = Field(
            default=None,
            description=(
                "Remote host alias to target that machine's Docker over SSH "
                "(resolved from the tunnel-manager inventory). Omit for the "
                "LOCAL Docker socket. Call cm_list_hosts to see the available "
                "aliases. Note: swarm-cluster actions (list_nodes/services) "
                "must target a swarm MANAGER node."
            ),
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman, kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str | list:
        """
        Manage container operations.
        """
        resolved = resolve_action(
            action,
            [
                "list_containers",
                "get_container_logs",
                "stop_container",
                "remove_container",
                "prune_containers",
                "exec_in_container",
            ],
            service=_SERVICE,
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_container_operations: {action}")

        try:
            dispatch = {
                "list_containers": lambda: _list_containers_action(
                    manager, all_containers
                ),
                "get_container_logs": lambda: _get_container_logs_action(
                    manager, container_id, tail
                ),
                "stop_container": lambda: _stop_container_action(manager, container_id),
                "remove_container": lambda: _remove_container_action(
                    manager, ctx, container_id, force
                ),
                "prune_containers": lambda: _prune_containers_action(manager, ctx),
                "exec_in_container": lambda: _exec_in_container_action(
                    manager, container_id, command, binary
                ),
            }
            handler = dispatch.get(action)
            if handler is None:
                return f"Error: Unknown action '{action}'"
            return await handler()
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"


async def _list_volumes_action(manager):
    return await run_blocking(manager.list_volumes)


async def _create_volume_action(manager, name):
    if not name:
        return "Error: 'name' is required for create_volume"
    return await run_blocking(manager.create_volume, name)


async def _remove_volume_action(manager, ctx, name, force):
    if not name:
        return "Error: 'name' is required"
    if ctx and not await ctx_confirm_destructive(ctx, "remove volume"):
        return {"status": "cancelled", "message": "Operation cancelled by user"}
    return await run_blocking(manager.remove_volume, name, force=force)


async def _prune_volumes_action(manager, ctx):
    if ctx and not await ctx_confirm_destructive(ctx, "prune volumes"):
        return {"status": "cancelled", "message": "Operation cancelled by user"}
    if ctx:
        await ctx_progress(ctx, 0, 100)
    return await run_blocking(manager.prune_volumes)


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
            description=(
                "Remote host alias to target that machine's Docker over SSH "
                "(resolved from the tunnel-manager inventory). Omit for the "
                "LOCAL Docker socket. Call cm_list_hosts to see the available "
                "aliases. Note: swarm-cluster actions (list_nodes/services) "
                "must target a swarm MANAGER node."
            ),
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
        resolved = resolve_action(
            action,
            ["list_volumes", "create_volume", "remove_volume", "prune_volumes"],
            service=_SERVICE,
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_volume_operations: {action}")

        try:
            if action == "list_volumes":
                return await _list_volumes_action(manager)
            if action == "create_volume":
                return await _create_volume_action(manager, name)
            if action == "remove_volume":
                return await _remove_volume_action(manager, ctx, name, force)
            if action == "prune_volumes":
                return await _prune_volumes_action(manager, ctx)
            return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"


async def _list_networks_action(manager):
    return await run_blocking(manager.list_networks)


async def _create_network_action(manager, network_id, driver):
    if not network_id:
        return "Error: 'network_id' is required for create_network"
    return await run_blocking(manager.create_network, network_id, driver=driver)


async def _remove_network_action(manager, ctx, network_id):
    if not network_id:
        return "Error: 'network_id' is required"
    if ctx and not await ctx_confirm_destructive(ctx, "remove network"):
        return {"status": "cancelled"}
    return await run_blocking(manager.remove_network, network_id)


async def _prune_networks_action(manager, ctx):
    if ctx and not await ctx_confirm_destructive(ctx, "prune networks"):
        return {"status": "cancelled"}
    if ctx:
        await ctx_progress(ctx, 0, 100)
    return await run_blocking(manager.prune_networks)


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
            description=(
                "Remote host alias to target that machine's Docker over SSH "
                "(resolved from the tunnel-manager inventory). Omit for the "
                "LOCAL Docker socket. Call cm_list_hosts to see the available "
                "aliases. Note: swarm-cluster actions (list_nodes/services) "
                "must target a swarm MANAGER node."
            ),
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
        resolved = resolve_action(
            action,
            ["list_networks", "create_network", "remove_network", "prune_networks"],
            service=_SERVICE,
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_network_operations: {action}")

        try:
            if action == "list_networks":
                return await _list_networks_action(manager)
            if action == "create_network":
                return await _create_network_action(manager, network_id, driver)
            if action == "remove_network":
                return await _remove_network_action(manager, ctx, network_id)
            if action == "prune_networks":
                return await _prune_networks_action(manager, ctx)
            return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"


async def _init_swarm_action(manager, ctx, advertise_addr):
    if ctx and not await ctx_confirm_destructive(ctx, "init swarm"):
        return {"status": "cancelled", "message": "Operation cancelled by user"}
    return await run_blocking(manager.init_swarm, advertise_addr)


async def _leave_swarm_action(manager, ctx, force):
    if ctx and not await ctx_confirm_destructive(ctx, "leave swarm"):
        return {"status": "cancelled", "message": "Operation cancelled by user"}
    return await run_blocking(manager.leave_swarm, force=force)


async def _list_swarm_nodes_action(manager):
    return await run_blocking(manager.list_nodes)


async def _list_swarm_services_action(manager):
    return await run_blocking(manager.list_services)


async def _create_swarm_service_action(manager, name, image, ports, mounts, replicas):
    if not name or not image:
        return "Error: 'name' and 'image' are required for create_service"
    p_ports = json.loads(ports) if ports else None
    p_mounts = json.loads(mounts) if mounts else None
    return await run_blocking(
        manager.create_service,
        name=name,
        image=image,
        ports=p_ports,
        mounts=p_mounts,
        replicas=replicas,
    )


async def _remove_swarm_service_action(manager, ctx, service_id):
    if not service_id:
        return "Error: 'service_id' is required"
    if ctx and not await ctx_confirm_destructive(ctx, "remove service"):
        return {"status": "cancelled", "message": "Operation cancelled by user"}
    return await run_blocking(manager.remove_service, service_id)


async def _inspect_swarm_node_action(manager, node_id):
    if not node_id:
        return "Error: 'node_id' is required"
    return await run_blocking(manager.inspect_node, node_id)


async def _update_swarm_node_action(
    manager, node_id, labels, role, availability, replace_labels
):
    if not node_id:
        return "Error: 'node_id' is required"
    p_labels = json.loads(labels) if labels else None
    if not any([p_labels, role, availability]):
        return "Error: provide at least one of 'labels', 'role', or 'availability'"
    return await run_blocking(
        manager.update_node,
        node_id,
        labels=p_labels,
        role=role,
        availability=availability,
        replace_labels=replace_labels,
    )


async def _remove_swarm_node_action(manager, ctx, node_id, force):
    if not node_id:
        return "Error: 'node_id' is required"
    if ctx and not await ctx_confirm_destructive(ctx, "remove node"):
        return {"status": "cancelled", "message": "Operation cancelled by user"}
    return await run_blocking(manager.remove_node, node_id, force=force)


async def _inspect_swarm_service_action(manager, service_id):
    if not service_id:
        return "Error: 'service_id' is required"
    return await run_blocking(manager.inspect_service, service_id)


async def _scale_swarm_service_action(manager, service_id, replicas):
    if not service_id:
        return "Error: 'service_id' is required"
    return await run_blocking(manager.scale_service, service_id, replicas)


async def _update_swarm_service_action(manager, service_id, replicas, updates):
    """`updates` bundles image/env/constraints/labels/force -- keeps this
    helper's own signature at 4 params instead of 8."""
    if not service_id:
        return "Error: 'service_id' is required"
    p_env = json.loads(updates["env"]) if updates["env"] else None
    p_constraints = (
        json.loads(updates["constraints"]) if updates["constraints"] else None
    )
    p_labels = json.loads(updates["labels"]) if updates["labels"] else None
    return await run_blocking(
        manager.update_service,
        service_id,
        image=updates["image"],
        replicas=replicas if replicas != 1 else None,
        env=p_env,
        constraints=p_constraints,
        labels=p_labels,
        force=updates["force"],
    )


async def _swarm_service_ps_action(manager, service_id):
    if not service_id:
        return "Error: 'service_id' is required"
    return await run_blocking(manager.service_ps, service_id)


async def _swarm_service_logs_action(manager, service_id, tail):
    if not service_id:
        return "Error: 'service_id' is required"
    return await run_blocking(manager.service_logs, service_id, tail=tail)


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
            "inspect_node",
            "update_node",
            "remove_node",
            "list_services",
            "inspect_service",
            "create_service",
            "update_service",
            "scale_service",
            "service_ps",
            "service_logs",
            "remove_service",
        ] = Field(
            description=(
                "Action to perform. Cluster actions ('list_nodes', 'inspect_node', "
                "'update_node', 'remove_node', '*_service') must target a swarm "
                "MANAGER (set 'host'). 'update_node' sets labels/role/availability "
                "(e.g. add the poweredge=true label or drain a node); 'update_service' "
                "/'scale_service' change a running service in place; 'service_ps'/"
                "'service_logs' inspect tasks and logs."
            )
        ),
        service_id: str | None = Field(default=None, description="Service ID or name"),
        node_id: str | None = Field(
            default=None,
            description="Node ID, short ID, or hostname (for *_node actions)",
        ),
        labels: str | None = Field(
            default=None,
            description=(
                'Labels as a JSON object, e.g. \'{"poweredge": "true"}\'. For '
                "update_node these are node labels (merged unless replace_labels); "
                "for update_service these are service labels (merged)."
            ),
        ),
        replace_labels: bool = Field(
            default=False,
            description="update_node: replace all labels instead of merging",
        ),
        role: str | None = Field(
            default=None, description="update_node: 'manager' or 'worker'"
        ),
        availability: str | None = Field(
            default=None,
            description="update_node: 'active', 'pause', or 'drain'",
        ),
        env: str | None = Field(
            default=None,
            description="update_service: environment as a JSON list of 'KEY=VALUE'",
        ),
        constraints: str | None = Field(
            default=None,
            description="update_service: placement constraints as a JSON list",
        ),
        tail: int = Field(
            default=100, description="service_logs: number of log lines to tail"
        ),
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
            description=(
                "Remote host alias to target that machine's Docker over SSH "
                "(resolved from the tunnel-manager inventory). Omit for the "
                "LOCAL Docker socket. Call cm_list_hosts to see the available "
                "aliases. Note: swarm-cluster actions (list_nodes/services) "
                "must target a swarm MANAGER node."
            ),
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman, kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str | list:
        """
        Manage swarm operations.
        """
        resolved = resolve_action(
            action,
            [
                "init_swarm",
                "leave_swarm",
                "list_nodes",
                "inspect_node",
                "update_node",
                "remove_node",
                "list_services",
                "inspect_service",
                "create_service",
                "update_service",
                "scale_service",
                "service_ps",
                "service_logs",
                "remove_service",
            ],
            service=_SERVICE,
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_swarm_operations: {action}")

        try:
            dispatch = {
                "init_swarm": lambda: _init_swarm_action(manager, ctx, advertise_addr),
                "leave_swarm": lambda: _leave_swarm_action(manager, ctx, force),
                "list_nodes": lambda: _list_swarm_nodes_action(manager),
                "list_services": lambda: _list_swarm_services_action(manager),
                "create_service": lambda: _create_swarm_service_action(
                    manager, name, image, ports, mounts, replicas
                ),
                "remove_service": lambda: _remove_swarm_service_action(
                    manager, ctx, service_id
                ),
                "inspect_node": lambda: _inspect_swarm_node_action(manager, node_id),
                "update_node": lambda: _update_swarm_node_action(
                    manager, node_id, labels, role, availability, replace_labels
                ),
                "remove_node": lambda: _remove_swarm_node_action(
                    manager, ctx, node_id, force
                ),
                "inspect_service": lambda: _inspect_swarm_service_action(
                    manager, service_id
                ),
                "scale_service": lambda: _scale_swarm_service_action(
                    manager, service_id, replicas
                ),
                "update_service": lambda: _update_swarm_service_action(
                    manager,
                    service_id,
                    replicas,
                    {
                        "image": image,
                        "env": env,
                        "constraints": constraints,
                        "labels": labels,
                        "force": force,
                    },
                ),
                "service_ps": lambda: _swarm_service_ps_action(manager, service_id),
                "service_logs": lambda: _swarm_service_logs_action(
                    manager, service_id, tail
                ),
            }
            handler = dispatch.get(action)
            if handler is None:
                return f"Error: Unknown action '{action}'"
            return await handler()
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"


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
            description=(
                "Remote host alias to target that machine's Docker over SSH "
                "(resolved from the tunnel-manager inventory). Omit for the "
                "LOCAL Docker socket. Call cm_list_hosts to see the available "
                "aliases. Note: swarm-cluster actions (list_nodes/services) "
                "must target a swarm MANAGER node."
            ),
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman, kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str:
        """
        Manage container manager system operations.
        """
        resolved = resolve_action(
            action, ["prune_system", "get_info", "get_version"], service=_SERVICE
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
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
                return await run_blocking(manager.prune_system, force=force, all=all)
            elif action == "get_info":
                return await run_blocking(manager.get_info)
            elif action == "get_version":
                return await run_blocking(manager.get_version)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"


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
        compose_file: str | None = Field(
            default=None, description="Path to compose file"
        ),
        host: str | None = Field(
            default=None,
            description=(
                "Remote host alias to target that machine's Docker over SSH "
                "(resolved from the tunnel-manager inventory). Omit for the "
                "LOCAL Docker socket. Call cm_list_hosts to see the available "
                "aliases. Note: swarm-cluster actions (list_nodes/services) "
                "must target a swarm MANAGER node."
            ),
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman, kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | str | list:
        """
        Manage docker-compose or podman-compose operations.
        """
        resolved = resolve_action(
            action, ["up", "down", "ps", "logs"], service=_SERVICE
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        if not compose_file:
            return "Error: 'compose_file' is required"
        manager = create_manager(manager_type, host=host)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_compose_operations: {action}")

        try:
            if action == "up":
                return await run_blocking(manager.compose_up, compose_file)
            elif action == "down":
                return await run_blocking(manager.compose_down, compose_file)
            elif action == "ps":
                return await run_blocking(manager.compose_ps, compose_file)
            elif action == "logs":
                return await run_blocking(manager.compose_logs, compose_file)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            return f"Error executing {action}: {type(e).__name__}"


def register_k8sworkloads_tools(mcp: FastMCP):
    """Register Kubernetes workload operations."""
    if os.environ.get("K8SWORKLOADSTOOL", "True").lower() not in ("true", "1", "yes"):
        return
    try:
        from container_manager_mcp.mcp.mcp_k8s_workloads import (
            register_k8sworkloads_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # Kubernetes client not installed, skip registration
        pass


def register_k8sconfig_tools(mcp: FastMCP):
    """Register Kubernetes configuration operations."""
    if os.environ.get("K8SCONFIGTOOL", "True").lower() not in ("true", "1", "yes"):
        return
    try:
        from container_manager_mcp.mcp.mcp_k8s_config import (
            register_k8sconfig_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # Kubernetes client not installed, skip registration
        pass


def register_k8snetworking_tools(mcp: FastMCP):
    """Register Kubernetes networking operations."""
    if os.environ.get("K8SNETWORKINGTOOL", "True").lower() not in ("true", "1", "yes"):
        return
    try:
        from container_manager_mcp.mcp.mcp_k8s_networking import (
            register_k8snetworking_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # Kubernetes client not installed, skip registration
        pass


def register_k8sstorage_tools(mcp: FastMCP):
    """Register Kubernetes storage operations."""
    if os.environ.get("K8SSTORAGETOOL", "True").lower() not in ("true", "1", "yes"):
        return
    try:
        from container_manager_mcp.mcp.mcp_k8s_storage import (
            register_k8sstorage_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # Kubernetes client not installed, skip registration
        pass


def register_k8srbac_tools(mcp: FastMCP):
    """Register Kubernetes RBAC and security operations."""
    if os.environ.get("K8SRBACTOOL", "True").lower() not in ("true", "1", "yes"):
        return
    try:
        from container_manager_mcp.mcp.mcp_k8s_rbac import (
            register_k8srbac_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # Kubernetes client not installed, skip registration
        pass


def register_k8scluster_tools(mcp: FastMCP):
    """Register Kubernetes cluster operations."""
    if os.environ.get("K8SCLUSTERTOOL", "True").lower() not in ("true", "1", "yes"):
        return
    try:
        from container_manager_mcp.mcp.mcp_k8s_cluster import (
            register_k8scluster_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # Kubernetes client not installed, skip registration
        pass


def register_k8sgovernance_tools(mcp: FastMCP):
    """Register Kubernetes governance operations."""
    if os.environ.get("K8SGOVERNANCETOOL", "True").lower() not in ("true", "1", "yes"):
        return
    try:
        from container_manager_mcp.mcp.mcp_k8s_governance import (
            register_k8sgovernance_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # Kubernetes client not installed, skip registration
        pass


def register_k8sobservability_tools(mcp: FastMCP):
    """Register Kubernetes observability operations."""
    if os.environ.get("K8SOBSERVABILITYTOOL", "True").lower() not in (
        "true",
        "1",
        "yes",
    ):
        return
    try:
        from container_manager_mcp.mcp.mcp_k8s_observability import (
            register_k8sobservability_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # Kubernetes client not installed, skip registration
        pass


def register_podman_tools(mcp: FastMCP):
    """Register Podman pod/network/volume tools."""
    if os.environ.get("PODMANTOOL", "True").lower() not in ("true", "1", "yes"):
        return
    try:
        from container_manager_mcp.mcp.mcp_podman import (
            register_podman_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # Podman client not installed, skip registration
        pass


def register_dockerswarm_tools(mcp: FastMCP):
    """Register Docker Swarm tools."""
    if os.environ.get("DOCKERSWARMTOOL", "True").lower() not in ("true", "1", "yes"):
        return
    try:
        from container_manager_mcp.mcp.mcp_docker_swarm import (
            register_dockerswarm_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # Docker client not installed, skip registration
        pass


def register_multicontext_tools(mcp: FastMCP):
    """Register multi-context container management tools."""
    if os.environ.get("MULTICONTEXTTOOL", "True").lower() not in ("true", "1", "yes"):
        return
    try:
        from container_manager_mcp.mcp.mcp_multi_context import (
            register_multicontext_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # Multi-context manager not available, skip registration
        pass


def register_doctor_tools(mcp: FastMCP):
    """Register the guided environment doctor tool (cm_doctor)."""
    if os.environ.get("DOCTORTOOL", "True").lower() not in ("true", "1", "yes"):
        return
    try:
        from container_manager_mcp.mcp.mcp_doctor import (
            register_doctor_tools as _register,
        )

        _register(mcp)
    except ImportError:
        # doctor engine unavailable, skip registration
        pass


def _container_port_matches(c, port: int) -> bool:
    """True if `c`'s port mappings expose the host-side `port`."""
    if not c.ports or c.ports == "none":
        return False
    for mapping in (m.strip() for m in c.ports.split(",")):
        if "->" not in mapping:
            continue
        host_part, _container_part = mapping.split("->", 1)
        if ":" not in host_part:
            continue
        _ip, host_port = host_part.rsplit(":", 1)
        if host_port == str(port):
            return True
    return False


async def _ingest_modality_sweep(name: str, lister, mapper, kwargs: dict) -> dict:
    """List one resource modality and push it through its kg_ingest mapper."""
    from agent_utilities.knowledge_graph.memory.native_ingest import NativeIngestError

    try:
        records = await run_blocking(lister)
        data = [
            r.model_dump() if hasattr(r, "model_dump") else r
            for r in records
            if r is not None
        ]
        ingested = mapper(data, **kwargs)
        return {"listed": len(data), "ingested": ingested}
    except NativeIngestError:
        raise
    except Exception:  # noqa: BLE001 — isolate source API failures
        return {"error": "Operation failed"}


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
            description=(
                "Remote host alias to target that machine's Docker over SSH "
                "(resolved from the tunnel-manager inventory). Omit for the "
                "LOCAL Docker socket. Call cm_list_hosts to see the available "
                "aliases. Note: swarm-cluster actions (list_nodes/services) "
                "must target a swarm MANAGER node."
            ),
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman, kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> list[dict] | str:
        """
        Locate the container actively using/mapping the specified port on the target host.
        """
        if ctx:
            ctx_log(ctx, logging.INFO, "Tracing a configured network endpoint")

        try:
            manager = create_manager(manager_type, host=host)
            containers = await run_blocking(manager.list_containers, all=True)
            return [
                c.model_dump() for c in containers if _container_port_matches(c, port)
            ]
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error tracing port {port}: {type(e).__name__}"
                )
            return f"Error tracing port {port}: {type(e).__name__}"

    @mcp.tool(
        annotations={
            "title": "Ingest Container Inventory into Knowledge Graph",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        tags={"misc", "kg"},
    )
    async def cm_ingest_inventory(
        modality: Literal[
            "all",
            "containers",
            "images",
            "volumes",
            "networks",
            "services",
            "nodes",
            "pods",
            "deployments",
            "namespaces",
            "k8s_services",
        ] = Field(
            default="all",
            description="Which resource inventory to ingest. 'all' sweeps containers/images/volumes/networks and (on a swarm manager) services/nodes, and (on a kubernetes manager) pods/deployments/namespaces/k8s_services.",
        ),
        host: str | None = Field(
            default=None,
            description=(
                "Remote host alias to target that machine's Docker over SSH "
                "(resolved from the tunnel-manager inventory). Omit for the LOCAL "
                "Docker socket. Swarm modalities (services/nodes) must target a "
                "swarm MANAGER node."
            ),
        ),
        all_containers: bool = Field(
            default=True,
            description="Include stopped containers when ingesting containers.",
        ),
        manager_type: str | None = Field(
            default=os.environ.get("CONTAINER_MANAGER_TYPE", None),
            description="Container manager: docker, podman (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict:
        """Natively ingest the container inventory into epistemic-graph as typed nodes.

        Lists resources via the real container-manager client and pushes them into the
        knowledge graph as ``:Container`` / ``:ContainerImage`` / ``:ContainerVolume`` /
        ``:ContainerNetwork`` / ``:SwarmService`` / ``:SwarmNode`` nodes (+ ``:usesImage`` /
        ``:runsOn`` links) through the authoritative native-ingest transaction.
        CONCEPT:AU-KG.ingest.enterprise-source-extractor.
        """
        from container_manager_mcp import kg_ingest

        if ctx:
            ctx_log(
                ctx,
                logging.INFO,
                f"Ingesting container inventory (modality={modality})",
            )

        manager = create_manager(manager_type, host=host)
        is_k8s = type(manager).__name__ == "KubernetesManager"
        if modality == "all":
            # Docker/swarm modalities always sweep (services/nodes no-op off-swarm);
            # the k8s modalities are only added when the active manager is Kubernetes.
            want = {"containers", "images", "volumes", "networks", "services", "nodes"}
            if is_k8s:
                want |= {"pods", "deployments", "namespaces", "k8s_services"}
        else:
            want = {modality}
        result: dict[str, Any] = {"host": host, "modalities": {}}

        sweeps = [
            (
                "containers",
                lambda: manager.list_containers(all=all_containers),
                kg_ingest.ingest_containers,
                {"host": host},
            ),
            ("images", manager.list_images, kg_ingest.ingest_images, {}),
            ("volumes", manager.list_volumes, kg_ingest.ingest_volumes, {}),
            ("networks", manager.list_networks, kg_ingest.ingest_networks, {}),
            ("services", manager.list_services, kg_ingest.ingest_services, {}),
            ("nodes", manager.list_nodes, kg_ingest.ingest_nodes, {}),
            ("pods", manager.list_pods, kg_ingest.ingest_pods, {}),
            # Deployment-shaped list_services on the Kubernetes manager.
            ("deployments", manager.list_services, kg_ingest.ingest_deployments, {}),
            ("namespaces", manager.list_namespaces, kg_ingest.ingest_namespaces, {}),
            (
                "k8s_services",
                manager.list_native_services,
                kg_ingest.ingest_k8s_services,
                {},
            ),
        ]
        for name, lister, mapper, kwargs in sweeps:
            if name in want:
                result["modalities"][name] = await _ingest_modality_sweep(
                    name, lister, mapper, kwargs
                )

        return result


def get_mcp_instance() -> tuple[Any, ...]:
    """Initialize and return the MCP instance."""
    load_config()
    args, mcp, middlewares = create_mcp_server(
        name="container-manager-mcp",
        version=__version__,
        instructions="Container Manager MCP Server - Manage Docker and Podman containers, images, volumes, networks, and swarm.",
    )

    # Tools target the per-call manager from create_manager(host=...), not a
    # Depends(get_client) client; condensed gates each register_*_tools via
    # setting("<TAG>TOOL", True). ContainerManagerBase is passed as the closest
    # importable client class for the verbose tier.

    # Single registration path: register_tool_surface() auto-discovers every
    # register_<tag>_tools in this module (base + the 8 themed k8s wrappers +
    # podman/docker/multi-context) and gates each on its <TAG>TOOL setting.
    register_tool_surface(
        mcp,
        client_cls=ContainerManagerBase,
        get_client=create_manager,
        service=_SERVICE,
        tools_module=sys.modules[__name__],
    )

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

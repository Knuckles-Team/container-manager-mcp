"""MCP tools for Kubernetes observability operations.

Themed dispatcher covering resource metrics (top/pod/node/cluster), autoscaler
metrics/history, watch/stream/events, field-selector listing, and debug helpers.
"""

import logging
from typing import Literal

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log

_UNHANDLED = object()  # sentinel: this action doesn't belong to this dispatch group


async def _dispatch_metrics_action(action, manager, name, namespace):
    if action == "top_pods":
        return await run_blocking(manager.top_pods, namespace=namespace)
    elif action == "top_nodes":
        return await run_blocking(manager.top_nodes)
    elif action == "get_pod_metrics":
        return await run_blocking(manager.get_pod_metrics, namespace)
    elif action == "get_node_metrics":
        return await run_blocking(manager.get_node_metrics)
    elif action == "get_pod_resource_usage":
        if not name or not namespace:
            return (
                "Error: 'name' and 'namespace' are required for get_pod_resource_usage"
            )
        return await run_blocking(manager.get_pod_resource_usage, name, namespace)
    elif action == "get_cluster_resource_summary":
        return await run_blocking(manager.get_cluster_resource_summary)
    return _UNHANDLED


async def _dispatch_autoscaler_action(
    action, manager, max_replicas, metrics, min_replicas, name, namespace
):
    if action == "get_autoscaler_metrics":
        if not name or not namespace:
            return (
                "Error: 'name' and 'namespace' are required for get_autoscaler_metrics"
            )
        return await run_blocking(manager.get_autoscaler_metrics, name, namespace)
    elif action == "set_autoscaler_metrics":
        if not name or not namespace or not metrics:
            return "Error: 'name', 'namespace', and 'metrics' are required for set_autoscaler_metrics"
        return await run_blocking(
            manager.set_autoscaler_metrics, name, namespace, metrics
        )
    elif action == "scale_deployment_autoscaler":
        if not name or not namespace or min_replicas is None or max_replicas is None:
            return "Error: 'name', 'namespace', 'min_replicas', and 'max_replicas' are required for scale_deployment_autoscaler"
        return await run_blocking(
            manager.scale_deployment_autoscaler,
            name,
            namespace,
            min_replicas,
            max_replicas,
        )
    elif action == "get_autoscaler_history":
        if not name or not namespace:
            return (
                "Error: 'name' and 'namespace' are required for get_autoscaler_history"
            )
        return await run_blocking(manager.get_autoscaler_history, name, namespace)
    return _UNHANDLED


async def _dispatch_watch_stream_action(
    action, manager, field_selector, name, namespace, resource_type, tail_lines
):
    if action == "watch_resource":
        if not resource_type or not name:
            return "Error: 'resource_type' and 'name' are required for watch_resource"
        return await run_blocking(
            manager.watch_resource, resource_type, name, namespace
        )
    elif action == "stream_pod_logs":
        if not name or not namespace:
            return "Error: 'name' and 'namespace' are required for stream_pod_logs"
        return await run_blocking(
            manager.stream_pod_logs, name, namespace, tail_lines or 100
        )
    elif action == "get_resource_events":
        if not resource_type or not name:
            return (
                "Error: 'resource_type' and 'name' are required for get_resource_events"
            )
        return await run_blocking(
            manager.get_resource_events, resource_type, name, namespace
        )
    elif action == "list_field_selector":
        if not resource_type or not field_selector:
            return "Error: 'resource_type' and 'field_selector' are required for list_field_selector"
        return await run_blocking(
            manager.list_field_selector,
            resource_type,
            field_selector,
            namespace,
        )
    return _UNHANDLED


async def _dispatch_debug_action(action, manager, name, namespace):
    if action == "debug_pod":
        if not name or not namespace:
            return "Error: 'name' and 'namespace' are required for debug_pod"
        return await run_blocking(manager.debug_pod, name, namespace)
    elif action == "debug_node":
        if not name:
            return "Error: 'name' is required for debug_node"
        return await run_blocking(manager.debug_node, name)
    elif action == "debug_service":
        if not name or not namespace:
            return "Error: 'name' and 'namespace' are required for debug_service"
        return await run_blocking(manager.debug_service, name, namespace)
    elif action == "debug_deployment":
        if not name or not namespace:
            return "Error: 'name' and 'namespace' are required for debug_deployment"
        return await run_blocking(manager.debug_deployment, name, namespace)
    return _UNHANDLED


_ACTION_GROUPS: dict[str, str] = {
    "top_pods": "metrics",
    "top_nodes": "metrics",
    "get_pod_metrics": "metrics",
    "get_node_metrics": "metrics",
    "get_pod_resource_usage": "metrics",
    "get_cluster_resource_summary": "metrics",
    "get_autoscaler_metrics": "autoscaler",
    "set_autoscaler_metrics": "autoscaler",
    "scale_deployment_autoscaler": "autoscaler",
    "get_autoscaler_history": "autoscaler",
    "watch_resource": "watch_stream",
    "stream_pod_logs": "watch_stream",
    "get_resource_events": "watch_stream",
    "list_field_selector": "watch_stream",
    "debug_pod": "debug",
    "debug_node": "debug",
    "debug_service": "debug",
    "debug_deployment": "debug",
}

_GROUP_FUNCS = {
    "metrics": _dispatch_metrics_action,
    "autoscaler": _dispatch_autoscaler_action,
    "watch_stream": _dispatch_watch_stream_action,
    "debug": _dispatch_debug_action,
}

_GROUP_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "metrics": (
        "name",
        "namespace",
    ),
    "autoscaler": (
        "max_replicas",
        "metrics",
        "min_replicas",
        "name",
        "namespace",
    ),
    "watch_stream": (
        "field_selector",
        "name",
        "namespace",
        "resource_type",
        "tail_lines",
    ),
    "debug": (
        "name",
        "namespace",
    ),
}


def register_k8sobservability_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Observability Operations",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        tags={"kubernetes", "observability", "metrics"},
    )
    async def cm_k8s_observability(
        action: Literal[
            # Metrics
            "top_pods",
            "top_nodes",
            "get_pod_metrics",
            "get_node_metrics",
            "get_pod_resource_usage",
            "get_cluster_resource_summary",
            # Autoscaler metrics
            "get_autoscaler_metrics",
            "set_autoscaler_metrics",
            "scale_deployment_autoscaler",
            "get_autoscaler_history",
            # Watch / stream / events
            "watch_resource",
            "stream_pod_logs",
            "get_resource_events",
            "list_field_selector",
            # Debug helpers
            "debug_pod",
            "debug_node",
            "debug_service",
            "debug_deployment",
        ] = Field(
            description="Observability action to perform (metrics, autoscaler metrics, watch/stream/events, debug helpers)."
        ),
        name: str | None = Field(
            default=None, description="Resource name for the operation"
        ),
        namespace: str | None = Field(
            default=None, description="Target namespace (default: from config)"
        ),
        resource_type: str | None = Field(
            default=None, description="Resource type for watch/events/field-selector"
        ),
        field_selector: str | None = Field(
            default=None, description="Field selector for list_field_selector"
        ),
        tail_lines: int | None = Field(
            default=None, description="Tail lines for stream_pod_logs (default: 100)"
        ),
        metrics: list | None = Field(
            default=None, description="Metrics for set_autoscaler_metrics"
        ),
        min_replicas: int | None = Field(
            default=None, description="Minimum replicas for scale_deployment_autoscaler"
        ),
        max_replicas: int | None = Field(
            default=None, description="Maximum replicas for scale_deployment_autoscaler"
        ),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers)."""
        manager = create_manager(manager_type or "kubernetes")
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_observability: {action}")

        try:
            group = _ACTION_GROUPS.get(action)
            if group is None:
                return f"Error: Unknown action '{action}'"
            all_values = {
                "field_selector": field_selector,
                "max_replicas": max_replicas,
                "metrics": metrics,
                "min_replicas": min_replicas,
                "name": name,
                "namespace": namespace,
                "resource_type": resource_type,
                "tail_lines": tail_lines,
            }
            group_kwargs = {n: all_values[n] for n in _GROUP_PARAM_NAMES[group]}
            return await _GROUP_FUNCS[group](action, manager, **group_kwargs)
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"

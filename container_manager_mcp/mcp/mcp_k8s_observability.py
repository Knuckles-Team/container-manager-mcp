"""MCP tools for Kubernetes observability operations.

Themed dispatcher covering resource metrics (top/pod/node/cluster), autoscaler
metrics/history, watch/stream/events, field-selector listing, and debug helpers.
"""

import logging
from collections.abc import Callable, Coroutine
from typing import Any, Literal

from agent_connector_sdk.mcp.concurrency import run_blocking
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


_AUTOSCALER_ACTIONS = {
    "get_autoscaler_metrics": (
        lambda v: not v["name"] or not v["namespace"],
        "'name' and 'namespace' are required for get_autoscaler_metrics",
        lambda v: (v["name"], v["namespace"]),
    ),
    "set_autoscaler_metrics": (
        lambda v: not v["name"] or not v["namespace"] or not v["metrics"],
        "'name', 'namespace', and 'metrics' are required for set_autoscaler_metrics",
        lambda v: (v["name"], v["namespace"], v["metrics"]),
    ),
    "scale_deployment_autoscaler": (
        lambda v: (
            not v["name"]
            or not v["namespace"]
            or v["min_replicas"] is None
            or v["max_replicas"] is None
        ),
        "'name', 'namespace', 'min_replicas', and 'max_replicas' are required for scale_deployment_autoscaler",
        lambda v: (v["name"], v["namespace"], v["min_replicas"], v["max_replicas"]),
    ),
    "get_autoscaler_history": (
        lambda v: not v["name"] or not v["namespace"],
        "'name' and 'namespace' are required for get_autoscaler_history",
        lambda v: (v["name"], v["namespace"]),
    ),
}


async def _dispatch_autoscaler_action(
    action, manager, max_replicas, metrics, min_replicas, name, namespace
):
    """Dispatch an autoscaler-metrics action via `_AUTOSCALER_ACTIONS`."""
    spec = _AUTOSCALER_ACTIONS.get(action)
    if spec is None:
        return _UNHANDLED
    is_missing, error_message, build_args = spec
    values = {
        "max_replicas": max_replicas,
        "metrics": metrics,
        "min_replicas": min_replicas,
        "name": name,
        "namespace": namespace,
    }
    if is_missing(values):
        return f"Error: {error_message}"
    return await run_blocking(getattr(manager, action), *build_args(values))


_WATCH_STREAM_ACTIONS = {
    "watch_resource": (
        ("resource_type", "name"),
        "'resource_type' and 'name' are required for watch_resource",
        lambda v: (v["resource_type"], v["name"], v["namespace"]),
    ),
    "stream_pod_logs": (
        ("name", "namespace"),
        "'name' and 'namespace' are required for stream_pod_logs",
        lambda v: (v["name"], v["namespace"], v["tail_lines"] or 100),
    ),
    "get_resource_events": (
        ("resource_type", "name"),
        "'resource_type' and 'name' are required for get_resource_events",
        lambda v: (v["resource_type"], v["name"], v["namespace"]),
    ),
    "list_field_selector": (
        ("resource_type", "field_selector"),
        "'resource_type' and 'field_selector' are required for list_field_selector",
        lambda v: (v["resource_type"], v["field_selector"], v["namespace"]),
    ),
}


async def _dispatch_watch_stream_action(
    action, manager, field_selector, name, namespace, resource_type, tail_lines
):
    """Dispatch a watch/stream/events action via `_WATCH_STREAM_ACTIONS`."""
    spec = _WATCH_STREAM_ACTIONS.get(action)
    if spec is None:
        return _UNHANDLED
    required, error_message, build_args = spec
    values = {
        "field_selector": field_selector,
        "name": name,
        "namespace": namespace,
        "resource_type": resource_type,
        "tail_lines": tail_lines,
    }
    if not all(values[name_] for name_ in required):
        return f"Error: {error_message}"
    return await run_blocking(getattr(manager, action), *build_args(values))


_DEBUG_ACTIONS = {
    "debug_pod": (
        ("name", "namespace"),
        "'name' and 'namespace' are required for debug_pod",
    ),
    "debug_node": (("name",), "'name' is required for debug_node"),
    "debug_service": (
        ("name", "namespace"),
        "'name' and 'namespace' are required for debug_service",
    ),
    "debug_deployment": (
        ("name", "namespace"),
        "'name' and 'namespace' are required for debug_deployment",
    ),
}


async def _dispatch_debug_action(action, manager, name, namespace):
    """Dispatch a debug action via `_DEBUG_ACTIONS`; manager method name == action."""
    spec = _DEBUG_ACTIONS.get(action)
    if spec is None:
        return _UNHANDLED
    required, error_message = spec
    values = {"name": name, "namespace": namespace}
    if not all(values[field] for field in required):
        return f"Error: {error_message}"
    return await run_blocking(
        getattr(manager, action), *(values[field] for field in required)
    )


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

_GROUP_FUNCS: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
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

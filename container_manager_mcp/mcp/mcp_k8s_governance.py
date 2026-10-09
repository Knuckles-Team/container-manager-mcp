"""MCP tools for Kubernetes governance operations.

Themed dispatcher covering ResourceQuotas, LimitRanges, PriorityClasses,
PodDisruptionBudgets, and HorizontalPodAutoscalers (full CRUD).
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


_RESOURCE_QUOTA_LIST = "list_resource_quotas"
_RESOURCE_QUOTA_NAME_ONLY = frozenset(
    {"describe_resource_quota", "delete_resource_quota"}
)
_RESOURCE_QUOTA_NAME_SPEC = frozenset(
    {"create_resource_quota", "update_resource_quota"}
)


async def _dispatch_resource_quota_action(action, manager, ns, name, namespace, spec):
    """Dispatch a ResourceQuota CRUD action; manager method name == action."""
    if action == _RESOURCE_QUOTA_LIST:
        return await run_blocking(manager.list_resource_quotas, namespace=namespace)
    if action in _RESOURCE_QUOTA_NAME_ONLY:
        if not name:
            return f"Error: 'name' is required for {action}"
        return await run_blocking(getattr(manager, action), name, ns)
    if action in _RESOURCE_QUOTA_NAME_SPEC:
        if not name or not spec:
            return f"Error: 'name' and 'spec' are required for {action}"
        return await run_blocking(getattr(manager, action), name, ns, spec)
    return _UNHANDLED


async def _dispatch_limit_range_action(action, manager, ns, name, namespace, spec):
    if action == "list_limit_ranges":
        return await run_blocking(manager.list_limit_ranges, namespace=namespace)
    elif action == "describe_limit_range":
        if not name:
            return "Error: 'name' is required for describe_limit_range"
        return await run_blocking(manager.describe_limit_range, name, ns)
    elif action == "create_limit_range":
        if not name or not spec:
            return "Error: 'name' and 'spec' are required for create_limit_range"
        return await run_blocking(manager.create_limit_range, name, ns, spec)
    elif action == "delete_limit_range":
        if not name:
            return "Error: 'name' is required for delete_limit_range"
        return await run_blocking(manager.delete_limit_range, name, ns)
    return _UNHANDLED


async def _dispatch_priority_class_action(action, manager, ns, name, spec):
    if action == "list_priority_classes":
        return await run_blocking(manager.list_priority_classes)
    elif action == "describe_priority_class":
        if not name:
            return "Error: 'name' is required for describe_priority_class"
        return await run_blocking(manager.describe_priority_class, name)
    elif action == "create_priority_class":
        if not name or not spec:
            return "Error: 'name' and 'spec' are required for create_priority_class"
        return await run_blocking(manager.create_priority_class, name, spec)
    elif action == "delete_priority_class":
        if not name:
            return "Error: 'name' is required for delete_priority_class"
        return await run_blocking(manager.delete_priority_class, name)
    return _UNHANDLED


async def _dispatch_pod_disruption_budget_action(
    action, manager, ns, name, namespace, spec
):
    if action == "list_pod_disruption_budgets":
        return await run_blocking(
            manager.list_pod_disruption_budgets, namespace=namespace
        )
    elif action == "describe_pod_disruption_budget":
        if not name:
            return "Error: 'name' is required for describe_pod_disruption_budget"
        return await run_blocking(manager.describe_pod_disruption_budget, name, ns)
    elif action == "create_pod_disruption_budget":
        if not name or not spec:
            return (
                "Error: 'name' and 'spec' are required for create_pod_disruption_budget"
            )
        return await run_blocking(manager.create_pod_disruption_budget, name, ns, spec)
    elif action == "delete_pod_disruption_budget":
        if not name:
            return "Error: 'name' is required for delete_pod_disruption_budget"
        return await run_blocking(manager.delete_pod_disruption_budget, name, ns)
    return _UNHANDLED


_HPA_LIST = "list_horizontal_pod_autoscalers"
_HPA_NAME_ONLY = frozenset(
    {"describe_horizontal_pod_autoscaler", "delete_horizontal_pod_autoscaler"}
)
_HPA_NAME_SPEC = frozenset(
    {"create_horizontal_pod_autoscaler", "update_horizontal_pod_autoscaler"}
)


async def _dispatch_horizontal_pod_autoscaler_action(
    action, manager, ns, name, namespace, spec
):
    """Dispatch a HorizontalPodAutoscaler CRUD action; manager method name == action."""
    if action == _HPA_LIST:
        return await run_blocking(
            manager.list_horizontal_pod_autoscalers, namespace=namespace
        )
    if action in _HPA_NAME_ONLY:
        if not name:
            return f"Error: 'name' is required for {action}"
        return await run_blocking(getattr(manager, action), name, ns)
    if action in _HPA_NAME_SPEC:
        if not name or not spec:
            return f"Error: 'name' and 'spec' are required for {action}"
        return await run_blocking(getattr(manager, action), name, ns, spec)
    return _UNHANDLED


_ACTION_GROUPS: dict[str, str] = {
    "list_resource_quotas": "resource_quota",
    "describe_resource_quota": "resource_quota",
    "create_resource_quota": "resource_quota",
    "update_resource_quota": "resource_quota",
    "delete_resource_quota": "resource_quota",
    "list_limit_ranges": "limit_range",
    "describe_limit_range": "limit_range",
    "create_limit_range": "limit_range",
    "delete_limit_range": "limit_range",
    "list_priority_classes": "priority_class",
    "describe_priority_class": "priority_class",
    "create_priority_class": "priority_class",
    "delete_priority_class": "priority_class",
    "list_pod_disruption_budgets": "pod_disruption_budget",
    "describe_pod_disruption_budget": "pod_disruption_budget",
    "create_pod_disruption_budget": "pod_disruption_budget",
    "delete_pod_disruption_budget": "pod_disruption_budget",
    "list_horizontal_pod_autoscalers": "horizontal_pod_autoscaler",
    "describe_horizontal_pod_autoscaler": "horizontal_pod_autoscaler",
    "create_horizontal_pod_autoscaler": "horizontal_pod_autoscaler",
    "update_horizontal_pod_autoscaler": "horizontal_pod_autoscaler",
    "delete_horizontal_pod_autoscaler": "horizontal_pod_autoscaler",
}

_GROUP_FUNCS: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
    "resource_quota": _dispatch_resource_quota_action,
    "limit_range": _dispatch_limit_range_action,
    "priority_class": _dispatch_priority_class_action,
    "pod_disruption_budget": _dispatch_pod_disruption_budget_action,
    "horizontal_pod_autoscaler": _dispatch_horizontal_pod_autoscaler_action,
}

_GROUP_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "resource_quota": (
        "name",
        "namespace",
        "spec",
    ),
    "limit_range": (
        "name",
        "namespace",
        "spec",
    ),
    "priority_class": (
        "name",
        "spec",
    ),
    "pod_disruption_budget": (
        "name",
        "namespace",
        "spec",
    ),
    "horizontal_pod_autoscaler": (
        "name",
        "namespace",
        "spec",
    ),
}


def register_k8sgovernance_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Governance Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "governance"},
    )
    async def cm_k8s_governance(
        action: Literal[
            # ResourceQuotas
            "list_resource_quotas",
            "describe_resource_quota",
            "create_resource_quota",
            "update_resource_quota",
            "delete_resource_quota",
            # LimitRanges
            "list_limit_ranges",
            "describe_limit_range",
            "create_limit_range",
            "delete_limit_range",
            # PriorityClasses
            "list_priority_classes",
            "describe_priority_class",
            "create_priority_class",
            "delete_priority_class",
            # PodDisruptionBudgets
            "list_pod_disruption_budgets",
            "describe_pod_disruption_budget",
            "create_pod_disruption_budget",
            "delete_pod_disruption_budget",
            # HorizontalPodAutoscalers
            "list_horizontal_pod_autoscalers",
            "describe_horizontal_pod_autoscaler",
            "create_horizontal_pod_autoscaler",
            "update_horizontal_pod_autoscaler",
            "delete_horizontal_pod_autoscaler",
        ] = Field(
            description="Governance action to perform (resource quotas, limit ranges, priority classes, PDBs, HPAs)."
        ),
        name: str | None = Field(
            default=None, description="Resource name for describe/create/update/delete"
        ),
        namespace: str | None = Field(
            default=None, description="Target namespace (default: from config)"
        ),
        spec: dict | None = Field(
            default=None,
            description="Resource specification for create/update operations",
        ),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs)."""
        manager = create_manager(manager_type or "kubernetes")
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_governance: {action}")

        try:
            ns = namespace or getattr(manager, "namespace", namespace)

            group = _ACTION_GROUPS.get(action)
            if group is None:
                return f"Error: Unknown action '{action}'"
            all_values = {
                "name": name,
                "namespace": namespace,
                "spec": spec,
            }
            group_kwargs = {n: all_values[n] for n in _GROUP_PARAM_NAMES[group]}
            return await _GROUP_FUNCS[group](action, manager, ns, **group_kwargs)
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"

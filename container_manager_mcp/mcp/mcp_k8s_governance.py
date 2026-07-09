"""MCP tools for Kubernetes governance operations.

Themed dispatcher covering ResourceQuotas, LimitRanges, PriorityClasses,
PodDisruptionBudgets, and HorizontalPodAutoscalers (full CRUD).
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log


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

            # ResourceQuotas
            if action == "list_resource_quotas":
                return await run_blocking(
                    manager.list_resource_quotas, namespace=namespace
                )
            elif action == "describe_resource_quota":
                if not name:
                    return "Error: 'name' is required for describe_resource_quota"
                return await run_blocking(manager.describe_resource_quota, name, ns)
            elif action == "create_resource_quota":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for create_resource_quota"
                return await run_blocking(manager.create_resource_quota, name, ns, spec)
            elif action == "update_resource_quota":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for update_resource_quota"
                return await run_blocking(manager.update_resource_quota, name, ns, spec)
            elif action == "delete_resource_quota":
                if not name:
                    return "Error: 'name' is required for delete_resource_quota"
                return await run_blocking(manager.delete_resource_quota, name, ns)

            # LimitRanges
            elif action == "list_limit_ranges":
                return await run_blocking(
                    manager.list_limit_ranges, namespace=namespace
                )
            elif action == "describe_limit_range":
                if not name:
                    return "Error: 'name' is required for describe_limit_range"
                return await run_blocking(manager.describe_limit_range, name, ns)
            elif action == "create_limit_range":
                if not name or not spec:
                    return (
                        "Error: 'name' and 'spec' are required for create_limit_range"
                    )
                return await run_blocking(manager.create_limit_range, name, ns, spec)
            elif action == "delete_limit_range":
                if not name:
                    return "Error: 'name' is required for delete_limit_range"
                return await run_blocking(manager.delete_limit_range, name, ns)

            # PriorityClasses
            elif action == "list_priority_classes":
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

            # PodDisruptionBudgets
            elif action == "list_pod_disruption_budgets":
                return await run_blocking(
                    manager.list_pod_disruption_budgets, namespace=namespace
                )
            elif action == "describe_pod_disruption_budget":
                if not name:
                    return (
                        "Error: 'name' is required for describe_pod_disruption_budget"
                    )
                return await run_blocking(
                    manager.describe_pod_disruption_budget, name, ns
                )
            elif action == "create_pod_disruption_budget":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for create_pod_disruption_budget"
                return await run_blocking(
                    manager.create_pod_disruption_budget, name, ns, spec
                )
            elif action == "delete_pod_disruption_budget":
                if not name:
                    return "Error: 'name' is required for delete_pod_disruption_budget"
                return await run_blocking(
                    manager.delete_pod_disruption_budget, name, ns
                )

            # HorizontalPodAutoscalers
            elif action == "list_horizontal_pod_autoscalers":
                return await run_blocking(
                    manager.list_horizontal_pod_autoscalers, namespace=namespace
                )
            elif action == "describe_horizontal_pod_autoscaler":
                if not name:
                    return "Error: 'name' is required for describe_horizontal_pod_autoscaler"
                return await run_blocking(
                    manager.describe_horizontal_pod_autoscaler, name, ns
                )
            elif action == "create_horizontal_pod_autoscaler":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for create_horizontal_pod_autoscaler"
                return await run_blocking(
                    manager.create_horizontal_pod_autoscaler, name, ns, spec
                )
            elif action == "update_horizontal_pod_autoscaler":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for update_horizontal_pod_autoscaler"
                return await run_blocking(
                    manager.update_horizontal_pod_autoscaler, name, ns, spec
                )
            elif action == "delete_horizontal_pod_autoscaler":
                if not name:
                    return (
                        "Error: 'name' is required for delete_horizontal_pod_autoscaler"
                    )
                return await run_blocking(
                    manager.delete_horizontal_pod_autoscaler, name, ns
                )

            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

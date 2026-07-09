"""MCP tools for advanced Kubernetes resource management operations.

This module provides advanced Kubernetes resource management for ResourceQuotas,
LimitRanges, PriorityClasses, PodDisruptionBudgets, HorizontalPodAutoscalers,
Jobs, CronJobs, and ReplicaSets.
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_advanced_resources_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Advanced Resource Management",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "advanced-resources"},
    )
    async def cm_k8s_advanced_resources(
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
            # Jobs
            "list_jobs",
            "describe_job",
            "create_job",
            "delete_job",
            # CronJobs
            "list_cron_jobs",
            "describe_cron_job",
            "create_cron_job",
            "delete_cron_job",
            # ReplicaSets
            "list_replica_sets",
            "describe_replica_set",
            # Cluster Operations
            "cordon_node",
            "uncordon_node",
            "drain_node",
            "cluster_info_dump",
            "get_node_conditions",
            "list_api_resources",
            "describe_api_resource",
            # Deployment Strategies
            "set_deployment_strategy",
            "get_deployment_strategy",
            "set_daemonset_update_strategy",
            "get_daemonset_update_strategy",
            "set_statefulset_update_strategy",
            "get_statefulset_update_strategy",
            "scale_replica_set",
        ] = Field(
            description="Action to perform. Advanced resource management operations."
        ),
        # Common parameters
        name: str | None = Field(default=None, description="Resource name for describe/create/update/delete operations"),
        namespace: str | None = Field(default=None, description="Target namespace (default: from config)"),
        spec: dict | None = Field(default=None, description="Resource specification for create/update operations"),
        grace_period_seconds: int | None = Field(default=None, description="Grace period for drain operation (default: 120)"),
        output_dir: str | None = Field(default=None, description="Output directory for cluster info dump"),
        replicas: int | None = Field(default=None, description="Number of replicas for scaling operations"),
    ) -> dict | list:
        """Manage advanced Kubernetes resources (ResourceQuotas, LimitRanges, PriorityClasses, PodDisruptionBudgets, HPAs, Jobs, CronJobs, ReplicaSets)."""
        
        ctx_log("Advanced resource management", action=action, name=name, namespace=namespace)
        
        @run_blocking
        def execute_operation():
            manager = create_manager("kubernetes")
            k8s_manager = getattr(manager, "k8s_manager", manager)
            
            # ResourceQuotas
            if action == "list_resource_quotas":
                return k8s_manager.list_resource_quotas(namespace=namespace)
            elif action == "describe_resource_quota":
                if not name:
                    raise ValueError("name is required for describe_resource_quota")
                return k8s_manager.describe_resource_quota(name, namespace or k8s_manager.namespace)
            elif action == "create_resource_quota":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_resource_quota")
                return k8s_manager.create_resource_quota(name, namespace or k8s_manager.namespace, spec)
            elif action == "update_resource_quota":
                if not name or not spec:
                    raise ValueError("name and spec are required for update_resource_quota")
                return k8s_manager.update_resource_quota(name, namespace or k8s_manager.namespace, spec)
            elif action == "delete_resource_quota":
                if not name:
                    raise ValueError("name is required for delete_resource_quota")
                return k8s_manager.delete_resource_quota(name, namespace or k8s_manager.namespace)
            
            # LimitRanges
            elif action == "list_limit_ranges":
                return k8s_manager.list_limit_ranges(namespace=namespace)
            elif action == "describe_limit_range":
                if not name:
                    raise ValueError("name is required for describe_limit_range")
                return k8s_manager.describe_limit_range(name, namespace or k8s_manager.namespace)
            elif action == "create_limit_range":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_limit_range")
                return k8s_manager.create_limit_range(name, namespace or k8s_manager.namespace, spec)
            elif action == "delete_limit_range":
                if not name:
                    raise ValueError("name is required for delete_limit_range")
                return k8s_manager.delete_limit_range(name, namespace or k8s_manager.namespace)
            
            # PriorityClasses
            elif action == "list_priority_classes":
                return k8s_manager.list_priority_classes()
            elif action == "describe_priority_class":
                if not name:
                    raise ValueError("name is required for describe_priority_class")
                return k8s_manager.describe_priority_class(name)
            elif action == "create_priority_class":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_priority_class")
                return k8s_manager.create_priority_class(name, spec)
            elif action == "delete_priority_class":
                if not name:
                    raise ValueError("name is required for delete_priority_class")
                return k8s_manager.delete_priority_class(name)
            
            # PodDisruptionBudgets
            elif action == "list_pod_disruption_budgets":
                return k8s_manager.list_pod_disruption_budgets(namespace=namespace)
            elif action == "describe_pod_disruption_budget":
                if not name:
                    raise ValueError("name is required for describe_pod_disruption_budget")
                return k8s_manager.describe_pod_disruption_budget(name, namespace or k8s_manager.namespace)
            elif action == "create_pod_disruption_budget":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_pod_disruption_budget")
                return k8s_manager.create_pod_disruption_budget(name, namespace or k8s_manager.namespace, spec)
            elif action == "delete_pod_disruption_budget":
                if not name:
                    raise ValueError("name is required for delete_pod_disruption_budget")
                return k8s_manager.delete_pod_disruption_budget(name, namespace or k8s_manager.namespace)
            
            # HorizontalPodAutoscalers
            elif action == "list_horizontal_pod_autoscalers":
                return k8s_manager.list_horizontal_pod_autoscalers(namespace=namespace)
            elif action == "describe_horizontal_pod_autoscaler":
                if not name:
                    raise ValueError("name is required for describe_horizontal_pod_autoscaler")
                return k8s_manager.describe_horizontal_pod_autoscaler(name, namespace or k8s_manager.namespace)
            elif action == "create_horizontal_pod_autoscaler":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_horizontal_pod_autoscaler")
                return k8s_manager.create_horizontal_pod_autoscaler(name, namespace or k8s_manager.namespace, spec)
            elif action == "update_horizontal_pod_autoscaler":
                if not name or not spec:
                    raise ValueError("name and spec are required for update_horizontal_pod_autoscaler")
                return k8s_manager.update_horizontal_pod_autoscaler(name, namespace or k8s_manager.namespace, spec)
            elif action == "delete_horizontal_pod_autoscaler":
                if not name:
                    raise ValueError("name is required for delete_horizontal_pod_autoscaler")
                return k8s_manager.delete_horizontal_pod_autoscaler(name, namespace or k8s_manager.namespace)
            
            # Jobs
            elif action == "list_jobs":
                return k8s_manager.list_jobs(namespace=namespace)
            elif action == "describe_job":
                if not name:
                    raise ValueError("name is required for describe_job")
                return k8s_manager.describe_job(name, namespace or k8s_manager.namespace)
            elif action == "create_job":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_job")
                return k8s_manager.create_job(name, namespace or k8s_manager.namespace, spec)
            elif action == "delete_job":
                if not name:
                    raise ValueError("name is required for delete_job")
                return k8s_manager.delete_job(name, namespace or k8s_manager.namespace)
            
            # CronJobs
            elif action == "list_cron_jobs":
                return k8s_manager.list_cron_jobs(namespace=namespace)
            elif action == "describe_cron_job":
                if not name:
                    raise ValueError("name is required for describe_cron_job")
                return k8s_manager.describe_cron_job(name, namespace or k8s_manager.namespace)
            elif action == "create_cron_job":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_cron_job")
                return k8s_manager.create_cron_job(name, namespace or k8s_manager.namespace, spec)
            elif action == "delete_cron_job":
                if not name:
                    raise ValueError("name is required for delete_cron_job")
                return k8s_manager.delete_cron_job(name, namespace or k8s_manager.namespace)
            
            # ReplicaSets
            elif action == "list_replica_sets":
                return k8s_manager.list_replica_sets(namespace=namespace)
            elif action == "describe_replica_set":
                if not name:
                    raise ValueError("name is required for describe_replica_set")
                return k8s_manager.describe_replica_set(name, namespace or k8s_manager.namespace)
            
            # Cluster Operations
            elif action == "cordon_node":
                if not name:
                    raise ValueError("name is required for cordon_node")
                return k8s_manager.cordon_node(name)
            elif action == "uncordon_node":
                if not name:
                    raise ValueError("name is required for uncordon_node")
                return k8s_manager.uncordon_node(name)
            elif action == "drain_node":
                if not name:
                    raise ValueError("name is required for drain_node")
                return k8s_manager.drain_node(name, grace_period_seconds or 120)
            elif action == "cluster_info_dump":
                if not output_dir:
                    raise ValueError("output_dir is required for cluster_info_dump")
                return k8s_manager.cluster_info_dump(output_dir)
            elif action == "get_node_conditions":
                if not name:
                    raise ValueError("name is required for get_node_conditions")
                return k8s_manager.get_node_conditions(name)
            elif action == "list_api_resources":
                return k8s_manager.list_api_resources()
            elif action == "describe_api_resource":
                if not name:
                    raise ValueError("name is required for describe_api_resource")
                return k8s_manager.describe_api_resource(name)
            
            # Deployment Strategies
            elif action == "set_deployment_strategy":
                if not name or not spec:
                    raise ValueError("name and spec are required for set_deployment_strategy")
                return k8s_manager.set_deployment_strategy(name, namespace or k8s_manager.namespace, spec)
            elif action == "get_deployment_strategy":
                if not name:
                    raise ValueError("name is required for get_deployment_strategy")
                return k8s_manager.get_deployment_strategy(name, namespace or k8s_manager.namespace)
            elif action == "set_daemonset_update_strategy":
                if not name or not spec:
                    raise ValueError("name and spec are required for set_daemonset_update_strategy")
                return k8s_manager.set_daemonset_update_strategy(name, namespace or k8s_manager.namespace, spec)
            elif action == "get_daemonset_update_strategy":
                if not name:
                    raise ValueError("name is required for get_daemonset_update_strategy")
                return k8s_manager.get_daemonset_update_strategy(name, namespace or k8s_manager.namespace)
            elif action == "set_statefulset_update_strategy":
                if not name or not spec:
                    raise ValueError("name and spec are required for set_statefulset_update_strategy")
                return k8s_manager.set_statefulset_update_strategy(name, namespace or k8s_manager.namespace, spec)
            elif action == "get_statefulset_update_strategy":
                if not name:
                    raise ValueError("name is required for get_statefulset_update_strategy")
                return k8s_manager.get_statefulset_update_strategy(name, namespace or k8s_manager.namespace)
            elif action == "scale_replica_set":
                if not name or replicas is None:
                    raise ValueError("name and replicas are required for scale_replica_set")
                return k8s_manager.scale_replica_set(name, namespace or k8s_manager.namespace, replicas)
            
            else:
                raise ValueError(f"Unknown action: {action}")
        
        return execute_operation()
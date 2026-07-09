"""MCP tools for advanced Kubernetes operations (Phases 3.1-3.7).

This module provides advanced operations including configuration/watch/steaming, monitoring,
autoscaling, output formatting, plugin system, debug operations, and auth operations.
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_advanced_comprehensive_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Advanced Comprehensive Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "advanced-comprehensive"},
    )
    async def cm_k8s_advanced_comprehensive(
        action: Literal[
            # Phase 3.1: Advanced Configuration (Watch/Streaming)
            "watch_resource",
            "stream_pod_logs",
            "get_resource_events",
            "list_field_selector",
            # Phase 3.2: Advanced Monitoring
            "get_pod_metrics",
            "get_node_metrics",
            "get_top_pods",
            "get_top_nodes",
            "get_pod_resource_usage",
            "get_cluster_resource_summary",
            # Phase 3.3: Advanced Autoscaling
            "get_autoscaler_metrics",
            "set_autoscaler_metrics",
            "scale_deployment_autoscaler",
            "get_autoscaler_history",
            # Phase 3.4: Advanced Output Formatting
            "format_output_json",
            "format_output_yaml",
            "format_output_table",
            "format_output_wide",
            "format_output_custom",
            # Phase 3.5: Plugin System
            "list_cluster_plugins",
            "describe_cluster_plugin",
            "test_cluster_plugin",
            # Phase 3.6: Debug Operations
            "debug_pod",
            "debug_node",
            "debug_service",
            "debug_deployment",
            # Phase 3.7: Auth Operations
            "get_cluster_info",
            "get_api_server_info",
            "validate_kubeconfig",
        ] = Field(
            description="Action to perform. Comprehensive advanced Kubernetes operations."
        ),
        # Common parameters
        name: str | None = Field(default=None, description="Resource name for operations"),
        namespace: str | None = Field(default=None, description="Target namespace (default: from config)"),
        resource_type: str | None = Field(default=None, description="Resource type for operations"),
        field_selector: str | None = Field(default=None, description="Field selector for filtering"),
        tail_lines: int | None = Field(default=None, description="Tail lines for log streaming"),
        expected_data: dict | None = Field(default=None, description="Expected data for comparison"),
        file_path: str | None = Field(default=None, description="File path for sync operations"),
        metrics: list | None = Field(default=None, description="Metrics for autoscaling"),
        min_replicas: int | None = Field(default=None, description="Minimum replicas for autoscaling"),
        max_replicas: int | None = Field(default=None, description="Maximum replicas for autoscaling"),
        plugin_type: str | None = Field(default=None, description="Plugin type (validating/mutating)"),
        test_resource: dict | None = Field(default=None, description="Test resource for plugin testing"),
        data: dict | None = Field(default=None, description="Data for formatting"),
        columns: list | None = Field(default=None, description="Columns for table formatting"),
        template: str | None = Field(default=None, description="Template for custom formatting"),
    ) -> dict | list | str:
        """Comprehensive advanced Kubernetes operations (watch/stream, monitoring, autoscaling, formatting, plugins, debug, auth)."""
        
        ctx_log("Advanced comprehensive operations", action=action, name=name, namespace=namespace)
        
        @run_blocking
        def execute_operation():
            manager = create_manager("kubernetes")
            k8s_manager = getattr(manager, "k8s_manager", manager)
            
            # Phase 3.1: Advanced Configuration
            if action == "watch_resource":
                if not resource_type or not name:
                    raise ValueError("resource_type and name are required for watch_resource")
                return k8s_manager.watch_resource(resource_type, name, namespace)
            elif action == "stream_pod_logs":
                if not name or not namespace:
                    raise ValueError("name and namespace are required for stream_pod_logs")
                return k8s_manager.stream_pod_logs(name, namespace, tail_lines or 100)
            elif action == "get_resource_events":
                if not resource_type or not name:
                    raise ValueError("resource_type and name are required for get_resource_events")
                return k8s_manager.get_resource_events(resource_type, name, namespace)
            elif action == "list_field_selector":
                if not resource_type or not field_selector:
                    raise ValueError("resource_type and field_selector are required for list_field_selector")
                return k8s_manager.list_field_selector(resource_type, field_selector, namespace)
            
            # Phase 3.2: Advanced Monitoring
            elif action == "get_pod_metrics":
                return k8s_manager.get_pod_metrics(namespace)
            elif action == "get_node_metrics":
                return k8s_manager.get_node_metrics()
            elif action == "get_top_pods":
                return k8s_manager.get_top_pods(namespace)
            elif action == "get_top_nodes":
                return k8s_manager.get_top_nodes()
            elif action == "get_pod_resource_usage":
                if not name or not namespace:
                    raise ValueError("name and namespace are required for get_pod_resource_usage")
                return k8s_manager.get_pod_resource_usage(name, namespace)
            elif action == "get_cluster_resource_summary":
                return k8s_manager.get_cluster_resource_summary()
            
            # Phase 3.3: Advanced Autoscaling
            elif action == "get_autoscaler_metrics":
                if not name or not namespace:
                    raise ValueError("name and namespace are required for get_autoscaler_metrics")
                return k8s_manager.get_autoscaler_metrics(name, namespace)
            elif action == "set_autoscaler_metrics":
                if not name or not namespace or not metrics:
                    raise ValueError("name, namespace, and metrics are required for set_autoscaler_metrics")
                return k8s_manager.set_autoscaler_metrics(name, namespace, metrics)
            elif action == "scale_deployment_autoscaler":
                if not name or not namespace or min_replicas is None or max_replicas is None:
                    raise ValueError("name, namespace, min_replicas, and max_replicas are required for scale_deployment_autoscaler")
                return k8s_manager.scale_deployment_autoscaler(name, namespace, min_replicas, max_replicas)
            elif action == "get_autoscaler_history":
                if not name or not namespace:
                    raise ValueError("name and namespace are required for get_autoscaler_history")
                return k8s_manager.get_autoscaler_history(name, namespace)
            
            # Phase 3.4: Advanced Output Formatting
            elif action == "format_output_json":
                if not data:
                    raise ValueError("data is required for format_output_json")
                return k8s_manager.format_output_json(data)
            elif action == "format_output_yaml":
                if not data:
                    raise ValueError("data is required for format_output_yaml")
                return k8s_manager.format_output_yaml(data)
            elif action == "format_output_table":
                if not data or not columns:
                    raise ValueError("data and columns are required for format_output_table")
                return k8s_manager.format_output_table(data, columns)
            elif action == "format_output_wide":
                if not data:
                    raise ValueError("data is required for format_output_wide")
                return k8s_manager.format_output_wide(data)
            elif action == "format_output_custom":
                if not data or not template:
                    raise ValueError("data and template are required for format_output_custom")
                return k8s_manager.format_output_custom(data, template)
            
            # Phase 3.5: Plugin System
            elif action == "list_cluster_plugins":
                return k8s_manager.list_cluster_plugins()
            elif action == "describe_cluster_plugin":
                if not name or not plugin_type:
                    raise ValueError("name and plugin_type are required for describe_cluster_plugin")
                return k8s_manager.describe_cluster_plugin(name, plugin_type)
            elif action == "test_cluster_plugin":
                if not name or not plugin_type or not test_resource:
                    raise ValueError("name, plugin_type, and test_resource are required for test_cluster_plugin")
                return k8s_manager.test_cluster_plugin(name, plugin_type, test_resource)
            
            # Phase 3.6: Debug Operations
            elif action == "debug_pod":
                if not name or not namespace:
                    raise ValueError("name and namespace are required for debug_pod")
                return k8s_manager.debug_pod(name, namespace)
            elif action == "debug_node":
                if not name:
                    raise ValueError("name is required for debug_node")
                return k8s_manager.debug_node(name)
            elif action == "debug_service":
                if not name or not namespace:
                    raise ValueError("name and namespace are required for debug_service")
                return k8s_manager.debug_service(name, namespace)
            elif action == "debug_deployment":
                if not name or not namespace:
                    raise ValueError("name and namespace are required for debug_deployment")
                return k8s_manager.debug_deployment(name, namespace)
            
            # Phase 3.7: Auth Operations
            elif action == "get_cluster_info":
                return k8s_manager.get_cluster_info()
            elif action == "get_api_server_info":
                return k8s_manager.get_api_server_info()
            elif action == "validate_kubeconfig":
                return k8s_manager.validate_kubeconfig()
            
            else:
                raise ValueError(f"Unknown action: {action}")
        
        return execute_operation()

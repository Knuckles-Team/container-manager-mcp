"""MCP tools for Kubernetes cluster operations.

Themed dispatcher covering nodes (list/inspect/cordon/drain/taint/affinity/
conditions), kubeconfig contexts, certificate signing requests, API resources,
cluster info, and admission plugins.
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8scluster_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Cluster Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "cluster"},
    )
    async def cm_k8s_cluster(
        action: Literal[
            # Nodes
            "list_nodes",
            "inspect_node",
            "cordon_node",
            "uncordon_node",
            "drain_node",
            "get_node_conditions",
            "taint_node",
            "untaint_node",
            "list_node_taints",
            "set_node_affinity",
            "get_node_affinity",
            "set_pod_anti_affinity",
            # Contexts
            "list_contexts",
            "use_context",
            "get_config",
            "rename_context",
            "validate_kubeconfig",
            # Certificate signing requests
            "list_csr",
            "approve_csr",
            "deny_csr",
            # API resources
            "list_api_resources",
            "describe_api_resource",
            # Cluster info
            "cluster_info_dump",
            "get_cluster_info",
            "get_api_server_info",
            # Admission plugins
            "list_cluster_plugins",
            "describe_cluster_plugin",
            "test_cluster_plugin",
        ] = Field(
            description="Cluster action to perform (nodes, contexts, CSRs, API resources, cluster info, admission plugins)."
        ),
        node_name: str | None = Field(default=None, description="Node name for node operations"),
        namespace: str | None = Field(default=None, description="Target namespace for affinity operations"),
        pod_name: str | None = Field(default=None, description="Pod name for affinity operations"),
        taints: list | None = Field(default=None, description="Taints for taint_node"),
        taint_key: str | None = Field(default=None, description="Taint key for untaint_node"),
        affinity: dict | None = Field(default=None, description="Affinity configuration for set_node_affinity"),
        anti_affinity: dict | None = Field(default=None, description="Anti-affinity configuration for set_pod_anti_affinity"),
        grace_period_seconds: int | None = Field(default=None, description="Grace period for drain (default: 120)"),
        context_name: str | None = Field(default=None, description="Context name"),
        new_context_name: str | None = Field(default=None, description="New context name for rename_context"),
        csr_name: str | None = Field(default=None, description="CertificateSigningRequest name"),
        reason: str | None = Field(default=None, description="Reason for deny_csr"),
        name: str | None = Field(default=None, description="Resource name (API resource, plugin)"),
        output_dir: str | None = Field(default=None, description="Output directory for cluster_info_dump"),
        plugin_type: str | None = Field(default=None, description="Plugin type (validating/mutating)"),
        test_resource: dict | None = Field(default=None, description="Test resource for test_cluster_plugin"),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins)."""
        manager = create_manager(manager_type or "kubernetes")
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_cluster: {action}")

        try:
            # Nodes
            if action == "list_nodes":
                return await run_blocking(manager.list_nodes)
            elif action == "inspect_node":
                if not node_name:
                    return "Error: 'node_name' is required for inspect_node"
                return await run_blocking(manager.inspect_node, node_name)
            elif action == "cordon_node":
                if not node_name:
                    return "Error: 'node_name' is required for cordon_node"
                return await run_blocking(manager.cordon_node, node_name)
            elif action == "uncordon_node":
                if not node_name:
                    return "Error: 'node_name' is required for uncordon_node"
                return await run_blocking(manager.uncordon_node, node_name)
            elif action == "drain_node":
                if not node_name:
                    return "Error: 'node_name' is required for drain_node"
                return await run_blocking(manager.drain_node, node_name, grace_period_seconds or 120)
            elif action == "get_node_conditions":
                if not node_name:
                    return "Error: 'node_name' is required for get_node_conditions"
                return await run_blocking(manager.get_node_conditions, node_name)
            elif action == "taint_node":
                if not node_name or not taints:
                    return "Error: 'node_name' and 'taints' are required for taint_node"
                return await run_blocking(manager.taint_node, node_name, taints)
            elif action == "untaint_node":
                if not node_name or not taint_key:
                    return "Error: 'node_name' and 'taint_key' are required for untaint_node"
                return await run_blocking(manager.untaint_node, node_name, taint_key)
            elif action == "list_node_taints":
                return await run_blocking(manager.list_node_taints)
            elif action == "set_node_affinity":
                if not pod_name or not namespace or not affinity:
                    return "Error: 'pod_name', 'namespace', and 'affinity' are required for set_node_affinity"
                return await run_blocking(manager.set_node_affinity, pod_name, namespace, affinity)
            elif action == "get_node_affinity":
                if not pod_name or not namespace:
                    return "Error: 'pod_name' and 'namespace' are required for get_node_affinity"
                return await run_blocking(manager.get_node_affinity, pod_name, namespace)
            elif action == "set_pod_anti_affinity":
                if not pod_name or not namespace or not anti_affinity:
                    return "Error: 'pod_name', 'namespace', and 'anti_affinity' are required for set_pod_anti_affinity"
                return await run_blocking(manager.set_pod_anti_affinity, pod_name, namespace, anti_affinity)

            # Contexts
            elif action == "list_contexts":
                return await run_blocking(manager.list_contexts)
            elif action == "use_context":
                if not context_name:
                    return "Error: 'context_name' is required for use_context"
                return await run_blocking(manager.use_context, context_name=context_name)
            elif action == "get_config":
                return await run_blocking(manager.get_config)
            elif action == "rename_context":
                if not context_name or not new_context_name:
                    return "Error: 'context_name' and 'new_context_name' are required for rename_context"
                return await run_blocking(
                    manager.rename_context, current_name=context_name, new_name=new_context_name
                )
            elif action == "validate_kubeconfig":
                return await run_blocking(manager.validate_kubeconfig)

            # Certificate signing requests
            elif action == "list_csr":
                return await run_blocking(manager.list_certificate_signing_requests)
            elif action == "approve_csr":
                if not csr_name:
                    return "Error: 'csr_name' is required for approve_csr"
                return await run_blocking(manager.approve_csr, csr_name)
            elif action == "deny_csr":
                if not csr_name:
                    return "Error: 'csr_name' is required for deny_csr"
                if reason:
                    return await run_blocking(manager.deny_csr, csr_name, reason)
                return await run_blocking(manager.deny_csr, csr_name)

            # API resources
            elif action == "list_api_resources":
                return await run_blocking(manager.list_api_resources)
            elif action == "describe_api_resource":
                if not name:
                    return "Error: 'name' is required for describe_api_resource"
                return await run_blocking(manager.describe_api_resource, name)

            # Cluster info
            elif action == "cluster_info_dump":
                if not output_dir:
                    return "Error: 'output_dir' is required for cluster_info_dump"
                return await run_blocking(manager.cluster_info_dump, output_dir)
            elif action == "get_cluster_info":
                return await run_blocking(manager.get_cluster_info)
            elif action == "get_api_server_info":
                return await run_blocking(manager.get_api_server_info)

            # Admission plugins
            elif action == "list_cluster_plugins":
                return await run_blocking(manager.list_cluster_plugins)
            elif action == "describe_cluster_plugin":
                if not name or not plugin_type:
                    return "Error: 'name' and 'plugin_type' are required for describe_cluster_plugin"
                return await run_blocking(manager.describe_cluster_plugin, name, plugin_type)
            elif action == "test_cluster_plugin":
                if not name or not plugin_type or not test_resource:
                    return "Error: 'name', 'plugin_type', and 'test_resource' are required for test_cluster_plugin"
                return await run_blocking(manager.test_cluster_plugin, name, plugin_type, test_resource)

            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

"""MCP tools for advanced Kubernetes RBAC operations.

This module provides advanced RBAC and security operations including ServiceAccount token management,
SubjectAccessReview, role aggregation, and pod security admission policies.
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_advanced_rbac_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Advanced RBAC Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "advanced-rbac"},
    )
    async def cm_k8s_advanced_rbac(
        action: Literal[
            # ServiceAccount Token Management
            "create_service_account_token",
            "list_service_account_tokens",
            "delete_service_account_token",
            # SubjectAccessReview
            "subject_access_review",
            "local_subject_access_review",
            # Role Aggregation
            "create_aggregated_cluster_role",
            "update_aggregated_cluster_role",
            # Pod Security Admission Policies
            "list_pod_security_policies",
            "describe_pod_security_policy",
            "create_pod_security_policy",
            "delete_pod_security_policy",
            "evaluate_pod_security",
        ] = Field(
            description="Action to perform. Advanced RBAC and security operations."
        ),
        # Common parameters
        name: str | None = Field(default=None, description="Resource name for operations"),
        namespace: str | None = Field(default=None, description="Target namespace (default: from config)"),
        spec: dict | None = Field(default=None, description="Specification for operations"),
        token_name: str | None = Field(default=None, description="Token name for ServiceAccount token operations"),
        aggregation_rule: dict | None = Field(default=None, description="Aggregation rule for ClusterRole operations"),
        pod_spec: dict | None = Field(default=None, description="Pod specification for security evaluation"),
    ) -> dict | list:
        """Manage advanced Kubernetes RBAC operations (ServiceAccount tokens, SubjectAccessReview, role aggregation, pod security policies)."""
        
        ctx_log("Advanced RBAC operations", action=action, name=name, namespace=namespace)
        
        @run_blocking
        def execute_operation():
            manager = create_manager("kubernetes")
            k8s_manager = getattr(manager, "k8s_manager", manager)
            
            # ServiceAccount Token Management
            if action == "create_service_account_token":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_service_account_token")
                return k8s_manager.create_service_account_token(name, namespace or k8s_manager.namespace, spec)
            elif action == "list_service_account_tokens":
                if not name:
                    raise ValueError("name is required for list_service_account_tokens")
                return k8s_manager.list_service_account_tokens(name, namespace or k8s_manager.namespace)
            elif action == "delete_service_account_token":
                if not name or not token_name:
                    raise ValueError("name and token_name are required for delete_service_account_token")
                return k8s_manager.delete_service_account_token(name, namespace or k8s_manager.namespace, token_name)
            
            # SubjectAccessReview
            elif action == "subject_access_review":
                if not spec:
                    raise ValueError("spec is required for subject_access_review")
                return k8s_manager.subject_access_review(spec)
            elif action == "local_subject_access_review":
                if not namespace or not spec:
                    raise ValueError("namespace and spec are required for local_subject_access_review")
                return k8s_manager.local_subject_access_review(namespace, spec)
            
            # Role Aggregation
            elif action == "create_aggregated_cluster_role":
                if not name or not aggregation_rule:
                    raise ValueError("name and aggregation_rule are required for create_aggregated_cluster_role")
                return k8s_manager.create_aggregated_cluster_role(name, aggregation_rule)
            elif action == "update_aggregated_cluster_role":
                if not name or not aggregation_rule:
                    raise ValueError("name and aggregation_rule are required for update_aggregated_cluster_role")
                return k8s_manager.update_aggregated_cluster_role(name, aggregation_rule)
            
            # Pod Security Admission Policies
            elif action == "list_pod_security_policies":
                return k8s_manager.list_pod_security_policies()
            elif action == "describe_pod_security_policy":
                if not name:
                    raise ValueError("name is required for describe_pod_security_policy")
                return k8s_manager.describe_pod_security_policy(name)
            elif action == "create_pod_security_policy":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_pod_security_policy")
                return k8s_manager.create_pod_security_policy(name, spec)
            elif action == "delete_pod_security_policy":
                if not name:
                    raise ValueError("name is required for delete_pod_security_policy")
                return k8s_manager.delete_pod_security_policy(name)
            elif action == "evaluate_pod_security":
                if not namespace or not pod_spec:
                    raise ValueError("namespace and pod_spec are required for evaluate_pod_security")
                return k8s_manager.evaluate_pod_security(namespace, pod_spec)
            
            else:
                raise ValueError(f"Unknown action: {action}")
        
        return execute_operation()
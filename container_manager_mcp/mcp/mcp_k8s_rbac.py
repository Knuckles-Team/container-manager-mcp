"""MCP tools for Kubernetes RBAC and security operations.

Themed dispatcher covering roles/clusterroles/bindings, serviceaccounts, auth
checks, ServiceAccount tokens, SubjectAccessReviews, aggregated cluster roles,
pod security policies, and ServiceAccount-secret mapping.
"""

import json
import logging
from typing import Literal

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log


def register_k8srbac_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes RBAC and Security Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "rbac", "security"},
    )
    async def cm_k8s_rbac(
        action: Literal[
            # Roles / bindings / service accounts
            "list_roles",
            "create_role",
            "delete_role",
            "list_cluster_roles",
            "list_rolebindings",
            "create_rolebinding",
            "delete_rolebinding",
            "list_cluster_rolebindings",
            "create_cluster_rolebinding",
            "delete_cluster_rolebinding",
            "list_serviceaccounts",
            "create_serviceaccount",
            "delete_serviceaccount",
            "auth_can_i",
            # ServiceAccount token management
            "create_service_account_token",
            "list_service_account_tokens",
            "delete_service_account_token",
            # SubjectAccessReview
            "subject_access_review",
            "local_subject_access_review",
            # Aggregated cluster roles
            "create_aggregated_cluster_role",
            "update_aggregated_cluster_role",
            # Pod security policies
            "list_pod_security_policies",
            "describe_pod_security_policy",
            "create_pod_security_policy",
            "delete_pod_security_policy",
            "evaluate_pod_security",
            # ServiceAccount-secret mapping
            "list_service_account_mapped_secrets",
            "map_secret_to_service_account",
            "unmap_secret_from_service_account",
        ] = Field(
            description="RBAC/security action to perform (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping)."
        ),
        namespace: str | None = Field(
            default=None, description="Target namespace (default: from config)"
        ),
        role_name: str | None = Field(default=None, description="Role name"),
        role_rules: str | None = Field(
            default=None, description="Role rules as JSON string"
        ),
        rolebinding_name: str | None = Field(
            default=None, description="RoleBinding name"
        ),
        role_ref: str | None = Field(
            default=None, description="Role reference as JSON string"
        ),
        subjects: str | None = Field(
            default=None, description="Subjects as JSON string"
        ),
        serviceaccount_name: str | None = Field(
            default=None, description="ServiceAccount name"
        ),
        auth_verb: str | None = Field(default=None, description="Verb for auth check"),
        auth_resource: str | None = Field(
            default=None, description="Resource for auth check"
        ),
        name: str | None = Field(
            default=None, description="Resource name for role/binding/SA operations"
        ),
        spec: dict | None = Field(
            default=None, description="Specification for role/binding/SA operations"
        ),
        token_name: str | None = Field(
            default=None, description="Token name for ServiceAccount token operations"
        ),
        aggregation_rule: dict | None = Field(
            default=None, description="Aggregation rule for ClusterRole operations"
        ),
        pod_spec: dict | None = Field(
            default=None, description="Pod specification for security evaluation"
        ),
        secret_name: str | None = Field(
            default=None, description="Secret name for ServiceAccount mapping"
        ),
        sa_name: str | None = Field(
            default=None, description="ServiceAccount name for secret mapping"
        ),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping)."""
        manager = create_manager(manager_type or "kubernetes")
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_rbac: {action}")

        try:
            ns = namespace or getattr(manager, "namespace", namespace)

            # Roles / bindings / service accounts
            if action == "list_roles":
                return await run_blocking(manager.list_roles, namespace=namespace)
            elif action == "create_role":
                if not role_name:
                    return "Error: 'role_name' is required for create_role"
                rules = json.loads(role_rules) if role_rules else None
                return await run_blocking(
                    manager.create_role,
                    name=role_name,
                    namespace=namespace,
                    rules=rules,
                )
            elif action == "delete_role":
                if not role_name:
                    return "Error: 'role_name' is required for delete_role"
                return await run_blocking(
                    manager.delete_role, name=role_name, namespace=namespace
                )
            elif action == "list_cluster_roles":
                return await run_blocking(manager.list_cluster_roles)
            elif action == "list_rolebindings":
                return await run_blocking(
                    manager.list_rolebindings, namespace=namespace
                )
            elif action == "create_rolebinding":
                if not rolebinding_name:
                    return (
                        "Error: 'rolebinding_name' is required for create_rolebinding"
                    )
                role_ref_dict = json.loads(role_ref) if role_ref else None
                subjects_list = json.loads(subjects) if subjects else None
                return await run_blocking(
                    manager.create_rolebinding,
                    name=rolebinding_name,
                    namespace=namespace,
                    role_ref=role_ref_dict,
                    subjects=subjects_list,
                )
            elif action == "delete_rolebinding":
                if not rolebinding_name:
                    return (
                        "Error: 'rolebinding_name' is required for delete_rolebinding"
                    )
                return await run_blocking(
                    manager.delete_rolebinding,
                    name=rolebinding_name,
                    namespace=namespace,
                )
            elif action == "list_cluster_rolebindings":
                return await run_blocking(manager.list_cluster_rolebindings)
            elif action == "create_cluster_rolebinding":
                if not rolebinding_name:
                    return "Error: 'rolebinding_name' is required for create_cluster_rolebinding"
                role_ref_dict = json.loads(role_ref) if role_ref else None
                subjects_list = json.loads(subjects) if subjects else None
                return await run_blocking(
                    manager.create_cluster_rolebinding,
                    name=rolebinding_name,
                    role_ref=role_ref_dict,
                    subjects=subjects_list,
                )
            elif action == "delete_cluster_rolebinding":
                if not rolebinding_name:
                    return "Error: 'rolebinding_name' is required for delete_cluster_rolebinding"
                return await run_blocking(
                    manager.delete_cluster_rolebinding, name=rolebinding_name
                )
            elif action == "list_serviceaccounts":
                return await run_blocking(
                    manager.list_serviceaccounts, namespace=namespace
                )
            elif action == "create_serviceaccount":
                if not serviceaccount_name:
                    return "Error: 'serviceaccount_name' is required for create_serviceaccount"
                return await run_blocking(
                    manager.create_serviceaccount,
                    name=serviceaccount_name,
                    namespace=namespace,
                )
            elif action == "delete_serviceaccount":
                if not serviceaccount_name:
                    return "Error: 'serviceaccount_name' is required for delete_serviceaccount"
                return await run_blocking(
                    manager.delete_serviceaccount,
                    name=serviceaccount_name,
                    namespace=namespace,
                )
            elif action == "auth_can_i":
                if not auth_verb or not auth_resource:
                    return "Error: 'auth_verb' and 'auth_resource' are required for auth_can_i"
                return await run_blocking(
                    manager.auth_can_i,
                    verb=auth_verb,
                    resource=auth_resource,
                    namespace=namespace,
                )

            # ServiceAccount token management
            elif action == "create_service_account_token":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for create_service_account_token"
                return await run_blocking(
                    manager.create_service_account_token, name, ns, spec
                )
            elif action == "list_service_account_tokens":
                if not name:
                    return "Error: 'name' is required for list_service_account_tokens"
                return await run_blocking(manager.list_service_account_tokens, name, ns)
            elif action == "delete_service_account_token":
                if not name or not token_name:
                    return "Error: 'name' and 'token_name' are required for delete_service_account_token"
                return await run_blocking(
                    manager.delete_service_account_token, name, ns, token_name
                )

            # SubjectAccessReview
            elif action == "subject_access_review":
                if not spec:
                    return "Error: 'spec' is required for subject_access_review"
                return await run_blocking(manager.subject_access_review, spec)
            elif action == "local_subject_access_review":
                if not namespace or not spec:
                    return "Error: 'namespace' and 'spec' are required for local_subject_access_review"
                return await run_blocking(
                    manager.local_subject_access_review, namespace, spec
                )

            # Aggregated cluster roles
            elif action == "create_aggregated_cluster_role":
                if not name or not aggregation_rule:
                    return "Error: 'name' and 'aggregation_rule' are required for create_aggregated_cluster_role"
                return await run_blocking(
                    manager.create_aggregated_cluster_role, name, aggregation_rule
                )
            elif action == "update_aggregated_cluster_role":
                if not name or not aggregation_rule:
                    return "Error: 'name' and 'aggregation_rule' are required for update_aggregated_cluster_role"
                return await run_blocking(
                    manager.update_aggregated_cluster_role, name, aggregation_rule
                )

            # Pod security policies
            elif action == "list_pod_security_policies":
                return await run_blocking(manager.list_pod_security_policies)
            elif action == "describe_pod_security_policy":
                if not name:
                    return "Error: 'name' is required for describe_pod_security_policy"
                return await run_blocking(manager.describe_pod_security_policy, name)
            elif action == "create_pod_security_policy":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for create_pod_security_policy"
                return await run_blocking(
                    manager.create_pod_security_policy, name, spec
                )
            elif action == "delete_pod_security_policy":
                if not name:
                    return "Error: 'name' is required for delete_pod_security_policy"
                return await run_blocking(manager.delete_pod_security_policy, name)
            elif action == "evaluate_pod_security":
                if not namespace or not pod_spec:
                    return "Error: 'namespace' and 'pod_spec' are required for evaluate_pod_security"
                return await run_blocking(
                    manager.evaluate_pod_security, namespace, pod_spec
                )

            # ServiceAccount-secret mapping
            elif action == "list_service_account_mapped_secrets":
                if not name:
                    return "Error: 'name' is required for list_service_account_mapped_secrets"
                return await run_blocking(
                    manager.list_service_account_mapped_secrets, name, ns
                )
            elif action == "map_secret_to_service_account":
                if not secret_name or not sa_name:
                    return "Error: 'secret_name' and 'sa_name' are required for map_secret_to_service_account"
                return await run_blocking(
                    manager.map_secret_to_service_account, secret_name, sa_name, ns
                )
            elif action == "unmap_secret_from_service_account":
                if not secret_name or not sa_name:
                    return "Error: 'secret_name' and 'sa_name' are required for unmap_secret_from_service_account"
                return await run_blocking(
                    manager.unmap_secret_from_service_account, secret_name, sa_name, ns
                )

            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}")
            return f"Error executing {action}: {type(e).__name__}"

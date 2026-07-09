"""MCP tools for Kubernetes RBAC and security operations.

This module provides comprehensive RBAC management including roles, rolebindings,
serviceaccounts, and authorization checks.
"""

import json
import logging
import os
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_rbac_tools(mcp: FastMCP):
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
        ] = Field(
            description="Action to perform. Must be one of: 'list_roles', 'create_role', 'delete_role', 'list_cluster_roles', 'list_rolebindings', 'create_rolebinding', 'delete_rolebinding', 'list_cluster_rolebindings', 'create_cluster_rolebinding', 'delete_cluster_rolebinding', 'list_serviceaccounts', 'create_serviceaccount', 'delete_serviceaccount', 'auth_can_i'"
        ),
        role_name: str | None = Field(default=None, description="Role name"),
        namespace: str | None = Field(default=None, description="Target namespace"),
        role_rules: str | None = Field(default=None, description="Role rules as JSON string"),
        rolebinding_name: str | None = Field(default=None, description="RoleBinding name"),
        role_ref: str | None = Field(default=None, description="Role reference as JSON string"),
        subjects: str | None = Field(default=None, description="Subjects as JSON string"),
        serviceaccount_name: str | None = Field(default=None, description="ServiceAccount name"),
        auth_verb: str | None = Field(default=None, description="Verb for auth check"),
        auth_resource: str | None = Field(default=None, description="Resource for auth check"),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list:
        """
        Manage Kubernetes RBAC and security (roles, rolebindings, serviceaccounts, auth).
        
        This tool provides comprehensive RBAC management including creating and managing
        roles, rolebindings, serviceaccounts, and checking authorization permissions.
        """
        # Determine manager type
        if manager_type is None:
            manager_type = "kubernetes"

        manager = create_manager(manager_type)
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_rbac: {action}")

        try:
            if action == "list_roles":
                return await run_blocking(manager.list_roles, namespace=namespace)
            elif action == "create_role":
                if not role_name:
                    return "Error: 'role_name' is required for create_role"
                rules = json.loads(role_rules) if role_rules else None
                return await run_blocking(
                    manager.create_role, name=role_name, namespace=namespace, rules=rules
                )
            elif action == "delete_role":
                if not role_name:
                    return "Error: 'role_name' is required for delete_role"
                return await run_blocking(manager.delete_role, name=role_name, namespace=namespace)
            elif action == "list_cluster_roles":
                return await run_blocking(manager.list_cluster_roles)
            elif action == "list_rolebindings":
                return await run_blocking(manager.list_rolebindings, namespace=namespace)
            elif action == "create_rolebinding":
                if not rolebinding_name:
                    return "Error: 'rolebinding_name' is required for create_rolebinding"
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
                    return "Error: 'rolebinding_name' is required for delete_rolebinding"
                return await run_blocking(
                    manager.delete_rolebinding, name=rolebinding_name, namespace=namespace
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
                return await run_blocking(manager.list_serviceaccounts, namespace=namespace)
            elif action == "create_serviceaccount":
                if not serviceaccount_name:
                    return "Error: 'serviceaccount_name' is required for create_serviceaccount"
                return await run_blocking(
                    manager.create_serviceaccount, name=serviceaccount_name, namespace=namespace
                )
            elif action == "delete_serviceaccount":
                if not serviceaccount_name:
                    return "Error: 'serviceaccount_name' is required for delete_serviceaccount"
                return await run_blocking(
                    manager.delete_serviceaccount, name=serviceaccount_name, namespace=namespace
                )
            elif action == "auth_can_i":
                if not auth_verb or not auth_resource:
                    return "Error: 'auth_verb' and 'auth_resource' are required for auth_can_i"
                return await run_blocking(
                    manager.auth_can_i, verb=auth_verb, resource=auth_resource, namespace=namespace
                )
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

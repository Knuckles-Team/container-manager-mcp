"""MCP tools for advanced Kubernetes networking operations.

This module provides advanced networking operations including ingress class management,
advanced NetworkPolicy rules, service account mapping, and DNS debugging tools.
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8s_advanced_networking_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Advanced Networking Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "advanced-networking"},
    )
    async def cm_k8s_advanced_networking(
        action: Literal[
            # Ingress Class Management
            "list_ingress_classes",
            "describe_ingress_class",
            "create_ingress_class",
            "set_default_ingress_class",
            # Advanced NetworkPolicy Rules
            "create_network_policy_with_cidr",
            "update_network_policy_rules",
            "test_network_policy_connectivity",
            # Service Account Mapping
            "list_service_account_mapped_secrets",
            "map_secret_to_service_account",
            "unmap_secret_from_service_account",
            # DNS Debugging Tools
            "check_dns_resolution",
            "list_dns_endpoints",
            "test_dns_connectivity",
        ] = Field(
            description="Action to perform. Advanced networking operations."
        ),
        # Common parameters
        name: str | None = Field(default=None, description="Resource name for operations"),
        namespace: str | None = Field(default=None, description="Target namespace (default: from config)"),
        spec: dict | None = Field(default=None, description="Specification for operations"),
        rules: list | None = Field(default=None, description="NetworkPolicy rules"),
        secret_name: str | None = Field(default=None, description="Secret name for ServiceAccount mapping"),
        sa_name: str | None = Field(default=None, description="ServiceAccount name for mapping"),
        pod_name: str | None = Field(default=None, description="Pod name for DNS operations"),
        hostname: str | None = Field(default=None, description="Hostname for DNS resolution"),
        service_name: str | None = Field(default=None, description="Service name for DNS operations"),
        target: str | None = Field(default=None, description="Target for DNS connectivity test"),
    ) -> dict | list:
        """Manage advanced Kubernetes networking operations (Ingress classes, advanced NetworkPolicies, ServiceAccount mapping, DNS debugging)."""
        
        ctx_log("Advanced networking operations", action=action, name=name, namespace=namespace)
        
        @run_blocking
        def execute_operation():
            manager = create_manager("kubernetes")
            k8s_manager = getattr(manager, "k8s_manager", manager)
            
            # Ingress Class Management
            if action == "list_ingress_classes":
                return k8s_manager.list_ingress_classes()
            elif action == "describe_ingress_class":
                if not name:
                    raise ValueError("name is required for describe_ingress_class")
                return k8s_manager.describe_ingress_class(name)
            elif action == "create_ingress_class":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_ingress_class")
                return k8s_manager.create_ingress_class(name, spec)
            elif action == "set_default_ingress_class":
                if not name:
                    raise ValueError("name is required for set_default_ingress_class")
                return k8s_manager.set_default_ingress_class(name)
            
            # Advanced NetworkPolicy Rules
            elif action == "create_network_policy_with_cidr":
                if not name or not spec:
                    raise ValueError("name and spec are required for create_network_policy_with_cidr")
                return k8s_manager.create_network_policy_with_cidr(name, namespace or k8s_manager.namespace, spec)
            elif action == "update_network_policy_rules":
                if not name or not rules:
                    raise ValueError("name and rules are required for update_network_policy_rules")
                return k8s_manager.update_network_policy_rules(name, namespace or k8s_manager.namespace, rules)
            elif action == "test_network_policy_connectivity":
                if not namespace or not name:
                    raise ValueError("namespace and policy_name are required for test_network_policy_connectivity")
                return k8s_manager.test_network_policy_connectivity(namespace, name)
            
            # Service Account Mapping
            elif action == "list_service_account_mapped_secrets":
                if not name:
                    raise ValueError("name is required for list_service_account_mapped_secrets")
                return k8s_manager.list_service_account_mapped_secrets(name, namespace or k8s_manager.namespace)
            elif action == "map_secret_to_service_account":
                if not secret_name or not sa_name:
                    raise ValueError("secret_name and sa_name are required for map_secret_to_service_account")
                return k8s_manager.map_secret_to_service_account(secret_name, sa_name, namespace or k8s_manager.namespace)
            elif action == "unmap_secret_from_service_account":
                if not secret_name or not sa_name:
                    raise ValueError("secret_name and sa_name are required for unmap_secret_from_service_account")
                return k8s_manager.unmap_secret_from_service_account(secret_name, sa_name, namespace or k8s_manager.namespace)
            
            # DNS Debugging Tools
            elif action == "check_dns_resolution":
                if not namespace or not pod_name or not hostname:
                    raise ValueError("namespace, pod_name, and hostname are required for check_dns_resolution")
                return k8s_manager.check_dns_resolution(namespace, pod_name, hostname)
            elif action == "list_dns_endpoints":
                if not namespace or not service_name:
                    raise ValueError("namespace and service_name are required for list_dns_endpoints")
                return k8s_manager.list_dns_endpoints(namespace, service_name)
            elif action == "test_dns_connectivity":
                if not namespace or not target:
                    raise ValueError("namespace and target are required for test_dns_connectivity")
                return k8s_manager.test_dns_connectivity(namespace, target)
            
            else:
                raise ValueError(f"Unknown action: {action}")
        
        return execute_operation()

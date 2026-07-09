"""MCP tools for Kubernetes networking operations.

Themed dispatcher covering ingress, ingress classes, network policies (incl.
CIDR / connectivity tests), endpoints/endpointslices, DNS debugging, and native
(core/v1) Services.
"""

import json
import logging
from typing import Literal

from agent_utilities.mcp_utilities import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log


def register_k8snetworking_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Networking Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "networking"},
    )
    async def cm_k8s_networking(
        action: Literal[
            # Ingress
            "list_ingress",
            "create_ingress",
            "delete_ingress",
            # Ingress classes
            "list_ingress_classes",
            "describe_ingress_class",
            "create_ingress_class",
            "set_default_ingress_class",
            # Network policies
            "list_networkpolicies",
            "create_networkpolicy",
            "delete_networkpolicy",
            "create_network_policy_with_cidr",
            "update_network_policy_rules",
            "test_network_policy_connectivity",
            # Endpoints
            "list_endpoints",
            "list_endpointslices",
            # DNS debugging
            "check_dns_resolution",
            "list_dns_endpoints",
            "test_dns_connectivity",
            # Native (core/v1) Services
            "list_k8s_services",
            "get_k8s_service",
            "create_k8s_service",
            "delete_k8s_service",
        ] = Field(
            description="Networking action to perform (ingress, ingress classes, network policies, endpoints, DNS, native services)."
        ),
        namespace: str | None = Field(
            default=None, description="Target namespace (default: from config)"
        ),
        ingress_name: str | None = Field(default=None, description="Ingress name"),
        ingress_spec: str | None = Field(
            default=None, description="Ingress spec as JSON string"
        ),
        netpol_name: str | None = Field(default=None, description="NetworkPolicy name"),
        netpol_spec: str | None = Field(
            default=None, description="NetworkPolicy spec as JSON string"
        ),
        name: str | None = Field(
            default=None,
            description="Resource name (ingress class, network policy, service)",
        ),
        spec: dict | None = Field(
            default=None,
            description="Specification for NetworkPolicy CIDR/rule and service ops",
        ),
        rules: list | None = Field(
            default=None,
            description="NetworkPolicy rules for update_network_policy_rules",
        ),
        pod_name: str | None = Field(
            default=None, description="Pod name for DNS operations"
        ),
        hostname: str | None = Field(
            default=None, description="Hostname for DNS resolution"
        ),
        service_name: str | None = Field(
            default=None, description="Service name for DNS endpoint lookup"
        ),
        target: str | None = Field(
            default=None, description="Target for DNS connectivity test"
        ),
        service_spec: dict | None = Field(
            default=None, description="Full spec dict for create_k8s_service"
        ),
        service_ports: list | None = Field(
            default=None, description="Ports list for create_k8s_service"
        ),
        service_selector: dict | None = Field(
            default=None, description="Selector for create_k8s_service"
        ),
        service_type: str = Field(
            default="ClusterIP", description="Service type for create_k8s_service"
        ),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services)."""
        manager = create_manager(manager_type or "kubernetes")
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_networking: {action}")

        try:
            ns = namespace or getattr(manager, "namespace", namespace)

            # Ingress
            if action == "list_ingress":
                return await run_blocking(manager.list_ingress, namespace=namespace)
            elif action == "create_ingress":
                if not ingress_name:
                    return "Error: 'ingress_name' is required for create_ingress"
                ing_spec = json.loads(ingress_spec) if ingress_spec else None
                return await run_blocking(
                    manager.create_ingress,
                    name=ingress_name,
                    namespace=namespace,
                    spec=ing_spec,
                )
            elif action == "delete_ingress":
                if not ingress_name:
                    return "Error: 'ingress_name' is required for delete_ingress"
                return await run_blocking(
                    manager.delete_ingress, name=ingress_name, namespace=namespace
                )

            # Ingress classes
            elif action == "list_ingress_classes":
                return await run_blocking(manager.list_ingress_classes)
            elif action == "describe_ingress_class":
                if not name:
                    return "Error: 'name' is required for describe_ingress_class"
                return await run_blocking(manager.describe_ingress_class, name)
            elif action == "create_ingress_class":
                if not name or not spec:
                    return (
                        "Error: 'name' and 'spec' are required for create_ingress_class"
                    )
                return await run_blocking(manager.create_ingress_class, name, spec)
            elif action == "set_default_ingress_class":
                if not name:
                    return "Error: 'name' is required for set_default_ingress_class"
                return await run_blocking(manager.set_default_ingress_class, name)

            # Network policies
            elif action == "list_networkpolicies":
                return await run_blocking(
                    manager.list_networkpolicies, namespace=namespace
                )
            elif action == "create_networkpolicy":
                if not netpol_name:
                    return "Error: 'netpol_name' is required for create_networkpolicy"
                np_spec = json.loads(netpol_spec) if netpol_spec else None
                return await run_blocking(
                    manager.create_networkpolicy,
                    name=netpol_name,
                    namespace=namespace,
                    spec=np_spec,
                )
            elif action == "delete_networkpolicy":
                if not netpol_name:
                    return "Error: 'netpol_name' is required for delete_networkpolicy"
                return await run_blocking(
                    manager.delete_networkpolicy, name=netpol_name, namespace=namespace
                )
            elif action == "create_network_policy_with_cidr":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for create_network_policy_with_cidr"
                return await run_blocking(
                    manager.create_network_policy_with_cidr, name, ns, spec
                )
            elif action == "update_network_policy_rules":
                if not name or not rules:
                    return "Error: 'name' and 'rules' are required for update_network_policy_rules"
                return await run_blocking(
                    manager.update_network_policy_rules, name, ns, rules
                )
            elif action == "test_network_policy_connectivity":
                if not namespace or not name:
                    return "Error: 'namespace' and 'name' (policy_name) are required for test_network_policy_connectivity"
                return await run_blocking(
                    manager.test_network_policy_connectivity, namespace, name
                )

            # Endpoints
            elif action == "list_endpoints":
                return await run_blocking(manager.list_endpoints, namespace=namespace)
            elif action == "list_endpointslices":
                return await run_blocking(
                    manager.list_endpointslices, namespace=namespace
                )

            # DNS debugging
            elif action == "check_dns_resolution":
                if not namespace or not pod_name or not hostname:
                    return "Error: 'namespace', 'pod_name', and 'hostname' are required for check_dns_resolution"
                return await run_blocking(
                    manager.check_dns_resolution, namespace, pod_name, hostname
                )
            elif action == "list_dns_endpoints":
                if not namespace or not service_name:
                    return "Error: 'namespace' and 'service_name' are required for list_dns_endpoints"
                return await run_blocking(
                    manager.list_dns_endpoints, namespace, service_name
                )
            elif action == "test_dns_connectivity":
                if not namespace or not target:
                    return "Error: 'namespace' and 'target' are required for test_dns_connectivity"
                return await run_blocking(
                    manager.test_dns_connectivity, namespace, target
                )

            # Native (core/v1) Services
            elif action == "list_k8s_services":
                return await run_blocking(
                    manager.list_native_services, namespace=namespace
                )
            elif action == "get_k8s_service":
                if not name:
                    return "Error: 'name' is required for get_k8s_service"
                return await run_blocking(manager.get_native_service, name, namespace)
            elif action == "create_k8s_service":
                if not name:
                    return "Error: 'name' is required for create_k8s_service"
                return await run_blocking(
                    manager.create_native_service,
                    name=name,
                    namespace=namespace,
                    spec=service_spec,
                    ports=service_ports,
                    selector=service_selector,
                    type=service_type,
                )
            elif action == "delete_k8s_service":
                if not name:
                    return "Error: 'name' is required for delete_k8s_service"
                return await run_blocking(
                    manager.delete_native_service, name, namespace
                )

            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

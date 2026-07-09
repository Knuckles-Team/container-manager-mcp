"""MCP tools for Kubernetes configuration operations.

Themed dispatcher covering ConfigMaps, Secrets, Namespaces, Events, CRDs,
labels/annotations, generic patching, and config/secret state tracking.
"""

import json
import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


def register_k8sconfig_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Configuration Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "config"},
    )
    async def cm_k8s_config(
        action: Literal[
            # ConfigMaps
            "list_configmaps",
            "create_configmap",
            # Secrets
            "list_secrets",
            "create_secret",
            # Namespaces
            "list_namespaces",
            "create_namespace",
            "delete_namespace",
            # Events
            "list_events",
            # CRDs / custom resources
            "list_crds",
            "describe_crd",
            "list_custom_resources",
            # Labels / annotations / patch
            "label_resource",
            "annotate_resource",
            "patch_resource",
            # Config / secret state tracking
            "compare_configmap_state",
            "sync_configmap_from_file",
            "get_secret_state_hash",
            "track_resource_version",
            "wait_for_resource_version",
        ] = Field(
            description="Configuration action to perform (configmaps, secrets, namespaces, events, CRDs, label/annotate/patch, state tracking)."
        ),
        namespace: str | None = Field(
            default=None, description="Target namespace (default: from config)"
        ),
        configmap_name: str | None = Field(default=None, description="ConfigMap name"),
        configmap_data: str | None = Field(
            default=None, description="ConfigMap data as JSON string"
        ),
        configmap_from_file: str | None = Field(
            default=None, description="Path to file for ConfigMap data"
        ),
        secret_name: str | None = Field(default=None, description="Secret name"),
        secret_type: str = Field(
            default="Opaque",
            description="Secret type (Opaque, kubernetes.io/dockerconfigjson, etc.)",
        ),
        secret_data: str | None = Field(
            default=None,
            description="Secret data as JSON string (base64-encoded values)",
        ),
        namespace_name: str | None = Field(
            default=None, description="Namespace name for create/delete namespace"
        ),
        field_selector: str | None = Field(
            default=None, description="Field selector for events"
        ),
        crd_name: str | None = Field(
            default=None, description="CRD name for describe_crd"
        ),
        crd_group: str | None = Field(
            default=None, description="CRD group for custom resources"
        ),
        crd_version: str | None = Field(
            default=None, description="CRD version for custom resources"
        ),
        crd_plural: str | None = Field(
            default=None, description="CRD plural name for custom resources"
        ),
        resource_type: str | None = Field(
            default=None,
            description="Resource type for label/annotate/patch/version operations",
        ),
        resource_name: str | None = Field(
            default=None, description="Resource name for label/annotate operations"
        ),
        labels: str | None = Field(default=None, description="Labels as JSON string"),
        annotations: str | None = Field(
            default=None, description="Annotations as JSON string"
        ),
        name: str | None = Field(
            default=None, description="Resource name for patch/state operations"
        ),
        patch_body: str | None = Field(
            default=None, description="Patch body as JSON string for patch_resource"
        ),
        patch_type: str = Field(
            default="strategic", description="Patch type: strategic, merge, or json"
        ),
        expected_data: dict | None = Field(
            default=None, description="Expected data for configmap state comparison"
        ),
        file_path: str | None = Field(
            default=None, description="File path for configmap sync"
        ),
        target_version: str | None = Field(
            default=None, description="Target resource version to wait for"
        ),
        timeout: int | None = Field(
            default=None, description="Timeout (seconds) for wait operations"
        ),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking)."""
        manager = create_manager(manager_type or "kubernetes")
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_config: {action}")

        try:
            # ConfigMaps
            if action == "list_configmaps":
                return await run_blocking(manager.list_configmaps, namespace=namespace)
            elif action == "create_configmap":
                if not configmap_name:
                    return "Error: 'configmap_name' is required for create_configmap"
                cm_data = json.loads(configmap_data) if configmap_data else None
                return await run_blocking(
                    manager.create_configmap,
                    name=configmap_name,
                    namespace=namespace,
                    data=cm_data,
                    from_file=configmap_from_file,
                )

            # Secrets
            elif action == "list_secrets":
                return await run_blocking(manager.list_secrets, namespace=namespace)
            elif action == "create_secret":
                if not secret_name:
                    return "Error: 'secret_name' is required for create_secret"
                secret_data_dict = json.loads(secret_data) if secret_data else None
                return await run_blocking(
                    manager.create_secret,
                    name=secret_name,
                    namespace=namespace,
                    secret_type=secret_type,
                    data=secret_data_dict,
                )

            # Namespaces
            elif action == "list_namespaces":
                return await run_blocking(manager.list_namespaces)
            elif action == "create_namespace":
                if not namespace_name:
                    return "Error: 'namespace_name' is required for create_namespace"
                return await run_blocking(manager.create_namespace, name=namespace_name)
            elif action == "delete_namespace":
                if not namespace_name:
                    return "Error: 'namespace_name' is required for delete_namespace"
                return await run_blocking(manager.delete_namespace, name=namespace_name)

            # Events
            elif action == "list_events":
                return await run_blocking(
                    manager.list_events,
                    namespace=namespace,
                    field_selector=field_selector,
                )

            # CRDs / custom resources
            elif action == "list_crds":
                return await run_blocking(manager.list_crds)
            elif action == "describe_crd":
                if not crd_name:
                    return "Error: 'crd_name' is required for describe_crd"
                return await run_blocking(manager.describe_crd, crd_name=crd_name)
            elif action == "list_custom_resources":
                if not crd_group or not crd_version or not crd_plural:
                    return "Error: 'crd_group', 'crd_version', and 'crd_plural' are required for list_custom_resources"
                return await run_blocking(
                    manager.list_custom_resources,
                    group=crd_group,
                    version=crd_version,
                    plural=crd_plural,
                    namespace=namespace,
                )

            # Labels / annotations / patch
            elif action == "label_resource":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for label_resource"
                labels_dict = json.loads(labels) if labels else None
                return await run_blocking(
                    manager.label_resource,
                    resource_type=resource_type,
                    name=resource_name,
                    namespace=namespace,
                    labels=labels_dict,
                )
            elif action == "annotate_resource":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for annotate_resource"
                annotations_dict = json.loads(annotations) if annotations else None
                return await run_blocking(
                    manager.annotate_resource,
                    resource_type=resource_type,
                    name=resource_name,
                    namespace=namespace,
                    annotations=annotations_dict,
                )
            elif action == "patch_resource":
                if not resource_type or not name:
                    return "Error: 'resource_type' and 'name' are required for patch_resource"
                patch = json.loads(patch_body) if patch_body else None
                return await run_blocking(
                    manager.patch_resource,
                    resource_type=resource_type,
                    name=name,
                    namespace=namespace,
                    patch_body=patch,
                    patch_type=patch_type,
                )

            # Config / secret state tracking
            elif action == "compare_configmap_state":
                if not name or not namespace or not expected_data:
                    return "Error: 'name', 'namespace', and 'expected_data' are required for compare_configmap_state"
                return await run_blocking(
                    manager.compare_configmap_state, name, namespace, expected_data
                )
            elif action == "sync_configmap_from_file":
                if not name or not namespace or not file_path:
                    return "Error: 'name', 'namespace', and 'file_path' are required for sync_configmap_from_file"
                return await run_blocking(
                    manager.sync_configmap_from_file, name, namespace, file_path
                )
            elif action == "get_secret_state_hash":
                if not name or not namespace:
                    return "Error: 'name' and 'namespace' are required for get_secret_state_hash"
                return await run_blocking(
                    manager.get_secret_state_hash, name, namespace
                )
            elif action == "track_resource_version":
                if not resource_type or not name:
                    return "Error: 'resource_type' and 'name' are required for track_resource_version"
                return await run_blocking(
                    manager.track_resource_version, resource_type, name, namespace
                )
            elif action == "wait_for_resource_version":
                if not resource_type or not name or not namespace or not target_version:
                    return "Error: 'resource_type', 'name', 'namespace', and 'target_version' are required for wait_for_resource_version"
                return await run_blocking(
                    manager.wait_for_resource_version,
                    resource_type,
                    name,
                    namespace,
                    target_version,
                    timeout or 60,
                )

            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

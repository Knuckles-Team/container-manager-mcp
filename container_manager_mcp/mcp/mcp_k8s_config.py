"""MCP tools for Kubernetes configuration operations.

Themed dispatcher covering ConfigMaps, Secrets, Namespaces, Events, CRDs,
labels/annotations, generic patching, and config/secret state tracking.
"""

import json
import logging
from typing import Literal

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log

_UNHANDLED = object()  # sentinel: this action doesn't belong to this dispatch group


async def _dispatch_configmap_action(
    action, manager, configmap_data, configmap_from_file, configmap_name, namespace
):
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
    return _UNHANDLED


async def _dispatch_secret_action(
    action, manager, namespace, secret_data, secret_name, secret_type
):
    if action == "list_secrets":
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
    return _UNHANDLED


async def _dispatch_namespace_action(action, manager, namespace_name):
    if action == "list_namespaces":
        return await run_blocking(manager.list_namespaces)
    elif action == "create_namespace":
        if not namespace_name:
            return "Error: 'namespace_name' is required for create_namespace"
        return await run_blocking(manager.create_namespace, name=namespace_name)
    elif action == "delete_namespace":
        if not namespace_name:
            return "Error: 'namespace_name' is required for delete_namespace"
        return await run_blocking(manager.delete_namespace, name=namespace_name)
    return _UNHANDLED


async def _dispatch_event_action(action, manager, field_selector, namespace):
    if action == "list_events":
        return await run_blocking(
            manager.list_events,
            namespace=namespace,
            field_selector=field_selector,
        )
    return _UNHANDLED


async def _dispatch_crd_action(
    action, manager, crd_group, crd_name, crd_plural, crd_version, namespace
):
    if action == "list_crds":
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
    return _UNHANDLED


async def _label_resource_action(manager, resource_type, resource_name, namespace, labels):
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


async def _annotate_resource_action(
    manager, resource_type, resource_name, namespace, annotations
):
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


async def _patch_resource_action(manager, resource_type, name, namespace, patch_body, patch_type):
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


async def _dispatch_resource_meta_action(
    action,
    manager,
    annotations,
    labels,
    name,
    namespace,
    patch_body,
    patch_type,
    resource_name,
    resource_type,
):
    if action == "label_resource":
        return await _label_resource_action(
            manager, resource_type, resource_name, namespace, labels
        )
    if action == "annotate_resource":
        return await _annotate_resource_action(
            manager, resource_type, resource_name, namespace, annotations
        )
    if action == "patch_resource":
        return await _patch_resource_action(
            manager, resource_type, name, namespace, patch_body, patch_type
        )
    return _UNHANDLED


async def _compare_configmap_state_action(manager, name, namespace, expected_data):
    if not name or not namespace or not expected_data:
        return "Error: 'name', 'namespace', and 'expected_data' are required for compare_configmap_state"
    return await run_blocking(
        manager.compare_configmap_state, name, namespace, expected_data
    )


async def _sync_configmap_from_file_action(manager, name, namespace, file_path):
    if not name or not namespace or not file_path:
        return "Error: 'name', 'namespace', and 'file_path' are required for sync_configmap_from_file"
    return await run_blocking(
        manager.sync_configmap_from_file, name, namespace, file_path
    )


async def _get_secret_state_hash_action(manager, name, namespace):
    if not name or not namespace:
        return "Error: 'name' and 'namespace' are required for get_secret_state_hash"
    return await run_blocking(manager.get_secret_state_hash, name, namespace)


async def _track_resource_version_action(manager, resource_type, name, namespace):
    if not resource_type or not name:
        return "Error: 'resource_type' and 'name' are required for track_resource_version"
    return await run_blocking(
        manager.track_resource_version, resource_type, name, namespace
    )


async def _wait_for_resource_version_action(
    manager, resource_type, name, namespace, target_version, timeout
):
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


async def _dispatch_state_tracking_action(
    action,
    manager,
    expected_data,
    file_path,
    name,
    namespace,
    resource_type,
    target_version,
    timeout,
):
    if action == "compare_configmap_state":
        return await _compare_configmap_state_action(
            manager, name, namespace, expected_data
        )
    if action == "sync_configmap_from_file":
        return await _sync_configmap_from_file_action(
            manager, name, namespace, file_path
        )
    if action == "get_secret_state_hash":
        return await _get_secret_state_hash_action(manager, name, namespace)
    if action == "track_resource_version":
        return await _track_resource_version_action(manager, resource_type, name, namespace)
    if action == "wait_for_resource_version":
        return await _wait_for_resource_version_action(
            manager, resource_type, name, namespace, target_version, timeout
        )
    return _UNHANDLED


_ACTION_GROUPS: dict[str, str] = {
    "list_configmaps": "configmap",
    "create_configmap": "configmap",
    "list_secrets": "secret",
    "create_secret": "secret",
    "list_namespaces": "namespace",
    "create_namespace": "namespace",
    "delete_namespace": "namespace",
    "list_events": "event",
    "list_crds": "crd",
    "describe_crd": "crd",
    "list_custom_resources": "crd",
    "label_resource": "resource_meta",
    "annotate_resource": "resource_meta",
    "patch_resource": "resource_meta",
    "compare_configmap_state": "state_tracking",
    "sync_configmap_from_file": "state_tracking",
    "get_secret_state_hash": "state_tracking",
    "track_resource_version": "state_tracking",
    "wait_for_resource_version": "state_tracking",
}

_GROUP_FUNCS = {
    "configmap": _dispatch_configmap_action,
    "secret": _dispatch_secret_action,
    "namespace": _dispatch_namespace_action,
    "event": _dispatch_event_action,
    "crd": _dispatch_crd_action,
    "resource_meta": _dispatch_resource_meta_action,
    "state_tracking": _dispatch_state_tracking_action,
}

_GROUP_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "configmap": (
        "configmap_data",
        "configmap_from_file",
        "configmap_name",
        "namespace",
    ),
    "secret": (
        "namespace",
        "secret_data",
        "secret_name",
        "secret_type",
    ),
    "namespace": ("namespace_name",),
    "event": (
        "field_selector",
        "namespace",
    ),
    "crd": (
        "crd_group",
        "crd_name",
        "crd_plural",
        "crd_version",
        "namespace",
    ),
    "resource_meta": (
        "annotations",
        "labels",
        "name",
        "namespace",
        "patch_body",
        "patch_type",
        "resource_name",
        "resource_type",
    ),
    "state_tracking": (
        "expected_data",
        "file_path",
        "name",
        "namespace",
        "resource_type",
        "target_version",
        "timeout",
    ),
}


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
            group = _ACTION_GROUPS.get(action)
            if group is None:
                return f"Error: Unknown action '{action}'"
            all_values = {
                "annotations": annotations,
                "configmap_data": configmap_data,
                "configmap_from_file": configmap_from_file,
                "configmap_name": configmap_name,
                "crd_group": crd_group,
                "crd_name": crd_name,
                "crd_plural": crd_plural,
                "crd_version": crd_version,
                "expected_data": expected_data,
                "field_selector": field_selector,
                "file_path": file_path,
                "labels": labels,
                "name": name,
                "namespace": namespace,
                "namespace_name": namespace_name,
                "patch_body": patch_body,
                "patch_type": patch_type,
                "resource_name": resource_name,
                "resource_type": resource_type,
                "secret_data": secret_data,
                "secret_name": secret_name,
                "secret_type": secret_type,
                "target_version": target_version,
                "timeout": timeout,
            }
            group_kwargs = {n: all_values[n] for n in _GROUP_PARAM_NAMES[group]}
            return await _GROUP_FUNCS[group](action, manager, **group_kwargs)
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"

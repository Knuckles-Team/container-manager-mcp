"""MCP tools for Kubernetes storage operations.

Themed dispatcher covering PersistentVolumes, PersistentVolumeClaims (incl.
expansion), StorageClasses (incl. default/provisioner), VolumeSnapshots, and
CSI drivers.
"""

import json
import logging
from typing import Literal

from agent_connector_sdk.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log


async def _dispatch_pv_action(action, manager, name, spec):
    if action == "list_persistent_volumes":
        return await run_blocking(manager.list_persistent_volumes)
    elif action == "create_persistent_volume":
        if not name or not spec:
            return "Error: 'name' and 'spec' are required for create_persistent_volume"
        return await run_blocking(manager.create_persistent_volume, name, spec)
    return None


async def _create_persistent_volume_claim_action(manager, pvc_name, namespace, pvc_spec):
    if not pvc_name:
        return "Error: 'pvc_name' is required for create_persistent_volume_claim"
    claim_spec = json.loads(pvc_spec) if pvc_spec else None
    return await run_blocking(
        manager.create_persistent_volume_claim,
        name=pvc_name,
        namespace=namespace,
        spec=claim_spec,
    )


async def _delete_persistent_volume_claim_action(manager, pvc_name, namespace):
    if not pvc_name:
        return "Error: 'pvc_name' is required for delete_persistent_volume_claim"
    return await run_blocking(
        manager.delete_persistent_volume_claim, name=pvc_name, namespace=namespace
    )


async def _expand_pvc_action(manager, pvc_name, namespace, pvc_size):
    if not pvc_name or not pvc_size:
        return "Error: 'pvc_name' and 'pvc_size' are required for expand_pvc"
    return await run_blocking(
        manager.expand_pvc, name=pvc_name, namespace=namespace, size=pvc_size
    )


async def _expand_persistent_volume_action(manager, name, namespace, size):
    if not name or not namespace or not size:
        return "Error: 'name', 'namespace', and 'size' are required for expand_persistent_volume"
    return await run_blocking(manager.expand_persistent_volume, name, namespace, size)


async def _dispatch_pvc_action(
    action, manager, name, namespace, pvc_name, pvc_size, pvc_spec, size
):
    if action == "list_persistent_volume_claims":
        return await run_blocking(
            manager.list_persistent_volume_claims, namespace=namespace
        )
    if action == "create_persistent_volume_claim":
        return await _create_persistent_volume_claim_action(manager, pvc_name, namespace, pvc_spec)
    if action == "delete_persistent_volume_claim":
        return await _delete_persistent_volume_claim_action(manager, pvc_name, namespace)
    if action == "expand_pvc":
        return await _expand_pvc_action(manager, pvc_name, namespace, pvc_size)
    if action == "expand_persistent_volume":
        return await _expand_persistent_volume_action(manager, name, namespace, size)
    return None


async def _dispatch_storage_class_action(action, manager, name, parameters, provisioner):
    if action == "list_storage_classes":
        return await run_blocking(manager.list_storage_classes)
    elif action == "create_storage_class":
        if not name or not provisioner:
            return "Error: 'name' and 'provisioner' are required for create_storage_class"
        return await run_blocking(
            manager.create_storage_class, name, provisioner, parameters
        )
    elif action == "set_default_storage_class":
        if not name:
            return "Error: 'name' is required for set_default_storage_class"
        return await run_blocking(manager.set_default_storage_class, name)
    elif action == "get_storage_class_provisioner":
        if not name:
            return "Error: 'name' is required for get_storage_class_provisioner"
        return await run_blocking(manager.get_storage_class_provisioner, name)
    return None


async def _dispatch_volume_snapshot_action(action, manager, name, namespace, spec):
    if action == "list_volume_snapshots":
        return await run_blocking(manager.list_volume_snapshots, namespace=namespace)
    elif action == "create_volume_snapshot":
        if not name or not namespace or not spec:
            return "Error: 'name', 'namespace', and 'spec' are required for create_volume_snapshot"
        return await run_blocking(
            manager.create_volume_snapshot, name, namespace, spec
        )
    return None


async def _dispatch_csi_driver_action(action, manager, driver_name, name):
    if action == "list_csi_drivers":
        return await run_blocking(manager.list_csi_drivers)
    elif action == "describe_csi_driver":
        if not name:
            return "Error: 'name' is required for describe_csi_driver"
        return await run_blocking(manager.describe_csi_driver, name)
    elif action == "get_csi_driver_capacity":
        if not driver_name:
            return "Error: 'driver_name' is required for get_csi_driver_capacity"
        return await run_blocking(manager.get_csi_driver_capacity, driver_name)
    return None


_ACTION_GROUPS: dict[str, str] = {
    "list_persistent_volumes": "pv",
    "create_persistent_volume": "pv",
    "list_persistent_volume_claims": "pvc",
    "create_persistent_volume_claim": "pvc",
    "delete_persistent_volume_claim": "pvc",
    "expand_pvc": "pvc",
    "expand_persistent_volume": "pvc",
    "list_storage_classes": "storage_class",
    "create_storage_class": "storage_class",
    "set_default_storage_class": "storage_class",
    "get_storage_class_provisioner": "storage_class",
    "list_volume_snapshots": "volume_snapshot",
    "create_volume_snapshot": "volume_snapshot",
    "list_csi_drivers": "csi_driver",
    "describe_csi_driver": "csi_driver",
    "get_csi_driver_capacity": "csi_driver",
}

_GROUP_FUNCS = {
    "pv": _dispatch_pv_action,
    "pvc": _dispatch_pvc_action,
    "storage_class": _dispatch_storage_class_action,
    "volume_snapshot": _dispatch_volume_snapshot_action,
    "csi_driver": _dispatch_csi_driver_action,
}

_GROUP_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "pv": ("name", "spec"),
    "pvc": ("name", "namespace", "pvc_name", "pvc_size", "pvc_spec", "size"),
    "storage_class": ("name", "parameters", "provisioner"),
    "volume_snapshot": ("name", "namespace", "spec"),
    "csi_driver": ("driver_name", "name"),
}


def register_k8sstorage_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Storage Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "storage"},
    )
    async def cm_k8s_storage(
        action: Literal[
            # Persistent Volumes
            "list_persistent_volumes",
            "create_persistent_volume",
            # Persistent Volume Claims
            "list_persistent_volume_claims",
            "create_persistent_volume_claim",
            "delete_persistent_volume_claim",
            "expand_pvc",
            "expand_persistent_volume",
            # Storage Classes
            "list_storage_classes",
            "create_storage_class",
            "set_default_storage_class",
            "get_storage_class_provisioner",
            # Volume Snapshots
            "list_volume_snapshots",
            "create_volume_snapshot",
            # CSI Drivers
            "list_csi_drivers",
            "describe_csi_driver",
            "get_csi_driver_capacity",
        ] = Field(
            description="Storage action to perform (PV, PVC, storage classes, volume snapshots, CSI drivers)."
        ),
        namespace: str | None = Field(
            default=None, description="Target namespace (default: from config)"
        ),
        pvc_name: str | None = Field(default=None, description="PVC name"),
        pvc_spec: str | None = Field(
            default=None, description="PVC spec as JSON string"
        ),
        pvc_size: str | None = Field(
            default=None, description="New PVC size for expansion"
        ),
        name: str | None = Field(
            default=None,
            description="Resource name (PV, storage class, snapshot, CSI driver)",
        ),
        spec: dict | None = Field(
            default=None, description="Specification for create operations"
        ),
        size: str | None = Field(
            default=None, description="Size for persistent volume expansion"
        ),
        provisioner: str | None = Field(
            default=None, description="Storage provisioner for create_storage_class"
        ),
        parameters: dict | None = Field(
            default=None, description="Storage parameters for create_storage_class"
        ),
        driver_name: str | None = Field(
            default=None, description="CSI driver name for capacity lookup"
        ),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers)."""
        manager = create_manager(manager_type or "kubernetes")
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_storage: {action}")

        try:
            group = _ACTION_GROUPS.get(action)
            if group is None:
                return f"Error: Unknown action '{action}'"
            all_values = {
                "driver_name": driver_name,
                "name": name,
                "namespace": namespace,
                "parameters": parameters,
                "provisioner": provisioner,
                "pvc_name": pvc_name,
                "pvc_size": pvc_size,
                "pvc_spec": pvc_spec,
                "size": size,
                "spec": spec,
            }
            group_kwargs = {n: all_values[n] for n in _GROUP_PARAM_NAMES[group]}
            return await _GROUP_FUNCS[group](action, manager, **group_kwargs)
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"

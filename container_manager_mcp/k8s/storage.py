"""StorageMixin for KubernetesManager (split from k8s_manager.py)."""

from typing import Any

import container_manager_mcp.k8s_manager as _km


class StorageMixin:
    def list_persistent_volumes(self) -> list[dict]:
        """List PersistentVolumes."""
        params: dict[str, Any] = {}
        try:
            pvs = self.core.list_persistent_volume().items
            result = [
                {
                    "name": pv.metadata.name,
                    "capacity": (
                        pv.spec.capacity.dict()
                        if hasattr(pv.spec.capacity, "dict")
                        else pv.spec.capacity if pv.spec and pv.spec.capacity else {}
                    ),
                    "access_modes": pv.spec.access_modes or [],
                    "reclaim_policy": (
                        pv.spec.persistent_volume_reclaim_policy if pv.spec else ""
                    ),
                    "status": pv.status.phase if pv.status else "unknown",
                    "created": self._ts(pv.metadata.creation_timestamp),
                }
                for pv in pvs
            ]
            self.log_action("list_persistent_volumes", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_persistent_volumes", params, error=e)
            raise RuntimeError(f"Failed to list persistent volumes: {str(e)}") from e

    def list_persistent_volume_claims(self, namespace: str | None = None) -> list[dict]:
        """List PersistentVolumeClaims in a namespace."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            pvcs = self.core.list_namespaced_persistent_volume_claim(ns).items
            result = [
                {
                    "name": pvc.metadata.name,
                    "namespace": pvc.metadata.namespace,
                    "capacity": (
                        pvc.spec.resources.requests.dict()
                        if hasattr(pvc.spec.resources.requests, "dict")
                        else (
                            pvc.spec.resources.requests
                            if pvc.spec
                            and pvc.spec.resources
                            and pvc.spec.resources.requests
                            else {}
                        )
                    ),
                    "access_modes": pvc.spec.access_modes or [],
                    "status": pvc.status.phase if pvc.status else "unknown",
                    "volume_name": pvc.spec.volume_name if pvc.spec else "",
                    "created": self._ts(pvc.metadata.creation_timestamp),
                }
                for pvc in pvcs
            ]
            self.log_action(
                "list_persistent_volume_claims", params, {"count": len(result)}
            )
            return result
        except _km.ApiException as e:
            self.log_action("list_persistent_volume_claims", params, error=e)
            raise RuntimeError(
                f"Failed to list persistent volume claims: {str(e)}"
            ) from e

    def create_persistent_volume_claim(
        self, name: str, namespace: str | None = None, spec: dict | None = None
    ) -> dict:
        """Create a PersistentVolumeClaim."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            ns = namespace or self.namespace
            pvc = _km.k8s_client.V1PersistentVolumeClaim(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=spec
            )
            created = self.core.create_namespaced_persistent_volume_claim(ns, pvc)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
            }
            self.log_action("create_persistent_volume_claim", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_persistent_volume_claim", params, error=e)
            raise RuntimeError(
                f"Failed to create persistent volume claim: {str(e)}"
            ) from e

    def delete_persistent_volume_claim(
        self, name: str, namespace: str | None = None
    ) -> dict:
        """Delete a PersistentVolumeClaim."""
        params = {"name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            self.core.delete_namespaced_persistent_volume_claim(name, ns)
            result = {"deleted": name}
            self.log_action("delete_persistent_volume_claim", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("delete_persistent_volume_claim", params, error=e)
            raise RuntimeError(
                f"Failed to delete persistent volume claim: {str(e)}"
            ) from e

    def list_storage_classes(self) -> list[dict]:
        """List StorageClasses."""
        params: dict[str, Any] = {}
        try:
            storage_classes = self.storage.list_storage_class().items
            result = [
                {
                    "name": sc.metadata.name,
                    "provisioner": sc.provisioner if sc.provisioner else "",
                    "reclaim_policy": sc.reclaim_policy if sc.reclaim_policy else "",
                    "volume_binding_mode": (
                        sc.volume_binding_mode if sc.volume_binding_mode else ""
                    ),
                    "allow_volume_expansion": (
                        sc.allow_volume_expansion
                        if sc.allow_volume_expansion
                        else False
                    ),
                }
                for sc in storage_classes
            ]
            self.log_action("list_storage_classes", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_storage_classes", params, error=e)
            raise RuntimeError(f"Failed to list storage classes: {str(e)}") from e

    def list_volume_snapshots(self, namespace: str | None = None) -> list[dict]:
        """List VolumeSnapshots (requires snapshot.k8s.io CRD)."""
        params: dict[str, Any] = {"namespace": namespace}
        try:
            # Volume snapshots are custom resources under snapshot.k8s.io/v1
            dynamic_client = _km.k8s_client.DynamicClient(self.core.api_client)
            resource = dynamic_client.resources.get(
                api_version="snapshot.storage.k8s.io/v1",
                kind="VolumeSnapshot",
            )
            items = resource.get(
                **({"namespace": namespace} if namespace else {})
            ).items
            result = [
                {
                    "name": item.metadata.name,
                    "namespace": item.metadata.namespace,
                    "status": item.status if hasattr(item, "status") else {},
                    "created": self._ts(item.metadata.creation_timestamp),
                }
                for item in items
            ]
            self.log_action("list_volume_snapshots", params, {"count": len(result)})
            return result
        except ImportError:
            raise RuntimeError("Dynamic client not available") from None
        except Exception as e:
            self.log_action("list_volume_snapshots", params, error=e)
            raise RuntimeError(f"Failed to list volume snapshots: {str(e)}") from e

    def expand_pvc(
        self, name: str, namespace: str | None = None, size: str | None = None
    ) -> dict:
        """Expand a PersistentVolumeClaim size."""
        params = {"name": name, "namespace": namespace, "size": size}
        try:
            ns = namespace or self.namespace
            pvc = self.core.read_namespaced_persistent_volume_claim(name, ns)

            # Update the resources request
            if not pvc.spec.resources:
                pvc.spec.resources = _km.k8s_client.V1ResourceRequirements(requests={})

            pvc.spec.resources.requests["storage"] = size

            self.core.patch_namespaced_persistent_volume_claim(
                name, ns, {"spec": {"resources": {"requests": {"storage": size}}}}
            )
            result = {"name": name, "namespace": ns, "expanded": True, "new_size": size}
            self.log_action("expand_pvc", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("expand_pvc", params, error=e)
            raise RuntimeError(f"Failed to expand PVC: {str(e)}") from e

    def list_csi_drivers(self) -> list[dict]:
        """List CSI drivers."""
        params: dict[str, Any] = {}
        try:
            storage_api = self.storage
            csidrivers = storage_api.list_csi_driver().items
            result = [
                {
                    "name": driver.metadata.name,
                    "attach_required": driver.spec.attach_required,
                    "pod_info_on_mount": (
                        driver.spec.pod_info_on_mount if driver.spec else None
                    ),
                    "storage_capacity": (
                        driver.spec.storage_capacity if driver.spec else None
                    ),
                    "created": self._ts(driver.metadata.creation_timestamp),
                }
                for driver in csidrivers
            ]
            self.log_action("list_csi_drivers", params, {"count": len(result)})
            return result
        except ImportError:
            raise RuntimeError("Storage client not available") from None
        except _km.ApiException as e:
            self.log_action("list_csi_drivers", params, error=e)
            raise RuntimeError(f"Failed to list CSI drivers: {str(e)}") from e

    def describe_csi_driver(self, name: str) -> dict:
        """Describe a CSI driver."""
        params = {"name": name}
        try:
            storage_api = self.storage
            driver = storage_api.read_csi_driver(name)
            result = {
                "name": driver.metadata.name,
                "spec": driver.spec,
                "created": self._ts(driver.metadata.creation_timestamp),
                "labels": driver.metadata.labels,
                "annotations": driver.metadata.annotations,
            }
            self.log_action("describe_csi_driver", params, result)
            return result
        except ImportError:
            raise RuntimeError("Storage client not available") from None
        except _km.ApiException as e:
            self.log_action("describe_csi_driver", params, error=e)
            raise RuntimeError(f"Failed to describe CSI driver: {str(e)}") from e

    def get_csi_driver_capacity(self, driver_name: str) -> dict:
        """Get CSI driver capacity information."""
        params = {"driver_name": driver_name}
        try:
            storage_api = self.storage
            driver = storage_api.read_csi_driver(driver_name)

            result = {
                "driver_name": driver_name,
                "attach_required": driver.spec.attach_required if driver.spec else None,
                "storage_capacity": (
                    driver.spec.storage_capacity if driver.spec else None
                ),
                "volume_lifecycle_modes": (
                    driver.spec.volume_lifecycle_modes if driver.spec else []
                ),
            }
            self.log_action("get_csi_driver_capacity", params, result)
            return result
        except ImportError:
            raise RuntimeError("Storage client not available") from None
        except _km.ApiException as e:
            self.log_action("get_csi_driver_capacity", params, error=e)
            raise RuntimeError(f"Failed to get CSI driver capacity: {str(e)}") from e

    def set_default_storage_class(self, name: str) -> dict:
        """Set the default StorageClass."""
        params = {"name": name}
        try:
            storage_api = self.storage
            storage_classes = storage_api.list_storage_class().items

            # Remove default from all existing classes
            for sc in storage_classes:
                if (
                    sc.metadata.annotations
                    and "storageclass.kubernetes.io/is-default-class"
                    in sc.metadata.annotations
                ):
                    del sc.metadata.annotations[
                        "storageclass.kubernetes.io/is-default-class"
                    ]
                    storage_api.patch_storage_class(sc.metadata.name, sc)

            # Set default on the specified class
            target_sc = storage_api.read_storage_class(name)
            if not target_sc.metadata.annotations:
                target_sc.metadata.annotations = {}
            target_sc.metadata.annotations[
                "storageclass.kubernetes.io/is-default-class"
            ] = "true"
            storage_api.patch_storage_class(name, target_sc)

            result = {"name": name, "status": "set_as_default"}
            self.log_action("set_default_storage_class", params, result)
            return result
        except ImportError:
            raise RuntimeError("Storage client not available") from None
        except _km.ApiException as e:
            self.log_action("set_default_storage_class", params, error=e)
            raise RuntimeError(f"Failed to set default StorageClass: {str(e)}") from e

    def get_storage_class_provisioner(self, name: str) -> dict:
        """Get StorageClass provisioner information."""
        params = {"name": name}
        try:
            storage_api = self.storage
            sc = storage_api.read_storage_class(name)

            result = {
                "name": name,
                "provisioner": sc.provisioner,
                "parameters": sc.parameters,
                "reclaim_policy": sc.reclaim_policy,
                "volume_binding_mode": sc.volume_binding_mode,
                "allow_volume_expansion": sc.allow_volume_expansion,
            }
            self.log_action("get_storage_class_provisioner", params, result)
            return result
        except ImportError:
            raise RuntimeError("Storage client not available") from None
        except _km.ApiException as e:
            self.log_action("get_storage_class_provisioner", params, error=e)
            raise RuntimeError(
                f"Failed to get StorageClass provisioner: {str(e)}"
            ) from e

    def expand_persistent_volume(self, name: str, namespace: str, size: str) -> dict:
        """Expand a PersistentVolumeClaim."""
        params = {"name": name, "namespace": namespace, "size": size}
        try:
            pvc = self.core.read_namespaced_persistent_volume_claim(name, namespace)

            # Update the PVC size
            if not pvc.spec.resources:
                pvc.spec.resources = _km.k8s_client.V1VolumeResourceRequirements()
            if not pvc.spec.resources.requests:
                pvc.spec.resources.requests = {}
            pvc.spec.resources.requests["storage"] = size

            self.core.patch_namespaced_persistent_volume_claim(name, namespace, pvc)

            result = {
                "name": name,
                "namespace": namespace,
                "size": size,
                "status": "expansion_requested",
            }
            self.log_action("expand_persistent_volume", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("expand_persistent_volume", params, error=e)
            raise RuntimeError(f"Failed to expand PersistentVolume: {str(e)}") from e

    def create_volume_snapshot(self, name: str, namespace: str, spec: dict) -> dict:
        """Create a VolumeSnapshot."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            # VolumeSnapshots are CRDs under snapshot.storage.k8s.io/v1.
            body = {
                "apiVersion": "snapshot.storage.k8s.io/v1",
                "kind": "VolumeSnapshot",
                "metadata": {"name": name, "namespace": namespace},
                "spec": spec,
            }
            created = self.custom_objects.create_namespaced_custom_object(
                "snapshot.storage.k8s.io", "v1", namespace, "volumesnapshots", body
            )
            result = {
                "name": name,
                "namespace": namespace,
                "status": "created",
                "created": self._ts(
                    (created.get("metadata") or {}).get("creationTimestamp")
                ),
            }
            self.log_action("create_volume_snapshot", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_volume_snapshot", params, error=e)
            raise RuntimeError(f"Failed to create VolumeSnapshot: {str(e)}") from e

    def create_storage_class(
        self, name: str, provisioner: str, parameters: dict | None = None
    ) -> dict:
        """Create a StorageClass."""
        params = {"name": name, "provisioner": provisioner, "parameters": parameters}
        try:
            storage_api = self.storage
            storage_class = _km.k8s_client.V1StorageClass(
                metadata=_km.k8s_client.V1ObjectMeta(name=name),
                provisioner=provisioner,
                parameters=parameters or {},
            )
            storage_api.create_storage_class(storage_class)
            result = {"name": name, "provisioner": provisioner, "status": "created"}
            self.log_action("create_storage_class", params, result)
            return result
        except Exception as e:
            self.log_action("create_storage_class", params, error=e)
            raise RuntimeError(f"Failed to create storage class: {str(e)}") from e

    def create_persistent_volume(self, name: str, spec: dict) -> dict:
        """Create a PersistentVolume."""
        params = {"name": name, "spec": spec}
        try:
            pv_spec = _km.k8s_client.V1PersistentVolumeSpec(**spec)
            pv = _km.k8s_client.V1PersistentVolume(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=pv_spec
            )
            self.core.create_persistent_volume(pv)
            result = {"name": name, "status": "created"}
            self.log_action("create_persistent_volume", params, result)
            return result
        except Exception as e:
            self.log_action("create_persistent_volume", params, error=e)
            raise RuntimeError(f"Failed to create persistent volume: {str(e)}") from e

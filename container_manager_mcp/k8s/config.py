"""ConfigMixin for KubernetesManager (split from k8s_manager.py)."""

from typing import Any

import container_manager_mcp.k8s_manager as _km


class ConfigMixin:
    _PATCH_TABLE: dict[str, tuple[str, str, bool]] = {
        "pod": ("core", "pod", True),
        "deployment": ("apps", "deployment", True),
        "service": ("core", "service", True),
        "configmap": ("core", "config_map", True),
        "secret": ("core", "secret", True),
        "namespace": ("core", "namespace", False),
        "node": ("core", "node", False),
        "ingress": ("networking", "ingress", True),
        "job": ("batch", "job", True),
        "cronjob": ("batch", "cron_job", True),
        "statefulset": ("apps", "stateful_set", True),
        "daemonset": ("apps", "daemon_set", True),
        "replicaset": ("apps", "replica_set", True),
        "persistentvolumeclaim": ("core", "persistent_volume_claim", True),
        "serviceaccount": ("core", "service_account", True),
        "role": ("rbac", "role", True),
        "rolebinding": ("rbac", "role_binding", True),
        "clusterrole": ("rbac", "cluster_role", False),
        "clusterrolebinding": ("rbac", "cluster_role_binding", False),
    }
    _PATCH_CONTENT_TYPES = {
        "strategic": "application/strategic-merge-patch+json",
        "merge": "application/merge-patch+json",
        "json": "application/json-patch+json",
    }

    def list_configmaps(self, namespace: str | None = None) -> list[dict]:
        """List all ConfigMaps."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            configmaps = self.core.list_namespaced_config_map(ns).items
            result = [
                {
                    "name": cm.metadata.name,
                    "namespace": cm.metadata.namespace,
                    "data_keys": list((cm.data or {}).keys()),
                    "created": self._ts(cm.metadata.creation_timestamp),
                }
                for cm in configmaps
            ]
            self.log_action("list_configmaps", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_configmaps", params, error=e)
            raise RuntimeError(f"Failed to list configmaps: {str(e)}") from e

    def create_configmap(
        self,
        name: str,
        namespace: str | None = None,
        data: dict | None = None,
        from_file: str | None = None,
    ) -> dict:
        """Create a ConfigMap."""
        params = {
            "name": name,
            "namespace": namespace,
            "data": data,
            "from_file": from_file,
        }
        try:
            ns = namespace or self.namespace
            cm_data = {}
            if from_file:
                with open(from_file) as f:
                    cm_data["config"] = f.read()
            if data:
                cm_data.update(data)

            configmap = _km.k8s_client.V1ConfigMap(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), data=cm_data or None
            )
            created = self.core.create_namespaced_config_map(ns, configmap)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
            }
            self.log_action("create_configmap", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_configmap", params, error=e)
            raise RuntimeError(f"Failed to create configmap: {str(e)}") from e

    def list_secrets(self, namespace: str | None = None) -> list[dict]:
        """List all Secrets (metadata only, no data values)."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            secrets = self.core.list_namespaced_secret(ns).items
            result = [
                {
                    "name": secret.metadata.name,
                    "namespace": secret.metadata.namespace,
                    "type": secret.type,
                    "created": self._ts(secret.metadata.creation_timestamp),
                }
                for secret in secrets
            ]
            self.log_action("list_secrets", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_secrets", params, error=e)
            raise RuntimeError(f"Failed to list secrets: {str(e)}") from e

    def create_secret(
        self,
        name: str,
        namespace: str | None = None,
        secret_type: str = "Opaque",
        data: dict | None = None,
    ) -> dict:
        """Create a Secret (data values should be base64-encoded)."""
        params = {
            "name": name,
            "namespace": namespace,
            "secret_type": secret_type,
            "data": data,
        }
        try:
            ns = namespace or self.namespace
            secret = _km.k8s_client.V1Secret(
                metadata=_km.k8s_client.V1ObjectMeta(name=name),
                type=secret_type,
                data=data,
            )
            created = self.core.create_namespaced_secret(ns, secret)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
            }
            self.log_action("create_secret", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_secret", params, error=e)
            raise RuntimeError(f"Failed to create secret: {str(e)}") from e

    def list_namespaces(self) -> list[dict]:
        """List all namespaces."""
        params: dict[str, Any] = {}
        try:
            namespaces = self.core.list_namespace().items
            result = [
                {
                    "name": ns.metadata.name,
                    "status": ns.status.phase if ns.status else "unknown",
                    "created": self._ts(ns.metadata.creation_timestamp),
                    "labels": ns.metadata.labels or {},
                }
                for ns in namespaces
            ]
            self.log_action("list_namespaces", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_namespaces", params, error=e)
            raise RuntimeError(f"Failed to list namespaces: {str(e)}") from e

    def list_events(
        self, namespace: str | None = None, field_selector: str | None = None
    ) -> list[dict]:
        """List events with optional filtering."""
        params = {"namespace": namespace, "field_selector": field_selector}
        try:
            ns = namespace or self.namespace
            if field_selector:
                events = self.core.list_namespaced_event(
                    ns, field_selector=field_selector
                ).items
            else:
                events = self.core.list_namespaced_event(ns).items
            result = [
                {
                    "type": event.type,
                    "reason": event.reason,
                    "message": event.message,
                    "source": event.source.dict() if event.source else {},
                    "first_seen": self._ts(event.first_timestamp),
                    "last_seen": self._ts(event.last_timestamp),
                    "count": event.count,
                }
                for event in events
            ]
            self.log_action("list_events", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_events", params, error=e)
            raise RuntimeError(f"Failed to list events: {str(e)}") from e

    def list_crds(self) -> list[dict]:
        """List Custom Resource Definitions."""
        params: dict[str, Any] = {}
        try:
            apiext = self.apiextensions
            crds = apiext.list_custom_resource_definition().items
            result = [
                {
                    "name": crd.metadata.name,
                    "group": crd.spec.group if crd.spec else "",
                    "scope": crd.spec.scope if crd.spec else "",
                    "names": (
                        crd.spec.names.dict() if crd.spec and crd.spec.names else {}
                    ),
                    "versions": [v.name for v in (crd.spec.versions or [])],
                    "created": self._ts(crd.metadata.creation_timestamp),
                }
                for crd in crds
            ]
            self.log_action("list_crds", params, {"count": len(result)})
            return result
        except ImportError:
            raise RuntimeError("API extensions client not available") from None
        except _km.ApiException as e:
            self.log_action("list_crds", params, error=e)
            raise RuntimeError(f"Failed to list CRDs: {str(e)}") from e

    def describe_crd(self, crd_name: str) -> dict:
        """Get detailed CRD information."""
        params = {"crd_name": crd_name}
        try:
            apiext = self.apiextensions
            crd = apiext.read_custom_resource_definition(crd_name)
            result = crd.to_dict()
            self.log_action("describe_crd", params, {"name": crd_name})
            return result
        except ImportError:
            raise RuntimeError("API extensions client not available") from None
        except _km.ApiException as e:
            self.log_action("describe_crd", params, error=e)
            raise RuntimeError(f"Failed to describe CRD: {str(e)}") from e

    def list_custom_resources(
        self, group: str, version: str, plural: str, namespace: str | None = None
    ) -> list[dict]:
        """List custom resources for a given CRD."""
        params = {
            "group": group,
            "version": version,
            "plural": plural,
            "namespace": namespace,
        }
        try:
            dynamic_client = _km.k8s_client.DynamicClient(self.core.api_client)

            if namespace:
                resource = dynamic_client.resources.get(
                    api_version=f"{group}/{version}",
                    kind=plural,
                    namespace=namespace,
                )
                items = resource.get().items
            else:
                resource = dynamic_client.resources.get(
                    api_version=f"{group}/{version}",
                    kind=plural,
                )
                items = resource.get().items

            result = [
                {
                    "name": item.metadata.name,
                    "namespace": item.metadata.namespace,
                    "created": self._ts(item.metadata.creation_timestamp),
                }
                for item in items
            ]
            self.log_action("list_custom_resources", params, {"count": len(result)})
            return result
        except ImportError:
            raise RuntimeError("Dynamic client not available") from None
        except Exception as e:
            self.log_action("list_custom_resources", params, error=e)
            raise RuntimeError(f"Failed to list custom resources: {str(e)}") from e

    def patch_resource(
        self,
        resource_type: str,
        name: str,
        namespace: str | None = None,
        patch_body: dict | None = None,
        patch_type: str = "strategic",
    ) -> dict:
        """Generic patch for any common Kubernetes resource kind.

        Dispatches on ``resource_type`` to the correct centralized client and
        ``patch_namespaced_<x>`` / ``patch_<x>`` method; unknown kinds fall back
        to the dynamic client resolved by kind.
        """
        params = {
            "resource_type": resource_type,
            "name": name,
            "namespace": namespace,
            "patch_type": patch_type,
        }
        ctype = self._PATCH_CONTENT_TYPES.get(
            patch_type, self._PATCH_CONTENT_TYPES["strategic"]
        )
        body: Any = patch_body if patch_body is not None else {}
        key = resource_type.lower().replace("-", "").replace("_", "")
        try:
            entry = self._PATCH_TABLE.get(key)
            if entry is not None:
                attr, suffix, namespaced = entry
                client = getattr(self, attr)

                def _invoke():
                    if namespaced:
                        ns = namespace or self.namespace
                        return getattr(client, f"patch_namespaced_{suffix}")(
                            name, ns, body
                        )
                    return getattr(client, f"patch_{suffix}")(name, body)

                # Force the requested patch content-type on the shared api_client
                # for the duration of the call, then restore it.
                api_client = client.api_client
                prev = api_client.default_headers.get("Content-Type")
                api_client.set_default_header("Content-Type", ctype)
                try:
                    _invoke()
                finally:
                    if prev is not None:
                        api_client.set_default_header("Content-Type", prev)
                    else:
                        api_client.default_headers.pop("Content-Type", None)
            else:
                # Unknown kind: resolve via the dynamic client by kind name.
                dynamic_client = _km.k8s_client.DynamicClient(self.core.api_client)
                resource = dynamic_client.resources.get(kind=resource_type)
                if namespace:
                    resource.patch(
                        body=body, name=name, namespace=namespace, content_type=ctype
                    )
                else:
                    resource.patch(body=body, name=name, content_type=ctype)
            result = {
                "resource_type": resource_type,
                "name": name,
                "namespace": namespace,
                "patched": True,
                "patch_type": patch_type,
            }
            self.log_action("patch_resource", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("patch_resource", params, error=e)
            raise RuntimeError(
                f"Failed to patch {resource_type} {name}: {str(e)}"
            ) from e

    def label_resource(
        self,
        resource_type: str,
        name: str,
        namespace: str | None = None,
        labels: dict | None = None,
    ) -> dict:
        """Label a Kubernetes resource (table-driven via patch_resource)."""
        params = {
            "resource_type": resource_type,
            "name": name,
            "namespace": namespace,
            "labels": labels,
        }
        self.patch_resource(
            resource_type,
            name,
            namespace=namespace,
            patch_body={"metadata": {"labels": labels or {}}},
            patch_type="merge",
        )
        result = {
            "resource_type": resource_type,
            "name": name,
            "labels_added": list(labels or {}),
        }
        self.log_action("label_resource", params, result)
        return result

    def annotate_resource(
        self,
        resource_type: str,
        name: str,
        namespace: str | None = None,
        annotations: dict | None = None,
    ) -> dict:
        """Annotate a Kubernetes resource (table-driven via patch_resource)."""
        params = {
            "resource_type": resource_type,
            "name": name,
            "namespace": namespace,
            "annotations": annotations,
        }
        self.patch_resource(
            resource_type,
            name,
            namespace=namespace,
            patch_body={"metadata": {"annotations": annotations or {}}},
            patch_type="merge",
        )
        result = {
            "resource_type": resource_type,
            "name": name,
            "annotations_added": list(annotations or {}),
        }
        self.log_action("annotate_resource", params, result)
        return result

    def create_namespace(self, name: str) -> dict:
        """Create a namespace."""
        params = {"name": name}
        try:
            namespace = _km.k8s_client.V1Namespace(
                metadata=_km.k8s_client.V1ObjectMeta(name=name)
            )
            created = self.core.create_namespace(namespace)
            result = {
                "name": created.metadata.name,
                "status": created.status.phase if created.status else "",
            }
            self.log_action("create_namespace", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_namespace", params, error=e)
            raise RuntimeError(f"Failed to create namespace: {str(e)}") from e

    def delete_namespace(self, name: str) -> dict:
        """Delete a namespace."""
        params = {"name": name}
        try:
            self.core.delete_namespace(name)
            result = {"deleted": name}
            self.log_action("delete_namespace", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("delete_namespace", params, error=e)
            raise RuntimeError(f"Failed to delete namespace: {str(e)}") from e

    def compare_configmap_state(
        self, name: str, namespace: str, expected_data: dict
    ) -> dict:
        """Compare ConfigMap state with expected data."""
        params = {"name": name, "namespace": namespace, "expected_data": expected_data}
        try:
            cm = self.core.read_namespaced_config_map(name, namespace)

            differences = {}
            actual_data = cm.data or {}

            # Check for differences
            for key, expected_value in expected_data.items():
                actual_value = actual_data.get(key)
                if actual_value != expected_value:
                    differences[key] = {
                        "expected": expected_value,
                        "actual": actual_value,
                    }

            # Check for extra keys
            for key in actual_data:
                if key not in expected_data:
                    differences[key] = {
                        "expected": None,
                        "actual": actual_data[key],
                        "extra": True,
                    }

            result = {
                "name": name,
                "namespace": namespace,
                "matches": len(differences) == 0,
                "differences": differences,
                "difference_count": len(differences),
            }
            self.log_action("compare_configmap_state", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("compare_configmap_state", params, error=e)
            raise RuntimeError(f"Failed to compare ConfigMap state: {str(e)}") from e

    def sync_configmap_from_file(
        self, name: str, namespace: str, file_path: str
    ) -> dict:
        """Sync ConfigMap from a file."""
        params = {"name": name, "namespace": namespace, "file_path": file_path}
        try:
            import os

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path) as f:
                content = f.read()

            # Create or update ConfigMap
            try:
                existing = self.core.read_namespaced_config_map(name, namespace)
                existing.data = {"content": content}
                self.core.patch_namespaced_config_map(name, namespace, existing)
                result = {"name": name, "namespace": namespace, "status": "updated"}
            except _km.ApiException as e:
                if e.status == 404:
                    # Create new ConfigMap
                    cm = _km.k8s_client.V1ConfigMap(
                        metadata=_km.k8s_client.V1ObjectMeta(name=name),
                        data={"content": content},
                    )
                    self.core.create_namespaced_config_map(namespace, cm)
                    result = {"name": name, "namespace": namespace, "status": "created"}
                else:
                    raise

            self.log_action("sync_configmap_from_file", params, result)
            return result
        except Exception as e:
            self.log_action("sync_configmap_from_file", params, error=e)
            raise RuntimeError(f"Failed to sync ConfigMap from file: {str(e)}") from e

    def get_secret_state_hash(self, name: str, namespace: str) -> dict:
        """Get hash of Secret state for comparison."""
        params = {"name": name, "namespace": namespace}
        try:
            import hashlib

            secret = self.core.read_namespaced_secret(name, namespace)
            data = secret.data or {}

            # Create hash of secret data
            hash_obj = hashlib.sha256()
            for key in sorted(data.keys()):
                hash_obj.update(key.encode())
                hash_obj.update(data[key].encode())

            result = {
                "name": name,
                "namespace": namespace,
                "hash": hash_obj.hexdigest(),
                "data_keys": list(data.keys()),
                "resource_version": secret.metadata.resource_version,
            }
            self.log_action("get_secret_state_hash", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("get_secret_state_hash", params, error=e)
            raise RuntimeError(f"Failed to get Secret state hash: {str(e)}") from e

    def track_resource_version(
        self, resource_type: str, name: str, namespace: str | None = None
    ) -> dict:
        """Track resource version for a resource."""
        params = {"resource_type": resource_type, "name": name, "namespace": namespace}
        try:
            if resource_type == "configmap":
                resource = self.core.read_namespaced_config_map(
                    name, namespace or self.namespace
                )
            elif resource_type == "secret":
                resource = self.core.read_namespaced_secret(
                    name, namespace or self.namespace
                )
            elif resource_type == "pod":
                resource = self.core.read_namespaced_pod(
                    name, namespace or self.namespace
                )
            elif resource_type == "deployment":
                apps_api = self.apps
                resource = apps_api.read_namespaced_deployment(
                    name, namespace or self.namespace
                )
            else:
                raise ValueError(f"Unsupported resource type: {resource_type}")

            result = {
                "resource_type": resource_type,
                "name": name,
                "namespace": namespace,
                "resource_version": resource.metadata.resource_version,
                "generation": resource.metadata.generation,
                "uid": resource.metadata.uid,
            }
            self.log_action("track_resource_version", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("track_resource_version", params, error=e)
            raise RuntimeError(f"Failed to track resource version: {str(e)}") from e

    def wait_for_resource_version(
        self,
        resource_type: str,
        name: str,
        namespace: str,
        target_version: str,
        timeout: int = 60,
    ) -> dict:
        """Wait for resource to reach a specific version."""
        params = {
            "resource_type": resource_type,
            "name": name,
            "namespace": namespace,
            "target_version": target_version,
            "timeout": timeout,
        }
        try:
            import time

            start_time = time.time()
            while time.time() - start_time < timeout:
                version_info = self.track_resource_version(
                    resource_type, name, namespace
                )
                if version_info["resource_version"] == target_version:
                    return {
                        "resource_type": resource_type,
                        "name": name,
                        "namespace": namespace,
                        "target_version": target_version,
                        "current_version": version_info["resource_version"],
                        "status": "reached",
                    }
                time.sleep(1)

            return {
                "resource_type": resource_type,
                "name": name,
                "namespace": namespace,
                "target_version": target_version,
                "status": "timeout",
            }
        except Exception as e:
            self.log_action("wait_for_resource_version", params, error=e)
            raise RuntimeError(f"Failed to wait for resource version: {str(e)}") from e

    def watch_resource(
        self, resource_type: str, name: str, namespace: str | None = None
    ) -> list[dict]:
        """Watch a specific resource for changes."""
        params = {"resource_type": resource_type, "name": name, "namespace": namespace}
        try:
            from kubernetes import watch

            if resource_type == "pod":
                stream = watch.Watch().stream(
                    self.core.list_namespaced_pod,
                    namespace=namespace or self.namespace,
                    field_selector=f"metadata.name={name}",
                )
            elif resource_type == "configmap":
                stream = watch.Watch().stream(
                    self.core.list_namespaced_config_map,
                    namespace=namespace or self.namespace,
                    field_selector=f"metadata.name={name}",
                )
            else:
                raise ValueError(
                    f"Unsupported resource type for watch: {resource_type}"
                )

            # Collect initial events (limited to recent history)
            events = []
            for event in stream:
                events.append(
                    {
                        "type": event["type"],
                        "object": {
                            "name": event["object"].metadata.name,
                            "resource_version": event[
                                "object"
                            ].metadata.resource_version,
                        },
                    }
                )
                if len(events) >= 10:  # Limit to recent events
                    stream.close()
                    break

            result = {
                "resource_type": resource_type,
                "name": name,
                "namespace": namespace,
                "events": events,
                "event_count": len(events),
            }
            self.log_action("watch_resource", params, result)
            return result
        except Exception as e:
            self.log_action("watch_resource", params, error=e)
            raise RuntimeError(f"Failed to watch resource: {str(e)}") from e

    def stream_pod_logs(
        self, pod_name: str, namespace: str, tail_lines: int = 100
    ) -> dict:
        """Stream pod logs."""
        params = {
            "pod_name": pod_name,
            "namespace": namespace,
            "tail_lines": tail_lines,
        }
        try:
            logs = self.core.read_namespaced_pod_log(
                pod_name, namespace, tail_lines=tail_lines, follow=False
            )

            result = {
                "pod_name": pod_name,
                "namespace": namespace,
                "log_lines": tail_lines,
                "logs": logs,
                "log_length": len(logs),
            }
            self.log_action("stream_pod_logs", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("stream_pod_logs", params, error=e)
            raise RuntimeError(f"Failed to stream pod logs: {str(e)}") from e

    def get_resource_events(
        self, resource_type: str, name: str, namespace: str | None = None
    ) -> list[dict]:
        """Get events for a specific resource."""
        params = {"resource_type": resource_type, "name": name, "namespace": namespace}
        try:
            events = self.core.list_namespaced_event(
                namespace=namespace or self.namespace
            ).items

            # Filter events for this resource
            resource_events = []
            for event in events:
                if event.involved_object and event.involved_object.name == name:
                    resource_events.append(
                        {
                            "type": event.type,
                            "reason": event.reason,
                            "message": event.message,
                            "first_timestamp": self._ts(event.first_timestamp),
                            "last_timestamp": self._ts(event.last_timestamp),
                            "count": event.count,
                        }
                    )

            result = {
                "resource_type": resource_type,
                "name": name,
                "namespace": namespace,
                "events": resource_events,
                "event_count": len(resource_events),
            }
            self.log_action("get_resource_events", params, result)
            return resource_events
        except _km.ApiException as e:
            self.log_action("get_resource_events", params, error=e)
            raise RuntimeError(f"Failed to get resource events: {str(e)}") from e

    def list_field_selector(
        self, resource_type: str, field_selector: str, namespace: str | None = None
    ) -> list[dict]:
        """List resources using field selector."""
        params = {
            "resource_type": resource_type,
            "field_selector": field_selector,
            "namespace": namespace,
        }
        try:
            if resource_type == "pod":
                resources = self.core.list_namespaced_pod(
                    namespace=namespace or self.namespace, field_selector=field_selector
                ).items
            elif resource_type == "node":
                resources = self.core.list_node(field_selector=field_selector).items
            else:
                raise ValueError(f"Unsupported resource type: {resource_type}")

            result = [
                {
                    "name": resource.metadata.name,
                    "namespace": resource.metadata.namespace,
                    "resource_version": resource.metadata.resource_version,
                }
                for resource in resources
            ]

            self.log_action("list_field_selector", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_field_selector", params, error=e)
            raise RuntimeError(f"Failed to list with field selector: {str(e)}") from e

"""GovernanceMixin for KubernetesManager (split from k8s_manager.py)."""

from typing import Any

import container_manager_mcp.k8s_manager as _km


class GovernanceMixin:
    def list_resource_quotas(self, namespace: str | None = None) -> list[dict]:
        """List ResourceQuotas in a namespace."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            quotas = self.core.list_namespaced_resource_quota(ns).items
            result = [
                {
                    "name": rq.metadata.name,
                    "namespace": rq.metadata.namespace,
                    "hard": rq.spec.hard.dict() if rq.spec and rq.spec.hard else {},
                    "used": (
                        rq.status.used.dict() if rq.status and rq.status.used else {}
                    ),
                    "created": self._ts(rq.metadata.creation_timestamp),
                }
                for rq in quotas
            ]
            self.log_action("list_resource_quotas", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_resource_quotas", params, error=e)
            raise RuntimeError("Failed to list resource quotas") from e

    def list_limit_ranges(self, namespace: str | None = None) -> list[dict]:
        """List LimitRanges in a namespace."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            limits = self.core.list_namespaced_limit_range(ns).items
            result = [
                {
                    "name": lr.metadata.name,
                    "namespace": lr.metadata.namespace,
                    "limits": lr.spec.limits if lr.spec else [],
                    "created": self._ts(lr.metadata.creation_timestamp),
                }
                for lr in limits
            ]
            self.log_action("list_limit_ranges", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_limit_ranges", params, error=e)
            raise RuntimeError("Failed to list limit ranges") from e

    def list_priority_classes(self) -> list[dict]:
        """List PriorityClasses."""
        params: dict[str, Any] = {}
        try:
            scheduling_api = self.scheduling
            pclasses = scheduling_api.list_priority_class().items
            result = [
                {
                    "name": pc.metadata.name,
                    "value": pc.value if pc.value else 0,
                    "global_default": pc.global_default if pc.global_default else False,
                    "description": (
                        pc.metadata.annotations.get("description")
                        if pc.metadata and pc.metadata.annotations
                        else ""
                    ),
                    "created": self._ts(pc.metadata.creation_timestamp),
                }
                for pc in pclasses
            ]
            self.log_action("list_priority_classes", params, {"count": len(result)})
            return result
        except ImportError:
            raise RuntimeError("Scheduling client not available") from None
        except _km.ApiException as e:
            self.log_action("list_priority_classes", params, error=e)
            raise RuntimeError("Failed to list priority classes") from e

    def list_pod_disruption_budgets(self, namespace: str | None = None) -> list[dict]:
        """List PodDisruptionBudgets in a namespace."""
        params = {"namespace": namespace}
        try:
            policy_api = self.policy
            ns = namespace or self.namespace
            pdbs = policy_api.list_namespaced_pod_disruption_budget(ns).items
            result = [
                {
                    "name": pdb.metadata.name,
                    "namespace": pdb.metadata.namespace,
                    "min_available": pdb.spec.min_available if pdb.spec else None,
                    "max_unavailable": pdb.spec.max_unavailable if pdb.spec else None,
                    "disruptions_allowed": (
                        pdb.status.disruptions_allowed if pdb.status else 0
                    ),
                    "created": self._ts(pdb.metadata.creation_timestamp),
                }
                for pdb in pdbs
            ]
            self.log_action(
                "list_pod_disruption_budgets", params, {"count": len(result)}
            )
            return result
        except ImportError:
            raise RuntimeError("Policy client not available") from None
        except _km.ApiException as e:
            self.log_action("list_pod_disruption_budgets", params, error=e)
            raise RuntimeError(
                f"Failed to list pod disruption budgets: {type(e).__name__}"
            ) from e

    def list_horizontal_pod_autoscalers(
        self, namespace: str | None = None
    ) -> list[dict]:
        """List HorizontalPodAutoscalers in a namespace."""
        params = {"namespace": namespace}
        try:
            autoscaling_api = self.autoscaling
            ns = namespace or self.namespace
            hpas = autoscaling_api.list_namespaced_horizontal_pod_autoscaler(ns).items
            result = [
                {
                    "name": hpa.metadata.name,
                    "namespace": hpa.metadata.namespace,
                    "min_replicas": hpa.spec.min_replicas if hpa.spec else 1,
                    "max_replicas": hpa.spec.max_replicas if hpa.spec else 0,
                    "current_replicas": (
                        hpa.status.current_replicas if hpa.status else 0
                    ),
                    "target_ref": (
                        hpa.spec.scale_target_ref.dict()
                        if hpa.spec and hpa.spec.scale_target_ref
                        else {}
                    ),
                    "created": self._ts(hpa.metadata.creation_timestamp),
                }
                for hpa in hpas
            ]
            self.log_action(
                "list_horizontal_pod_autoscalers", params, {"count": len(result)}
            )
            return result
        except ImportError:
            raise RuntimeError("Autoscaling client not available") from None
        except _km.ApiException as e:
            self.log_action("list_horizontal_pod_autoscalers", params, error=e)
            raise RuntimeError("Failed to list HPAs") from e

    def describe_resource_quota(self, name: str, namespace: str) -> dict:
        """Describe a ResourceQuota."""
        params = {"name": name, "namespace": namespace}
        try:
            rq = self.core.read_namespaced_resource_quota(name, namespace)
            result = {
                "name": rq.metadata.name,
                "namespace": rq.metadata.namespace,
                "status": rq.status,
                "spec": rq.spec,
                "created": self._ts(rq.metadata.creation_timestamp),
                "labels": rq.metadata.labels,
                "annotations": rq.metadata.annotations,
            }
            self.log_action("describe_resource_quota", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("describe_resource_quota", params, error=e)
            raise RuntimeError("Failed to describe ResourceQuota") from e

    def create_resource_quota(self, name: str, namespace: str, spec: dict) -> dict:
        """Create a ResourceQuota."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            quota_spec = _km.k8s_client.V1ResourceQuotaSpec(**spec)
            quota = _km.k8s_client.V1ResourceQuota(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=quota_spec
            )
            created = self.core.create_namespaced_resource_quota(namespace, quota)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
                "status": "created",
                "created": self._ts(created.metadata.creation_timestamp),
            }
            self.log_action("create_resource_quota", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_resource_quota", params, error=e)
            raise RuntimeError("Failed to create ResourceQuota") from e

    def update_resource_quota(self, name: str, namespace: str, spec: dict) -> dict:
        """Update a ResourceQuota."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            existing = self.core.read_namespaced_resource_quota(name, namespace)
            quota_spec = _km.k8s_client.V1ResourceQuotaSpec(**spec)
            existing.spec = quota_spec
            updated = self.core.patch_namespaced_resource_quota(
                name, namespace, existing
            )
            result = {
                "name": updated.metadata.name,
                "namespace": updated.metadata.namespace,
                "status": "updated",
                "updated": self._ts(updated.metadata.creation_timestamp),
            }
            self.log_action("update_resource_quota", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("update_resource_quota", params, error=e)
            raise RuntimeError("Failed to update ResourceQuota") from e

    def delete_resource_quota(self, name: str, namespace: str) -> dict:
        """Delete a ResourceQuota."""
        params = {"name": name, "namespace": namespace}
        try:
            self.core.delete_namespaced_resource_quota(name, namespace)
            result = {"name": name, "namespace": namespace, "status": "deleted"}
            self.log_action("delete_resource_quota", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("delete_resource_quota", params, error=e)
            raise RuntimeError("Failed to delete ResourceQuota") from e

    def describe_limit_range(self, name: str, namespace: str) -> dict:
        """Describe a LimitRange."""
        params = {"name": name, "namespace": namespace}
        try:
            lr = self.core.read_namespaced_limit_range(name, namespace)
            result = {
                "name": lr.metadata.name,
                "namespace": lr.metadata.namespace,
                "spec": lr.spec,
                "created": self._ts(lr.metadata.creation_timestamp),
                "labels": lr.metadata.labels,
                "annotations": lr.metadata.annotations,
            }
            self.log_action("describe_limit_range", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("describe_limit_range", params, error=e)
            raise RuntimeError("Failed to describe LimitRange") from e

    def create_limit_range(self, name: str, namespace: str, spec: dict) -> dict:
        """Create a LimitRange."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            limit_spec = _km.k8s_client.V1LimitRangeSpec(**spec)
            limit_range = _km.k8s_client.V1LimitRange(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=limit_spec
            )
            created = self.core.create_namespaced_limit_range(namespace, limit_range)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
                "status": "created",
                "created": self._ts(created.metadata.creation_timestamp),
            }
            self.log_action("create_limit_range", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_limit_range", params, error=e)
            raise RuntimeError("Failed to create LimitRange") from e

    def delete_limit_range(self, name: str, namespace: str) -> dict:
        """Delete a LimitRange."""
        params = {"name": name, "namespace": namespace}
        try:
            self.core.delete_namespaced_limit_range(name, namespace)
            result = {"name": name, "namespace": namespace, "status": "deleted"}
            self.log_action("delete_limit_range", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("delete_limit_range", params, error=e)
            raise RuntimeError("Failed to delete LimitRange") from e

    def describe_priority_class(self, name: str) -> dict:
        """Describe a PriorityClass."""
        params = {"name": name}
        try:
            scheduling_api = self.scheduling
            pc = scheduling_api.read_priority_class(name)
            result = {
                "name": pc.metadata.name,
                "value": pc.value,
                "global_default": pc.global_default,
                "preemption_policy": pc.preemption_policy,
                "description": pc.metadata.annotations.get("description", ""),
                "created": self._ts(pc.metadata.creation_timestamp),
                "labels": pc.metadata.labels,
                "annotations": pc.metadata.annotations,
            }
            self.log_action("describe_priority_class", params, result)
            return result
        except ImportError:
            raise RuntimeError("Scheduling client not available") from None
        except _km.ApiException as e:
            self.log_action("describe_priority_class", params, error=e)
            raise RuntimeError("Failed to describe PriorityClass") from e

    def create_priority_class(self, name: str, spec: dict) -> dict:
        """Create a PriorityClass."""
        params = {"name": name, "spec": spec}
        try:
            scheduling_api = self.scheduling
            priority_class = _km.k8s_client.V1PriorityClass(
                metadata=_km.k8s_client.V1ObjectMeta(name=name),
                value=spec.get("value", 1000),
                global_default=spec.get("global_default", False),
                preemption_policy=spec.get("preemption_policy", "PreemptLowerPriority"),
                description=spec.get("description", ""),
            )
            created = scheduling_api.create_priority_class(priority_class)
            result = {
                "name": created.metadata.name,
                "status": "created",
                "created": self._ts(created.metadata.creation_timestamp),
            }
            self.log_action("create_priority_class", params, result)
            return result
        except ImportError:
            raise RuntimeError("Scheduling client not available") from None
        except _km.ApiException as e:
            self.log_action("create_priority_class", params, error=e)
            raise RuntimeError("Failed to create PriorityClass") from e

    def delete_priority_class(self, name: str) -> dict:
        """Delete a PriorityClass."""
        params = {"name": name}
        try:
            scheduling_api = self.scheduling
            scheduling_api.delete_priority_class(name)
            result = {"name": name, "status": "deleted"}
            self.log_action("delete_priority_class", params, result)
            return result
        except ImportError:
            raise RuntimeError("Scheduling client not available") from None
        except _km.ApiException as e:
            self.log_action("delete_priority_class", params, error=e)
            raise RuntimeError("Failed to delete PriorityClass") from e

    def describe_pod_disruption_budget(self, name: str, namespace: str) -> dict:
        """Describe a PodDisruptionBudget."""
        params = {"name": name, "namespace": namespace}
        try:
            policy_api = self.policy
            pdb = policy_api.read_namespaced_pod_disruption_budget(name, namespace)
            result = {
                "name": pdb.metadata.name,
                "namespace": pdb.metadata.namespace,
                "spec": pdb.spec,
                "status": pdb.status,
                "created": self._ts(pdb.metadata.creation_timestamp),
                "labels": pdb.metadata.labels,
                "annotations": pdb.metadata.annotations,
            }
            self.log_action("describe_pod_disruption_budget", params, result)
            return result
        except ImportError:
            raise RuntimeError("Policy client not available") from None
        except _km.ApiException as e:
            self.log_action("describe_pod_disruption_budget", params, error=e)
            raise RuntimeError(
                f"Failed to describe PodDisruptionBudget: {type(e).__name__}"
            ) from e

    def create_pod_disruption_budget(
        self, name: str, namespace: str, spec: dict
    ) -> dict:
        """Create a PodDisruptionBudget."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            policy_api = self.policy
            pdb_spec = _km.k8s_client.V1PodDisruptionBudgetSpec(**spec)
            pdb = _km.k8s_client.V1PodDisruptionBudget(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=pdb_spec
            )
            created = policy_api.create_namespaced_pod_disruption_budget(namespace, pdb)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
                "status": "created",
                "created": self._ts(created.metadata.creation_timestamp),
            }
            self.log_action("create_pod_disruption_budget", params, result)
            return result
        except ImportError:
            raise RuntimeError("Policy client not available") from None
        except _km.ApiException as e:
            self.log_action("create_pod_disruption_budget", params, error=e)
            raise RuntimeError("Failed to create PodDisruptionBudget") from e

    def delete_pod_disruption_budget(self, name: str, namespace: str) -> dict:
        """Delete a PodDisruptionBudget."""
        params = {"name": name, "namespace": namespace}
        try:
            policy_api = self.policy
            policy_api.delete_namespaced_pod_disruption_budget(name, namespace)
            result = {"name": name, "namespace": namespace, "status": "deleted"}
            self.log_action("delete_pod_disruption_budget", params, result)
            return result
        except ImportError:
            raise RuntimeError("Policy client not available") from None
        except _km.ApiException as e:
            self.log_action("delete_pod_disruption_budget", params, error=e)
            raise RuntimeError("Failed to delete PodDisruptionBudget") from e

    def describe_horizontal_pod_autoscaler(self, name: str, namespace: str) -> dict:
        """Describe a HorizontalPodAutoscaler."""
        params = {"name": name, "namespace": namespace}
        try:
            autoscaling_api = self.autoscaling
            hpa = autoscaling_api.read_namespaced_horizontal_pod_autoscaler(
                name, namespace
            )
            result = {
                "name": hpa.metadata.name,
                "namespace": hpa.metadata.namespace,
                "spec": hpa.spec,
                "status": hpa.status,
                "created": self._ts(hpa.metadata.creation_timestamp),
                "labels": hpa.metadata.labels,
                "annotations": hpa.metadata.annotations,
            }
            self.log_action("describe_horizontal_pod_autoscaler", params, result)
            return result
        except ImportError:
            raise RuntimeError("Autoscaling client not available") from None
        except _km.ApiException as e:
            self.log_action("describe_horizontal_pod_autoscaler", params, error=e)
            raise RuntimeError(
                f"Failed to describe HorizontalPodAutoscaler: {type(e).__name__}"
            ) from e

    def create_horizontal_pod_autoscaler(
        self, name: str, namespace: str, spec: dict
    ) -> dict:
        """Create a HorizontalPodAutoscaler."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            autoscaling_api = self.autoscaling
            hpa_spec = _km.k8s_client.V2HorizontalPodAutoscalerSpec(**spec)
            hpa = _km.k8s_client.V2HorizontalPodAutoscaler(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=hpa_spec
            )
            created = autoscaling_api.create_namespaced_horizontal_pod_autoscaler(
                namespace, hpa
            )
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
                "status": "created",
                "created": self._ts(created.metadata.creation_timestamp),
            }
            self.log_action("create_horizontal_pod_autoscaler", params, result)
            return result
        except ImportError:
            raise RuntimeError("Autoscaling client not available") from None
        except _km.ApiException as e:
            self.log_action("create_horizontal_pod_autoscaler", params, error=e)
            raise RuntimeError(
                f"Failed to create HorizontalPodAutoscaler: {type(e).__name__}"
            ) from e

    def update_horizontal_pod_autoscaler(
        self, name: str, namespace: str, spec: dict
    ) -> dict:
        """Update a HorizontalPodAutoscaler."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            autoscaling_api = self.autoscaling
            existing = autoscaling_api.read_namespaced_horizontal_pod_autoscaler(
                name, namespace
            )
            hpa_spec = _km.k8s_client.V2HorizontalPodAutoscalerSpec(**spec)
            existing.spec = hpa_spec
            updated = autoscaling_api.patch_namespaced_horizontal_pod_autoscaler(
                name, namespace, existing
            )
            result = {
                "name": updated.metadata.name,
                "namespace": updated.metadata.namespace,
                "status": "updated",
                "updated": self._ts(updated.metadata.creation_timestamp),
            }
            self.log_action("update_horizontal_pod_autoscaler", params, result)
            return result
        except ImportError:
            raise RuntimeError("Autoscaling client not available") from None
        except _km.ApiException as e:
            self.log_action("update_horizontal_pod_autoscaler", params, error=e)
            raise RuntimeError(
                f"Failed to update HorizontalPodAutoscaler: {type(e).__name__}"
            ) from e

    def delete_horizontal_pod_autoscaler(self, name: str, namespace: str) -> dict:
        """Delete a HorizontalPodAutoscaler."""
        params = {"name": name, "namespace": namespace}
        try:
            autoscaling_api = self.autoscaling
            autoscaling_api.delete_namespaced_horizontal_pod_autoscaler(name, namespace)
            result = {"name": name, "namespace": namespace, "status": "deleted"}
            self.log_action("delete_horizontal_pod_autoscaler", params, result)
            return result
        except ImportError:
            raise RuntimeError("Autoscaling client not available") from None
        except _km.ApiException as e:
            self.log_action("delete_horizontal_pod_autoscaler", params, error=e)
            raise RuntimeError(
                f"Failed to delete HorizontalPodAutoscaler: {type(e).__name__}"
            ) from e

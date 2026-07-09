"""ClusterNodesMixin for KubernetesManager (split from k8s_manager.py)."""

from typing import Any

import container_manager_mcp.k8s_manager as _km


class ClusterNodesMixin:
    def list_nodes(self) -> list[dict]:
        params: dict[str, Any] = {}
        try:
            result = []
            for node in self.core.list_node().items:
                meta = node.metadata
                labels = meta.labels or {}
                conditions = (node.status.conditions or []) if node.status else []
                ready = next(
                    (c.status for c in conditions if c.type == "Ready"), "Unknown"
                )
                result.append(
                    {
                        "id": (meta.uid or "unknown")[:12],
                        "hostname": meta.name,
                        "role": self._node_role(labels),
                        "status": "ready" if ready == "True" else "not ready",
                        "availability": (
                            "drain" if node.spec.unschedulable else "active"
                        ),
                        "created": self._ts(meta.creation_timestamp),
                        "updated": "unknown",
                    }
                )
            self.log_action("list_nodes", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_nodes", params, error=e)
            raise RuntimeError(f"Failed to list nodes: {str(e)}") from e

    def inspect_node(self, node_id: str) -> dict:
        params = {"node_id": node_id}
        try:
            node = self.core.read_node(node_id)
            result = self._node_summary(node)
            self.log_action("inspect_node", params, {"id": node_id})
            return result
        except _km.ApiException as e:
            self.log_action("inspect_node", params, error=e)
            raise RuntimeError(f"Failed to inspect node: {str(e)}") from e

    def update_node(
        self,
        node_id: str,
        labels: dict[str, str] | None = None,
        role: str | None = None,
        availability: str | None = None,
        replace_labels: bool = False,
    ) -> dict:
        """Update a node's labels, cordon state, or control-plane role label.

        ``availability``: ``active`` uncordons; ``pause``/``drain`` cordon
        (``drain`` also evicts non-mirror pods). ``role`` toggles the
        ``node-role.kubernetes.io/control-plane`` label.
        """
        params = {
            "node_id": node_id,
            "labels": labels,
            "role": role,
            "availability": availability,
            "replace_labels": replace_labels,
        }
        try:
            node = self.core.read_node(node_id)
            current = dict(node.metadata.labels or {})
            new_labels = current
            if labels is not None:
                if replace_labels:
                    # null out existing labels not in the new set
                    new_labels = {k: None for k in current}  # type: ignore
                    new_labels.update(labels)
                else:
                    new_labels = {**current, **labels}
            if role == "manager":
                new_labels["node-role.kubernetes.io/control-plane"] = ""
            elif role == "worker":
                new_labels["node-role.kubernetes.io/control-plane"] = None  # type: ignore

            body: dict[str, Any] = {"metadata": {"labels": new_labels}}
            if availability is not None:
                body["spec"] = {"unschedulable": availability in ("pause", "drain")}
            self.core.patch_node(node_id, body)

            if availability == "drain":
                self._drain_node(node_id)

            result = self._node_summary(self.core.read_node(node_id))
            self.log_action("update_node", params, {"id": node_id})
            return result
        except _km.ApiException as e:
            self.log_action("update_node", params, error=e)
            raise RuntimeError(f"Failed to update node: {str(e)}") from e

    def remove_node(self, node_id: str, force: bool = False) -> dict:
        params = {"node_id": node_id, "force": force}
        try:
            self.core.patch_node(node_id, {"spec": {"unschedulable": True}})
            self._drain_node(node_id)
            self.core.delete_node(node_id)
            result = {"removed": node_id}
            self.log_action("remove_node", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("remove_node", params, error=e)
            raise RuntimeError(f"Failed to remove node: {str(e)}") from e

    def taint_node(self, node_name: str, taints: list[dict]) -> dict:
        """Taint a node with specified taints."""
        params = {"node_name": node_name, "taints": taints}
        try:
            node = self.core.read_node(node_name)
            current_taints = node.spec.taints or []
            new_taints = [_km.k8s_client.V1Taint(**taint) for taint in taints]
            self.core.patch_node(
                node_name,
                {"spec": {"taints": current_taints + new_taints}},
            )
            result = {"node": node_name, "taints_added": len(taints)}
            self.log_action("taint_node", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("taint_node", params, error=e)
            raise RuntimeError(f"Failed to taint node: {str(e)}") from e

    def list_contexts(self) -> list[dict]:
        """List kubeconfig contexts."""
        params: dict[str, Any] = {}
        try:
            contexts, current_context = _km.k8s_config.list_kube_config_contexts()
            result = [
                {"name": name, "is_current": name == current_context}
                for name in contexts
            ]
            self.log_action("list_contexts", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("list_contexts", params, error=e)
            raise RuntimeError(f"Failed to list contexts: {str(e)}") from e

    def use_context(self, context_name: str) -> dict:
        """Switch kubeconfig context."""
        params = {"context_name": context_name}
        try:
            _km.k8s_config.load_kube_config(context=context_name)
            result = {"current_context": context_name}
            self.log_action("use_context", params, result)
            return result
        except Exception as e:
            self.log_action("use_context", params, error=e)
            raise RuntimeError(f"Failed to use context: {str(e)}") from e

    def get_config(self) -> dict:
        """Get kubeconfig information."""
        params: dict[str, Any] = {}
        try:
            contexts, current_context = _km.k8s_config.list_kube_config_contexts()
            clusters = _km.k8s_config.list_kube_config_clusters()
            users = _km.k8s_config.list_kube_config_users()
            result = {
                "current_context": current_context,
                "contexts": contexts,
                "clusters": list(clusters.keys()) if clusters else [],
                "users": list(users.keys()) if users else [],
            }
            self.log_action("get_config", params, result)
            return result
        except Exception as e:
            self.log_action("get_config", params, error=e)
            raise RuntimeError(f"Failed to get config: {str(e)}") from e

    def rename_context(self, current_name: str, new_name: str) -> dict:
        """Rename a kubeconfig context."""
        params = {"current_name": current_name, "new_name": new_name}
        try:
            _km.k8s_config.rename_context(current_name, new_name)
            result = {"renamed": f"{current_name} -> {new_name}"}
            self.log_action("rename_context", params, result)
            return result
        except Exception as e:
            self.log_action("rename_context", params, error=e)
            raise RuntimeError(f"Failed to rename context: {str(e)}") from e

    def cordon_node(self, node_name: str, unschedulable: bool = True) -> dict:
        """Mark a node as unschedulable (cordon) or schedulable (uncordon)."""
        params = {"node_name": node_name, "unschedulable": unschedulable}
        try:
            body = {"spec": {"unschedulable": unschedulable}}
            self.core.patch_node(node_name, body)
            result = {
                "node": node_name,
                "unschedulable": unschedulable,
                "status": "cordoned" if unschedulable else "uncordoned",
            }
            self.log_action("cordon_node", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("cordon_node", params, error=e)
            raise RuntimeError(f"Failed to cordon/uncordon node: {str(e)}") from e

    def drain_node(
        self, node_name: str, force: bool = False, grace_period_seconds: int = 120
    ) -> dict:
        """Safely evict all pods from a node (drain)."""
        params = {
            "node_name": node_name,
            "force": force,
            "grace_period_seconds": grace_period_seconds,
        }
        try:
            # First cordon the node
            self.cordon_node(node_name, unschedulable=True)

            # Get all pods on the node
            pods = self.core.list_pod_for_all_namespaces(
                field_selector=f"spec.nodeName={node_name}"
            ).items

            evicted = []
            for pod in pods:
                try:
                    # Skip DaemonSet pods and pods with local storage
                    if pod.metadata.owner_references and any(
                        owner.kind == "DaemonSet"
                        for owner in pod.metadata.owner_references
                    ):
                        continue

                    # Delete the pod
                    self.core.delete_namespaced_pod(
                        pod.metadata.name, pod.metadata.namespace
                    )
                    evicted.append(
                        {"name": pod.metadata.name, "namespace": pod.metadata.namespace}
                    )
                except _km.ApiException:
                    pass

            result = {
                "node": node_name,
                "evicted_pods": len(evicted),
                "evicted": evicted,
                "status": "drained",
            }
            self.log_action("drain_node", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("drain_node", params, error=e)
            raise RuntimeError(f"Failed to drain node: {str(e)}") from e

    def cluster_info_dump(self, output_dir: str = "/tmp/k8s-cluster-dump") -> dict:
        """Dump cluster information for debugging."""
        params = {"output_dir": output_dir}
        try:
            import json as json_module
            import os as os_module

            os_module.makedirs(output_dir, exist_ok=True)

            dump_info = {}

            # Get nodes
            nodes = self.core.list_node().items
            dump_info["nodes"] = [node.metadata.name for node in nodes]

            # Get namespaces
            namespaces = self.core.list_namespace().items
            dump_info["namespaces"] = [ns.metadata.name for ns in namespaces]

            # Get pods per namespace
            dump_info["pods"] = {}
            for ns in namespaces:
                pods = self.core.list_namespaced_pod(ns.metadata.name).items
                dump_info["pods"][ns.metadata.name] = [
                    pod.metadata.name for pod in pods
                ]

            # Get services
            services = self.core.list_service_for_all_namespaces().items
            dump_info["services"] = [
                {"name": svc.metadata.name, "namespace": svc.metadata.namespace}
                for svc in services
            ]

            # Write to file
            dump_file = os_module.path.join(output_dir, "cluster-dump.json")
            with open(dump_file, "w") as f:
                json_module.dump(dump_info, f, indent=2)

            result = {"output_dir": output_dir, "file": dump_file, "status": "dumped"}
            self.log_action("cluster_info_dump", params, result)
            return result
        except Exception as e:
            self.log_action("cluster_info_dump", params, error=e)
            raise RuntimeError(f"Failed to dump cluster info: {str(e)}") from e

    def list_node_conditions(self, node_name: str | None = None) -> list[dict]:
        """List node conditions for all nodes or a specific node."""
        params = {"node_name": node_name}
        try:
            if node_name:
                node = self.core.read_node(node_name)
                nodes = [node]
            else:
                nodes = self.core.list_node().items

            result = []
            for node in nodes:
                if node.status and node.status.conditions:
                    conditions = [
                        {
                            "type": cond.type,
                            "status": cond.status,
                            "reason": cond.reason,
                            "message": cond.message,
                            "last_transition_time": self._ts(cond.last_transition_time),
                        }
                        for cond in node.status.conditions
                    ]
                    result.append(
                        {
                            "node": node.metadata.name,
                            "conditions": conditions,
                        }
                    )

            self.log_action("list_node_conditions", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_node_conditions", params, error=e)
            raise RuntimeError(f"Failed to list node conditions: {str(e)}") from e

    def api_resources(self) -> list[dict]:
        """List all API resources available in the cluster."""
        params: dict[str, Any] = {}
        try:
            resources = self.core.get_api_resources()
            result = [
                {
                    "name": res.name,
                    "namespaced": (
                        res.namespaced if hasattr(res, "namespaced") else False
                    ),
                    "kind": res.kind,
                    "verbs": res.verbs if hasattr(res, "verbs") else [],
                    "short_names": (
                        res.short_names if hasattr(res, "short_names") else []
                    ),
                }
                for res in resources.resources
                if hasattr(res, "resources")
            ]
            self.log_action("api_resources", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("api_resources", params, error=e)
            raise RuntimeError(f"Failed to list API resources: {str(e)}") from e

    def list_certificate_signing_requests(self) -> list[dict]:
        """List CertificateSigningRequests."""
        params: dict[str, Any] = {}
        try:
            cert_api = self.certificates
            csrs = cert_api.list_certificate_signing_request().items
            result = [
                {
                    "name": csr.metadata.name,
                    "requester": csr.spec.requester_name if csr.spec else "",
                    "signer_name": csr.spec.signer_name if csr.spec else "",
                    "usages": csr.spec.usages if csr.spec else [],
                    "status": csr.status.conditions if csr.status else [],
                    "created": self._ts(csr.metadata.creation_timestamp),
                }
                for csr in csrs
            ]
            self.log_action(
                "list_certificate_signing_requests", params, {"count": len(result)}
            )
            return result
        except ImportError:
            raise RuntimeError("Certificates client not available") from None
        except _km.ApiException as e:
            self.log_action("list_certificate_signing_requests", params, error=e)
            raise RuntimeError(f"Failed to list CSRs: {str(e)}") from e

    def approve_csr(self, csr_name: str) -> dict:
        """Approve a CertificateSigningRequest."""
        params = {"csr_name": csr_name}
        try:
            cert_api = self.certificates
            body = {
                "status": {
                    "conditions": [
                        {
                            "type": "Approved",
                            "status": "True",
                            "reason": "Approved by MCP",
                            "message": "This CSR was approved programmatically",
                        }
                    ]
                }
            }
            cert_api.approve_certificate_signing_request(csr_name, body)
            result = {"name": csr_name, "status": "approved"}
            self.log_action("approve_csr", params, result)
            return result
        except ImportError:
            raise RuntimeError("Certificates client not available") from None
        except _km.ApiException as e:
            self.log_action("approve_csr", params, error=e)
            raise RuntimeError(f"Failed to approve CSR: {str(e)}") from e

    def deny_csr(self, csr_name: str, reason: str = "Denied by MCP") -> dict:
        """Deny a CertificateSigningRequest."""
        params = {"csr_name": csr_name, "reason": reason}
        try:
            cert_api = self.certificates
            body = {
                "status": {
                    "conditions": [
                        {
                            "type": "Denied",
                            "status": "True",
                            "reason": "Denied by MCP",
                            "message": reason,
                        }
                    ]
                }
            }
            cert_api.deny_certificate_signing_request(csr_name, body)
            result = {"name": csr_name, "status": "denied", "reason": reason}
            self.log_action("deny_csr", params, result)
            return result
        except ImportError:
            raise RuntimeError("Certificates client not available") from None
        except _km.ApiException as e:
            self.log_action("deny_csr", params, error=e)
            raise RuntimeError(f"Failed to deny CSR: {str(e)}") from e

    def uncordon_node(self, node_name: str) -> dict:
        """Uncordon a node (mark it as schedulable)."""
        params = {"node_name": node_name}
        try:
            body = {"spec": {"unschedulable": False}}
            self.core.patch_node(node_name, body)
            result = {"node_name": node_name, "status": "uncordoned"}
            self.log_action("uncordon_node", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("uncordon_node", params, error=e)
            raise RuntimeError(f"Failed to uncordon node: {str(e)}") from e

    def get_node_conditions(self, node_name: str) -> dict:
        """Get detailed conditions for a node."""
        params = {"node_name": node_name}
        try:
            node = self.core.read_node(node_name)
            conditions = []
            if node.status and node.status.conditions:
                for condition in node.status.conditions:
                    conditions.append(
                        {
                            "type": condition.type,
                            "status": condition.status,
                            "reason": condition.reason,
                            "message": condition.message,
                            "last_transition_time": self._ts(
                                condition.last_transition_time
                            ),
                        }
                    )

            result = {
                "node_name": node_name,
                "conditions": conditions,
                "ready": all(
                    c["type"] == "Ready" and c["status"] == "True"
                    for c in conditions
                    if c["type"] == "Ready"
                ),
            }
            self.log_action("get_node_conditions", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("get_node_conditions", params, error=e)
            raise RuntimeError(f"Failed to get node conditions: {str(e)}") from e

    def list_api_resources(self) -> list[dict]:
        """List all available API resources."""
        params: dict[str, Any] = {}
        try:
            discovery_api = _km.k8s_client.DiscoveryV1API()
            resources = discovery_api.server_resources_for_all_api_groups()

            result = []
            for group in resources.resources:
                for api_resource in group.api_resources:
                    result.append(
                        {
                            "name": api_resource.name,
                            "namespaced": api_resource.namespaced,
                            "kind": api_resource.kind,
                            "group": group.group_version,
                            "verbs": api_resource.verbs,
                            "short_names": api_resource.short_names or [],
                        }
                    )

            self.log_action("list_api_resources", params, {"count": len(result)})
            return result
        except ImportError:
            raise RuntimeError("Discovery client not available") from None
        except _km.ApiException as e:
            self.log_action("list_api_resources", params, error=e)
            raise RuntimeError(f"Failed to list API resources: {str(e)}") from e

    def describe_api_resource(self, resource_name: str) -> dict:
        """Describe a specific API resource."""
        params = {"resource_name": resource_name}
        try:
            discovery_api = _km.k8s_client.DiscoveryV1API()
            resources = discovery_api.server_resources_for_all_api_groups()

            for group in resources.resources:
                for api_resource in group.api_resources:
                    if api_resource.name == resource_name:
                        result = {
                            "name": api_resource.name,
                            "namespaced": api_resource.namespaced,
                            "kind": api_resource.kind,
                            "group": group.group_version,
                            "verbs": api_resource.verbs,
                            "short_names": api_resource.short_names or [],
                            "categories": api_resource.categories or [],
                        }
                        self.log_action("describe_api_resource", params, result)
                        return result

            raise ValueError(f"API resource '{resource_name}' not found")
        except ImportError:
            raise RuntimeError("Discovery client not available") from None
        except _km.ApiException as e:
            self.log_action("describe_api_resource", params, error=e)
            raise RuntimeError(f"Failed to describe API resource: {str(e)}") from e

    def untaint_node(self, node_name: str, taint_key: str) -> dict:
        """Remove a taint from a node."""
        params = {"node_name": node_name, "taint_key": taint_key}
        try:
            node = self.core.read_node(node_name)

            # Remove the specified taint
            if node.spec and node.spec.taints:
                node.spec.taints = [t for t in node.spec.taints if t.key != taint_key]

            self.core.patch_node(node_name, node)
            result = {
                "node_name": node_name,
                "taint_key": taint_key,
                "status": "untainted",
            }
            self.log_action("untaint_node", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("untaint_node", params, error=e)
            raise RuntimeError(f"Failed to untaint node: {str(e)}") from e

    def list_node_taints(self) -> list[dict]:
        """List all node taints."""
        params: dict[str, Any] = {}
        try:
            nodes = self.core.list_node().items
            result = []
            for node in nodes:
                if node.spec and node.spec.taints:
                    for taint in node.spec.taints:
                        result.append(
                            {
                                "node_name": node.metadata.name,
                                "taint_key": taint.key,
                                "taint_value": taint.value,
                                "taint_effect": taint.effect,
                            }
                        )

            self.log_action("list_node_taints", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_node_taints", params, error=e)
            raise RuntimeError(f"Failed to list node taints: {str(e)}") from e

    def set_node_affinity(self, pod_name: str, namespace: str, affinity: dict) -> dict:
        """Set node affinity for a pod."""
        params = {"pod_name": pod_name, "namespace": namespace, "affinity": affinity}
        try:
            pod = self.core.read_namespaced_pod(pod_name, namespace)

            # Initialize affinity if not present
            if not pod.spec:
                pod.spec = _km.k8s_client.V1PodSpec()
            if not pod.spec.affinity:
                pod.spec.affinity = _km.k8s_client.V1Affinity()

            # Set node affinity
            pod.spec.affinity.node_affinity = _km.k8s_client.V1NodeAffinity(**affinity)

            self.core.patch_namespaced_pod(pod_name, namespace, pod)
            result = {
                "pod_name": pod_name,
                "namespace": namespace,
                "status": "affinity_set",
            }
            self.log_action("set_node_affinity", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("set_node_affinity", params, error=e)
            raise RuntimeError(f"Failed to set node affinity: {str(e)}") from e

    def get_node_affinity(self, pod_name: str, namespace: str) -> dict:
        """Get node affinity for a pod."""
        params = {"pod_name": pod_name, "namespace": namespace}
        try:
            pod = self.core.read_namespaced_pod(pod_name, namespace)

            result = {
                "pod_name": pod_name,
                "namespace": namespace,
                "node_affinity": (
                    pod.spec.affinity.node_affinity._asdict()
                    if (pod.spec.affinity and pod.spec.affinity.node_affinity)
                    else None
                ),
            }
            self.log_action("get_node_affinity", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("get_node_affinity", params, error=e)
            raise RuntimeError(f"Failed to get node affinity: {str(e)}") from e

    def set_pod_anti_affinity(
        self, pod_name: str, namespace: str, anti_affinity: dict
    ) -> dict:
        """Set pod anti-affinity for a pod."""
        params = {
            "pod_name": pod_name,
            "namespace": namespace,
            "anti_affinity": anti_affinity,
        }
        try:
            pod = self.core.read_namespaced_pod(pod_name, namespace)

            # Initialize affinity if not present
            if not pod.spec:
                pod.spec = _km.k8s_client.V1PodSpec()
            if not pod.spec.affinity:
                pod.spec.affinity = _km.k8s_client.V1Affinity()

            # Set pod anti-affinity
            pod.spec.affinity.pod_anti_affinity = _km.k8s_client.V1PodAntiAffinity(
                **anti_affinity
            )

            self.core.patch_namespaced_pod(pod_name, namespace, pod)
            result = {
                "pod_name": pod_name,
                "namespace": namespace,
                "status": "anti_affinity_set",
            }
            self.log_action("set_pod_anti_affinity", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("set_pod_anti_affinity", params, error=e)
            raise RuntimeError(f"Failed to set pod anti-affinity: {str(e)}") from e

    def list_cluster_plugins(self) -> list[dict]:
        """List cluster plugins (dynamic admission controllers)."""
        params: dict[str, Any] = {}
        try:
            admission_api = self.admission
            validating_webhooks = (
                admission_api.list_validating_webhook_configuration().items
            )
            mutating_webhooks = (
                admission_api.list_mutating_webhook_configuration().items
            )

            result = []
            for webhook in validating_webhooks:
                result.append(
                    {
                        "name": webhook.metadata.name,
                        "type": "validating",
                        "webhooks": len(webhook.webhooks or []),
                        "created": self._ts(webhook.metadata.creation_timestamp),
                    }
                )

            for webhook in mutating_webhooks:
                result.append(
                    {
                        "name": webhook.metadata.name,
                        "type": "mutating",
                        "webhooks": len(webhook.webhooks or []),
                        "created": self._ts(webhook.metadata.creation_timestamp),
                    }
                )

            self.log_action("list_cluster_plugins", params, {"count": len(result)})
            return result
        except ImportError:
            raise RuntimeError("Admission client not available") from None
        except _km.ApiException as e:
            self.log_action("list_cluster_plugins", params, error=e)
            raise RuntimeError(f"Failed to list cluster plugins: {str(e)}") from e

    def describe_cluster_plugin(self, name: str, plugin_type: str) -> dict:
        """Describe a cluster plugin."""
        params = {"name": name, "plugin_type": plugin_type}
        try:
            admission_api = self.admission

            if plugin_type == "validating":
                plugin = admission_api.read_validating_webhook_configuration(name)
            elif plugin_type == "mutating":
                plugin = admission_api.read_mutating_webhook_configuration(name)
            else:
                raise ValueError(f"Invalid plugin type: {plugin_type}")

            result = {
                "name": name,
                "type": plugin_type,
                "webhooks": plugin.webhooks if plugin.webhooks else [],
                "created": self._ts(plugin.metadata.creation_timestamp),
                "labels": plugin.metadata.labels,
                "annotations": plugin.metadata.annotations,
            }
            self.log_action("describe_cluster_plugin", params, result)
            return result
        except ImportError:
            raise RuntimeError("Admission client not available") from None
        except _km.ApiException as e:
            self.log_action("describe_cluster_plugin", params, error=e)
            raise RuntimeError(f"Failed to describe cluster plugin: {str(e)}") from e

    def test_cluster_plugin(
        self, name: str, plugin_type: str, test_resource: dict
    ) -> dict:
        """Test a cluster plugin with a test resource."""
        params = {
            "name": name,
            "plugin_type": plugin_type,
            "test_resource": test_resource,
        }
        try:
            # This is a simplified test - in production would create actual test resource
            plugin_info = self.describe_cluster_plugin(name, plugin_type)

            result = {
                "name": name,
                "type": plugin_type,
                "test_result": "simulated",
                "webhook_count": len(plugin_info["webhooks"]),
                "test_resource": test_resource,
            }
            self.log_action("test_cluster_plugin", params, result)
            return result
        except Exception as e:
            self.log_action("test_cluster_plugin", params, error=e)
            raise RuntimeError(f"Failed to test cluster plugin: {str(e)}") from e

    def get_cluster_info(self) -> dict:
        """Get cluster information."""
        params: dict[str, Any] = {}
        try:
            version = self.version_api.get_code()
            nodes = self.core.list_node().items

            result = {
                "version": version.git_version,
                "platform": version.platform,
                "nodes_count": len(nodes),
                "kubernetes_version": f"{version.major}.{version.minor}",
            }
            self.log_action("get_cluster_info", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("get_cluster_info", params, error=e)
            raise RuntimeError(f"Failed to get cluster info: {str(e)}") from e

    def get_api_server_info(self) -> dict:
        """Get API server information."""
        params: dict[str, Any] = {}
        try:
            discovery_api = _km.k8s_client.DiscoveryV1API()
            server_groups = discovery_api.server_groups()

            result = {
                "api_groups": [group.name for group in server_groups.groups],
                "api_groups_count": len(server_groups.groups),
                "server_version": server_groups.server_version,
            }
            self.log_action("get_api_server_info", params, result)
            return result
        except ImportError:
            raise RuntimeError("Discovery client not available") from None
        except _km.ApiException as e:
            self.log_action("get_api_server_info", params, error=e)
            raise RuntimeError(f"Failed to get API server info: {str(e)}") from e

    def validate_kubeconfig(self) -> dict:
        """Validate kubeconfig."""
        params: dict[str, Any] = {}
        try:
            from kubernetes import config

            config.load_kube_config()

            result = {
                "status": "valid",
                "context": (
                    config.list_kube_config_contexts()[0]
                    if config.list_kube_config_contexts()
                    else None
                ),
            }
            self.log_action("validate_kubeconfig", params, result)
            return result
        except Exception as e:
            self.log_action("validate_kubeconfig", params, error=e)
            return {"status": "invalid", "error": str(e)}

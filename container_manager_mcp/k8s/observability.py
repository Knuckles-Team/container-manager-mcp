"""ObservabilityMixin for KubernetesManager (split from k8s_manager.py)."""

from typing import Any

import container_manager_mcp.k8s_manager as _km


class ObservabilityMixin:
    def top_pods(self, namespace: str | None = None) -> list[dict]:
        """Get resource usage for pods using metrics server."""
        params = {"namespace": namespace}
        try:
            # Metrics come from the metrics.k8s.io aggregated API via the
            # CustomObjects client (metrics-server exposes it as a CRD group).
            ns = namespace or self.namespace
            metrics = self.custom_objects.list_namespaced_custom_object(
                "metrics.k8s.io", "v1beta1", ns, "pods"
            )
            pod_metrics = metrics.get("items", [])
            result = [
                {
                    "name": metric["metadata"]["name"],
                    "namespace": metric["metadata"].get("namespace", ns),
                    "cpu": (
                        metric["containers"][0]["usage"]["cpu"]
                        if metric.get("containers")
                        else "N/A"
                    ),
                    "memory": (
                        metric["containers"][0]["usage"]["memory"]
                        if metric.get("containers")
                        else "N/A"
                    ),
                    "containers": [
                        {
                            "name": container["name"],
                            "cpu": container.get("usage", {}).get("cpu", "N/A"),
                            "memory": container.get("usage", {}).get("memory", "N/A"),
                        }
                        for container in (metric.get("containers") or [])
                    ],
                }
                for metric in pod_metrics
            ]
            self.log_action(
                "top_pods", params, {"count": len(result), "source": "metrics_server"}
            )
            return result
        except (ImportError, AttributeError):
            # Fallback to basic pod info if metrics API not available
            ns = namespace or self.namespace
            pods = self.core.list_namespaced_pod(ns).items
            result = [
                {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "cpu": "N/A (metrics server required)",
                    "memory": "N/A (metrics server required)",
                    "containers": [],
                }
                for pod in pods
            ]
            self.log_action(
                "top_pods",
                params,
                {"count": len(result), "note": "Metrics server API not available"},
            )
            return result
        except _km.ApiException as e:
            # Metrics server might not be installed, fall back to basic info
            if "NotFound" in str(e) or "ServiceUnavailable" in str(e):
                ns = namespace or self.namespace
                pods = self.core.list_namespaced_pod(ns).items
                result = [
                    {
                        "name": pod.metadata.name,
                        "namespace": pod.metadata.namespace,
                        "cpu": "N/A (metrics server not installed)",
                        "memory": "N/A (metrics server not installed)",
                        "containers": [],
                    }
                    for pod in pods
                ]
                self.log_action(
                    "top_pods",
                    params,
                    {"count": len(result), "note": "Metrics server not installed"},
                )
                return result
            self.log_action("top_pods", params, error=e)
            raise RuntimeError(f"Failed to get pod metrics: {str(e)}") from e

    def top_nodes(self) -> list[dict]:
        """Get resource usage for nodes using metrics server."""
        params: dict[str, Any] = {}
        try:
            # Node metrics from the metrics.k8s.io aggregated API via the
            # CustomObjects client.
            metrics = self.custom_objects.list_cluster_custom_object(
                "metrics.k8s.io", "v1beta1", "nodes"
            )
            node_metrics = metrics.get("items", [])
            result = [
                {
                    "name": metric["metadata"]["name"],
                    "cpu": metric.get("usage", {}).get("cpu", "N/A"),
                    "memory": metric.get("usage", {}).get("memory", "N/A"),
                }
                for metric in node_metrics
            ]
            self.log_action(
                "top_nodes", params, {"count": len(result), "source": "metrics_server"}
            )
            return result
        except (ImportError, AttributeError):
            # Fallback to basic node info if metrics API not available
            nodes = self.core.list_node().items
            result = [
                {
                    "name": node.metadata.name,
                    "cpu": "N/A (metrics server required)",
                    "memory": "N/A (metrics server required)",
                    "capacity": (
                        node.status.allocatable.dict()
                        if hasattr(node.status.allocatable, "dict")
                        else (
                            node.status.allocatable
                            if node.status and node.status.allocatable
                            else {}
                        )
                    ),
                }
                for node in nodes
            ]
            self.log_action(
                "top_nodes",
                params,
                {"count": len(result), "note": "Metrics server API not available"},
            )
            return result
        except _km.ApiException as e:
            # Metrics server might not be installed, fall back to basic info
            if "NotFound" in str(e) or "ServiceUnavailable" in str(e):
                nodes = self.core.list_node().items
                result = [
                    {
                        "name": node.metadata.name,
                        "cpu": "N/A (metrics server not installed)",
                        "memory": "N/A (metrics server not installed)",
                        "capacity": (
                            node.status.allocatable.dict()
                            if hasattr(node.status.allocatable, "dict")
                            else (
                                node.status.allocatable
                                if node.status and node.status.allocatable
                                else {}
                            )
                        ),
                    }
                    for node in nodes
                ]
                self.log_action(
                    "top_nodes",
                    params,
                    {"count": len(result), "note": "Metrics server not installed"},
                )
                return result
            self.log_action("top_nodes", params, error=e)
            raise RuntimeError(f"Failed to get node metrics: {str(e)}") from e

    def get_pod_metrics(self, namespace: str | None = None) -> list[dict]:
        """Get pod metrics."""
        params: dict[str, Any] = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            metrics = self.custom_objects.list_namespaced_custom_object(
                "metrics.k8s.io", "v1beta1", ns, "pods"
            )
            pod_metrics = metrics.get("items", [])
            result = [
                {
                    "name": metric["metadata"]["name"],
                    "namespace": metric["metadata"].get("namespace", ns),
                    "containers": [
                        {
                            "name": container["name"],
                            "cpu": container.get("usage", {}).get("cpu", ""),
                            "memory": container.get("usage", {}).get("memory", ""),
                        }
                        for container in (metric.get("containers") or [])
                    ],
                    "timestamp": self._ts(metric.get("timestamp")),
                }
                for metric in pod_metrics
            ]
            self.log_action("get_pod_metrics", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("get_pod_metrics", params, error=e)
            raise RuntimeError(f"Failed to get pod metrics: {str(e)}") from e

    def get_node_metrics(self) -> list[dict]:
        """Get node metrics."""
        params: dict[str, Any] = {}
        try:
            metrics = self.custom_objects.list_cluster_custom_object(
                "metrics.k8s.io", "v1beta1", "nodes"
            )
            node_metrics = metrics.get("items", [])
            result = [
                {
                    "name": metric["metadata"]["name"],
                    "cpu": metric.get("usage", {}).get("cpu", ""),
                    "memory": metric.get("usage", {}).get("memory", ""),
                    "timestamp": self._ts(metric.get("timestamp")),
                }
                for metric in node_metrics
            ]
            self.log_action("get_node_metrics", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("get_node_metrics", params, error=e)
            raise RuntimeError(f"Failed to get node metrics: {str(e)}") from e

    def get_top_pods(self, namespace: str | None = None) -> list[dict]:
        """Get top pods by resource usage."""
        params: dict[str, Any] = {"namespace": namespace}
        try:
            pod_metrics = self.get_pod_metrics(namespace)

            # Sort by CPU usage
            sorted_pods = sorted(
                pod_metrics,
                key=lambda x: sum(
                    float(c["cpu"].replace("m", "")) if c["cpu"].endswith("m") else 0
                    for c in x["containers"]
                ),
                reverse=True,
            )

            result = sorted_pods[:10]  # Top 10
            self.log_action("get_top_pods", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("get_top_pods", params, error=e)
            raise RuntimeError(f"Failed to get top pods: {str(e)}") from e

    def get_top_nodes(self) -> list[dict]:
        """Get top nodes by resource usage."""
        params: dict[str, Any] = {}
        try:
            node_metrics = self.get_node_metrics()

            # Sort by CPU usage
            sorted_nodes = sorted(
                node_metrics,
                key=lambda x: (
                    float(x["cpu"].replace("m", "")) if x["cpu"].endswith("m") else 0
                ),
                reverse=True,
            )

            result = sorted_nodes
            self.log_action("get_top_nodes", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("get_top_nodes", params, error=e)
            raise RuntimeError(f"Failed to get top nodes: {str(e)}") from e

    def get_pod_resource_usage(self, pod_name: str, namespace: str) -> dict:
        """Get detailed resource usage for a pod."""
        params = {"pod_name": pod_name, "namespace": namespace}
        try:
            pod_metrics = self.get_pod_metrics(namespace)

            for metric in pod_metrics:
                if metric["name"] == pod_name:
                    result = {
                        "pod_name": pod_name,
                        "namespace": namespace,
                        "containers": metric["containers"],
                        "timestamp": metric["timestamp"],
                    }
                    self.log_action("get_pod_resource_usage", params, result)
                    return result

            raise ValueError(f"Pod {pod_name} not found in metrics")
        except Exception as e:
            self.log_action("get_pod_resource_usage", params, error=e)
            raise RuntimeError(f"Failed to get pod resource usage: {str(e)}") from e

    def get_cluster_resource_summary(self) -> dict:
        """Get cluster-wide resource summary."""
        params: dict[str, Any] = {}
        try:
            nodes = self.core.list_node().items
            pods = self.core.list_pod_for_all_namespaces().items
            node_metrics = self.get_node_metrics()
            pod_metrics = self.get_pod_metrics()

            total_cpu = sum(
                float(n["cpu"].replace("m", "")) if n["cpu"].endswith("m") else 0
                for n in node_metrics
            )
            total_memory = sum(
                (
                    float(n["memory"].replace("Mi", ""))
                    if n["memory"].endswith("Mi")
                    else 0
                )
                for n in node_metrics
            )

            result = {
                "nodes_count": len(nodes),
                "pods_count": len(pods),
                "total_cpu_millicores": total_cpu,
                "total_memory_mib": total_memory,
                "node_metrics_count": len(node_metrics),
                "pod_metrics_count": len(pod_metrics),
            }
            self.log_action("get_cluster_resource_summary", params, result)
            return result
        except Exception as e:
            self.log_action("get_cluster_resource_summary", params, error=e)
            raise RuntimeError(
                f"Failed to get cluster resource summary: {str(e)}"
            ) from e

    def get_autoscaler_metrics(self, name: str, namespace: str) -> dict:
        """Get metrics for a HorizontalPodAutoscaler."""
        params = {"name": name, "namespace": namespace}
        try:
            autoscaling_api = self.autoscaling
            hpa = autoscaling_api.read_namespaced_horizontal_pod_autoscaler(
                name, namespace
            )

            result = {
                "name": name,
                "namespace": namespace,
                "min_replicas": hpa.spec.min_replicas if hpa.spec else None,
                "max_replicas": hpa.spec.max_replicas if hpa.spec else None,
                "current_replicas": hpa.status.current_replicas if hpa.status else None,
                "target_replicas": hpa.status.desired_replicas if hpa.status else None,
                "metrics": hpa.spec.metrics if hpa.spec else [],
                "conditions": hpa.status.conditions if hpa.status else [],
            }
            self.log_action("get_autoscaler_metrics", params, result)
            return result
        except ImportError:
            raise RuntimeError("Autoscaling client not available") from None
        except _km.ApiException as e:
            self.log_action("get_autoscaler_metrics", params, error=e)
            raise RuntimeError(f"Failed to get autoscaler metrics: {str(e)}") from e

    def set_autoscaler_metrics(
        self, name: str, namespace: str, metrics: list[dict]
    ) -> dict:
        """Set metrics for a HorizontalPodAutoscaler."""
        params = {"name": name, "namespace": namespace, "metrics": metrics}
        try:
            autoscaling_api = self.autoscaling
            hpa = autoscaling_api.read_namespaced_horizontal_pod_autoscaler(
                name, namespace
            )

            # Convert metrics to proper format
            metric_specs = []
            for metric in metrics:
                if metric["type"] == "Resource":
                    metric_spec = _km.k8s_client.V2MetricSpec(
                        type="Resource",
                        resource=_km.k8s_client.V2ResourceMetricSource(
                            name=metric["resource"],
                            target=_km.k8s_client.V2MetricTarget(
                                type=metric["target_type"],
                                average_utilization=metric.get("average_utilization"),
                            ),
                        ),
                    )
                    metric_specs.append(metric_spec)

            if hpa.spec:
                hpa.spec.metrics = metric_specs

            autoscaling_api.patch_namespaced_horizontal_pod_autoscaler(
                name, namespace, hpa
            )
            result = {
                "name": name,
                "namespace": namespace,
                "status": "updated",
                "metrics_count": len(metrics),
            }
            self.log_action("set_autoscaler_metrics", params, result)
            return result
        except ImportError:
            raise RuntimeError("Autoscaling client not available") from None
        except _km.ApiException as e:
            self.log_action("set_autoscaler_metrics", params, error=e)
            raise RuntimeError(f"Failed to set autoscaler metrics: {str(e)}") from e

    def scale_deployment_autoscaler(
        self, name: str, namespace: str, min_replicas: int, max_replicas: int
    ) -> dict:
        """Scale deployment autoscaler bounds."""
        params = {
            "name": name,
            "namespace": namespace,
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
        }
        try:
            autoscaling_api = self.autoscaling
            hpa = autoscaling_api.read_namespaced_horizontal_pod_autoscaler(
                name, namespace
            )

            if hpa.spec:
                hpa.spec.min_replicas = min_replicas
                hpa.spec.max_replicas = max_replicas

            autoscaling_api.patch_namespaced_horizontal_pod_autoscaler(
                name, namespace, hpa
            )
            result = {
                "name": name,
                "namespace": namespace,
                "min_replicas": min_replicas,
                "max_replicas": max_replicas,
                "status": "scaled",
            }
            self.log_action("scale_deployment_autoscaler", params, result)
            return result
        except ImportError:
            raise RuntimeError("Autoscaling client not available") from None
        except _km.ApiException as e:
            self.log_action("scale_deployment_autoscaler", params, error=e)
            raise RuntimeError(f"Failed to scale autoscaler: {str(e)}") from e

    def get_autoscaler_history(self, name: str, namespace: str) -> dict:
        """Get autoscaler scaling history."""
        params = {"name": name, "namespace": namespace}
        try:
            # Get events for the HPA
            events = self.core.list_namespaced_event(namespace=namespace).items

            hpa_events = []
            for event in events:
                if (
                    event.involved_object
                    and event.involved_object.name == name
                    and event.involved_object.kind == "HorizontalPodAutoscaler"
                ):
                    hpa_events.append(
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
                "name": name,
                "namespace": namespace,
                "events": hpa_events,
                "event_count": len(hpa_events),
            }
            self.log_action("get_autoscaler_history", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("get_autoscaler_history", params, error=e)
            raise RuntimeError(f"Failed to get autoscaler history: {str(e)}") from e

    def debug_pod(self, pod_name: str, namespace: str) -> dict:
        """Debug a pod by gathering diagnostic information."""
        params = {"pod_name": pod_name, "namespace": namespace}
        try:
            pod = self.core.read_namespaced_pod(pod_name, namespace)
            events = self.get_resource_events("pod", pod_name, namespace)
            logs = self.stream_pod_logs(pod_name, namespace, tail_lines=50)

            result = {
                "pod_name": pod_name,
                "namespace": namespace,
                "phase": pod.status.phase if pod.status else None,
                "conditions": pod.status.conditions if pod.status else [],
                "events": events,
                "logs_sample": logs["logs"][:500] if logs else "",
                "restart_count": (
                    sum(c.restart_count for c in pod.status.container_statuses if c)
                    if pod.status
                    else 0
                ),
            }
            self.log_action("debug_pod", params, result)
            return result
        except Exception as e:
            self.log_action("debug_pod", params, error=e)
            raise RuntimeError(f"Failed to debug pod: {str(e)}") from e

    def debug_node(self, node_name: str) -> dict:
        """Debug a node by gathering diagnostic information."""
        params = {"node_name": node_name}
        try:
            node = self.core.read_node(node_name)
            conditions = self.get_node_conditions(node_name)
            pods = self.core.list_pod_for_all_namespaces(
                field_selector=f"spec.nodeName={node_name}"
            ).items

            result = {
                "node_name": node_name,
                "conditions": conditions["conditions"],
                "ready": conditions["ready"],
                "pods_on_node": len(pods),
                "unschedulable": node.spec.unschedulable if node.spec else False,
                "node_info": node.status.node_info if node.status else None,
            }
            self.log_action("debug_node", params, result)
            return result
        except Exception as e:
            self.log_action("debug_node", params, error=e)
            raise RuntimeError(f"Failed to debug node: {str(e)}") from e

    def debug_service(self, service_name: str, namespace: str) -> dict:
        """Debug a service by gathering diagnostic information."""
        params = {"service_name": service_name, "namespace": namespace}
        try:
            service = self.core.read_namespaced_service(service_name, namespace)
            endpoints = self.core.read_namespaced_endpoints(service_name, namespace)
            pods = self.core.list_namespaced_pod(
                namespace, label_selector=f"app={service_name}"
            ).items

            result = {
                "service_name": service_name,
                "namespace": namespace,
                "type": service.spec.type,
                "cluster_ip": service.spec.cluster_ip,
                "ports": service.spec.ports if service.spec else [],
                "endpoints": endpoints.subsets if endpoints.subsets else [],
                "target_pods": len(pods),
                "selector": service.spec.selector if service.spec else None,
            }
            self.log_action("debug_service", params, result)
            return result
        except Exception as e:
            self.log_action("debug_service", params, error=e)
            raise RuntimeError(f"Failed to debug service: {str(e)}") from e

    def debug_deployment(self, deployment_name: str, namespace: str) -> dict:
        """Debug a deployment by gathering diagnostic information."""
        params = {"deployment_name": deployment_name, "namespace": namespace}
        try:
            apps_api = self.apps
            deployment = apps_api.read_namespaced_deployment(deployment_name, namespace)
            replicasets = apps_api.list_namespaced_replica_set(
                namespace, label_selector=f"app={deployment_name}"
            ).items
            pods = self.core.list_namespaced_pod(
                namespace, label_selector=f"app={deployment_name}"
            ).items

            result = {
                "deployment_name": deployment_name,
                "namespace": namespace,
                "replicas": deployment.spec.replicas if deployment.spec else None,
                "available_replicas": (
                    deployment.status.available_replicas if deployment.status else None
                ),
                "updated_replicas": (
                    deployment.status.updated_replicas if deployment.status else None
                ),
                "replicasets_count": len(replicasets),
                "pods_count": len(pods),
                "conditions": deployment.status.conditions if deployment.status else [],
            }
            self.log_action("debug_deployment", params, result)
            return result
        except Exception as e:
            self.log_action("debug_deployment", params, error=e)
            raise RuntimeError(f"Failed to debug deployment: {str(e)}") from e

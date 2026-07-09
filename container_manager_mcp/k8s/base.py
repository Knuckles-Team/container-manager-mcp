"""Base class for :class:`KubernetesManager`: __init__ + shared helpers."""

import os
from typing import Any
import container_manager_mcp.k8s_manager as _km


class _K8sBase:
    def __init__(
        self,
        context: str | None = None,
        namespace: str | None = None,
        silent: bool = False,
        log_file: str | None = None,
    ):
        super().__init__(silent, log_file)
        if _km.k8s_client is None:
            raise ImportError(
                "Please install the kubernetes client: "
                "pip install container-manager-mcp[kubernetes]"
            )
        self.namespace = namespace or os.environ.get(
            "CONTAINER_MANAGER_K8S_NAMESPACE", "default"
        )
        context = context or os.environ.get("CONTAINER_MANAGER_KUBECONTEXT")
        try:
            if os.environ.get("KUBERNETES_SERVICE_HOST"):
                _km.k8s_config.load_incluster_config()
            else:
                _km.k8s_config.load_kube_config(context=context)
        except Exception as e:
            self.logger.error(f"Failed to load kubeconfig: {str(e)}")
            raise RuntimeError(
                f"Could not load Kubernetes config (context={context or 'default'}): "
                f"{str(e)}"
            ) from e
        self.core = _km.k8s_client.CoreV1Api()
        self.apps = _km.k8s_client.AppsV1Api()
        # Centralized API-group clients (all constructed once here so mixins /
        # methods never re-instantiate them inline).
        self.networking = _km.k8s_client.NetworkingV1Api()
        self.batch = _km.k8s_client.BatchV1Api()
        self.rbac = _km.k8s_client.RbacAuthorizationV1Api()
        self.authz = _km.k8s_client.AuthorizationV1Api()
        self.authn = _km.k8s_client.AuthenticationV1Api()
        self.storage = _km.k8s_client.StorageV1Api()
        self.scheduling = _km.k8s_client.SchedulingV1Api()
        self.policy = _km.k8s_client.PolicyV1Api()
        self.autoscaling = _km.k8s_client.AutoscalingV2Api()
        self.certificates = _km.k8s_client.CertificatesV1Api()
        self.discovery = _km.k8s_client.DiscoveryV1Api()
        self.admission = _km.k8s_client.AdmissionregistrationV1Api()
        self.custom_objects = _km.k8s_client.CustomObjectsApi()
        self.version_api = _km.k8s_client.VersionApi()
        self.apiextensions = _km.k8s_client.ApiextensionsV1Api()
    @staticmethod
    def _unsupported(op: str) -> RuntimeError:
        return RuntimeError(
            f"'{op}' is a node-local Docker operation not available on the "
            "Kubernetes backend; use create_service / list_services / "
            "scale_service / service_ps / service_logs instead."
        )
    @staticmethod
    def _ts(value: Any) -> str:
        """Render a Kubernetes datetime (or string) as ISO 8601."""
        if value is None:
            return "unknown"
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)
    @staticmethod
    def _node_role(labels: dict[str, str]) -> str:
        """Map Kubernetes role labels onto Swarm vocabulary."""
        for key in labels or {}:
            if key in (
                "node-role.kubernetes.io/control-plane",
                "node-role.kubernetes.io/master",
            ):
                return "manager"
        return "worker"
    @staticmethod
    def _parse_mounts(mounts: list[str] | None) -> tuple[list, list]:
        """Translate ``source:target`` bind strings into volumes + mounts.

        Mirrors the homelab's host-bind-mount model: each mount becomes a
        ``hostPath`` volume. Returns ``(volumes, volume_mounts)``.
        """
        volumes: list = []
        volume_mounts: list = []
        for idx, raw in enumerate(mounts or []):
            parts = raw.split(":")
            if len(parts) < 2:
                continue
            source, target = parts[0], parts[1]
            name = f"vol{idx}"
            volumes.append(
                _km.k8s_client.V1Volume(
                    name=name, host_path=_km.k8s_client.V1HostPathVolumeSource(path=source)
                )
            )
            volume_mounts.append(_km.k8s_client.V1VolumeMount(name=name, mount_path=target))
        return volumes, volume_mounts
    @staticmethod
    def _constraints_to_node_selector(constraints: list[str] | None) -> dict[str, str]:
        """Translate Swarm placement constraints into a nodeSelector.

        Handles ``node.labels.<k> == <v>`` and ``node.hostname == <v>``.
        """
        selector: dict[str, str] = {}
        for c in constraints or []:
            expr = c.replace("==", "=").replace("!=", "=")
            if "=" not in expr:
                continue
            lhs, rhs = (s.strip() for s in expr.split("=", 1))
            if lhs.startswith("node.labels."):
                selector[lhs[len("node.labels.") :]] = rhs
            elif lhs == "node.hostname":
                selector["kubernetes.io/hostname"] = rhs
        return selector
    def get_version(self) -> dict:
        params: dict[str, Any] = {}
        try:
            version = self.version_api.get_code()
            result = {
                "version": getattr(version, "git_version", "unknown"),
                "api_version": f"{getattr(version, 'major', '')}.{getattr(version, 'minor', '')}",
                "os": getattr(version, "platform", "unknown"),
                "arch": getattr(version, "go_version", "unknown"),
                "build_time": getattr(version, "build_date", "unknown"),
            }
            self.log_action("get_version", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("get_version", params, error=e)
            raise RuntimeError(f"Failed to get version: {str(e)}") from e
    def get_info(self) -> dict:
        params: dict[str, Any] = {}
        try:
            nodes = self.core.list_node().items
            pods = self.core.list_pod_for_all_namespaces().items
            running = sum(1 for p in pods if (p.status and p.status.phase == "Running"))
            first = nodes[0].status.node_info if nodes else None
            result = {
                "containers_total": len(pods),
                "containers_running": running,
                "images": 0,
                "driver": "kubernetes",
                "platform": (
                    f"{first.os_image} {first.architecture}" if first else "kubernetes"
                ),
                "nodes": len(nodes),
                "namespace": self.namespace,
            }
            self.log_action("get_info", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("get_info", params, error=e)
            raise RuntimeError(f"Failed to get info: {str(e)}") from e
    def init_swarm(self, advertise_addr: str | None = None) -> dict:
        result = {
            "status": "kubernetes",
            "note": "cluster lifecycle is managed by RKE2/k3s; nothing to init",
        }
        self.log_action("init_swarm", {"advertise_addr": advertise_addr}, result)
        return result
    def leave_swarm(self, force: bool = False) -> dict:
        result = {
            "status": "kubernetes",
            "note": "node membership is managed by RKE2/k3s; use remove_node to drain",
        }
        self.log_action("leave_swarm", {"force": force}, result)
        return result
    def _node_summary(self, node) -> dict:
        meta = node.metadata
        info = node.status.node_info if node.status else None
        addresses = (node.status.addresses or []) if node.status else []
        addr = next(
            (a.address for a in addresses if a.type == "InternalIP"),
            "unknown",
        )
        return {
            "id": meta.uid or "unknown",
            "hostname": meta.name,
            "role": self._node_role(meta.labels or {}),
            "availability": "drain" if node.spec.unschedulable else "active",
            "state": "ready",
            "addr": addr,
            "labels": meta.labels or {},
            "engine_version": getattr(info, "kubelet_version", "unknown"),
            "platform": {
                "os": getattr(info, "os_image", "unknown"),
                "arch": getattr(info, "architecture", "unknown"),
            },
            "manager": self._node_role(meta.labels or {}) == "manager",
        }
    def _drain_node(self, node_id: str) -> None:
        """Evict non-mirror, non-daemonset pods from a cordoned node."""
        field = f"spec.nodeName={node_id}"
        pods = self.core.list_pod_for_all_namespaces(field_selector=field).items
        for pod in pods:
            owners = pod.metadata.owner_references or []
            if any(o.kind == "DaemonSet" for o in owners):
                continue
            if (pod.metadata.annotations or {}).get("kubernetes.io/config.mirror"):
                continue
            eviction = _km.k8s_client.V1Eviction(
                metadata=_km.k8s_client.V1ObjectMeta(
                    name=pod.metadata.name, namespace=pod.metadata.namespace
                )
            )
            try:
                self.core.create_namespaced_pod_eviction(
                    pod.metadata.name, pod.metadata.namespace, eviction
                )
            except _km.ApiException:
                continue
    def _deployment_summary(self, dep) -> dict:
        spec = dep.spec
        containers = spec.template.spec.containers if spec and spec.template else []
        image = containers[0].image if containers else "unknown"
        ports = []
        if containers and containers[0].ports:
            ports = [str(p.container_port) for p in containers[0].ports]
        return {
            "id": (dep.metadata.uid or "unknown")[:12],
            "name": dep.metadata.name,
            "namespace": dep.metadata.namespace,
            "image": image,
            "replicas": spec.replicas if spec else 0,
            "ports": ", ".join(ports) if ports else "none",
            "created": self._ts(dep.metadata.creation_timestamp),
            "updated": "unknown",
        }

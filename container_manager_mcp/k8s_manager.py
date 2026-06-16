#!/usr/bin/env python
"""Kubernetes backend for the container manager.

``KubernetesManager`` implements the same :class:`ContainerManagerBase`
interface the Swarm (``DockerManager``) backend exposes, so the MCP tool
surface (``list_nodes``/``create_service``/``scale_service``/...) works
unchanged when ``CONTAINER_MANAGER_TYPE=kubernetes``. Swarm-shaped verbs map
onto Kubernetes objects:

    list_nodes      -> CoreV1Api.list_node
    create_service  -> AppsV1Api Deployment (+ Service when ports published)
    scale_service   -> patch_namespaced_deployment_scale
    update_node     -> patch_node (labels / cordon / role label)
    service_ps      -> list pods for the deployment
    service_logs    -> pod logs
    compose_up      -> kompose convert + kubectl apply

Genuinely node-local Docker operations (run_container, image/volume/network
management) are not meaningful against a cluster and raise ``RuntimeError`` —
mirroring how ``PodmanManager`` raises for Swarm verbs it cannot serve.
"""

import os
import subprocess
import tempfile
from typing import Any

from container_manager_mcp.container_manager import ContainerManagerBase
from container_manager_mcp.models import (
    ContainerInfo,
    ImageInfo,
    NetworkInfo,
    VolumeInfo,
)

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
    from kubernetes.client.rest import ApiException
except ImportError:
    k8s_client = None  # type: ignore
    k8s_config = None  # type: ignore
    ApiException = Exception  # type: ignore


class KubernetesManager(ContainerManagerBase):
    """ContainerManagerBase implementation backed by a Kubernetes cluster."""

    def __init__(
        self,
        context: str | None = None,
        namespace: str | None = None,
        silent: bool = False,
        log_file: str | None = None,
    ):
        super().__init__(silent, log_file)
        if k8s_client is None:
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
                k8s_config.load_incluster_config()
            else:
                k8s_config.load_kube_config(context=context)
        except Exception as e:
            self.logger.error(f"Failed to load kubeconfig: {str(e)}")
            raise RuntimeError(
                f"Could not load Kubernetes config (context={context or 'default'}): "
                f"{str(e)}"
            ) from e
        self.core = k8s_client.CoreV1Api()
        self.apps = k8s_client.AppsV1Api()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
                k8s_client.V1Volume(
                    name=name, host_path=k8s_client.V1HostPathVolumeSource(path=source)
                )
            )
            volume_mounts.append(k8s_client.V1VolumeMount(name=name, mount_path=target))
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

    # ------------------------------------------------------------------
    # Version / info
    # ------------------------------------------------------------------
    def get_version(self) -> dict:
        params: dict[str, Any] = {}
        try:
            version = k8s_client.VersionApi().get_code()
            result = {
                "version": getattr(version, "git_version", "unknown"),
                "api_version": f"{getattr(version, 'major', '')}.{getattr(version, 'minor', '')}",
                "os": getattr(version, "platform", "unknown"),
                "arch": getattr(version, "go_version", "unknown"),
                "build_time": getattr(version, "build_date", "unknown"),
            }
            self.log_action("get_version", params, result)
            return result
        except ApiException as e:
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
        except ApiException as e:
            self.log_action("get_info", params, error=e)
            raise RuntimeError(f"Failed to get info: {str(e)}") from e

    # ------------------------------------------------------------------
    # Cluster lifecycle (no-ops: RKE2/k3s owns this, not the MCP)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------
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
        except ApiException as e:
            self.log_action("list_nodes", params, error=e)
            raise RuntimeError(f"Failed to list nodes: {str(e)}") from e

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

    def inspect_node(self, node_id: str) -> dict:
        params = {"node_id": node_id}
        try:
            node = self.core.read_node(node_id)
            result = self._node_summary(node)
            self.log_action("inspect_node", params, {"id": node_id})
            return result
        except ApiException as e:
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
        except ApiException as e:
            self.log_action("update_node", params, error=e)
            raise RuntimeError(f"Failed to update node: {str(e)}") from e

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
            eviction = k8s_client.V1Eviction(
                metadata=k8s_client.V1ObjectMeta(
                    name=pod.metadata.name, namespace=pod.metadata.namespace
                )
            )
            try:
                self.core.create_namespaced_pod_eviction(
                    pod.metadata.name, pod.metadata.namespace, eviction
                )
            except ApiException:
                continue

    def remove_node(self, node_id: str, force: bool = False) -> dict:
        params = {"node_id": node_id, "force": force}
        try:
            self.core.patch_node(node_id, {"spec": {"unschedulable": True}})
            self._drain_node(node_id)
            self.core.delete_node(node_id)
            result = {"removed": node_id}
            self.log_action("remove_node", params, result)
            return result
        except ApiException as e:
            self.log_action("remove_node", params, error=e)
            raise RuntimeError(f"Failed to remove node: {str(e)}") from e

    # ------------------------------------------------------------------
    # Service (Deployment) operations
    # ------------------------------------------------------------------
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

    def list_services(self) -> list[dict]:
        params: dict[str, Any] = {}
        try:
            deps = self.apps.list_deployment_for_all_namespaces().items
            result = [self._deployment_summary(d) for d in deps]
            self.log_action("list_services", params, {"count": len(result)})
            return result
        except ApiException as e:
            self.log_action("list_services", params, error=e)
            raise RuntimeError(f"Failed to list services: {str(e)}") from e

    def create_service(
        self,
        name: str,
        image: str,
        replicas: int = 1,
        ports: dict[str, str] | None = None,
        mounts: list[str] | None = None,
    ) -> dict:
        params = {
            "name": name,
            "image": image,
            "replicas": replicas,
            "ports": ports,
            "mounts": mounts,
        }
        try:
            volumes, volume_mounts = self._parse_mounts(mounts)
            container_ports = []
            if ports:
                container_ports = [
                    k8s_client.V1ContainerPort(container_port=int(cp.split("/")[0]))
                    for cp in ports
                ]
            container = k8s_client.V1Container(
                name=name,
                image=image,
                ports=container_ports or None,
                volume_mounts=volume_mounts or None,
            )
            pod_spec = k8s_client.V1PodSpec(
                containers=[container], volumes=volumes or None
            )
            template = k8s_client.V1PodTemplateSpec(
                metadata=k8s_client.V1ObjectMeta(labels={"app": name}),
                spec=pod_spec,
            )
            dep = k8s_client.V1Deployment(
                metadata=k8s_client.V1ObjectMeta(name=name, labels={"app": name}),
                spec=k8s_client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=k8s_client.V1LabelSelector(match_labels={"app": name}),
                    template=template,
                ),
            )
            created = self.apps.create_namespaced_deployment(self.namespace, dep)

            if ports:
                svc_ports = [
                    k8s_client.V1ServicePort(
                        port=int(host_port),
                        target_port=int(container_port.split("/")[0]),
                    )
                    for container_port, host_port in ports.items()
                ]
                svc = k8s_client.V1Service(
                    metadata=k8s_client.V1ObjectMeta(name=name),
                    spec=k8s_client.V1ServiceSpec(
                        selector={"app": name}, ports=svc_ports
                    ),
                )
                self.core.create_namespaced_service(self.namespace, svc)

            result = self._deployment_summary(created)
            self.log_action("create_service", params, result)
            return result
        except ApiException as e:
            self.log_action("create_service", params, error=e)
            raise RuntimeError(f"Failed to create service: {str(e)}") from e

    def remove_service(self, service_id: str) -> dict:
        params = {"service_id": service_id}
        try:
            self.apps.delete_namespaced_deployment(service_id, self.namespace)
            try:
                self.core.delete_namespaced_service(service_id, self.namespace)
            except ApiException:
                pass
            result = {"removed": service_id}
            self.log_action("remove_service", params, result)
            return result
        except ApiException as e:
            self.log_action("remove_service", params, error=e)
            raise RuntimeError(f"Failed to remove service: {str(e)}") from e

    def inspect_service(self, service_id: str) -> dict:
        params = {"service_id": service_id}
        try:
            dep = self.apps.read_namespaced_deployment(service_id, self.namespace)
            result = dep.to_dict()
            self.log_action("inspect_service", params, {"id": service_id})
            return result
        except ApiException as e:
            self.log_action("inspect_service", params, error=e)
            raise RuntimeError(f"Failed to inspect service: {str(e)}") from e

    def scale_service(self, service_id: str, replicas: int) -> dict:
        params = {"service_id": service_id, "replicas": replicas}
        try:
            self.apps.patch_namespaced_deployment_scale(
                service_id, self.namespace, {"spec": {"replicas": replicas}}
            )
            result = {"service": service_id, "replicas": replicas, "scaled": True}
            self.log_action("scale_service", params, result)
            return result
        except ApiException as e:
            self.log_action("scale_service", params, error=e)
            raise RuntimeError(f"Failed to scale service: {str(e)}") from e

    def update_service(
        self,
        service_id: str,
        image: str | None = None,
        replicas: int | None = None,
        env: list[str] | None = None,
        constraints: list[str] | None = None,
        labels: dict[str, str] | None = None,
        force: bool = False,
    ) -> dict:
        """Patch a Deployment in place (image, replicas, env, nodeSelector)."""
        params = {
            "service_id": service_id,
            "image": image,
            "replicas": replicas,
            "env": env,
            "constraints": constraints,
            "labels": labels,
            "force": force,
        }
        try:
            container: dict[str, Any] = {"name": service_id}
            if image:
                container["image"] = image
            if env is not None:
                container["env"] = [
                    {"name": k, "value": v}
                    for k, _, v in (e.partition("=") for e in env)
                ]
            pod_spec: dict[str, Any] = {"containers": [container]}
            if constraints is not None:
                pod_spec["nodeSelector"] = self._constraints_to_node_selector(
                    constraints
                )
            template_meta: dict[str, Any] = {}
            if force:
                template_meta["annotations"] = {
                    "container-manager/restartedAt": "force"
                }
            spec: dict[str, Any] = {"template": {"spec": pod_spec}}
            if template_meta:
                spec["template"]["metadata"] = template_meta
            if replicas is not None:
                spec["replicas"] = replicas
            body: dict[str, Any] = {"spec": spec}
            if labels is not None:
                body["metadata"] = {"labels": labels}
            self.apps.patch_namespaced_deployment(service_id, self.namespace, body)
            result = {"service": service_id, "updated": True, "image": image}
            self.log_action("update_service", params, result)
            return result
        except ApiException as e:
            self.log_action("update_service", params, error=e)
            raise RuntimeError(f"Failed to update service: {str(e)}") from e

    def service_ps(self, service_id: str) -> list[dict]:
        params = {"service_id": service_id}
        try:
            pods = self.core.list_namespaced_pod(
                self.namespace, label_selector=f"app={service_id}"
            ).items
            result = []
            for pod in pods:
                status = pod.status
                state = status.phase if status else "unknown"
                error = ""
                for cs in (status.container_statuses or []) if status else []:
                    waiting = cs.state.waiting if cs.state else None
                    if waiting and waiting.reason:
                        error = waiting.reason
                        break
                result.append(
                    {
                        "id": pod.metadata.name,
                        "node": pod.spec.node_name or "",
                        "desired_state": "Running",
                        "state": state,
                        "error": error,
                        "timestamp": self._ts(pod.metadata.creation_timestamp),
                    }
                )
            self.log_action("service_ps", params, {"count": len(result)})
            return result
        except ApiException as e:
            self.log_action("service_ps", params, error=e)
            raise RuntimeError(f"Failed to list service tasks: {str(e)}") from e

    def service_logs(self, service_id: str, tail: int = 100) -> dict:
        params = {"service_id": service_id, "tail": tail}
        try:
            pods = self.core.list_namespaced_pod(
                self.namespace, label_selector=f"app={service_id}"
            ).items
            chunks = []
            for pod in pods:
                try:
                    log = self.core.read_namespaced_pod_log(
                        pod.metadata.name,
                        self.namespace,
                        tail_lines=tail,
                        timestamps=True,
                    )
                except ApiException:
                    log = ""
                chunks.append(f"=== {pod.metadata.name} ===\n{log}")
            result = {"service": service_id, "logs": "\n".join(chunks)}
            self.log_action("service_logs", params, {"service": service_id})
            return result
        except ApiException as e:
            self.log_action("service_logs", params, error=e)
            raise RuntimeError(f"Failed to get service logs: {str(e)}") from e

    # ------------------------------------------------------------------
    # Compose -> manifests (kompose convert + kubectl apply)
    # ------------------------------------------------------------------
    def compose_up(
        self, compose_file: str, detach: bool = True, build: bool = False
    ) -> str:
        params = {"compose_file": compose_file, "detach": detach, "build": build}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "manifests.yaml")
                convert = subprocess.run(
                    ["kompose", "convert", "-f", compose_file, "-o", out],
                    capture_output=True,
                    text=True,
                )
                if convert.returncode != 0:
                    raise RuntimeError(convert.stderr)
                apply = subprocess.run(
                    ["kubectl", "apply", "-n", self.namespace, "-f", out],
                    capture_output=True,
                    text=True,
                )
                if apply.returncode != 0:
                    raise RuntimeError(apply.stderr)
                self.log_action("compose_up", params, apply.stdout)
                return apply.stdout
        except Exception as e:
            self.log_action("compose_up", params, error=e)
            raise RuntimeError(f"Failed to apply manifests: {str(e)}") from e

    def compose_down(self, compose_file: str) -> str:
        params = {"compose_file": compose_file}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "manifests.yaml")
                convert = subprocess.run(
                    ["kompose", "convert", "-f", compose_file, "-o", out],
                    capture_output=True,
                    text=True,
                )
                if convert.returncode != 0:
                    raise RuntimeError(convert.stderr)
                delete = subprocess.run(
                    ["kubectl", "delete", "-n", self.namespace, "-f", out],
                    capture_output=True,
                    text=True,
                )
                if delete.returncode != 0:
                    raise RuntimeError(delete.stderr)
                self.log_action("compose_down", params, delete.stdout)
                return delete.stdout
        except Exception as e:
            self.log_action("compose_down", params, error=e)
            raise RuntimeError(f"Failed to delete manifests: {str(e)}") from e

    def compose_ps(self, compose_file: str) -> str:
        params = {"compose_file": compose_file}
        try:
            result = subprocess.run(
                ["kubectl", "get", "all", "-n", self.namespace],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_ps", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_ps", params, error=e)
            raise RuntimeError(f"Failed to list resources: {str(e)}") from e

    def compose_logs(self, compose_file: str, service: str | None = None) -> str:
        params = {"compose_file": compose_file, "service": service}
        try:
            cmd = ["kubectl", "logs", "-n", self.namespace]
            cmd += [f"deploy/{service}"] if service else ["--all-containers"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_logs", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_logs", params, error=e)
            raise RuntimeError(f"Failed to get logs: {str(e)}") from e

    # ------------------------------------------------------------------
    # Node-local Docker operations — not meaningful against a cluster
    # ------------------------------------------------------------------
    def list_images(self) -> list[ImageInfo]:
        raise self._unsupported("list_images")

    def pull_image(self, image: str, tag: str = "latest", platform: str | None = None):
        raise self._unsupported("pull_image")

    def remove_image(self, image: str, force: bool = False) -> dict:
        raise self._unsupported("remove_image")

    def prune_images(self, force: bool = False, all: bool = False) -> dict:
        raise self._unsupported("prune_images")

    def list_containers(self, all: bool = False) -> list[ContainerInfo]:
        raise self._unsupported("list_containers")

    def run_container(
        self,
        image: str,
        name: str | None = None,
        command: str | None = None,
        detach: bool = False,
        ports: dict[str, str] | None = None,
        volumes: dict[str, dict] | None = None,
        environment: dict[str, str] | list[str] | None = None,
        labels: dict[str, str] | None = None,
    ) -> dict:
        raise self._unsupported("run_container")

    def inspect_container(self, container_id: str) -> dict:
        raise self._unsupported("inspect_container")

    def stop_container(self, container_id: str, timeout: int = 10) -> dict:
        raise self._unsupported("stop_container")

    def remove_container(self, container_id: str, force: bool = False) -> dict:
        raise self._unsupported("remove_container")

    def prune_containers(self) -> dict:
        raise self._unsupported("prune_containers")

    def get_container_logs(self, container_id: str, tail: str = "50") -> str:
        raise self._unsupported("get_container_logs")

    def exec_in_container(
        self, container_id: str, command: list[str], detach: bool = False
    ) -> dict:
        raise self._unsupported("exec_in_container")

    def list_volumes(self) -> list[VolumeInfo]:
        raise self._unsupported("list_volumes")

    def create_volume(self, name: str) -> VolumeInfo:
        raise self._unsupported("create_volume")

    def remove_volume(self, name: str, force: bool = False) -> dict:
        raise self._unsupported("remove_volume")

    def prune_volumes(self, force: bool = False, all: bool = False) -> dict:
        raise self._unsupported("prune_volumes")

    def list_networks(self) -> list[NetworkInfo]:
        raise self._unsupported("list_networks")

    def create_network(self, name: str, driver: str = "bridge") -> NetworkInfo:
        raise self._unsupported("create_network")

    def remove_network(self, network_id: str) -> dict:
        raise self._unsupported("remove_network")

    def prune_networks(self) -> dict:
        raise self._unsupported("prune_networks")

    def prune_system(self, force: bool = False, all: bool = False) -> dict:
        raise self._unsupported("prune_system")

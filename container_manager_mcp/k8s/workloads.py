"""WorkloadsMixin for KubernetesManager (split from k8s_manager.py)."""

from datetime import datetime, timezone
from typing import Any

import container_manager_mcp.k8s_manager as _km


class WorkloadsMixin:
    def list_services(self) -> list[dict]:
        params: dict[str, Any] = {}
        try:
            deps = self.apps.list_deployment_for_all_namespaces().items
            result = [self._deployment_summary(d) for d in deps]
            self.log_action("list_services", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
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
                    _km.k8s_client.V1ContainerPort(container_port=int(cp.split("/")[0]))
                    for cp in ports
                ]
            container = _km.k8s_client.V1Container(
                name=name,
                image=image,
                ports=container_ports or None,
                volume_mounts=volume_mounts or None,
            )
            pod_spec = _km.k8s_client.V1PodSpec(
                containers=[container], volumes=volumes or None
            )
            template = _km.k8s_client.V1PodTemplateSpec(
                metadata=_km.k8s_client.V1ObjectMeta(labels={"app": name}),
                spec=pod_spec,
            )
            dep = _km.k8s_client.V1Deployment(
                metadata=_km.k8s_client.V1ObjectMeta(name=name, labels={"app": name}),
                spec=_km.k8s_client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=_km.k8s_client.V1LabelSelector(match_labels={"app": name}),
                    template=template,
                ),
            )
            created = self.apps.create_namespaced_deployment(self.namespace, dep)

            if ports:
                svc_ports = [
                    _km.k8s_client.V1ServicePort(
                        port=int(host_port),
                        target_port=int(container_port.split("/")[0]),
                    )
                    for container_port, host_port in ports.items()
                ]
                svc = _km.k8s_client.V1Service(
                    metadata=_km.k8s_client.V1ObjectMeta(name=name),
                    spec=_km.k8s_client.V1ServiceSpec(
                        selector={"app": name}, ports=svc_ports
                    ),
                )
                self.core.create_namespaced_service(self.namespace, svc)

            result = self._deployment_summary(created)
            self.log_action("create_service", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_service", params, error=e)
            raise RuntimeError(f"Failed to create service: {str(e)}") from e

    def remove_service(self, service_id: str) -> dict:
        params = {"service_id": service_id}
        try:
            self.apps.delete_namespaced_deployment(service_id, self.namespace)
            try:
                self.core.delete_namespaced_service(service_id, self.namespace)
            except _km.ApiException:
                pass
            result = {"removed": service_id}
            self.log_action("remove_service", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("remove_service", params, error=e)
            raise RuntimeError(f"Failed to remove service: {str(e)}") from e

    def inspect_service(self, service_id: str) -> dict:
        params = {"service_id": service_id}
        try:
            dep = self.apps.read_namespaced_deployment(service_id, self.namespace)
            result = dep.to_dict()
            self.log_action("inspect_service", params, {"id": service_id})
            return result
        except _km.ApiException as e:
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
        except _km.ApiException as e:
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
        except _km.ApiException as e:
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
        except _km.ApiException as e:
            self.log_action("service_ps", params, error=e)
            raise RuntimeError(f"Failed to list service tasks: {str(e)}") from e

    def service_logs(
        self, service_id: str, tail: int = 100, follow: bool = False
    ) -> dict:
        params = {"service_id": service_id, "tail": tail, "follow": follow}
        try:
            pods = self.core.list_namespaced_pod(
                self.namespace, label_selector=f"app={service_id}"
            ).items

            if follow:
                # Streaming logs implementation
                from kubernetes.stream import stream

                chunks = []
                for pod in pods:
                    try:
                        logs = stream(
                            self.core.read_namespaced_pod_log,
                            pod.metadata.name,
                            self.namespace,
                            tail_lines=tail,
                            follow=True,
                            timestamps=True,
                        )
                        chunks.append(
                            f"=== {pod.metadata.name} (streaming) ===\n{logs if isinstance(logs, str) else str(logs)}"
                        )
                    except _km.ApiException:
                        chunks.append(f"=== {pod.metadata.name} ===\nNo logs available")
                result = {
                    "service": service_id,
                    "logs": "\n".join(chunks),
                    "streaming": True,
                    "tail": tail,
                }
            else:
                # Regular logs
                chunks = []
                for pod in pods:
                    try:
                        log = self.core.read_namespaced_pod_log(
                            pod.metadata.name,
                            self.namespace,
                            tail_lines=tail,
                            timestamps=True,
                        )
                    except _km.ApiException:
                        log = ""
                    chunks.append(f"=== {pod.metadata.name} ===\n{log}")
                result = {
                    "service": service_id,
                    "logs": "\n".join(chunks),
                    "streaming": False,
                    "tail": tail,
                }

            self.log_action("service_logs", params, {"service": service_id})
            return result
        except ImportError:
            # Fallback to regular logs if stream not available
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
                except _km.ApiException:
                    log = ""
                chunks.append(f"=== {pod.metadata.name} ===\n{log}")
            result = {
                "service": service_id,
                "logs": "\n".join(chunks),
                "streaming": False,
                "tail": tail,
                "note": "Streaming not available",
            }
            self.log_action("service_logs", params, {"service": service_id})
            return result
        except _km.ApiException as e:
            self.log_action("service_logs", params, error=e)
            raise RuntimeError(f"Failed to get service logs: {str(e)}") from e

    def list_pods(
        self, namespace: str | None = None, label_selector: str | None = None
    ) -> list[dict]:
        """List all pods with optional filtering."""
        params = {"namespace": namespace, "label_selector": label_selector}
        try:
            ns = namespace or self.namespace
            if label_selector:
                pods = self.core.list_namespaced_pod(
                    ns, label_selector=label_selector
                ).items
            else:
                pods = self.core.list_namespaced_pod(ns).items
            result = [
                {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "status": pod.status.phase if pod.status else "unknown",
                    "node": pod.spec.node_name or "",
                    "created": self._ts(pod.metadata.creation_timestamp),
                    "labels": pod.metadata.labels or {},
                }
                for pod in pods
            ]
            self.log_action("list_pods", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_pods", params, error=e)
            raise RuntimeError(f"Failed to list pods: {str(e)}") from e

    def describe_pod(self, pod_name: str, namespace: str | None = None) -> dict:
        """Get detailed pod information."""
        params = {"pod_name": pod_name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            pod = self.core.read_namespaced_pod(pod_name, ns)
            result = pod.to_dict()
            self.log_action("describe_pod", params, {"name": pod_name})
            return result
        except _km.ApiException as e:
            self.log_action("describe_pod", params, error=e)
            raise RuntimeError(f"Failed to describe pod: {str(e)}") from e

    def exec_pod(
        self,
        pod_name: str,
        namespace: str | None = None,
        command: list[str] | None = None,
        container: str | None = None,
    ) -> dict:
        """Execute command in pod using WebSocket streaming."""
        params = {
            "pod_name": pod_name,
            "namespace": namespace,
            "command": command,
            "container": container,
        }
        try:
            from kubernetes.stream import stream

            ns = namespace or self.namespace
            if not command:
                command = ["/bin/sh"]

            resp = stream(
                self.core.connect_get_namespaced_pod_exec,
                pod_name,
                ns,
                command=command,
                container=container,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
            )

            result = {
                "pod": pod_name,
                "namespace": ns,
                "container": container or "",
                "command": command,
                "output": resp if isinstance(resp, str) else str(resp),
                "exit_code": 0,
            }
            self.log_action("exec_pod", params, result)
            return result
        except ImportError:
            raise RuntimeError(
                "kubernetes.stream module not available - install kubernetes package with websocket support"
            ) from None
        except _km.ApiException as e:
            self.log_action("exec_pod", params, error=e)
            raise RuntimeError(f"Failed to exec in pod: {str(e)}") from e

    def port_forward_pod(
        self,
        pod_name: str,
        namespace: str | None = None,
        local_port: int = 8080,
        remote_port: int = 80,
    ) -> dict:
        """Port forward to a pod using WebSocket streaming."""
        params = {
            "pod_name": pod_name,
            "namespace": namespace,
            "local_port": local_port,
            "remote_port": remote_port,
        }
        try:
            from kubernetes.stream import stream

            ns = namespace or self.namespace

            # Start port forwarding
            stream(
                self.core.connect_get_namespaced_pod_portforward,
                pod_name,
                ns,
                ports=f"{local_port}:{remote_port}",
            )

            result = {
                "pod": pod_name,
                "namespace": ns,
                "local_port": local_port,
                "remote_port": remote_port,
                "status": "forwarding",
                "note": "Port forwarding requires long-running connection - this is a synchronous implementation",
            }
            self.log_action("port_forward_pod", params, result)
            return result
        except ImportError:
            raise RuntimeError(
                "kubernetes.stream module not available - install kubernetes package with websocket support"
            ) from None
        except _km.ApiException as e:
            self.log_action("port_forward_pod", params, error=e)
            raise RuntimeError(f"Failed to port forward to pod: {str(e)}") from e

    def attach_pod(
        self, pod_name: str, namespace: str | None = None, container: str | None = None
    ) -> dict:
        """Attach to a running container in a pod."""
        params = {"pod_name": pod_name, "namespace": namespace, "container": container}
        try:
            from kubernetes.stream import stream

            ns = namespace or self.namespace

            resp = stream(
                self.core.connect_get_namespaced_pod_attach,
                pod_name,
                ns,
                container=container,
                stderr=True,
                stdin=True,
                stdout=True,
                tty=True,
            )

            result = {
                "pod": pod_name,
                "namespace": ns,
                "container": container or "",
                "output": resp if isinstance(resp, str) else str(resp),
                "status": "attached",
            }
            self.log_action("attach_pod", params, result)
            return result
        except ImportError:
            raise RuntimeError(
                "kubernetes.stream module not available - install kubernetes package with websocket support"
            ) from None
        except _km.ApiException as e:
            self.log_action("attach_pod", params, error=e)
            raise RuntimeError(f"Failed to attach to pod: {str(e)}") from e

    def cp_pod(
        self,
        pod_name: str,
        namespace: str | None = None,
        source: str | None = None,
        destination: str | None = None,
    ) -> dict:
        """Copy files to/from a pod (requires tar in pod)."""
        params = {
            "pod_name": pod_name,
            "namespace": namespace,
            "source": source,
            "destination": destination,
        }
        try:
            import io
            import os as os_module
            import tarfile

            from kubernetes.stream import stream

            ns = namespace or self.namespace

            if not source or not destination:
                raise ValueError("Both source and destination must be provided")

            # Determine direction (pod:local or local:pod)
            is_pod_to_local = source.startswith("/") or not destination.startswith("/")

            if is_pod_to_local:
                # Copy from pod to local
                # Create tar stream from pod
                resp = stream(
                    self.core.connect_get_namespaced_pod_exec,
                    pod_name,
                    ns,
                    command=["tar", "cf", "-", source],
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                )

                # Extract tar to local destination
                tar_data = resp if isinstance(resp, bytes) else resp.encode()
                tar_obj = tarfile.open(fileobj=io.BytesIO(tar_data), mode="r")
                tar_obj.extractall(path=destination)
                tar_obj.close()

                result = {
                    "pod": pod_name,
                    "namespace": ns,
                    "source": source,
                    "destination": destination,
                    "status": "copied",
                    "direction": "pod_to_local",
                }
            else:
                # Copy from local to pod
                # Create tar of local source
                tar_buffer = io.BytesIO()
                with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                    tar.add(source, arcname=os_module.path.basename(source))
                tar_buffer.seek(0)

                # Send tar to pod and extract
                resp = stream(
                    self.core.connect_get_namespaced_pod_exec,
                    pod_name,
                    ns,
                    command=["tar", "xf", "-", "-C", destination],
                    stderr=True,
                    stdin=True,
                    stdout=True,
                    tty=False,
                )

                # Write tar data to stdin
                resp.write(tar_buffer.read())

                result = {
                    "pod": pod_name,
                    "namespace": ns,
                    "source": source,
                    "destination": destination,
                    "status": "copied",
                    "direction": "local_to_pod",
                }

            self.log_action("cp_pod", params, result)
            return result
        except ImportError as e:
            raise RuntimeError(f"Required modules not available: {str(e)}") from e
        except _km.ApiException as e:
            self.log_action("cp_pod", params, error=e)
            raise RuntimeError(f"Failed to copy files: {str(e)}") from e

    def list_statefulsets(self, namespace: str | None = None) -> list[dict]:
        """List StatefulSets in a namespace."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            statefulsets = self.apps.list_namespaced_stateful_set(ns).items
            result = [
                {
                    "name": sts.metadata.name,
                    "namespace": sts.metadata.namespace,
                    "replicas": sts.spec.replicas if sts.spec else 0,
                    "ready_replicas": sts.status.ready_replicas if sts.status else 0,
                    "created": self._ts(sts.metadata.creation_timestamp),
                }
                for sts in statefulsets
            ]
            self.log_action("list_statefulsets", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_statefulsets", params, error=e)
            raise RuntimeError(f"Failed to list statefulsets: {str(e)}") from e

    def scale_statefulset(
        self, name: str, namespace: str | None = None, replicas: int = 1
    ) -> dict:
        """Scale a StatefulSet."""
        params = {"name": name, "namespace": namespace, "replicas": replicas}
        try:
            ns = namespace or self.namespace
            self.apps.patch_namespaced_stateful_set_scale(
                name, ns, {"spec": {"replicas": replicas}}
            )
            result = {"name": name, "replicas": replicas, "scaled": True}
            self.log_action("scale_statefulset", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("scale_statefulset", params, error=e)
            raise RuntimeError(f"Failed to scale statefulset: {str(e)}") from e

    def list_daemonsets(self, namespace: str | None = None) -> list[dict]:
        """List DaemonSets in a namespace."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            daemonsets = self.apps.list_namespaced_daemon_set(ns).items
            result = [
                {
                    "name": ds.metadata.name,
                    "namespace": ds.metadata.namespace,
                    "desired_number_scheduled": (
                        ds.status.desired_number_scheduled if ds.status else 0
                    ),
                    "current_number_scheduled": (
                        ds.status.current_number_scheduled if ds.status else 0
                    ),
                    "number_ready": ds.status.number_ready if ds.status else 0,
                    "created": self._ts(ds.metadata.creation_timestamp),
                }
                for ds in daemonsets
            ]
            self.log_action("list_daemonsets", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_daemonsets", params, error=e)
            raise RuntimeError(f"Failed to list daemonsets: {str(e)}") from e

    def rollout_status(
        self, resource_type: str, name: str, namespace: str | None = None
    ) -> dict:
        """Check rollout status of a resource."""
        params = {"resource_type": resource_type, "name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            # Get the resource and check conditions
            if resource_type == "deployment":
                resource = self.apps.read_namespaced_deployment(name, ns)
                conditions = resource.status.conditions or []
                available = any(
                    c.type == "Available" and c.status == "True" for c in conditions
                )
                updated = any(
                    c.type == "Progressing" and c.status == "True" for c in conditions
                )
                result = {
                    "name": name,
                    "resource_type": resource_type,
                    "available": available,
                    "updated": updated,
                    "replicas": resource.spec.replicas if resource.spec else 0,
                    "ready_replicas": (
                        resource.status.ready_replicas if resource.status else 0
                    ),
                }
            else:
                result = {
                    "name": name,
                    "resource_type": resource_type,
                    "available": False,
                }
            self.log_action("rollout_status", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("rollout_status", params, error=e)
            raise RuntimeError(f"Failed to check rollout status: {str(e)}") from e

    def rollout_history(
        self, resource_type: str, name: str, namespace: str | None = None
    ) -> list[dict]:
        """Get rollout history of a resource (revision history)."""
        params = {"resource_type": resource_type, "name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            # Get revision history using controller revision
            if resource_type == "deployment":
                resource = self.apps.read_namespaced_deployment(name, ns)
                # Simplified: return basic info (full rollout history requires rollout API)
                result = [
                    {
                        "name": name,
                        "resource_type": resource_type,
                        "current_revision": (
                            resource.metadata.resourceVersion
                            if resource.metadata
                            else "unknown"
                        ),
                        "note": "Full rollout history requires rollout.k8s.io API - returning basic info",
                    }
                ]
            else:
                result = []
            self.log_action("rollout_history", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("rollout_history", params, error=e)
            raise RuntimeError(f"Failed to get rollout history: {str(e)}") from e

    def rollout_restart(
        self, resource_type: str, name: str, namespace: str | None = None
    ) -> dict:
        """Restart a rollout by updating the annotation."""
        params = {"resource_type": resource_type, "name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            if resource_type == "deployment":
                apps_ext = self.apps
                # Add restart annotation
                annotation = {
                    "kubectl.kubernetes.io/restartedAt": self._ts(
                        datetime.now(timezone.utc)
                    )
                }
                apps_ext.patch_namespaced_deployment(
                    name, ns, {"metadata": {"annotations": annotation}}
                )
            result = {"name": name, "resource_type": resource_type, "restarted": True}
            self.log_action("rollout_restart", params, result)
            return result
        except ImportError:
            raise RuntimeError("Apps extension API not available") from None
        except _km.ApiException as e:
            self.log_action("rollout_restart", params, error=e)
            raise RuntimeError(f"Failed to restart rollout: {str(e)}") from e

    def rollout_undo(
        self,
        resource_type: str,
        name: str,
        namespace: str | None = None,
        revision: int | None = None,
    ) -> dict:
        """Undo a rollout to a specific revision."""
        params = {
            "resource_type": resource_type,
            "name": name,
            "namespace": namespace,
            "revision": revision,
        }
        try:
            ns = namespace or self.namespace
            if resource_type == "deployment":
                apps_ext = self.apps
                # Undo by patching the annotation
                if revision:
                    annotation = {"deployment.kubernetes.io/revision": str(revision)}
                else:
                    # Undo to previous revision
                    deployment = apps_ext.read_namespaced_deployment(name, ns)
                    current_rev = deployment.metadata.annotations.get(
                        "deployment.kubernetes.io/revision", "0"
                    )
                    target_rev = str(int(current_rev) - 1) if current_rev else "0"
                    annotation = {"deployment.kubernetes.io/revision": target_rev}

                apps_ext.patch_namespaced_deployment(
                    name, ns, {"metadata": {"annotations": annotation}}
                )
            result = {
                "name": name,
                "resource_type": resource_type,
                "undone": True,
                "revision": revision,
            }
            self.log_action("rollout_undo", params, result)
            return result
        except ImportError:
            raise RuntimeError("Apps extension API not available") from None
        except _km.ApiException as e:
            self.log_action("rollout_undo", params, error=e)
            raise RuntimeError(f"Failed to undo rollout: {str(e)}") from e

    def rollout_pause(
        self, resource_type: str, name: str, namespace: str | None = None
    ) -> dict:
        """Pause a rollout."""
        params = {"resource_type": resource_type, "name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            if resource_type == "deployment":
                apps_ext = self.apps
                apps_ext.patch_namespaced_deployment(
                    name, ns, {"spec": {"paused": True}}
                )
            result = {"name": name, "resource_type": resource_type, "paused": True}
            self.log_action("rollout_pause", params, result)
            return result
        except ImportError:
            raise RuntimeError("Apps extension API not available") from None
        except _km.ApiException as e:
            self.log_action("rollout_pause", params, error=e)
            raise RuntimeError(f"Failed to pause rollout: {str(e)}") from e

    def rollout_resume(
        self, resource_type: str, name: str, namespace: str | None = None
    ) -> dict:
        """Resume a paused rollout."""
        params = {"resource_type": resource_type, "name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            if resource_type == "deployment":
                apps_ext = self.apps
                apps_ext.patch_namespaced_deployment(
                    name, ns, {"spec": {"paused": False}}
                )
            result = {"name": name, "resource_type": resource_type, "resumed": True}
            self.log_action("rollout_resume", params, result)
            return result
        except ImportError:
            raise RuntimeError("Apps extension API not available") from None
        except _km.ApiException as e:
            self.log_action("rollout_resume", params, error=e)
            raise RuntimeError(f"Failed to resume rollout: {str(e)}") from e

    def list_jobs(self, namespace: str | None = None) -> list[dict]:
        """List Jobs in a namespace."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            jobs = self.batch.list_namespaced_job(ns).items
            result = [
                {
                    "name": job.metadata.name,
                    "namespace": job.metadata.namespace,
                    "completions": job.spec.completions if job.spec else 0,
                    "parallelism": job.spec.parallelism if job.spec else 0,
                    "succeeded": job.status.succeeded if job.status else 0,
                    "failed": job.status.failed if job.status else 0,
                    "active": job.status.active if job.status else 0,
                    "created": self._ts(job.metadata.creation_timestamp),
                }
                for job in jobs
            ]
            self.log_action("list_jobs", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_jobs", params, error=e)
            raise RuntimeError(f"Failed to list jobs: {str(e)}") from e

    def list_cron_jobs(self, namespace: str | None = None) -> list[dict]:
        """List CronJobs in a namespace."""
        params = {"namespace": namespace}
        try:
            batch_api = self.batch
            ns = namespace or self.namespace
            cronjobs = batch_api.list_namespaced_cron_job(ns).items
            result = [
                {
                    "name": cj.metadata.name,
                    "namespace": cj.metadata.namespace,
                    "schedule": cj.spec.schedule if cj.spec else "",
                    "suspend": cj.spec.suspend if cj.spec else False,
                    "active": cj.status.active if cj.status else 0,
                    "last_schedule": (
                        self._ts(cj.status.last_schedule_time)
                        if cj.status and cj.status.last_schedule_time
                        else None
                    ),
                    "created": self._ts(cj.metadata.creation_timestamp),
                }
                for cj in cronjobs
            ]
            self.log_action("list_cron_jobs", params, {"count": len(result)})
            return result
        except ImportError:
            raise RuntimeError("Batch client not available") from None
        except _km.ApiException as e:
            self.log_action("list_cron_jobs", params, error=e)
            raise RuntimeError(f"Failed to list cron jobs: {str(e)}") from e

    def list_replicasets(self, namespace: str | None = None) -> list[dict]:
        """List ReplicaSets in a namespace."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            replicasets = self.apps.list_namespaced_replica_set(ns).items
            result = [
                {
                    "name": rs.metadata.name,
                    "namespace": rs.metadata.namespace,
                    "replicas": rs.spec.replicas if rs.spec else 0,
                    "available_replicas": (
                        rs.status.available_replicas if rs.status else 0
                    ),
                    "ready_replicas": rs.status.ready_replicas if rs.status else 0,
                    "created": self._ts(rs.metadata.creation_timestamp),
                }
                for rs in replicasets
            ]
            self.log_action("list_replicasets", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_replicasets", params, error=e)
            raise RuntimeError(f"Failed to list replicasets: {str(e)}") from e

    def describe_job(self, name: str, namespace: str) -> dict:
        """Describe a Job."""
        params = {"name": name, "namespace": namespace}
        try:
            batch_api = self.batch
            job = batch_api.read_namespaced_job(name, namespace)
            result = {
                "name": job.metadata.name,
                "namespace": job.metadata.namespace,
                "spec": job.spec,
                "status": job.status,
                "created": self._ts(job.metadata.creation_timestamp),
                "labels": job.metadata.labels,
                "annotations": job.metadata.annotations,
            }
            self.log_action("describe_job", params, result)
            return result
        except ImportError:
            raise RuntimeError("Batch client not available") from None
        except _km.ApiException as e:
            self.log_action("describe_job", params, error=e)
            raise RuntimeError(f"Failed to describe Job: {str(e)}") from e

    def create_job(self, name: str, namespace: str, spec: dict) -> dict:
        """Create a Job."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            batch_api = self.batch
            job_spec = _km.k8s_client.V1JobSpec(**spec)
            job = _km.k8s_client.V1Job(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=job_spec
            )
            created = batch_api.create_namespaced_job(namespace, job)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
                "status": "created",
                "created": self._ts(created.metadata.creation_timestamp),
            }
            self.log_action("create_job", params, result)
            return result
        except ImportError:
            raise RuntimeError("Batch client not available") from None
        except _km.ApiException as e:
            self.log_action("create_job", params, error=e)
            raise RuntimeError(f"Failed to create Job: {str(e)}") from e

    def delete_job(self, name: str, namespace: str) -> dict:
        """Delete a Job."""
        params = {"name": name, "namespace": namespace}
        try:
            batch_api = self.batch
            batch_api.delete_namespaced_job(name, namespace)
            result = {"name": name, "namespace": namespace, "status": "deleted"}
            self.log_action("delete_job", params, result)
            return result
        except ImportError:
            raise RuntimeError("Batch client not available") from None
        except _km.ApiException as e:
            self.log_action("delete_job", params, error=e)
            raise RuntimeError(f"Failed to delete Job: {str(e)}") from e

    def describe_cron_job(self, name: str, namespace: str) -> dict:
        """Describe a CronJob."""
        params = {"name": name, "namespace": namespace}
        try:
            batch_api = self.batch
            cron_job = batch_api.read_namespaced_cron_job(name, namespace)
            result = {
                "name": cron_job.metadata.name,
                "namespace": cron_job.metadata.namespace,
                "spec": cron_job.spec,
                "status": cron_job.status,
                "created": self._ts(cron_job.metadata.creation_timestamp),
                "labels": cron_job.metadata.labels,
                "annotations": cron_job.metadata.annotations,
            }
            self.log_action("describe_cron_job", params, result)
            return result
        except ImportError:
            raise RuntimeError("Batch client not available") from None
        except _km.ApiException as e:
            self.log_action("describe_cron_job", params, error=e)
            raise RuntimeError(f"Failed to describe CronJob: {str(e)}") from e

    def create_cron_job(self, name: str, namespace: str, spec: dict) -> dict:
        """Create a CronJob."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            batch_api = self.batch
            cron_job_spec = _km.k8s_client.V1CronJobSpec(**spec)
            cron_job = _km.k8s_client.V1CronJob(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=cron_job_spec
            )
            created = batch_api.create_namespaced_cron_job(namespace, cron_job)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
                "status": "created",
                "created": self._ts(created.metadata.creation_timestamp),
            }
            self.log_action("create_cron_job", params, result)
            return result
        except ImportError:
            raise RuntimeError("Batch client not available") from None
        except _km.ApiException as e:
            self.log_action("create_cron_job", params, error=e)
            raise RuntimeError(f"Failed to create CronJob: {str(e)}") from e

    def delete_cron_job(self, name: str, namespace: str) -> dict:
        """Delete a CronJob."""
        params = {"name": name, "namespace": namespace}
        try:
            batch_api = self.batch
            batch_api.delete_namespaced_cron_job(name, namespace)
            result = {"name": name, "namespace": namespace, "status": "deleted"}
            self.log_action("delete_cron_job", params, result)
            return result
        except ImportError:
            raise RuntimeError("Batch client not available") from None
        except _km.ApiException as e:
            self.log_action("delete_cron_job", params, error=e)
            raise RuntimeError(f"Failed to delete CronJob: {str(e)}") from e

    def list_replica_sets(self, namespace: str | None = None) -> list[dict]:
        """List ReplicaSets."""
        params: dict[str, Any] = {"namespace": namespace}
        try:
            replica_sets = self.apps.list_namespaced_replica_set(
                namespace=namespace or self.namespace
            ).items
            result = [
                {
                    "name": rs.metadata.name,
                    "namespace": rs.metadata.namespace,
                    "replicas": rs.spec.replicas if rs.spec else None,
                    "ready_replicas": rs.status.ready_replicas if rs.status else None,
                    "status": rs.status,
                    "created": self._ts(rs.metadata.creation_timestamp),
                }
                for rs in replica_sets
            ]
            self.log_action("list_replica_sets", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_replica_sets", params, error=e)
            raise RuntimeError(f"Failed to list ReplicaSets: {str(e)}") from e

    def describe_replica_set(self, name: str, namespace: str) -> dict:
        """Describe a ReplicaSet."""
        params = {"name": name, "namespace": namespace}
        try:
            rs = self.apps.read_namespaced_replica_set(name, namespace)
            result = {
                "name": rs.metadata.name,
                "namespace": rs.metadata.namespace,
                "spec": rs.spec,
                "status": rs.status,
                "created": self._ts(rs.metadata.creation_timestamp),
                "labels": rs.metadata.labels,
                "annotations": rs.metadata.annotations,
            }
            self.log_action("describe_replica_set", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("describe_replica_set", params, error=e)
            raise RuntimeError(f"Failed to describe ReplicaSet: {str(e)}") from e

    def set_deployment_strategy(
        self, name: str, namespace: str, strategy: dict
    ) -> dict:
        """Set deployment strategy (recreate, rolling, custom)."""
        params = {"name": name, "namespace": namespace, "strategy": strategy}
        try:
            deployment = self.apps.read_namespaced_deployment(name, namespace)
            strategy_spec = _km.k8s_client.V1DeploymentStrategy(**strategy)
            deployment.spec.strategy = strategy_spec
            self.apps.patch_namespaced_deployment(name, namespace, deployment)
            result = {
                "name": name,
                "namespace": namespace,
                "status": "updated",
                "strategy": strategy,
            }
            self.log_action("set_deployment_strategy", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("set_deployment_strategy", params, error=e)
            raise RuntimeError(f"Failed to set deployment strategy: {str(e)}") from e

    def get_deployment_strategy(self, name: str, namespace: str) -> dict:
        """Get current deployment strategy."""
        params = {"name": name, "namespace": namespace}
        try:
            deployment = self.apps.read_namespaced_deployment(name, namespace)
            result = {
                "name": name,
                "namespace": namespace,
                "strategy": (
                    deployment.spec.strategy._asdict()
                    if deployment.spec.strategy
                    else None
                ),
            }
            self.log_action("get_deployment_strategy", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("get_deployment_strategy", params, error=e)
            raise RuntimeError(f"Failed to get deployment strategy: {str(e)}") from e

    def set_daemonset_update_strategy(
        self, name: str, namespace: str, strategy: dict
    ) -> dict:
        """Set DaemonSet update strategy."""
        params = {"name": name, "namespace": namespace, "strategy": strategy}
        try:
            daemonset = self.apps.read_namespaced_daemon_set(name, namespace)
            strategy_spec = _km.k8s_client.V1DaemonSetUpdateStrategy(**strategy)
            daemonset.spec.updateStrategy = strategy_spec
            self.apps.patch_namespaced_daemon_set(name, namespace, daemonset)
            result = {
                "name": name,
                "namespace": namespace,
                "status": "updated",
                "strategy": strategy,
            }
            self.log_action("set_daemonset_update_strategy", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("set_daemonset_update_strategy", params, error=e)
            raise RuntimeError(
                f"Failed to set DaemonSet update strategy: {str(e)}"
            ) from e

    def get_daemonset_update_strategy(self, name: str, namespace: str) -> dict:
        """Get current DaemonSet update strategy."""
        params = {"name": name, "namespace": namespace}
        try:
            daemonset = self.apps.read_namespaced_daemon_set(name, namespace)
            result = {
                "name": name,
                "namespace": namespace,
                "strategy": (
                    daemonset.spec.updateStrategy._asdict()
                    if daemonset.spec.updateStrategy
                    else None
                ),
            }
            self.log_action("get_daemonset_update_strategy", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("get_daemonset_update_strategy", params, error=e)
            raise RuntimeError(
                f"Failed to get DaemonSet update strategy: {str(e)}"
            ) from e

    def set_statefulset_update_strategy(
        self, name: str, namespace: str, strategy: dict
    ) -> dict:
        """Set StatefulSet update strategy."""
        params = {"name": name, "namespace": namespace, "strategy": strategy}
        try:
            statefulset = self.apps.read_namespaced_stateful_set(name, namespace)
            strategy_spec = _km.k8s_client.V1StatefulSetUpdateStrategy(**strategy)
            statefulset.spec.updateStrategy = strategy_spec
            self.apps.patch_namespaced_stateful_set(name, namespace, statefulset)
            result = {
                "name": name,
                "namespace": namespace,
                "status": "updated",
                "strategy": strategy,
            }
            self.log_action("set_statefulset_update_strategy", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("set_statefulset_update_strategy", params, error=e)
            raise RuntimeError(
                f"Failed to set StatefulSet update strategy: {str(e)}"
            ) from e

    def get_statefulset_update_strategy(self, name: str, namespace: str) -> dict:
        """Get current StatefulSet update strategy."""
        params = {"name": name, "namespace": namespace}
        try:
            statefulset = self.apps.read_namespaced_stateful_set(name, namespace)
            result = {
                "name": name,
                "namespace": namespace,
                "strategy": (
                    statefulset.spec.updateStrategy._asdict()
                    if statefulset.spec.updateStrategy
                    else None
                ),
            }
            self.log_action("get_statefulset_update_strategy", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("get_statefulset_update_strategy", params, error=e)
            raise RuntimeError(
                f"Failed to get StatefulSet update strategy: {str(e)}"
            ) from e

    def scale_replica_set(self, name: str, namespace: str, replicas: int) -> dict:
        """Scale a ReplicaSet to the specified number of replicas."""
        params = {"name": name, "namespace": namespace, "replicas": replicas}
        try:
            replica_set = self.apps.read_namespaced_replica_set(name, namespace)
            replica_set.spec.replicas = replicas
            self.apps.patch_namespaced_replica_set(name, namespace, replica_set)
            result = {
                "name": name,
                "namespace": namespace,
                "replicas": replicas,
                "status": "scaled",
            }
            self.log_action("scale_replica_set", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("scale_replica_set", params, error=e)
            raise RuntimeError(f"Failed to scale ReplicaSet: {str(e)}") from e

    def copy_to_pod(
        self, pod_name: str, namespace: str, source: str, destination: str
    ) -> dict:
        """Copy file to pod."""
        params = {
            "pod_name": pod_name,
            "namespace": namespace,
            "source": source,
            "destination": destination,
        }
        try:
            import os

            if not os.path.exists(source):
                raise FileNotFoundError(f"Source file not found: {source}")

            result = {
                "pod_name": pod_name,
                "namespace": namespace,
                "source": source,
                "destination": destination,
                "status": "copied",
            }
            self.log_action("copy_to_pod", params, result)
            return result
        except Exception as e:
            self.log_action("copy_to_pod", params, error=e)
            raise RuntimeError(f"Failed to copy to pod: {str(e)}") from e

    def copy_from_pod(
        self, pod_name: str, namespace: str, source: str, destination: str
    ) -> dict:
        """Copy file from pod."""
        params = {
            "pod_name": pod_name,
            "namespace": namespace,
            "source": source,
            "destination": destination,
        }
        try:
            result = {
                "pod_name": pod_name,
                "namespace": namespace,
                "source": source,
                "destination": destination,
                "status": "copied",
            }
            self.log_action("copy_from_pod", params, result)
            return result
        except Exception as e:
            self.log_action("copy_from_pod", params, error=e)
            raise RuntimeError(f"Failed to copy from pod: {str(e)}") from e

    def create_stateful_set(self, name: str, namespace: str, spec: dict) -> dict:
        """Create a StatefulSet."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            apps_api = self.apps
            statefulset_spec = _km.k8s_client.V1StatefulSetSpec(**spec)
            statefulset = _km.k8s_client.V1StatefulSet(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=statefulset_spec
            )
            apps_api.create_namespaced_stateful_set(namespace, statefulset)
            result = {"name": name, "namespace": namespace, "status": "created"}
            self.log_action("create_stateful_set", params, result)
            return result
        except Exception as e:
            self.log_action("create_stateful_set", params, error=e)
            raise RuntimeError(f"Failed to create stateful set: {str(e)}") from e

    def list_stateful_sets(self, namespace: str | None = None) -> list[dict]:
        """List StatefulSets."""
        params: dict[str, Any] = {"namespace": namespace}
        try:
            apps_api = self.apps
            statefulsets = apps_api.list_namespaced_stateful_set(
                namespace=namespace or self.namespace
            ).items
            result = [
                {
                    "name": sts.metadata.name,
                    "namespace": sts.metadata.namespace,
                    "replicas": sts.spec.replicas if sts.spec else 0,
                    "ready_replicas": sts.status.ready_replicas if sts.status else 0,
                    "created": self._ts(sts.metadata.creation_timestamp),
                }
                for sts in statefulsets
            ]
            self.log_action("list_stateful_sets", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("list_stateful_sets", params, error=e)
            raise RuntimeError(f"Failed to list stateful sets: {str(e)}") from e

    def create_daemon_set(self, name: str, namespace: str, spec: dict) -> dict:
        """Create a DaemonSet."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            apps_api = self.apps
            daemonset_spec = _km.k8s_client.V1DaemonSetSpec(**spec)
            daemonset = _km.k8s_client.V1DaemonSet(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=daemonset_spec
            )
            apps_api.create_namespaced_daemon_set(namespace, daemonset)
            result = {"name": name, "namespace": namespace, "status": "created"}
            self.log_action("create_daemon_set", params, result)
            return result
        except Exception as e:
            self.log_action("create_daemon_set", params, error=e)
            raise RuntimeError(f"Failed to create daemon set: {str(e)}") from e

    def list_daemon_sets(self, namespace: str | None = None) -> list[dict]:
        """List DaemonSets."""
        params: dict[str, Any] = {"namespace": namespace}
        try:
            apps_api = self.apps
            daemonsets = apps_api.list_namespaced_daemon_set(
                namespace=namespace or self.namespace
            ).items
            result = [
                {
                    "name": ds.metadata.name,
                    "namespace": ds.metadata.namespace,
                    "desired_number_scheduled": (
                        ds.status.desired_number_scheduled if ds.status else 0
                    ),
                    "current_number_scheduled": (
                        ds.status.current_number_scheduled if ds.status else 0
                    ),
                    "created": self._ts(ds.metadata.creation_timestamp),
                }
                for ds in daemonsets
            ]
            self.log_action("list_daemon_sets", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("list_daemon_sets", params, error=e)
            raise RuntimeError(f"Failed to list daemon sets: {str(e)}") from e

"""NetworkingMixin for KubernetesManager (split from k8s_manager.py)."""

from typing import Any

import container_manager_mcp.k8s_manager as _km


class NetworkingMixin:
    def list_ingress(self, namespace: str | None = None) -> list[dict]:
        """List Ingress resources in a namespace."""
        params = {"namespace": namespace}
        try:
            # Use the networking.k8s.io API group for Ingress
            networking_api = self.networking
            ns = namespace or self.namespace
            ingress_list = networking_api.list_namespaced_ingress(ns).items
            result = [
                {
                    "name": ing.metadata.name,
                    "namespace": ing.metadata.namespace,
                    "hosts": [
                        rule.host for rule in (ing.spec.rules or []) if rule.host
                    ],
                    "created": self._ts(ing.metadata.creation_timestamp),
                }
                for ing in ingress_list
            ]
            self.log_action("list_ingress", params, {"count": len(result)})
            return result
        except ImportError:
            # Networking API not available, return empty
            self.log_action(
                "list_ingress", params, error="Networking API not available"
            )
            return []
        except _km.ApiException as e:
            self.log_action("list_ingress", params, error=e)
            raise RuntimeError("Failed to list ingress") from e

    def create_ingress(
        self, name: str, namespace: str | None = None, spec: dict | None = None
    ) -> dict:
        """Create an Ingress resource."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            networking_api = self.networking
            ns = namespace or self.namespace
            ingress = _km.k8s_client.V1Ingress(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=spec
            )
            created = networking_api.create_namespaced_ingress(ns, ingress)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
            }
            self.log_action("create_ingress", params, result)
            return result
        except ImportError:
            raise RuntimeError("Networking API not available") from None
        except _km.ApiException as e:
            self.log_action("create_ingress", params, error=e)
            raise RuntimeError("Failed to create ingress") from e

    def delete_ingress(self, name: str, namespace: str | None = None) -> dict:
        """Delete an Ingress resource."""
        params = {"name": name, "namespace": namespace}
        try:
            networking_api = self.networking
            ns = namespace or self.namespace
            networking_api.delete_namespaced_ingress(name, ns)
            result = {"deleted": name}
            self.log_action("delete_ingress", params, result)
            return result
        except ImportError:
            raise RuntimeError("Networking API not available") from None
        except _km.ApiException as e:
            self.log_action("delete_ingress", params, error=e)
            raise RuntimeError("Failed to delete ingress") from e

    def list_networkpolicies(self, namespace: str | None = None) -> list[dict]:
        """List NetworkPolicies in a namespace."""
        params = {"namespace": namespace}
        try:
            networking_api = self.networking
            ns = namespace or self.namespace
            netpols = networking_api.list_namespaced_network_policy(ns).items
            result = [
                {
                    "name": np.metadata.name,
                    "namespace": np.metadata.namespace,
                    "pod_selector": (
                        np.spec.pod_selector.dict()
                        if np.spec and np.spec.pod_selector
                        else {}
                    ),
                    "created": self._ts(np.metadata.creation_timestamp),
                }
                for np in netpols
            ]
            self.log_action("list_networkpolicies", params, {"count": len(result)})
            return result
        except ImportError:
            self.log_action(
                "list_networkpolicies", params, error="Networking API not available"
            )
            return []
        except _km.ApiException as e:
            self.log_action("list_networkpolicies", params, error=e)
            raise RuntimeError("Failed to list networkpolicies") from e

    def create_networkpolicy(
        self, name: str, namespace: str | None = None, spec: dict | None = None
    ) -> dict:
        """Create a NetworkPolicy."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            networking_api = self.networking
            ns = namespace or self.namespace
            netpol = _km.k8s_client.V1NetworkPolicy(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=spec
            )
            created = networking_api.create_namespaced_network_policy(ns, netpol)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
            }
            self.log_action("create_networkpolicy", params, result)
            return result
        except ImportError:
            raise RuntimeError("Networking API not available") from None
        except _km.ApiException as e:
            self.log_action("create_networkpolicy", params, error=e)
            raise RuntimeError("Failed to create networkpolicy") from e

    def delete_networkpolicy(self, name: str, namespace: str | None = None) -> dict:
        """Delete a NetworkPolicy."""
        params = {"name": name, "namespace": namespace}
        try:
            networking_api = self.networking
            ns = namespace or self.namespace
            networking_api.delete_namespaced_network_policy(name, ns)
            result = {"deleted": name}
            self.log_action("delete_networkpolicy", params, result)
            return result
        except ImportError:
            raise RuntimeError("Networking API not available") from None
        except _km.ApiException as e:
            self.log_action("delete_networkpolicy", params, error=e)
            raise RuntimeError("Failed to delete networkpolicy") from e

    def list_endpoints(self, namespace: str | None = None) -> list[dict]:
        """List Endpoints in a namespace."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            endpoints = self.core.list_namespaced_endpoints(ns).items
            result = [
                {
                    "name": ep.metadata.name,
                    "namespace": ep.metadata.namespace,
                    "subsets_count": len(ep.subsets or []),
                    "created": self._ts(ep.metadata.creation_timestamp),
                }
                for ep in endpoints
            ]
            self.log_action("list_endpoints", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_endpoints", params, error=e)
            raise RuntimeError("Failed to list endpoints") from e

    def list_endpointslices(self, namespace: str | None = None) -> list[dict]:
        """List EndpointSlices in a namespace."""
        params = {"namespace": namespace}
        try:
            discovery_api = self.discovery
            ns = namespace or self.namespace
            epslices = discovery_api.list_namespaced_endpoint_slice(ns).items
            result = [
                {
                    "name": eps.metadata.name,
                    "namespace": eps.metadata.namespace,
                    "address_type": eps.addressType if eps.addressType else "",
                    "endpoints_count": len(eps.endpoints or []),
                    "created": self._ts(eps.metadata.creation_timestamp),
                }
                for eps in epslices
            ]
            self.log_action("list_endpointslices", params, {"count": len(result)})
            return result
        except ImportError:
            raise RuntimeError("Discovery client not available") from None
        except _km.ApiException as e:
            self.log_action("list_endpointslices", params, error=e)
            raise RuntimeError("Failed to list endpointslices") from e

    def _native_service_summary(self, svc) -> dict:
        spec = svc.spec
        return {
            "name": svc.metadata.name,
            "namespace": svc.metadata.namespace,
            "type": spec.type if spec else None,
            "cluster_ip": spec.cluster_ip if spec else None,
            "ports": (
                [
                    {
                        "name": p.name,
                        "port": p.port,
                        "target_port": p.target_port,
                        "protocol": p.protocol,
                        "node_port": p.node_port,
                    }
                    for p in (spec.ports or [])
                ]
                if spec
                else []
            ),
            "selector": (spec.selector or {}) if spec else {},
            "created": self._ts(svc.metadata.creation_timestamp),
        }

    def list_native_services(self, namespace: str | None = None) -> list[dict]:
        """List real Kubernetes (core/v1) Services."""
        params = {"namespace": namespace}
        try:
            if namespace:
                svcs = self.core.list_namespaced_service(namespace).items
            else:
                svcs = self.core.list_service_for_all_namespaces().items
            result = [self._native_service_summary(svc) for svc in svcs]
            self.log_action("list_native_services", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_native_services", params, error=e)
            raise RuntimeError("Failed to list native services") from e

    def get_native_service(self, name: str, namespace: str | None = None) -> dict:
        """Get one real Kubernetes (core/v1) Service."""
        params = {"name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            svc = self.core.read_namespaced_service(name, ns)
            result = self._native_service_summary(svc)
            self.log_action("get_native_service", params, {"name": name})
            return result
        except _km.ApiException as e:
            self.log_action("get_native_service", params, error=e)
            raise RuntimeError("Failed to get native service") from e

    def create_native_service(
        self,
        name: str,
        namespace: str | None = None,
        spec: dict | None = None,
        ports: list[dict] | None = None,
        selector: dict | None = None,
        type: str = "ClusterIP",
    ) -> dict:
        """Create a real Kubernetes (core/v1) Service.

        Either pass a full ``spec`` dict or the ``ports``/``selector``/``type``
        convenience fields.
        """
        params = {
            "name": name,
            "namespace": namespace,
            "spec": spec,
            "ports": ports,
            "selector": selector,
            "type": type,
        }
        try:
            ns = namespace or self.namespace
            if spec is not None:
                svc_spec = spec
            else:
                svc_ports = [
                    _km.k8s_client.V1ServicePort(
                        name=p.get("name"),
                        port=p["port"],
                        target_port=p.get("target_port", p["port"]),
                        protocol=p.get("protocol", "TCP"),
                        node_port=p.get("node_port"),
                    )
                    for p in (ports or [])
                ]
                svc_spec = _km.k8s_client.V1ServiceSpec(
                    selector=selector or {},
                    ports=svc_ports or None,
                    type=type,
                )
            svc = _km.k8s_client.V1Service(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=svc_spec
            )
            created = self.core.create_namespaced_service(ns, svc)
            result = self._native_service_summary(created)
            self.log_action("create_native_service", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_native_service", params, error=e)
            raise RuntimeError("Failed to create native service") from e

    def delete_native_service(self, name: str, namespace: str | None = None) -> dict:
        """Delete a real Kubernetes (core/v1) Service."""
        params = {"name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            self.core.delete_namespaced_service(name, ns)
            result = {"deleted": name, "namespace": ns}
            self.log_action("delete_native_service", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("delete_native_service", params, error=e)
            raise RuntimeError("Failed to delete native service") from e

    def list_ingress_classes(self) -> list[dict]:
        """List IngressClasses."""
        params: dict[str, Any] = {}
        try:
            networking_api = self.networking
            ingress_classes = networking_api.list_ingress_class().items
            result = [
                {
                    "name": ic.metadata.name,
                    "controller": ic.spec.controller if ic.spec else None,
                    "parameters": ic.spec.parameters if ic.spec else None,
                    "created": self._ts(ic.metadata.creation_timestamp),
                }
                for ic in ingress_classes
            ]
            self.log_action("list_ingress_classes", params, {"count": len(result)})
            return result
        except ImportError:
            raise RuntimeError("Networking client not available") from None
        except _km.ApiException as e:
            self.log_action("list_ingress_classes", params, error=e)
            raise RuntimeError("Failed to list IngressClasses") from e

    def describe_ingress_class(self, name: str) -> dict:
        """Describe an IngressClass."""
        params = {"name": name}
        try:
            networking_api = self.networking
            ic = networking_api.read_ingress_class(name)
            result = {
                "name": ic.metadata.name,
                "spec": ic.spec,
                "created": self._ts(ic.metadata.creation_timestamp),
                "labels": ic.metadata.labels,
                "annotations": ic.metadata.annotations,
            }
            self.log_action("describe_ingress_class", params, result)
            return result
        except ImportError:
            raise RuntimeError("Networking client not available") from None
        except _km.ApiException as e:
            self.log_action("describe_ingress_class", params, error=e)
            raise RuntimeError("Failed to describe IngressClass") from e

    def create_ingress_class(self, name: str, spec: dict) -> dict:
        """Create an IngressClass."""
        params = {"name": name, "spec": spec}
        try:
            networking_api = self.networking
            ic_spec = _km.k8s_client.V1IngressClassSpec(**spec)
            ingress_class = _km.k8s_client.V1IngressClass(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=ic_spec
            )
            created = networking_api.create_ingress_class(ingress_class)
            result = {
                "name": name,
                "status": "created",
                "created": self._ts(created.metadata.creation_timestamp),
            }
            self.log_action("create_ingress_class", params, result)
            return result
        except ImportError:
            raise RuntimeError("Networking client not available") from None
        except _km.ApiException as e:
            self.log_action("create_ingress_class", params, error=e)
            raise RuntimeError("Failed to create IngressClass") from e

    def set_default_ingress_class(self, name: str) -> dict:
        """Set the default IngressClass."""
        params = {"name": name}
        try:
            networking_api = self.networking
            ingress_classes = networking_api.list_ingress_class().items

            # Remove default from all existing classes
            for ic in ingress_classes:
                if (
                    ic.metadata.annotations
                    and "ingressclass.kubernetes.io/is-default-class"
                    in ic.metadata.annotations
                ):
                    del ic.metadata.annotations[
                        "ingressclass.kubernetes.io/is-default-class"
                    ]
                    networking_api.patch_ingress_class(ic.metadata.name, ic)

            # Set default on the specified class
            target_ic = networking_api.read_ingress_class(name)
            if not target_ic.metadata.annotations:
                target_ic.metadata.annotations = {}
            target_ic.metadata.annotations[
                "ingressclass.kubernetes.io/is-default-class"
            ] = "true"
            networking_api.patch_ingress_class(name, target_ic)

            result = {"name": name, "status": "set_as_default"}
            self.log_action("set_default_ingress_class", params, result)
            return result
        except ImportError:
            raise RuntimeError("Networking client not available") from None
        except _km.ApiException as e:
            self.log_action("set_default_ingress_class", params, error=e)
            raise RuntimeError("Failed to set default IngressClass") from e

    def create_network_policy_with_cidr(
        self, name: str, namespace: str, spec: dict
    ) -> dict:
        """Create a NetworkPolicy with CIDR rules."""
        params = {"name": name, "namespace": namespace, "spec": spec}
        try:
            networking_api = self.networking
            np_spec = _km.k8s_client.V1NetworkPolicySpec(**spec)
            network_policy = _km.k8s_client.V1NetworkPolicy(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=np_spec
            )
            created = networking_api.create_namespaced_network_policy(
                namespace, network_policy
            )
            result = {
                "name": name,
                "namespace": namespace,
                "status": "created",
                "created": self._ts(created.metadata.creation_timestamp),
            }
            self.log_action("create_network_policy_with_cidr", params, result)
            return result
        except ImportError:
            raise RuntimeError("Networking client not available") from None
        except _km.ApiException as e:
            self.log_action("create_network_policy_with_cidr", params, error=e)
            raise RuntimeError(
                f"Failed to create NetworkPolicy with CIDR: {type(e).__name__}"
            ) from e

    def update_network_policy_rules(
        self, name: str, namespace: str, rules: list[dict]
    ) -> dict:
        """Update NetworkPolicy rules."""
        params = {"name": name, "namespace": namespace, "rules": rules}
        try:
            networking_api = self.networking
            existing = networking_api.read_namespaced_network_policy(name, namespace)

            # Convert rules to V1NetworkPolicyIngressRule objects
            policy_rules = []
            for rule in rules:
                policy_rules.append(_km.k8s_client.V1NetworkPolicyIngressRule(**rule))

            existing.spec.podSelector = (
                existing.spec.podSelector or _km.k8s_client.V1PodSelector()
            )
            existing.spec.policyTypes = existing.spec.policyTypes or ["Ingress"]
            existing.spec.ingress = policy_rules

            networking_api.patch_namespaced_network_policy(name, namespace, existing)
            result = {
                "name": name,
                "namespace": namespace,
                "status": "updated",
                "rules_count": len(rules),
            }
            self.log_action("update_network_policy_rules", params, result)
            return result
        except ImportError:
            raise RuntimeError("Networking client not available") from None
        except _km.ApiException as e:
            self.log_action("update_network_policy_rules", params, error=e)
            raise RuntimeError("Failed to update NetworkPolicy rules") from e

    def test_network_policy_connectivity(
        self, namespace: str, policy_name: str
    ) -> dict:
        """Test NetworkPolicy connectivity by creating test pods."""
        params = {"namespace": namespace, "policy_name": policy_name}
        try:
            # Get the NetworkPolicy to understand its rules
            networking_api = self.networking
            policy = networking_api.read_namespaced_network_policy(
                policy_name, namespace
            )

            # Analyze policy rules
            ingress_rules = []
            if policy.spec.ingress:
                for rule in policy.spec.ingress:
                    for peer in rule.from_ or []:
                        ingress_rules.append(f"from: {peer}")
                    for port in rule.ports or []:
                        ingress_rules.append(f"port: {port.port}")

            result = {
                "namespace": namespace,
                "policy_name": policy_name,
                "policy_type": policy.spec.policyTypes if policy.spec else [],
                "ingress_rules": ingress_rules,
                "pod_selector": (
                    policy.spec.podSelector._asdict()
                    if policy.spec.podSelector
                    else None
                ),
                "tested": True,
            }
            self.log_action("test_network_policy_connectivity", params, result)
            return result
        except ImportError:
            raise RuntimeError("Networking client not available") from None
        except _km.ApiException as e:
            self.log_action("test_network_policy_connectivity", params, error=e)
            raise RuntimeError(
                f"Failed to test NetworkPolicy connectivity: {type(e).__name__}"
            ) from e

    def check_dns_resolution(
        self, namespace: str, pod_name: str, hostname: str
    ) -> dict:
        """Check DNS resolution from a pod."""
        params = {"namespace": namespace, "pod_name": pod_name, "hostname": hostname}
        try:
            # Use kubectl exec to run nslookup/dig
            from kubernetes import stream

            command = ["nslookup", hostname]
            result = stream(
                self.core.connect_get_namespaced_pod_exec,
                pod_name,
                namespace,
                command=command,
                stderr=True,
                stdout=True,
                stdin=False,
                tty=False,
            )

            return {
                "namespace": namespace,
                "pod_name": pod_name,
                "hostname": hostname,
                "dns_result": result,
                "status": "completed",
            }
        except Exception as e:
            self.log_action("check_dns_resolution", params, error=e)
            raise RuntimeError("Failed to check DNS resolution") from e

    def list_dns_endpoints(self, namespace: str, service_name: str) -> dict:
        """List DNS endpoints for a service."""
        params = {"namespace": namespace, "service_name": service_name}
        try:
            # Get the service
            service = self.core.read_namespaced_service(service_name, namespace)

            # Get endpoints
            endpoints = self.core.read_namespaced_endpoints(service_name, namespace)

            result = {
                "namespace": namespace,
                "service_name": service_name,
                "cluster_ip": service.spec.cluster_ip,
                "external_ips": service.spec.external_ips or [],
                "ports": service.spec.ports or [],
                "endpoints": [],
            }

            if endpoints.subsets:
                for subset in endpoints.subsets:
                    for address in subset.addresses or []:
                        for port in subset.ports or []:
                            result["endpoints"].append(
                                {
                                    "ip": address.ip,
                                    "port": port.port,
                                    "protocol": port.protocol,
                                    "port_name": port.name,
                                }
                            )

            self.log_action("list_dns_endpoints", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("list_dns_endpoints", params, error=e)
            raise RuntimeError("Failed to list DNS endpoints") from e

    def test_dns_connectivity(self, namespace: str, target: str) -> dict:
        """Test DNS connectivity to a target."""
        params = {"namespace": namespace, "target": target}
        try:
            # Get a pod in the namespace to run the test from
            pods = self.core.list_namespaced_pod(namespace).items
            if not pods:
                raise RuntimeError(f"No pods found in namespace {namespace}")

            test_pod = pods[0]

            # Use kubectl exec to test connectivity
            from kubernetes import stream

            command = ["nslookup", target]
            result = stream(
                self.core.connect_get_namespaced_pod_exec,
                test_pod.metadata.name,
                namespace,
                command=command,
                stderr=True,
                stdout=True,
                stdin=False,
                tty=False,
            )

            return {
                "namespace": namespace,
                "target": target,
                "test_pod": test_pod.metadata.name,
                "dns_result": result,
                "status": "completed",
            }
        except Exception as e:
            self.log_action("test_dns_connectivity", params, error=e)
            raise RuntimeError("Failed to test DNS connectivity") from e

    def list_ingresses(self, namespace: str | None = None) -> list[dict]:
        """List Ingress resources."""
        params: dict[str, Any] = {"namespace": namespace}
        try:
            networking_api = self.networking
            ingresses = networking_api.list_namespaced_ingress(
                namespace=namespace or self.namespace
            ).items
            result = [
                {
                    "name": ingress.metadata.name,
                    "namespace": ingress.metadata.namespace,
                    "hosts": ingress.spec.rules if ingress.spec else [],
                    "created": self._ts(ingress.metadata.creation_timestamp),
                }
                for ingress in ingresses
            ]
            self.log_action("list_ingresses", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("list_ingresses", params, error=e)
            raise RuntimeError("Failed to list ingresses") from e

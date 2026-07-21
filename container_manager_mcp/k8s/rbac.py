"""RbacMixin for KubernetesManager (split from k8s_manager.py)."""

from typing import Any

import container_manager_mcp.k8s_manager as _km


class RbacMixin:
    def list_roles(self, namespace: str | None = None) -> list[dict]:
        """List Roles in a namespace."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            roles = self.rbac.list_namespaced_role(ns).items
            result = [
                {
                    "name": role.metadata.name,
                    "namespace": role.metadata.namespace,
                    "created": self._ts(role.metadata.creation_timestamp),
                    "rules_count": len(role.rules or []),
                }
                for role in roles
            ]
            self.log_action("list_roles", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_roles", params, error=e)
            raise RuntimeError("Failed to list roles") from e

    def create_role(
        self, name: str, namespace: str | None = None, rules: list[dict] | None = None
    ) -> dict:
        """Create a Role with specified rules."""
        params = {"name": name, "namespace": namespace, "rules": rules}
        try:
            ns = namespace or self.namespace
            role_rules = [_km.k8s_client.V1PolicyRule(**rule) for rule in (rules or [])]
            role = _km.k8s_client.V1Role(
                metadata=_km.k8s_client.V1ObjectMeta(name=name),
                rules=role_rules or None,
            )
            created = self.rbac.create_namespaced_role(ns, role)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
            }
            self.log_action("create_role", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_role", params, error=e)
            raise RuntimeError("Failed to create role") from e

    def delete_role(self, name: str, namespace: str | None = None) -> dict:
        """Delete a Role."""
        params = {"name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            self.rbac.delete_namespaced_role(name, ns)
            result = {"deleted": name}
            self.log_action("delete_role", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("delete_role", params, error=e)
            raise RuntimeError("Failed to delete role") from e

    def list_cluster_roles(self) -> list[dict]:
        """List ClusterRoles."""
        params: dict[str, Any] = {}
        try:
            cluster_roles = self.rbac.list_cluster_role().items
            result = [
                {
                    "name": cr.metadata.name,
                    "created": self._ts(cr.metadata.creation_timestamp),
                    "rules_count": len(cr.rules or []),
                }
                for cr in cluster_roles
            ]
            self.log_action("list_cluster_roles", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_cluster_roles", params, error=e)
            raise RuntimeError("Failed to list cluster roles") from e

    def list_rolebindings(self, namespace: str | None = None) -> list[dict]:
        """List RoleBindings in a namespace."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            rolebindings = self.rbac.list_namespaced_role_binding(ns).items
            result = [
                {
                    "name": rb.metadata.name,
                    "namespace": rb.metadata.namespace,
                    "role_ref": rb.role_ref.dict() if rb.role_ref else {},
                    "subjects_count": len(rb.subjects or []),
                    "created": self._ts(rb.metadata.creation_timestamp),
                }
                for rb in rolebindings
            ]
            self.log_action("list_rolebindings", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_rolebindings", params, error=e)
            raise RuntimeError("Failed to list rolebindings") from e

    def create_rolebinding(
        self,
        name: str,
        namespace: str | None = None,
        role_ref: dict | None = None,
        subjects: list[dict] | None = None,
    ) -> dict:
        """Create a RoleBinding."""
        params = {
            "name": name,
            "namespace": namespace,
            "role_ref": role_ref,
            "subjects": subjects,
        }
        try:
            ns = namespace or self.namespace
            role_ref_obj = _km.k8s_client.V1RoleRef(**role_ref) if role_ref else None
            subjects_objs = [_km.k8s_client.V1Subject(**s) for s in (subjects or [])]
            rolebinding = _km.k8s_client.V1RoleBinding(
                metadata=_km.k8s_client.V1ObjectMeta(name=name),
                role_ref=role_ref_obj,
                subjects=subjects_objs or None,
            )
            created = self.rbac.create_namespaced_role_binding(ns, rolebinding)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
            }
            self.log_action("create_rolebinding", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_rolebinding", params, error=e)
            raise RuntimeError("Failed to create rolebinding") from e

    def delete_rolebinding(self, name: str, namespace: str | None = None) -> dict:
        """Delete a RoleBinding."""
        params = {"name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            self.rbac.delete_namespaced_role_binding(name, ns)
            result = {"deleted": name}
            self.log_action("delete_rolebinding", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("delete_rolebinding", params, error=e)
            raise RuntimeError("Failed to delete rolebinding") from e

    def list_serviceaccounts(self, namespace: str | None = None) -> list[dict]:
        """List ServiceAccounts in a namespace."""
        params = {"namespace": namespace}
        try:
            ns = namespace or self.namespace
            serviceaccounts = self.core.list_namespaced_service_account(ns).items
            result = [
                {
                    "name": sa.metadata.name,
                    "namespace": sa.metadata.namespace,
                    "secrets_count": len(sa.secrets or []),
                    "created": self._ts(sa.metadata.creation_timestamp),
                }
                for sa in serviceaccounts
            ]
            self.log_action("list_serviceaccounts", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_serviceaccounts", params, error=e)
            raise RuntimeError("Failed to list serviceaccounts") from e

    def create_serviceaccount(self, name: str, namespace: str | None = None) -> dict:
        """Create a ServiceAccount."""
        params = {"name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            serviceaccount = _km.k8s_client.V1ServiceAccount(
                metadata=_km.k8s_client.V1ObjectMeta(name=name)
            )
            created = self.core.create_namespaced_service_account(ns, serviceaccount)
            result = {
                "name": created.metadata.name,
                "namespace": created.metadata.namespace,
            }
            self.log_action("create_serviceaccount", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_serviceaccount", params, error=e)
            raise RuntimeError("Failed to create serviceaccount") from e

    def delete_serviceaccount(self, name: str, namespace: str | None = None) -> dict:
        """Delete a ServiceAccount."""
        params = {"name": name, "namespace": namespace}
        try:
            ns = namespace or self.namespace
            self.core.delete_namespaced_service_account(name, ns)
            result = {"deleted": name}
            self.log_action("delete_serviceaccount", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("delete_serviceaccount", params, error=e)
            raise RuntimeError("Failed to delete serviceaccount") from e

    def auth_can_i(
        self, verb: str, resource: str, namespace: str | None = None
    ) -> dict:
        """Check if a user has permission to perform an action (kubectl auth can-i)."""
        params = {"verb": verb, "resource": resource, "namespace": namespace}
        try:
            # Create a SelfSubjectAccessReview
            access_review = _km.k8s_client.V1SelfSubjectAccessReview(
                spec=_km.k8s_client.V1SelfSubjectAccessReviewSpec(
                    verb=verb, resource=resource, namespace=namespace
                )
            )
            response = self.authz.create_self_subject_access_review(access_review)
            result = {
                "allowed": response.status.allowed if response.status else False,
                "reason": response.status.reason if response.status else "",
                "verb": verb,
                "resource": resource,
            }
            self.log_action("auth_can_i", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("auth_can_i", params, error=e)
            raise RuntimeError("Failed to check authorization") from e

    def list_cluster_rolebindings(self) -> list[dict]:
        """List ClusterRoleBindings."""
        params: dict[str, Any] = {}
        try:
            cluster_rolebindings = self.rbac.list_cluster_role_binding().items
            result = [
                {
                    "name": crb.metadata.name,
                    "role_ref": crb.role_ref.dict() if crb.role_ref else {},
                    "subjects_count": len(crb.subjects or []),
                    "created": self._ts(crb.metadata.creation_timestamp),
                }
                for crb in cluster_rolebindings
            ]
            self.log_action("list_cluster_rolebindings", params, {"count": len(result)})
            return result
        except _km.ApiException as e:
            self.log_action("list_cluster_rolebindings", params, error=e)
            raise RuntimeError("Failed to list cluster rolebindings") from e

    def create_cluster_rolebinding(
        self,
        name: str,
        role_ref: dict | None = None,
        subjects: list[dict] | None = None,
    ) -> dict:
        """Create a ClusterRoleBinding."""
        params = {"name": name, "role_ref": role_ref, "subjects": subjects}
        try:
            role_ref_obj = _km.k8s_client.V1RoleRef(**role_ref) if role_ref else None
            subjects_objs = [_km.k8s_client.V1Subject(**s) for s in (subjects or [])]
            cluster_rolebinding = _km.k8s_client.V1ClusterRoleBinding(
                metadata=_km.k8s_client.V1ObjectMeta(name=name),
                role_ref=role_ref_obj,
                subjects=subjects_objs or None,
            )
            created = self.rbac.create_cluster_role_binding(cluster_rolebinding)
            result = {"name": created.metadata.name}
            self.log_action("create_cluster_rolebinding", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("create_cluster_rolebinding", params, error=e)
            raise RuntimeError("Failed to create cluster rolebinding") from e

    def delete_cluster_rolebinding(self, name: str) -> dict:
        """Delete a ClusterRoleBinding."""
        params = {"name": name}
        try:
            self.rbac.delete_cluster_role_binding(name)
            result = {"deleted": name}
            self.log_action("delete_cluster_rolebinding", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("delete_cluster_rolebinding", params, error=e)
            raise RuntimeError("Failed to delete cluster rolebinding") from e

    def create_service_account_token(
        self, name: str, namespace: str, token_spec: dict
    ) -> dict:
        """Create a ServiceAccount token."""
        params = {"name": name, "namespace": namespace, "token_spec": token_spec}
        try:
            auth_api = self.authn
            token_request = _km.k8s_client.V1TokenRequest(
                metadata=_km.k8s_client.V1ObjectMeta(
                    name=f"{name}-token",
                    namespace=namespace,
                    annotations={"kubernetes.io/service-account.name": name},
                ),
                spec=_km.k8s_client.V1TokenRequestSpec(**token_spec),
            )
            created = auth_api.create_namespaced_token_request(namespace, token_request)
            result = {
                "name": created.metadata.name,
                "namespace": namespace,
                "service_account": name,
                "status": "created",
                "token": created.status.token if created.status else None,
            }
            self.log_action("create_service_account_token", params, result)
            return result
        except ImportError:
            raise RuntimeError("Authentication client not available") from None
        except _km.ApiException as e:
            self.log_action("create_service_account_token", params, error=e)
            raise RuntimeError(
                f"Failed to create ServiceAccount token: {type(e).__name__}"
            ) from e

    def list_service_account_tokens(self, name: str, namespace: str) -> list[dict]:
        """List ServiceAccount tokens."""
        params = {"name": name, "namespace": namespace}
        try:
            auth_api = self.authn
            token_requests = auth_api.list_namespaced_token_request(namespace).items

            # Filter for tokens belonging to this service account
            service_account_tokens = []
            for tr in token_requests:
                sa_name = tr.metadata.annotations.get(
                    "kubernetes.io/service-account.name"
                )
                if sa_name == name:
                    service_account_tokens.append(
                        {
                            "name": tr.metadata.name,
                            "namespace": tr.metadata.namespace,
                            "created": self._ts(tr.metadata.creation_timestamp),
                            "status": tr.status,
                        }
                    )

            result = {
                "service_account": name,
                "namespace": namespace,
                "tokens": service_account_tokens,
                "count": len(service_account_tokens),
            }
            self.log_action("list_service_account_tokens", params, result)
            return service_account_tokens
        except ImportError:
            raise RuntimeError("Authentication client not available") from None
        except _km.ApiException as e:
            self.log_action("list_service_account_tokens", params, error=e)
            raise RuntimeError("Failed to list ServiceAccount tokens") from e

    def delete_service_account_token(
        self, name: str, namespace: str, token_name: str
    ) -> dict:
        """Delete a ServiceAccount token."""
        params = {"name": name, "namespace": namespace, "token_name": token_name}
        try:
            auth_api = self.authn
            auth_api.delete_namespaced_token_request(token_name, namespace)
            result = {
                "service_account": name,
                "namespace": namespace,
                "token_name": token_name,
                "status": "deleted",
            }
            self.log_action("delete_service_account_token", params, result)
            return result
        except ImportError:
            raise RuntimeError("Authentication client not available") from None
        except _km.ApiException as e:
            self.log_action("delete_service_account_token", params, error=e)
            raise RuntimeError(
                f"Failed to delete ServiceAccount token: {type(e).__name__}"
            ) from e

    def subject_access_review(self, spec: dict) -> dict:
        """Perform a SubjectAccessReview."""
        params = {"spec": spec}
        try:
            auth_api = self.authz
            sar = _km.k8s_client.V1SubjectAccessReview(
                spec=_km.k8s_client.V1SubjectAccessReviewSpec(**spec)
            )
            response = auth_api.create_subject_access_review(sar)
            result = {
                "allowed": response.status.allowed,
                "reason": response.status.reason,
                "denied": not response.status.allowed,
            }
            self.log_action("subject_access_review", params, result)
            return result
        except ImportError:
            raise RuntimeError("Authorization client not available") from None
        except _km.ApiException as e:
            self.log_action("subject_access_review", params, error=e)
            raise RuntimeError(
                f"Failed to perform SubjectAccessReview: {type(e).__name__}"
            ) from e

    def local_subject_access_review(self, namespace: str, spec: dict) -> dict:
        """Perform a LocalSubjectAccessReview."""
        params = {"namespace": namespace, "spec": spec}
        try:
            auth_api = self.authz
            lsar = _km.k8s_client.V1LocalSubjectAccessReview(
                spec=_km.k8s_client.V1SubjectAccessReviewSpec(**spec)
            )
            response = auth_api.create_namespaced_local_subject_access_review(
                namespace, lsar
            )
            result = {
                "namespace": namespace,
                "allowed": response.status.allowed,
                "reason": response.status.reason,
                "denied": not response.status.allowed,
            }
            self.log_action("local_subject_access_review", params, result)
            return result
        except ImportError:
            raise RuntimeError("Authorization client not available") from None
        except _km.ApiException as e:
            self.log_action("local_subject_access_review", params, error=e)
            raise RuntimeError(
                f"Failed to perform LocalSubjectAccessReview: {type(e).__name__}"
            ) from e

    def create_aggregated_cluster_role(self, name: str, aggregation_rule: dict) -> dict:
        """Create an aggregated ClusterRole."""
        params = {"name": name, "aggregation_rule": aggregation_rule}
        try:
            rbac_api = self.rbac
            cluster_role = _km.k8s_client.V1ClusterRole(
                metadata=_km.k8s_client.V1ObjectMeta(name=name),
                aggregation_rule=_km.k8s_client.V1AggregationRule(**aggregation_rule),
            )
            rbac_api.create_cluster_role(cluster_role)
            result = {
                "name": name,
                "status": "created",
                "aggregation_rule": aggregation_rule,
            }
            self.log_action("create_aggregated_cluster_role", params, result)
            return result
        except ImportError:
            raise RuntimeError("RBAC client not available") from None
        except _km.ApiException as e:
            self.log_action("create_aggregated_cluster_role", params, error=e)
            raise RuntimeError(
                f"Failed to create aggregated ClusterRole: {type(e).__name__}"
            ) from e

    def update_aggregated_cluster_role(self, name: str, aggregation_rule: dict) -> dict:
        """Update an aggregated ClusterRole."""
        params = {"name": name, "aggregation_rule": aggregation_rule}
        try:
            rbac_api = self.rbac
            existing = rbac_api.read_cluster_role(name)
            existing.aggregation_rule = _km.k8s_client.V1AggregationRule(
                **aggregation_rule
            )
            rbac_api.patch_cluster_role(name, existing)
            result = {
                "name": name,
                "status": "updated",
                "aggregation_rule": aggregation_rule,
            }
            self.log_action("update_aggregated_cluster_role", params, result)
            return result
        except ImportError:
            raise RuntimeError("RBAC client not available") from None
        except _km.ApiException as e:
            self.log_action("update_aggregated_cluster_role", params, error=e)
            raise RuntimeError(
                f"Failed to update aggregated ClusterRole: {type(e).__name__}"
            ) from e

    def list_pod_security_policies(self) -> list[dict]:
        """List PodSecurityPolicies (deprecated)."""
        params: dict[str, Any] = {}
        try:
            policy_api = _km.k8s_client.PolicyV1beta1Api()
            psps = policy_api.list_pod_security_policy().items
            result = [
                {
                    "name": psp.metadata.name,
                    "spec": psp.spec,
                    "created": self._ts(psp.metadata.creation_timestamp),
                }
                for psp in psps
            ]
            self.log_action(
                "list_pod_security_policies", params, {"count": len(result)}
            )
            return result
        except ImportError:
            raise RuntimeError("Policy client not available") from None
        except _km.ApiException as e:
            self.log_action("list_pod_security_policies", params, error=e)
            raise RuntimeError("Failed to list PodSecurityPolicies") from e

    def describe_pod_security_policy(self, name: str) -> dict:
        """Describe a PodSecurityPolicy."""
        params = {"name": name}
        try:
            policy_api = _km.k8s_client.PolicyV1beta1Api()
            psp = policy_api.read_pod_security_policy(name)
            result = {
                "name": psp.metadata.name,
                "spec": psp.spec,
                "created": self._ts(psp.metadata.creation_timestamp),
                "labels": psp.metadata.labels,
                "annotations": psp.metadata.annotations,
            }
            self.log_action("describe_pod_security_policy", params, result)
            return result
        except ImportError:
            raise RuntimeError("Policy client not available") from None
        except _km.ApiException as e:
            self.log_action("describe_pod_security_policy", params, error=e)
            raise RuntimeError("Failed to describe PodSecurityPolicy") from e

    def create_pod_security_policy(self, name: str, spec: dict) -> dict:
        """Create a PodSecurityPolicy."""
        params = {"name": name, "spec": spec}
        try:
            policy_api = _km.k8s_client.PolicyV1beta1Api()
            psp_spec = _km.k8s_client.V1PodSecurityPolicySpec(**spec)
            psp = _km.k8s_client.V1PodSecurityPolicy(
                metadata=_km.k8s_client.V1ObjectMeta(name=name), spec=psp_spec
            )
            created = policy_api.create_pod_security_policy(psp)
            result = {
                "name": name,
                "status": "created",
                "created": self._ts(created.metadata.creation_timestamp),
            }
            self.log_action("create_pod_security_policy", params, result)
            return result
        except ImportError:
            raise RuntimeError("Policy client not available") from None
        except _km.ApiException as e:
            self.log_action("create_pod_security_policy", params, error=e)
            raise RuntimeError("Failed to create PodSecurityPolicy") from e

    def delete_pod_security_policy(self, name: str) -> dict:
        """Delete a PodSecurityPolicy."""
        params = {"name": name}
        try:
            policy_api = _km.k8s_client.PolicyV1beta1Api()
            policy_api.delete_pod_security_policy(name)
            result = {"name": name, "status": "deleted"}
            self.log_action("delete_pod_security_policy", params, result)
            return result
        except ImportError:
            raise RuntimeError("Policy client not available") from None
        except _km.ApiException as e:
            self.log_action("delete_pod_security_policy", params, error=e)
            raise RuntimeError("Failed to delete PodSecurityPolicy") from e

    def evaluate_pod_security(self, namespace: str, pod_spec: dict) -> dict:
        """Evaluate pod security against policies."""
        params = {"namespace": namespace, "pod_spec": pod_spec}
        try:
            auth_api = self.authn

            # Create a temporary subject access review for pod creation
            spec = {
                "resourceAttributes": {
                    "namespace": namespace,
                    "resource": "pods",
                    "verb": "create",
                }
            }

            sar = _km.k8s_client.V1SubjectAccessReview(
                spec=_km.k8s_client.V1SubjectAccessReviewSpec(**spec)
            )
            response = auth_api.create_subject_access_review(sar)

            result = {
                "namespace": namespace,
                "allowed": response.status.allowed,
                "reason": response.status.reason,
                "evaluated": True,
            }
            self.log_action("evaluate_pod_security", params, result)
            return result
        except ImportError:
            raise RuntimeError("Authentication client not available") from None
        except _km.ApiException as e:
            self.log_action("evaluate_pod_security", params, error=e)
            raise RuntimeError("Failed to evaluate pod security") from e

    def list_service_account_mapped_secrets(
        self, name: str, namespace: str
    ) -> list[dict]:
        """List secrets mapped to a ServiceAccount."""
        params = {"name": name, "namespace": namespace}
        try:
            sa = self.core.read_namespaced_service_account(name, namespace)
            secret_names = sa.secrets or []

            result = []
            for secret_ref in secret_names:
                try:
                    secret = self.core.read_namespaced_secret(
                        secret_ref.name, namespace
                    )
                    result.append(
                        {
                            "name": secret.metadata.name,
                            "type": secret.type,
                            "created": self._ts(secret.metadata.creation_timestamp),
                        }
                    )
                except _km.ApiException:
                    # Secret might not exist anymore
                    pass

            self.log_action(
                "list_service_account_mapped_secrets", params, {"count": len(result)}
            )
            return result
        except _km.ApiException as e:
            self.log_action("list_service_account_mapped_secrets", params, error=e)
            raise RuntimeError(
                f"Failed to list ServiceAccount mapped secrets: {type(e).__name__}"
            ) from e

    def map_secret_to_service_account(
        self, secret_name: str, sa_name: str, namespace: str
    ) -> dict:
        """Map a secret to a ServiceAccount."""
        params = {
            "secret_name": secret_name,
            "sa_name": sa_name,
            "namespace": namespace,
        }
        try:
            sa = self.core.read_namespaced_service_account(sa_name, namespace)

            # Add secret to ServiceAccount
            if not sa.secrets:
                sa.secrets = []

            # Check if secret is already mapped
            for secret_ref in sa.secrets:
                if secret_ref.name == secret_name:
                    return {
                        "secret_name": secret_name,
                        "sa_name": sa_name,
                        "namespace": namespace,
                        "status": "already_mapped",
                    }

            sa.secrets.append(_km.k8s_client.V1ObjectReference(name=secret_name))
            self.core.patch_namespaced_service_account(sa_name, namespace, sa)

            result = {
                "secret_name": secret_name,
                "sa_name": sa_name,
                "namespace": namespace,
                "status": "mapped",
            }
            self.log_action("map_secret_to_service_account", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("map_secret_to_service_account", params, error=e)
            raise RuntimeError(
                f"Failed to map secret to ServiceAccount: {type(e).__name__}"
            ) from e

    def unmap_secret_from_service_account(
        self, secret_name: str, sa_name: str, namespace: str
    ) -> dict:
        """Unmap a secret from a ServiceAccount."""
        params = {
            "secret_name": secret_name,
            "sa_name": sa_name,
            "namespace": namespace,
        }
        try:
            sa = self.core.read_namespaced_service_account(sa_name, namespace)

            # Remove secret from ServiceAccount
            if sa.secrets:
                sa.secrets = [s for s in sa.secrets if s.name != secret_name]

            self.core.patch_namespaced_service_account(sa_name, namespace, sa)

            result = {
                "secret_name": secret_name,
                "sa_name": sa_name,
                "namespace": namespace,
                "status": "unmapped",
            }
            self.log_action("unmap_secret_from_service_account", params, result)
            return result
        except _km.ApiException as e:
            self.log_action("unmap_secret_from_service_account", params, error=e)
            raise RuntimeError(
                f"Failed to unmap secret from ServiceAccount: {type(e).__name__}"
            ) from e

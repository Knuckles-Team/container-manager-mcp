"""ManagementMixin for KubernetesManager: generic resource management.

Whole-cluster management through one typed tool per operation (each public
method here is served as ``k8s_<method>`` by the verbose tier):

* any-kind reads, applies, deletes and drift checks through the dynamic client
  (``get_resource`` / ``list_resources`` / ``apply_manifest`` /
  ``delete_resource`` / ``export_manifest`` / ``diff_manifest``);
* deployment helpers that take a namespace (``describe_deployment`` /
  ``scale_deployment``) and pod logs with container/previous selection;
* Secret reads that return metadata unless ``reveal=True``;
* ExternalSecret listing and a force-sync that proves the refresh happened.

Secret values never leave the connector by default: Secret bodies are
redacted, including the client-side-apply annotation that would otherwise
carry the values, and exports for persistence in a git repository refuse
Secret data outright.
"""

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import container_manager_mcp.k8s_manager as _km

if TYPE_CHECKING:
    from container_manager_mcp.k8s.base import _K8sBase as _Base
else:
    _Base = object

REDACTED = "<redacted>"
LAST_APPLIED = "kubectl.kubernetes.io/last-applied-configuration"
FIELD_MANAGER = "container-manager-mcp"
ESO_GROUP = "external-secrets.io"
ESO_VERSION = "v1"
ESO_PLURAL = "externalsecrets"
FORCE_SYNC = "force-sync"
MAX_WAIT_SECONDS = 300

# Server-populated metadata a manifest kept in git must not carry.
_VOLATILE_METADATA = (
    "managedFields",
    "resourceVersion",
    "uid",
    "creationTimestamp",
    "generation",
    "selfLink",
    "finalizers",
    "ownerReferences",
    "deletionTimestamp",
    "deletionGracePeriodSeconds",
)
_VOLATILE_ANNOTATIONS = (
    LAST_APPLIED,
    "deployment.kubernetes.io/revision",
    "kubectl.kubernetes.io/restartedAt",
    "force-sync",
)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_secret(obj: dict) -> bool:
    return obj.get("kind") == "Secret" and obj.get("apiVersion", "v1") == "v1"


def _data_digest(data: dict | None) -> str:
    """One digest over every key and value: detects change, reveals nothing."""
    canonical = json.dumps(data or {}, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def redact_secret(obj: dict) -> dict:
    """A Secret body with values replaced; keys and a change digest kept."""
    out = dict(obj)
    data = obj.get("data") or {}
    string_data = obj.get("stringData") or {}
    if data:
        out["data"] = dict.fromkeys(data, REDACTED)
    if string_data:
        out["stringData"] = dict.fromkeys(string_data, REDACTED)
    metadata = dict(out.get("metadata") or {})
    annotations = dict(metadata.get("annotations") or {})
    if LAST_APPLIED in annotations:
        annotations[LAST_APPLIED] = REDACTED
        metadata["annotations"] = annotations
        out["metadata"] = metadata
    out["dataDigest"] = _data_digest({**data, **string_data})
    out["redacted"] = True
    return out


def clean_for_git(obj: dict) -> dict:
    """A live object reduced to the fields a services/ manifest keeps."""
    out = {k: v for k, v in obj.items() if k != "status"}
    metadata = {
        k: v
        for k, v in (out.get("metadata") or {}).items()
        if k not in _VOLATILE_METADATA
    }
    annotations = {
        k: v
        for k, v in (metadata.get("annotations") or {}).items()
        if k not in _VOLATILE_ANNOTATIONS
    }
    if annotations:
        metadata["annotations"] = annotations
    else:
        metadata.pop("annotations", None)
    out["metadata"] = metadata
    if out.get("kind") == "Service":
        spec = dict(out.get("spec") or {})
        for key in ("clusterIP", "clusterIPs"):
            spec.pop(key, None)
        out["spec"] = spec
    return out


def _diff_paths(desired: Any, live: Any, path: str = "") -> list[str]:
    """JSON paths where ``desired`` disagrees with ``live`` (desired keys only)."""
    if isinstance(desired, dict) and isinstance(live, dict):
        found: list[str] = []
        for key, value in desired.items():
            found += _diff_paths(value, live.get(key), f"{path}.{key}")
        return found
    if isinstance(desired, list) and isinstance(live, list):
        if len(desired) != len(live):
            return [path or "."]
        found = []
        for index, (want, have) in enumerate(zip(desired, live, strict=True)):
            found += _diff_paths(want, have, f"{path}[{index}]")
        return found
    return [] if desired == live else [path or "."]


def _load_documents(manifest_yaml: str) -> list[dict]:
    import yaml

    docs = [d for d in yaml.safe_load_all(manifest_yaml) if d]
    for doc in docs:
        if not isinstance(doc, dict) or "kind" not in doc or "apiVersion" not in doc:
            raise ValueError("every manifest document needs apiVersion and kind")
    return docs


class ManagementMixin(_Base):
    # ----------------------------------------------------------------- helpers
    def _dynamic(self) -> Any:
        cached = getattr(self, "_dynamic_client", None)
        if cached is None:
            from kubernetes.dynamic import DynamicClient

            cached = DynamicClient(self.core.api_client)
            self._dynamic_client = cached
        return cached

    def _resource(self, api_version: str, kind: str) -> Any:
        return self._dynamic().resources.get(api_version=api_version, kind=kind)

    def _sanitize(self, obj: Any) -> dict:
        if obj is None:
            return {}
        if hasattr(obj, "to_dict") and not isinstance(obj, dict):
            return obj.to_dict()
        return self.core.api_client.sanitize_for_serialization(obj)

    def _read(self, api_version: str, kind: str, name: str, namespace: str | None):
        resource = self._resource(api_version, kind)
        if getattr(resource, "namespaced", False):
            return resource, resource.get(
                name=name, namespace=namespace or self.namespace
            )
        return resource, resource.get(name=name)

    # --------------------------------------------------------- generic reads
    def get_resource(
        self,
        api_version: str,
        kind: str,
        name: str,
        namespace: str | None = None,
        reveal: bool = False,
    ) -> dict:
        """Read any object by apiVersion and kind (CRDs included).

        Secret values are redacted unless ``reveal`` is true; the redacted body
        keeps the key names and a ``dataDigest`` that changes when any value
        changes.
        """
        params = {"api_version": api_version, "kind": kind, "name": name}
        _, live = self._read(api_version, kind, name, namespace)
        body = live.to_dict()
        if _is_secret(body) and not reveal:
            body = redact_secret(body)
        self.log_action("get_resource", params, {"kind": kind})
        return body

    def list_resources(
        self,
        api_version: str,
        kind: str,
        namespace: str | None = None,
        label_selector: str | None = None,
        all_namespaces: bool = False,
    ) -> list[dict]:
        """List objects of any kind as name/namespace/created/labels summaries."""
        params = {"api_version": api_version, "kind": kind, "all": all_namespaces}
        resource = self._resource(api_version, kind)
        kwargs: dict[str, Any] = {}
        if label_selector:
            kwargs["label_selector"] = label_selector
        if getattr(resource, "namespaced", False) and not all_namespaces:
            kwargs["namespace"] = namespace or self.namespace
        items = resource.get(**kwargs).to_dict().get("items") or []
        result = [
            {
                "name": (item.get("metadata") or {}).get("name"),
                "namespace": (item.get("metadata") or {}).get("namespace"),
                "created": (item.get("metadata") or {}).get("creationTimestamp"),
                "labels": (item.get("metadata") or {}).get("labels") or {},
            }
            for item in items
        ]
        self.log_action("list_resources", params, {"count": len(result)})
        return result

    # ------------------------------------------------------- apply / delete
    def apply_manifest(
        self,
        manifest_yaml: str,
        namespace: str | None = None,
        dry_run: bool = False,
        force_conflicts: bool = True,
    ) -> list[dict]:
        """Server-side apply one or more YAML documents (kubectl apply --server-side).

        Namespaced documents without ``metadata.namespace`` go to ``namespace``.
        Returns one summary per document; object bodies are not echoed.
        """
        results = []
        for doc in _load_documents(manifest_yaml):
            resource = self._resource(doc["apiVersion"], doc["kind"])
            metadata = doc.setdefault("metadata", {})
            kwargs: dict[str, Any] = {
                "body": doc,
                "name": metadata.get("name"),
                "field_manager": FIELD_MANAGER,
                "force_conflicts": force_conflicts,
            }
            if getattr(resource, "namespaced", False):
                metadata.setdefault("namespace", namespace or self.namespace)
                kwargs["namespace"] = metadata["namespace"]
            if dry_run:
                kwargs["dry_run"] = "All"
            applied = resource.server_side_apply(**kwargs)
            applied_meta = applied.to_dict().get("metadata") or {}
            results.append(
                {
                    "kind": doc["kind"],
                    "name": metadata.get("name"),
                    "namespace": metadata.get("namespace"),
                    "resource_version": applied_meta.get("resourceVersion"),
                    "dry_run": dry_run,
                    "applied": True,
                }
            )
        self.log_action("apply_manifest", {"documents": len(results)}, results)
        return results

    def delete_resource(
        self,
        api_version: str,
        kind: str,
        name: str,
        namespace: str | None = None,
        propagation_policy: str = "Background",
    ) -> dict:
        """Delete any object by apiVersion and kind."""
        resource = self._resource(api_version, kind)
        kwargs: dict[str, Any] = {
            "name": name,
            "body": {"propagationPolicy": propagation_policy},
        }
        ns = None
        if getattr(resource, "namespaced", False):
            ns = namespace or self.namespace
            kwargs["namespace"] = ns
        resource.delete(**kwargs)
        result = {"kind": kind, "name": name, "namespace": ns, "deleted": True}
        self.log_action("delete_resource", {"kind": kind, "name": name}, result)
        return result

    # ------------------------------------------------------ GitOps tie-in
    def export_manifest(
        self,
        api_version: str,
        kind: str,
        name: str,
        namespace: str | None = None,
    ) -> dict:
        """The live object as a clean manifest to persist in services/<name>/k8s/.

        Status, managed fields, server ids and rollout bookkeeping annotations
        are stripped. A Secret is refused: its values belong in OpenBao behind
        an ExternalSecret, never in git.
        """
        import yaml

        _, live = self._read(api_version, kind, name, namespace)
        body = live.to_dict()
        if _is_secret(body):
            raise ValueError(
                "Secrets are not exported to git; store the values in OpenBao and "
                "persist the ExternalSecret that projects them instead"
            )
        clean = clean_for_git(body)
        result = {
            "kind": kind,
            "name": name,
            "namespace": (clean.get("metadata") or {}).get("namespace"),
            "manifest_yaml": yaml.safe_dump(clean, sort_keys=False),
        }
        self.log_action("export_manifest", {"kind": kind, "name": name}, {"ok": True})
        return result

    def _live_only_paths(
        self, resource: Any, doc: dict, ns: str | None, live_body: dict
    ) -> list[str] | None:
        """Fields set live (by another field manager) that the manifest lacks.

        A server-side apply keeps fields other managers own, so a live-only
        addition (an env var patched in by hand, say) is invisible to the apply
        dry run. The manifest is dry-run CREATED under a probe name to get the
        server-defaulted object it declares, and the live object's spec, data,
        labels and annotations are compared against that. ``None`` when the
        probe cannot be created (the kind rejects a second name, a quota).
        """
        probe = json.loads(json.dumps(doc))
        probe["metadata"] = {
            k: v
            for k, v in probe["metadata"].items()
            if k in ("name", "namespace", "labels", "annotations")
        }
        probe["metadata"]["name"] = f"{doc['metadata']['name']}-drift-probe"[:253]
        kwargs: dict[str, Any] = {"body": probe, "dry_run": "All"}
        if ns:
            kwargs["namespace"] = ns
        try:
            declared_text = json.dumps(resource.create(**kwargs).to_dict())
        except Exception:  # noqa: BLE001 - any refusal means "cannot tell"
            return None
        # Values the server derives from the name follow the real name.
        declared = json.loads(
            declared_text.replace(
                json.dumps(probe["metadata"]["name"])[1:-1],
                json.dumps(doc["metadata"]["name"])[1:-1],
            )
        )

        def scope(body: dict) -> dict:
            clean = clean_for_git(body)
            meta = clean.get("metadata") or {}
            out = {
                k: clean[k]
                for k in ("spec", "data", "binaryData", "stringData")
                if k in clean
            }
            out["metadata"] = {
                "labels": meta.get("labels") or {},
                "annotations": meta.get("annotations") or {},
            }
            return out

        return _diff_paths(scope(live_body), scope(declared))

    def diff_manifest(
        self, manifest_yaml: str, namespace: str | None = None
    ) -> list[dict]:
        """Compare a manifest (e.g. from services/<name>/) with the live cluster.

        Each document is server-side applied as a dry run and the would-be
        object is compared with the live one; ``drift`` is true when applying
        the manifest would change the cluster (or the object is absent).
        Secret documents are compared by digest only.
        """
        results = []
        for doc in _load_documents(manifest_yaml):
            resource = self._resource(doc["apiVersion"], doc["kind"])
            metadata = doc.setdefault("metadata", {})
            ns = None
            if getattr(resource, "namespaced", False):
                ns = metadata.setdefault("namespace", namespace or self.namespace)
            entry: dict[str, Any] = {
                "kind": doc["kind"],
                "name": metadata.get("name"),
                "namespace": ns,
            }
            try:
                live = resource.get(
                    name=metadata.get("name"), **({"namespace": ns} if ns else {})
                )
            except _km.ApiException as exc:
                if getattr(exc, "status", None) == 404:
                    results.append(
                        {**entry, "drift": True, "live": "absent", "paths": []}
                    )
                    continue
                raise
            kwargs: dict[str, Any] = {
                "body": doc,
                "name": metadata.get("name"),
                "field_manager": FIELD_MANAGER,
                "force_conflicts": True,
                "dry_run": "All",
            }
            if ns:
                kwargs["namespace"] = ns
            would_be = resource.server_side_apply(**kwargs).to_dict()
            live_body = live.to_dict()
            live_only: list[str] | None = []
            if _is_secret(live_body):
                changed = _data_digest(would_be.get("data")) != _data_digest(
                    live_body.get("data")
                )
                paths = [".data"] if changed else []
            else:
                paths = _diff_paths(clean_for_git(would_be), clean_for_git(live_body))
                live_only = self._live_only_paths(resource, doc, ns, live_body)
            results.append(
                {
                    **entry,
                    "drift": bool(paths or live_only),
                    "live": "present",
                    "paths": paths,
                    "live_only_paths": live_only,
                }
            )
        self.log_action("diff_manifest", {"documents": len(results)}, {"ok": True})
        return results

    # ------------------------------------------------- deployments / logs
    def describe_deployment(self, name: str, namespace: str | None = None) -> dict:
        """Describe a Deployment: replicas, strategy, conditions, containers, events."""
        ns = namespace or self.namespace
        dep = self.apps.read_namespaced_deployment(name, ns)
        spec, status = dep.spec, dep.status
        containers = spec.template.spec.containers if spec and spec.template else []
        events = self.core.list_namespaced_event(
            ns,
            field_selector=f"involvedObject.kind=Deployment,involvedObject.name={name}",
        ).items
        result = {
            "name": name,
            "namespace": ns,
            "revision": (dep.metadata.annotations or {}).get(
                "deployment.kubernetes.io/revision"
            ),
            "generation": dep.metadata.generation,
            "observed_generation": getattr(status, "observed_generation", None),
            "replicas": spec.replicas if spec else None,
            "ready_replicas": getattr(status, "ready_replicas", None) or 0,
            "updated_replicas": getattr(status, "updated_replicas", None) or 0,
            "available_replicas": getattr(status, "available_replicas", None) or 0,
            "paused": bool(getattr(spec, "paused", False)),
            "strategy": getattr(getattr(spec, "strategy", None), "type", None),
            "selector": getattr(getattr(spec, "selector", None), "match_labels", None)
            or {},
            "node_selector": getattr(spec.template.spec, "node_selector", None) or {},
            "containers": [
                {
                    "name": c.name,
                    "image": c.image,
                    "resources": c.resources.to_dict() if c.resources else {},
                }
                for c in containers
            ],
            "conditions": [
                {
                    "type": c.type,
                    "status": c.status,
                    "reason": c.reason,
                    "message": c.message,
                }
                for c in (getattr(status, "conditions", None) or [])
            ],
            "events": [
                {
                    "type": e.type,
                    "reason": e.reason,
                    "message": e.message,
                    "count": e.count,
                }
                for e in events
            ],
        }
        self.log_action("describe_deployment", {"name": name}, {"ok": True})
        return result

    def scale_deployment(
        self, name: str, replicas: int, namespace: str | None = None
    ) -> dict:
        """Scale a Deployment in any namespace."""
        ns = namespace or self.namespace
        self.apps.patch_namespaced_deployment_scale(
            name, ns, {"spec": {"replicas": replicas}}
        )
        result = {"name": name, "namespace": ns, "replicas": replicas, "scaled": True}
        self.log_action("scale_deployment", {"name": name}, result)
        return result

    def read_pod_logs(
        self,
        pod_name: str,
        namespace: str | None = None,
        container: str | None = None,
        tail_lines: int = 200,
        previous: bool = False,
        since_seconds: int | None = None,
    ) -> dict:
        """Read a pod's logs, optionally one container's or the previous instance's."""
        ns = namespace or self.namespace
        kwargs: dict[str, Any] = {"tail_lines": tail_lines, "previous": previous}
        if container:
            kwargs["container"] = container
        if since_seconds:
            kwargs["since_seconds"] = since_seconds
        logs = self.core.read_namespaced_pod_log(pod_name, ns, **kwargs)
        result = {
            "pod_name": pod_name,
            "namespace": ns,
            "container": container,
            "previous": previous,
            "logs": logs,
        }
        self.log_action(
            "read_pod_logs", {"pod_name": pod_name}, {"length": len(logs or "")}
        )
        return result

    # ------------------------------------------------------------- secrets
    def get_secret(
        self, name: str, namespace: str | None = None, reveal: bool = False
    ) -> dict:
        """Read a Secret's metadata and key names; values only with ``reveal=True``."""
        ns = namespace or self.namespace
        secret = self.core.read_namespaced_secret(name, ns)
        data = secret.data or {}
        result: dict[str, Any] = {
            "name": name,
            "namespace": ns,
            "type": secret.type,
            "keys": sorted(data),
            "data_digest": _data_digest(data),
            "labels": secret.metadata.labels or {},
            "owner": [
                o.kind + "/" + o.name for o in (secret.metadata.owner_references or [])
            ],
            "revealed": reveal,
        }
        if reveal:
            import base64

            result["data"] = {
                k: base64.b64decode(v).decode("utf-8", "replace")
                for k, v in data.items()
            }
        self.log_action(
            "get_secret", {"name": name, "reveal": reveal}, {"keys": len(data)}
        )
        return result

    # ------------------------------------------------------ ExternalSecrets
    def _external_secret(self, name: str, namespace: str) -> dict:
        return self.custom_objects.get_namespaced_custom_object(
            ESO_GROUP, ESO_VERSION, namespace, ESO_PLURAL, name
        )

    @staticmethod
    def _external_secret_summary(item: dict) -> dict:
        metadata = item.get("metadata") or {}
        spec = item.get("spec") or {}
        status = item.get("status") or {}
        ready = next(
            (c for c in status.get("conditions") or [] if c.get("type") == "Ready"), {}
        )
        return {
            "name": metadata.get("name"),
            "namespace": metadata.get("namespace"),
            "target": (spec.get("target") or {}).get("name") or metadata.get("name"),
            "store": (spec.get("secretStoreRef") or {}).get("name"),
            "refresh_interval": spec.get("refreshInterval"),
            "ready": ready.get("status"),
            "reason": ready.get("reason"),
            "message": ready.get("message"),
            "refresh_time": status.get("refreshTime"),
            "synced_resource_version": status.get("syncedResourceVersion"),
        }

    def list_external_secrets(
        self, namespace: str | None = None, all_namespaces: bool = False
    ) -> list[dict]:
        """List ExternalSecrets with their sync condition and last refresh time."""
        if all_namespaces:
            body = self.custom_objects.list_cluster_custom_object(
                ESO_GROUP, ESO_VERSION, ESO_PLURAL
            )
        else:
            body = self.custom_objects.list_namespaced_custom_object(
                ESO_GROUP, ESO_VERSION, namespace or self.namespace, ESO_PLURAL
            )
        result = [self._external_secret_summary(i) for i in body.get("items") or []]
        self.log_action(
            "list_external_secrets", {"all": all_namespaces}, {"count": len(result)}
        )
        return result

    def _target_digest(self, name: str, namespace: str) -> str | None:
        try:
            return _data_digest(self.core.read_namespaced_secret(name, namespace).data)
        except _km.ApiException as exc:
            if getattr(exc, "status", None) == 404:
                return None
            raise

    def _wait_for_refresh(
        self, name: str, ns: str, before: str | None, deadline: float
    ) -> dict:
        while True:
            summary = self._external_secret_summary(self._external_secret(name, ns))
            if summary["refresh_time"] != before and summary["ready"] == "True":
                return {**summary, "refreshed": True}
            if time.monotonic() >= deadline:
                return {**summary, "refreshed": False}
            time.sleep(2)

    def _wait_for_target(self, target: str, ns: str, deadline: float) -> str | None:
        while time.monotonic() < deadline:
            digest = self._target_digest(target, ns)
            if digest is not None:
                return digest
            time.sleep(2)
        return None

    def force_sync_external_secret(
        self,
        name: str,
        namespace: str | None = None,
        recreate_target: bool = False,
        timeout_seconds: int = 60,
    ) -> dict:
        """Force an ExternalSecret to re-read its store now, and prove it did.

        Sets the ``force-sync`` annotation to a new value, then waits until the
        controller reports a newer ``refreshTime`` with Ready=True. When the
        refresh does not happen in time, or ``recreate_target`` is true, the
        target Secret is deleted so the controller must recreate it from the
        store (the step that works when an annotation alone is ignored, e.g. a
        controller queue stuck behind failing objects). Reports the target
        Secret's data digest before and after; values are never returned.
        """
        ns = namespace or self.namespace
        timeout = max(5, min(int(timeout_seconds), MAX_WAIT_SECONDS))
        current = self._external_secret_summary(self._external_secret(name, ns))
        target = current["target"]
        digest_before = self._target_digest(target, ns)
        stamp = str(int(time.time()))
        self.custom_objects.patch_namespaced_custom_object(
            ESO_GROUP,
            ESO_VERSION,
            ns,
            ESO_PLURAL,
            name,
            {"metadata": {"annotations": {FORCE_SYNC: stamp}}},
        )
        deadline = time.monotonic() + timeout
        method = "annotation"
        outcome = self._wait_for_refresh(name, ns, current["refresh_time"], deadline)
        if recreate_target or not outcome["refreshed"]:
            method = "annotation+recreate-target"
            try:
                self.core.delete_namespaced_secret(target, ns)
            except _km.ApiException as exc:
                if getattr(exc, "status", None) != 404:
                    raise
            # Nudge the controller again so it notices the missing target.
            self.custom_objects.patch_namespaced_custom_object(
                ESO_GROUP,
                ESO_VERSION,
                ns,
                ESO_PLURAL,
                name,
                {"metadata": {"annotations": {FORCE_SYNC: f"{stamp}-recreate"}}},
            )
            deadline = time.monotonic() + timeout
            digest_after = self._wait_for_target(target, ns, deadline)
            outcome = self._external_secret_summary(self._external_secret(name, ns))
            outcome["refreshed"] = (
                digest_after is not None and outcome["ready"] == "True"
            )
        else:
            digest_after = self._target_digest(target, ns)
        result = {
            **outcome,
            "method": method,
            "target_present": digest_after is not None,
            "target_changed": digest_before != digest_after,
            "requested_at": _now(),
        }
        self.log_action(
            "force_sync_external_secret", {"name": name}, {"ok": result["refreshed"]}
        )
        return result

    # ------------------------------------------------------------ topology
    @staticmethod
    def _pv_node(pv) -> str | None:
        affinity = getattr(getattr(pv.spec, "node_affinity", None), "required", None)
        for term in getattr(affinity, "node_selector_terms", None) or []:
            for expr in term.match_expressions or []:
                if expr.key == "kubernetes.io/hostname" and expr.values:
                    return expr.values[0]
        return None

    def cluster_topology(self) -> dict:
        """Cluster topology and data locations: nodes, and where each volume's data lives.

        Each node lists its roles, readiness, capacity and labels; each
        PersistentVolume lists its claim, storage class, the node whose disk holds
        node-local data and its host path. Node hostnames equal the inventory host
        aliases, so a volume routes to the physical host that stores it.
        """
        nodes = []
        for node in self.core.list_node().items:
            labels = node.metadata.labels or {}
            ready = next(
                (c.status for c in (node.status.conditions or []) if c.type == "Ready"),
                "Unknown",
            )
            nodes.append(
                {
                    "name": node.metadata.name,
                    "roles": sorted(
                        k.split("/", 1)[1]
                        for k in labels
                        if k.startswith("node-role.kubernetes.io/")
                    ),
                    "ready": ready,
                    "unschedulable": bool(node.spec.unschedulable),
                    "capacity": dict(node.status.capacity or {}),
                    "allocatable": dict(node.status.allocatable or {}),
                    "architecture": labels.get("kubernetes.io/arch"),
                    "labels": labels,
                }
            )
        volumes = []
        for pv in self.core.list_persistent_volume().items:
            claim = pv.spec.claim_ref
            host_path = getattr(pv.spec.host_path, "path", None) or getattr(
                pv.spec.local, "path", None
            )
            volumes.append(
                {
                    "name": pv.metadata.name,
                    "claim": f"{claim.namespace}/{claim.name}" if claim else None,
                    "storage_class": pv.spec.storage_class_name,
                    "capacity": (pv.spec.capacity or {}).get("storage"),
                    "node": self._pv_node(pv),
                    "path": host_path,
                    "phase": getattr(pv.status, "phase", None),
                }
            )
        result = {
            "nodes": nodes,
            "persistent_volumes": volumes,
            "namespaces": sorted(
                ns.metadata.name for ns in self.core.list_namespace().items
            ),
        }
        self.log_action("cluster_topology", {}, {"nodes": len(nodes)})
        return result

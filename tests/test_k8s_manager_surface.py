"""Structural guards for the split ``KubernetesManager`` mixin subpackage.

These tests do not need a cluster or the kubernetes client to be functional —
they only introspect the composed class and its mixins.
"""

import collections
import inspect

import container_manager_mcp.k8s.manager as man
from container_manager_mcp.k8s_manager import KubernetesManager

MIXINS = [
    man.WorkloadsMixin,
    man.ConfigMixin,
    man.NetworkingMixin,
    man.StorageMixin,
    man.RbacMixin,
    man.ClusterNodesMixin,
    man.GovernanceMixin,
    man.ObservabilityMixin,
    man.ComposeMixin,
    man.InteropMixin,
    man.UnsupportedMixin,
    man._K8sBase,
]

# The new public methods added in the correctness pass, plus a representative
# slice of the deduped baseline surface that must survive the split.
NEW_METHODS = {
    "patch_resource",
    "list_native_services",
    "get_native_service",
    "create_native_service",
    "delete_native_service",
}
BASELINE_METHODS = {
    # Swarm-parity ABC surface
    "list_nodes",
    "inspect_node",
    "update_node",
    "remove_node",
    "list_services",
    "create_service",
    "scale_service",
    "update_service",
    "service_ps",
    "service_logs",
    "init_swarm",
    "leave_swarm",
    # deduped methods (one copy must remain)
    "exec_pod",
    "attach_pod",
    "port_forward_pod",
    "create_ingress",
    "list_persistent_volumes",
    "list_persistent_volume_claims",
    "create_persistent_volume_claim",
    "list_storage_classes",
    "list_volume_snapshots",
    "cordon_node",
    "drain_node",
    "taint_node",
    "cluster_info_dump",
    "list_jobs",
    "list_cron_jobs",
    "list_resource_quotas",
    "list_limit_ranges",
    "list_priority_classes",
    "list_pod_disruption_budgets",
    "list_horizontal_pod_autoscalers",
    # RBAC (repaired)
    "list_roles",
    "create_role",
    "auth_can_i",
    "create_aggregated_cluster_role",
    # label/annotate now delegate to patch_resource
    "label_resource",
    "annotate_resource",
}


def _public_methods(cls):
    return {n for n, _ in inspect.getmembers(cls, predicate=inspect.isfunction)}


def test_no_method_defined_in_more_than_one_mixin():
    owners = collections.defaultdict(list)
    for mx in MIXINS:
        for name, val in mx.__dict__.items():
            if name.startswith("__"):
                continue
            if inspect.isfunction(val) or isinstance(val, staticmethod):
                owners[name].append(mx.__name__)
    dupes = {n: o for n, o in owners.items() if len(o) > 1}
    assert not dupes, f"methods defined in multiple mixins: {dupes}"


def test_composed_class_has_no_silently_shadowed_methods():
    # Every public method resolves to exactly one defining mixin/base in the MRO.
    method_names = _public_methods(KubernetesManager)
    defining = collections.defaultdict(list)
    for name in method_names:
        for klass in KubernetesManager.__mro__:
            if name in klass.__dict__:
                defining[name].append(klass.__name__)
    # A name may legitimately live on ContainerManagerBase AND be overridden by
    # exactly one k8s mixin; assert no name is defined by two *k8s* mixins.
    k8s_owned = {mx.__name__ for mx in MIXINS}
    for name, klasses in defining.items():
        k8s_defs = [k for k in klasses if k in k8s_owned]
        assert len(k8s_defs) <= 1, f"{name} defined by multiple k8s mixins: {k8s_defs}"


def test_composed_class_exposes_expected_surface():
    surface = _public_methods(KubernetesManager)
    missing = (BASELINE_METHODS | NEW_METHODS) - surface
    assert not missing, f"missing expected methods: {sorted(missing)}"


def test_no_duplicate_public_methods_on_composed_class():
    names = [
        n
        for n, _ in inspect.getmembers(KubernetesManager, predicate=inspect.isfunction)
    ]
    dupes = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dupes, f"duplicate public methods: {dupes}"


def test_patch_point_symbols_are_module_attributes():
    import container_manager_mcp.k8s_manager as km

    assert hasattr(km, "k8s_client")
    assert hasattr(km, "k8s_config")
    assert hasattr(km, "ApiException")
    assert km.KubernetesManager is KubernetesManager

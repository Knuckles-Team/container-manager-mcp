"""Capability-parity matrix — the objective 100% coverage gate + regression guard.

This test introspects the ACTUAL registered MCP tools (built via
``get_mcp_instance``) and asserts that every action in an explicit
expected-capability matrix is present on the live tool's ``action`` enum. It is
what makes "100% docker / podman / kubernetes coverage incl. advanced use cases"
an objective, enforced property rather than a claim: if a themed tool loses an
action (a refactor drops a dispatch branch), this test FAILS.

Adding a capability = add its action to the matrix below (and to the tool). The
matrix is deliberately explicit and readable so the coverage surface is auditable
in one place.
"""

from __future__ import annotations

import asyncio
import sys
from functools import lru_cache

import pytest

from container_manager_mcp.mcp_server import get_mcp_instance


@lru_cache(maxsize=1)
def _tool_actions() -> dict[str, set[str]]:
    """Map every registered tool name → the set of actions its schema enumerates."""
    # get_mcp_instance() parses sys.argv for server flags (--port, etc.); isolate it
    # so the test is hermetic under any pytest invocation (e.g. `-p no:cacheprovider`).
    _saved_argv = sys.argv
    sys.argv = [_saved_argv[0] if _saved_argv else "container-manager-mcp"]
    try:
        _args, mcp, _mw = get_mcp_instance()
    finally:
        sys.argv = _saved_argv
    tools = asyncio.run(mcp.list_tools())
    out: dict[str, set[str]] = {}
    for tool in tools:
        schema = getattr(tool, "inputSchema", None) or getattr(tool, "parameters", {})
        action = (schema.get("properties") or {}).get("action") or {}
        enum = action.get("enum")
        if enum:
            out[tool.name] = set(enum)
    return out


# --- The 10 user-checklist actions that MUST exist (tool -> actions) ----------
CHECKLIST: dict[str, list[str]] = {
    "cm_k8s_workloads": ["list_pods", "describe_pod"],
    "cm_k8s_config": [
        "list_secrets",
        "create_secret",
        "list_namespaces",
        "patch_resource",
        "annotate_resource",
    ],
    "cm_k8s_networking": ["list_k8s_services", "list_ingress"],
    "cm_k8s_rbac": ["list_roles"],
}


# --- Full expected-capability matrix (the coverage gate) ----------------------
# KUBERNETES — the 8 themed cm_k8s_* tools' key actions (workloads/config/
# networking/storage/rbac/cluster/governance/observability), covering advanced ops.
KUBE_MATRIX: dict[str, list[str]] = {
    "cm_k8s_workloads": [
        "list_pods",
        "describe_pod",
        "exec_pod",
        "port_forward_pod",
        "list_statefulsets",
        "create_stateful_set",
        "scale_statefulset",
        "list_daemonsets",
        "create_daemon_set",
        "list_replicasets",
        "list_jobs",
        "create_job",
        "list_cron_jobs",
        "create_cron_job",
        "rollout_status",
        "rollout_restart",
        "rollout_undo",
    ],
    "cm_k8s_config": [
        "list_configmaps",
        "create_configmap",
        "list_secrets",
        "create_secret",
        "list_namespaces",
        "create_namespace",
        "delete_namespace",
        "list_events",
        "list_crds",
        "list_custom_resources",
        "label_resource",
        "annotate_resource",
        "patch_resource",
    ],
    "cm_k8s_networking": [
        "list_ingress",
        "create_ingress",
        "delete_ingress",
        "list_ingress_classes",
        "list_networkpolicies",
        "create_networkpolicy",
        "list_endpoints",
        "check_dns_resolution",
        "list_k8s_services",
        "get_k8s_service",
        "create_k8s_service",
        "delete_k8s_service",
    ],
    "cm_k8s_storage": [
        "list_persistent_volumes",
        "create_persistent_volume",
        "list_persistent_volume_claims",
        "create_persistent_volume_claim",
        "delete_persistent_volume_claim",
        "expand_pvc",
        "list_storage_classes",
        "create_storage_class",
        "list_volume_snapshots",
        "create_volume_snapshot",
        "list_csi_drivers",
    ],
    "cm_k8s_rbac": [
        "list_roles",
        "create_role",
        "delete_role",
        "list_cluster_roles",
        "list_rolebindings",
        "create_rolebinding",
        "list_cluster_rolebindings",
        "list_serviceaccounts",
        "create_serviceaccount",
        "auth_can_i",
        "subject_access_review",
        "list_pod_security_policies",
    ],
    "cm_k8s_cluster": [
        "list_nodes",
        "inspect_node",
        "cordon_node",
        "uncordon_node",
        "drain_node",
        "taint_node",
        "untaint_node",
        "list_contexts",
        "use_context",
        "get_config",
        "list_csr",
        "approve_csr",
        "list_api_resources",
    ],
    "cm_k8s_governance": [
        "list_resource_quotas",
        "create_resource_quota",
        "delete_resource_quota",
        "list_limit_ranges",
        "create_limit_range",
        "list_priority_classes",
        "create_priority_class",
        "list_pod_disruption_budgets",
        "create_pod_disruption_budget",
        "list_horizontal_pod_autoscalers",
        "create_horizontal_pod_autoscaler",
    ],
    "cm_k8s_observability": [
        "top_pods",
        "top_nodes",
        "get_pod_metrics",
        "get_node_metrics",
        "get_cluster_resource_summary",
        "watch_resource",
        "stream_pod_logs",
        "get_resource_events",
        "debug_pod",
        "debug_node",
        "debug_deployment",
    ],
}

# DOCKER — the full cm_docker_advanced surface (swarm / service / stack / config /
# secret / node). Enumerated from mcp/mcp_docker_advanced.py.
DOCKER_MATRIX: dict[str, list[str]] = {
    "cm_docker_advanced": [
        "docker_swarm_init",
        "docker_swarm_join",
        "docker_swarm_leave",
        "docker_service_create",
        "docker_service_list",
        "docker_service_update",
        "docker_service_rm",
        "docker_service_logs",
        "docker_service_ps",
        "docker_stack_deploy",
        "docker_stack_services",
        "docker_stack_rm",
        "docker_config_create",
        "docker_config_list",
        "docker_secret_create",
        "docker_secret_list",
        "docker_node_ls",
        "docker_node_update",
        "docker_node_inspect",
    ],
}

# PODMAN — the full cm_podman_advanced surface (kube gen/play, checkpoint/restore,
# pods, networks, volumes, system). Enumerated from mcp/mcp_podman_advanced.py.
PODMAN_MATRIX: dict[str, list[str]] = {
    "cm_podman_advanced": [
        "podman_generate_kube_yaml",
        "podman_play_kube_yaml",
        "podman_checkpoint",
        "podman_restore",
        "podman_pod_create",
        "podman_pod_list",
        "podman_pod_stats",
        "podman_pod_top",
        "podman_pod_inspect",
        "podman_pod_logs",
        "podman_pod_stop",
        "podman_pod_rm",
        "podman_network_create",
        "podman_network_list",
        "podman_network_inspect",
        "podman_volume_create",
        "podman_volume_list",
        "podman_volume_inspect",
        "podman_system_prune",
        "podman_health_check",
    ],
}

# The whole coverage surface across the three runtimes.
CAPABILITY_MATRIX: dict[str, list[str]] = {
    **KUBE_MATRIX,
    **DOCKER_MATRIX,
    **PODMAN_MATRIX,
}


def _cases() -> list[tuple[str, str]]:
    return [
        (tool, action)
        for tool, actions in CAPABILITY_MATRIX.items()
        for action in actions
    ]


def test_all_expected_tools_are_registered():
    """Every tool named in the matrix is actually registered with an action enum."""
    live = _tool_actions()
    missing = [t for t in CAPABILITY_MATRIX if t not in live]
    assert not missing, f"expected tools absent from the live surface: {missing}"


@pytest.mark.parametrize("tool,action", _cases(), ids=lambda v: v)
def test_capability_action_present(tool: str, action: str):
    """Coverage gate: each expected action must exist on the live tool's enum."""
    live = _tool_actions()
    assert tool in live, f"tool {tool!r} not registered"
    assert action in live[tool], (
        f"MISSING capability: {tool}.{action} — the tool no longer exposes this "
        f"action (coverage regression). Live actions: {sorted(live[tool])}"
    )


@pytest.mark.parametrize(
    "tool,action",
    [(t, a) for t, acts in CHECKLIST.items() for a in acts],
    ids=lambda v: v,
)
def test_user_checklist_action_present(tool: str, action: str):
    """The 10 user-checklist actions must all exist on their themed tools."""
    live = _tool_actions()
    assert action in live.get(tool, set()), f"checklist action absent: {tool}.{action}"


def test_matrix_guards_every_runtime():
    """Sanity: the matrix covers all three runtimes with a meaningful surface."""
    assert len(DOCKER_MATRIX["cm_docker_advanced"]) >= 15
    assert len(PODMAN_MATRIX["cm_podman_advanced"]) >= 15
    # all 8 themed kubernetes tools represented
    assert len([t for t in KUBE_MATRIX if t.startswith("cm_k8s_")]) == 8

"""Characterization tests for CXA-FL-CONTAINERMANAGERMCP-01's
register_k8srbac_tools.cm_k8s_rbac (CCN 73), before decomposition.

cm_k8s_rbac is a 29-branch action dispatcher (roles/clusterroles/bindings,
serviceaccounts, auth checks, ServiceAccount tokens, SubjectAccessReviews,
aggregated cluster roles, pod security policies, ServiceAccount-secret
mapping). For every action: which manager.<method> it calls and with
exactly which args (including the `json.loads(...)`-parsed variants for
create_role/create_rolebinding/create_cluster_rolebinding), each
required-param guard's exact error string with the manager left uncalled,
the unknown-action fallback, and that a manager exception is caught and
formatted as "Error executing {action}: {exc_type_name}".

Harness pattern (`_capture_tool`) matches the repo's own precedent in
tests/test_multi_context_manager.py: bypass the @mcp.tool decorator to get
the plain coroutine function, and monkeypatch module-level `create_manager`
to return a MagicMock so no real Kubernetes client is touched.

Guard tests pass the guarded parameter(s) explicitly as `None`: the raw
(undecorated) tool function's real Python defaults are FastMCP
`Field(default=None, ...)` sentinel objects, not `None` itself (FastMCP's
own decorator normally resolves those via pydantic before the call), so an
omitted guarded param would be a truthy FieldInfo object and the guard
would never fire when calling the bare function this way.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from container_manager_mcp.mcp import mcp_k8s_rbac


@pytest.fixture(autouse=True)
def _restore_create_manager():
    original = mcp_k8s_rbac.create_manager
    yield
    mcp_k8s_rbac.create_manager = original


def _capture_tool(register_fn):
    captured = {}

    def tool_decorator(*args, **kwargs):
        def wrapper(fn):
            captured["fn"] = fn
            return fn

        return wrapper

    fake_mcp = MagicMock()
    fake_mcp.tool = tool_decorator
    register_fn(fake_mcp)
    return captured["fn"]


def _make_tool():
    manager = MagicMock()
    mcp_k8s_rbac.create_manager = lambda manager_type=None: manager
    tool = _capture_tool(mcp_k8s_rbac.register_k8srbac_tools)
    return manager, tool


def test_cm_k8s_rbac_list_roles_dispatches_to_list_roles():
    manager, tool = _make_tool()
    manager.list_roles.return_value = "SENTINEL_RESULT_list_roles"
    result = asyncio.run(
        tool(
            action="list_roles",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_roles"
    manager.list_roles.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_rbac_create_role_dispatches_to_create_role():
    manager, tool = _make_tool()
    manager.create_role.return_value = "SENTINEL_RESULT_create_role"
    result = asyncio.run(
        tool(
            action="create_role",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_create_role"
    manager.create_role.assert_called_once_with(
        name="role_name_val",
        namespace="namespace_val",
        rules=[{"verbs": ["get"], "resources": ["pods"]}],
    )


def test_cm_k8s_rbac_create_role_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="create_role",
            role_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'role_name' is required for create_role"
    manager.create_role.assert_not_called()


def test_cm_k8s_rbac_delete_role_dispatches_to_delete_role():
    manager, tool = _make_tool()
    manager.delete_role.return_value = "SENTINEL_RESULT_delete_role"
    result = asyncio.run(
        tool(
            action="delete_role",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_delete_role"
    manager.delete_role.assert_called_once_with(
        name="role_name_val", namespace="namespace_val"
    )


def test_cm_k8s_rbac_delete_role_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="delete_role",
            role_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'role_name' is required for delete_role"
    manager.delete_role.assert_not_called()


def test_cm_k8s_rbac_list_cluster_roles_dispatches_to_list_cluster_roles():
    manager, tool = _make_tool()
    manager.list_cluster_roles.return_value = "SENTINEL_RESULT_list_cluster_roles"
    result = asyncio.run(
        tool(
            action="list_cluster_roles",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_cluster_roles"
    manager.list_cluster_roles.assert_called_once_with()


def test_cm_k8s_rbac_list_rolebindings_dispatches_to_list_rolebindings():
    manager, tool = _make_tool()
    manager.list_rolebindings.return_value = "SENTINEL_RESULT_list_rolebindings"
    result = asyncio.run(
        tool(
            action="list_rolebindings",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_rolebindings"
    manager.list_rolebindings.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_rbac_create_rolebinding_dispatches_to_create_rolebinding():
    manager, tool = _make_tool()
    manager.create_rolebinding.return_value = "SENTINEL_RESULT_create_rolebinding"
    result = asyncio.run(
        tool(
            action="create_rolebinding",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_create_rolebinding"
    manager.create_rolebinding.assert_called_once_with(
        name="rolebinding_name_val",
        namespace="namespace_val",
        role_ref={"kind": "Role", "name": "r1"},
        subjects=[{"kind": "User", "name": "u1"}],
    )


def test_cm_k8s_rbac_create_rolebinding_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="create_rolebinding",
            rolebinding_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'rolebinding_name' is required for create_rolebinding"
    manager.create_rolebinding.assert_not_called()


def test_cm_k8s_rbac_delete_rolebinding_dispatches_to_delete_rolebinding():
    manager, tool = _make_tool()
    manager.delete_rolebinding.return_value = "SENTINEL_RESULT_delete_rolebinding"
    result = asyncio.run(
        tool(
            action="delete_rolebinding",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_delete_rolebinding"
    manager.delete_rolebinding.assert_called_once_with(
        name="rolebinding_name_val", namespace="namespace_val"
    )


def test_cm_k8s_rbac_delete_rolebinding_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="delete_rolebinding",
            rolebinding_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'rolebinding_name' is required for delete_rolebinding"
    manager.delete_rolebinding.assert_not_called()


def test_cm_k8s_rbac_list_cluster_rolebindings_dispatches_to_list_cluster_rolebindings():
    manager, tool = _make_tool()
    manager.list_cluster_rolebindings.return_value = (
        "SENTINEL_RESULT_list_cluster_rolebindings"
    )
    result = asyncio.run(
        tool(
            action="list_cluster_rolebindings",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_cluster_rolebindings"
    manager.list_cluster_rolebindings.assert_called_once_with()


def test_cm_k8s_rbac_create_cluster_rolebinding_dispatches_to_create_cluster_rolebinding():
    manager, tool = _make_tool()
    manager.create_cluster_rolebinding.return_value = (
        "SENTINEL_RESULT_create_cluster_rolebinding"
    )
    result = asyncio.run(
        tool(
            action="create_cluster_rolebinding",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_create_cluster_rolebinding"
    manager.create_cluster_rolebinding.assert_called_once_with(
        name="rolebinding_name_val",
        role_ref={"kind": "Role", "name": "r1"},
        subjects=[{"kind": "User", "name": "u1"}],
    )


def test_cm_k8s_rbac_create_cluster_rolebinding_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="create_cluster_rolebinding",
            rolebinding_name=None,
            ctx=None,
        )
    )
    assert (
        result == "Error: 'rolebinding_name' is required for create_cluster_rolebinding"
    )
    manager.create_cluster_rolebinding.assert_not_called()


def test_cm_k8s_rbac_delete_cluster_rolebinding_dispatches_to_delete_cluster_rolebinding():
    manager, tool = _make_tool()
    manager.delete_cluster_rolebinding.return_value = (
        "SENTINEL_RESULT_delete_cluster_rolebinding"
    )
    result = asyncio.run(
        tool(
            action="delete_cluster_rolebinding",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_delete_cluster_rolebinding"
    manager.delete_cluster_rolebinding.assert_called_once_with(
        name="rolebinding_name_val"
    )


def test_cm_k8s_rbac_delete_cluster_rolebinding_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="delete_cluster_rolebinding",
            rolebinding_name=None,
            ctx=None,
        )
    )
    assert (
        result == "Error: 'rolebinding_name' is required for delete_cluster_rolebinding"
    )
    manager.delete_cluster_rolebinding.assert_not_called()


def test_cm_k8s_rbac_list_serviceaccounts_dispatches_to_list_serviceaccounts():
    manager, tool = _make_tool()
    manager.list_serviceaccounts.return_value = "SENTINEL_RESULT_list_serviceaccounts"
    result = asyncio.run(
        tool(
            action="list_serviceaccounts",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_serviceaccounts"
    manager.list_serviceaccounts.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_rbac_create_serviceaccount_dispatches_to_create_serviceaccount():
    manager, tool = _make_tool()
    manager.create_serviceaccount.return_value = "SENTINEL_RESULT_create_serviceaccount"
    result = asyncio.run(
        tool(
            action="create_serviceaccount",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_create_serviceaccount"
    manager.create_serviceaccount.assert_called_once_with(
        name="serviceaccount_name_val", namespace="namespace_val"
    )


def test_cm_k8s_rbac_create_serviceaccount_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="create_serviceaccount",
            serviceaccount_name=None,
            ctx=None,
        )
    )
    assert (
        result == "Error: 'serviceaccount_name' is required for create_serviceaccount"
    )
    manager.create_serviceaccount.assert_not_called()


def test_cm_k8s_rbac_delete_serviceaccount_dispatches_to_delete_serviceaccount():
    manager, tool = _make_tool()
    manager.delete_serviceaccount.return_value = "SENTINEL_RESULT_delete_serviceaccount"
    result = asyncio.run(
        tool(
            action="delete_serviceaccount",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_delete_serviceaccount"
    manager.delete_serviceaccount.assert_called_once_with(
        name="serviceaccount_name_val", namespace="namespace_val"
    )


def test_cm_k8s_rbac_delete_serviceaccount_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="delete_serviceaccount",
            serviceaccount_name=None,
            ctx=None,
        )
    )
    assert (
        result == "Error: 'serviceaccount_name' is required for delete_serviceaccount"
    )
    manager.delete_serviceaccount.assert_not_called()


def test_cm_k8s_rbac_auth_can_i_dispatches_to_auth_can_i():
    manager, tool = _make_tool()
    manager.auth_can_i.return_value = "SENTINEL_RESULT_auth_can_i"
    result = asyncio.run(
        tool(
            action="auth_can_i",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_auth_can_i"
    manager.auth_can_i.assert_called_once_with(
        verb="auth_verb_val", resource="auth_resource_val", namespace="namespace_val"
    )


def test_cm_k8s_rbac_auth_can_i_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="auth_can_i",
            auth_resource=None,
            auth_verb=None,
            ctx=None,
        )
    )
    assert (
        result == "Error: 'auth_verb' and 'auth_resource' are required for auth_can_i"
    )
    manager.auth_can_i.assert_not_called()


def test_cm_k8s_rbac_create_service_account_token_dispatches_to_create_service_account_token():
    manager, tool = _make_tool()
    manager.create_service_account_token.return_value = (
        "SENTINEL_RESULT_create_service_account_token"
    )
    result = asyncio.run(
        tool(
            action="create_service_account_token",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_create_service_account_token"
    manager.create_service_account_token.assert_called_once_with(
        "name_val", "namespace_val", {"spec_k": "spec_v"}
    )


def test_cm_k8s_rbac_create_service_account_token_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="create_service_account_token",
            name=None,
            spec=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'name' and 'spec' are required for create_service_account_token"
    )
    manager.create_service_account_token.assert_not_called()


def test_cm_k8s_rbac_list_service_account_tokens_dispatches_to_list_service_account_tokens():
    manager, tool = _make_tool()
    manager.list_service_account_tokens.return_value = (
        "SENTINEL_RESULT_list_service_account_tokens"
    )
    result = asyncio.run(
        tool(
            action="list_service_account_tokens",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_service_account_tokens"
    manager.list_service_account_tokens.assert_called_once_with(
        "name_val", "namespace_val"
    )


def test_cm_k8s_rbac_list_service_account_tokens_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="list_service_account_tokens",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for list_service_account_tokens"
    manager.list_service_account_tokens.assert_not_called()


def test_cm_k8s_rbac_delete_service_account_token_dispatches_to_delete_service_account_token():
    manager, tool = _make_tool()
    manager.delete_service_account_token.return_value = (
        "SENTINEL_RESULT_delete_service_account_token"
    )
    result = asyncio.run(
        tool(
            action="delete_service_account_token",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_delete_service_account_token"
    manager.delete_service_account_token.assert_called_once_with(
        "name_val", "namespace_val", "token_name_val"
    )


def test_cm_k8s_rbac_delete_service_account_token_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="delete_service_account_token",
            name=None,
            token_name=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'name' and 'token_name' are required for delete_service_account_token"
    )
    manager.delete_service_account_token.assert_not_called()


def test_cm_k8s_rbac_subject_access_review_dispatches_to_subject_access_review():
    manager, tool = _make_tool()
    manager.subject_access_review.return_value = "SENTINEL_RESULT_subject_access_review"
    result = asyncio.run(
        tool(
            action="subject_access_review",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_subject_access_review"
    manager.subject_access_review.assert_called_once_with({"spec_k": "spec_v"})


def test_cm_k8s_rbac_subject_access_review_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="subject_access_review",
            spec=None,
            ctx=None,
        )
    )
    assert result == "Error: 'spec' is required for subject_access_review"
    manager.subject_access_review.assert_not_called()


def test_cm_k8s_rbac_local_subject_access_review_dispatches_to_local_subject_access_review():
    manager, tool = _make_tool()
    manager.local_subject_access_review.return_value = (
        "SENTINEL_RESULT_local_subject_access_review"
    )
    result = asyncio.run(
        tool(
            action="local_subject_access_review",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_local_subject_access_review"
    manager.local_subject_access_review.assert_called_once_with(
        "namespace_val", {"spec_k": "spec_v"}
    )


def test_cm_k8s_rbac_local_subject_access_review_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="local_subject_access_review",
            namespace=None,
            spec=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'namespace' and 'spec' are required for local_subject_access_review"
    )
    manager.local_subject_access_review.assert_not_called()


def test_cm_k8s_rbac_create_aggregated_cluster_role_dispatches_to_create_aggregated_cluster_role():
    manager, tool = _make_tool()
    manager.create_aggregated_cluster_role.return_value = (
        "SENTINEL_RESULT_create_aggregated_cluster_role"
    )
    result = asyncio.run(
        tool(
            action="create_aggregated_cluster_role",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_create_aggregated_cluster_role"
    manager.create_aggregated_cluster_role.assert_called_once_with(
        "name_val", {"aggregation_rule_k": "aggregation_rule_v"}
    )


def test_cm_k8s_rbac_create_aggregated_cluster_role_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="create_aggregated_cluster_role",
            aggregation_rule=None,
            name=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'name' and 'aggregation_rule' are required for create_aggregated_cluster_role"
    )
    manager.create_aggregated_cluster_role.assert_not_called()


def test_cm_k8s_rbac_update_aggregated_cluster_role_dispatches_to_update_aggregated_cluster_role():
    manager, tool = _make_tool()
    manager.update_aggregated_cluster_role.return_value = (
        "SENTINEL_RESULT_update_aggregated_cluster_role"
    )
    result = asyncio.run(
        tool(
            action="update_aggregated_cluster_role",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_update_aggregated_cluster_role"
    manager.update_aggregated_cluster_role.assert_called_once_with(
        "name_val", {"aggregation_rule_k": "aggregation_rule_v"}
    )


def test_cm_k8s_rbac_update_aggregated_cluster_role_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="update_aggregated_cluster_role",
            aggregation_rule=None,
            name=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'name' and 'aggregation_rule' are required for update_aggregated_cluster_role"
    )
    manager.update_aggregated_cluster_role.assert_not_called()


def test_cm_k8s_rbac_list_pod_security_policies_dispatches_to_list_pod_security_policies():
    manager, tool = _make_tool()
    manager.list_pod_security_policies.return_value = (
        "SENTINEL_RESULT_list_pod_security_policies"
    )
    result = asyncio.run(
        tool(
            action="list_pod_security_policies",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_pod_security_policies"
    manager.list_pod_security_policies.assert_called_once_with()


def test_cm_k8s_rbac_describe_pod_security_policy_dispatches_to_describe_pod_security_policy():
    manager, tool = _make_tool()
    manager.describe_pod_security_policy.return_value = (
        "SENTINEL_RESULT_describe_pod_security_policy"
    )
    result = asyncio.run(
        tool(
            action="describe_pod_security_policy",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_describe_pod_security_policy"
    manager.describe_pod_security_policy.assert_called_once_with("name_val")


def test_cm_k8s_rbac_describe_pod_security_policy_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="describe_pod_security_policy",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for describe_pod_security_policy"
    manager.describe_pod_security_policy.assert_not_called()


def test_cm_k8s_rbac_create_pod_security_policy_dispatches_to_create_pod_security_policy():
    manager, tool = _make_tool()
    manager.create_pod_security_policy.return_value = (
        "SENTINEL_RESULT_create_pod_security_policy"
    )
    result = asyncio.run(
        tool(
            action="create_pod_security_policy",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_create_pod_security_policy"
    manager.create_pod_security_policy.assert_called_once_with(
        "name_val", {"spec_k": "spec_v"}
    )


def test_cm_k8s_rbac_create_pod_security_policy_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="create_pod_security_policy",
            name=None,
            spec=None,
            ctx=None,
        )
    )
    assert (
        result == "Error: 'name' and 'spec' are required for create_pod_security_policy"
    )
    manager.create_pod_security_policy.assert_not_called()


def test_cm_k8s_rbac_delete_pod_security_policy_dispatches_to_delete_pod_security_policy():
    manager, tool = _make_tool()
    manager.delete_pod_security_policy.return_value = (
        "SENTINEL_RESULT_delete_pod_security_policy"
    )
    result = asyncio.run(
        tool(
            action="delete_pod_security_policy",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_delete_pod_security_policy"
    manager.delete_pod_security_policy.assert_called_once_with("name_val")


def test_cm_k8s_rbac_delete_pod_security_policy_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="delete_pod_security_policy",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for delete_pod_security_policy"
    manager.delete_pod_security_policy.assert_not_called()


def test_cm_k8s_rbac_evaluate_pod_security_dispatches_to_evaluate_pod_security():
    manager, tool = _make_tool()
    manager.evaluate_pod_security.return_value = "SENTINEL_RESULT_evaluate_pod_security"
    result = asyncio.run(
        tool(
            action="evaluate_pod_security",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_evaluate_pod_security"
    manager.evaluate_pod_security.assert_called_once_with(
        "namespace_val", {"pod_spec_k": "pod_spec_v"}
    )


def test_cm_k8s_rbac_evaluate_pod_security_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="evaluate_pod_security",
            namespace=None,
            pod_spec=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'namespace' and 'pod_spec' are required for evaluate_pod_security"
    )
    manager.evaluate_pod_security.assert_not_called()


def test_cm_k8s_rbac_list_service_account_mapped_secrets_dispatches_to_list_service_account_mapped_secrets():
    manager, tool = _make_tool()
    manager.list_service_account_mapped_secrets.return_value = (
        "SENTINEL_RESULT_list_service_account_mapped_secrets"
    )
    result = asyncio.run(
        tool(
            action="list_service_account_mapped_secrets",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_service_account_mapped_secrets"
    manager.list_service_account_mapped_secrets.assert_called_once_with(
        "name_val", "namespace_val"
    )


def test_cm_k8s_rbac_list_service_account_mapped_secrets_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="list_service_account_mapped_secrets",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for list_service_account_mapped_secrets"
    manager.list_service_account_mapped_secrets.assert_not_called()


def test_cm_k8s_rbac_map_secret_to_service_account_dispatches_to_map_secret_to_service_account():
    manager, tool = _make_tool()
    manager.map_secret_to_service_account.return_value = (
        "SENTINEL_RESULT_map_secret_to_service_account"
    )
    result = asyncio.run(
        tool(
            action="map_secret_to_service_account",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_map_secret_to_service_account"
    manager.map_secret_to_service_account.assert_called_once_with(
        "secret_name_val", "sa_name_val", "namespace_val"
    )


def test_cm_k8s_rbac_map_secret_to_service_account_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="map_secret_to_service_account",
            sa_name=None,
            secret_name=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'secret_name' and 'sa_name' are required for map_secret_to_service_account"
    )
    manager.map_secret_to_service_account.assert_not_called()


def test_cm_k8s_rbac_unmap_secret_from_service_account_dispatches_to_unmap_secret_from_service_account():
    manager, tool = _make_tool()
    manager.unmap_secret_from_service_account.return_value = (
        "SENTINEL_RESULT_unmap_secret_from_service_account"
    )
    result = asyncio.run(
        tool(
            action="unmap_secret_from_service_account",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_unmap_secret_from_service_account"
    manager.unmap_secret_from_service_account.assert_called_once_with(
        "secret_name_val", "sa_name_val", "namespace_val"
    )


def test_cm_k8s_rbac_unmap_secret_from_service_account_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="unmap_secret_from_service_account",
            sa_name=None,
            secret_name=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'secret_name' and 'sa_name' are required for unmap_secret_from_service_account"
    )
    manager.unmap_secret_from_service_account.assert_not_called()


def test_cm_k8s_rbac_unknown_action_returns_error():
    manager, tool = _make_tool()
    result = asyncio.run(tool(action="bogus_action_xyz", ctx=None))
    assert result == "Error: Unknown action 'bogus_action_xyz'"


def test_cm_k8s_rbac_manager_exception_is_caught_and_formatted():
    manager, tool = _make_tool()
    manager.list_roles.side_effect = RuntimeError("boom")
    result = asyncio.run(
        tool(
            action="list_roles",
            namespace="namespace_val",
            role_name="role_name_val",
            role_rules='[{"verbs": ["get"], "resources": ["pods"]}]',
            rolebinding_name="rolebinding_name_val",
            role_ref='{"kind": "Role", "name": "r1"}',
            subjects='[{"kind": "User", "name": "u1"}]',
            serviceaccount_name="serviceaccount_name_val",
            auth_verb="auth_verb_val",
            auth_resource="auth_resource_val",
            name="name_val",
            spec={"spec_k": "spec_v"},
            token_name="token_name_val",
            aggregation_rule={"aggregation_rule_k": "aggregation_rule_v"},
            pod_spec={"pod_spec_k": "pod_spec_v"},
            secret_name="secret_name_val",
            sa_name="sa_name_val",
            ctx=None,
        )
    )
    assert result == "Error executing list_roles: RuntimeError"

"""Characterization tests for CXA-FL-CONTAINERMANAGERMCP-02's
register_k8sgovernance_tools.cm_k8s_governance (CCN 52), before
decomposition.

cm_k8s_governance is a 22-branch action dispatcher (ResourceQuotas,
LimitRanges, PriorityClasses, PodDisruptionBudgets,
HorizontalPodAutoscalers) -- five near-identical list/describe/create/
[update/]delete CRUD groups. For every action: which manager.<method> it
calls and with exactly which args (list actions use `namespace` directly,
the rest use the `ns = namespace or getattr(manager, "namespace",
namespace)` fallback), each required-param guard's exact error string
with the manager left uncalled, the unknown-action fallback, and
manager-exception formatting.

Harness pattern (`_capture_tool`) matches the repo's own precedent in
tests/test_multi_context_manager.py. Guard tests pass the guarded
parameter(s) explicitly as `None` (the bare function's real defaults are
FastMCP FieldInfo objects, not None).
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from container_manager_mcp.mcp import mcp_k8s_governance


@pytest.fixture(autouse=True)
def _restore_create_manager():
    original = mcp_k8s_governance.create_manager
    yield
    mcp_k8s_governance.create_manager = original


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
    mcp_k8s_governance.create_manager = lambda manager_type=None: manager
    tool = _capture_tool(mcp_k8s_governance.register_k8sgovernance_tools)
    return manager, tool


def test_cm_k8s_governance_list_resource_quotas_dispatches_to_list_resource_quotas():
    manager, tool = _make_tool()
    manager.list_resource_quotas.return_value = "SENTINEL_RESULT_list_resource_quotas"
    result = asyncio.run(tool(
        action="list_resource_quotas",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_resource_quotas"
    manager.list_resource_quotas.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_governance_describe_resource_quota_dispatches_to_describe_resource_quota():
    manager, tool = _make_tool()
    manager.describe_resource_quota.return_value = "SENTINEL_RESULT_describe_resource_quota"
    result = asyncio.run(tool(
        action="describe_resource_quota",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_describe_resource_quota"
    manager.describe_resource_quota.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_governance_describe_resource_quota_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="describe_resource_quota",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for describe_resource_quota"
    manager.describe_resource_quota.assert_not_called()


def test_cm_k8s_governance_create_resource_quota_dispatches_to_create_resource_quota():
    manager, tool = _make_tool()
    manager.create_resource_quota.return_value = "SENTINEL_RESULT_create_resource_quota"
    result = asyncio.run(tool(
        action="create_resource_quota",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_resource_quota"
    manager.create_resource_quota.assert_called_once_with("name_val", "namespace_val", {"spec_k": "spec_v"})


def test_cm_k8s_governance_create_resource_quota_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_resource_quota",
        name=None, spec=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'spec' are required for create_resource_quota"
    manager.create_resource_quota.assert_not_called()


def test_cm_k8s_governance_update_resource_quota_dispatches_to_update_resource_quota():
    manager, tool = _make_tool()
    manager.update_resource_quota.return_value = "SENTINEL_RESULT_update_resource_quota"
    result = asyncio.run(tool(
        action="update_resource_quota",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_update_resource_quota"
    manager.update_resource_quota.assert_called_once_with("name_val", "namespace_val", {"spec_k": "spec_v"})


def test_cm_k8s_governance_update_resource_quota_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="update_resource_quota",
        name=None, spec=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'spec' are required for update_resource_quota"
    manager.update_resource_quota.assert_not_called()


def test_cm_k8s_governance_delete_resource_quota_dispatches_to_delete_resource_quota():
    manager, tool = _make_tool()
    manager.delete_resource_quota.return_value = "SENTINEL_RESULT_delete_resource_quota"
    result = asyncio.run(tool(
        action="delete_resource_quota",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_delete_resource_quota"
    manager.delete_resource_quota.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_governance_delete_resource_quota_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="delete_resource_quota",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for delete_resource_quota"
    manager.delete_resource_quota.assert_not_called()


def test_cm_k8s_governance_list_limit_ranges_dispatches_to_list_limit_ranges():
    manager, tool = _make_tool()
    manager.list_limit_ranges.return_value = "SENTINEL_RESULT_list_limit_ranges"
    result = asyncio.run(tool(
        action="list_limit_ranges",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_limit_ranges"
    manager.list_limit_ranges.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_governance_describe_limit_range_dispatches_to_describe_limit_range():
    manager, tool = _make_tool()
    manager.describe_limit_range.return_value = "SENTINEL_RESULT_describe_limit_range"
    result = asyncio.run(tool(
        action="describe_limit_range",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_describe_limit_range"
    manager.describe_limit_range.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_governance_describe_limit_range_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="describe_limit_range",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for describe_limit_range"
    manager.describe_limit_range.assert_not_called()


def test_cm_k8s_governance_create_limit_range_dispatches_to_create_limit_range():
    manager, tool = _make_tool()
    manager.create_limit_range.return_value = "SENTINEL_RESULT_create_limit_range"
    result = asyncio.run(tool(
        action="create_limit_range",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_limit_range"
    manager.create_limit_range.assert_called_once_with("name_val", "namespace_val", {"spec_k": "spec_v"})


def test_cm_k8s_governance_create_limit_range_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_limit_range",
        name=None, spec=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'spec' are required for create_limit_range"
    manager.create_limit_range.assert_not_called()


def test_cm_k8s_governance_delete_limit_range_dispatches_to_delete_limit_range():
    manager, tool = _make_tool()
    manager.delete_limit_range.return_value = "SENTINEL_RESULT_delete_limit_range"
    result = asyncio.run(tool(
        action="delete_limit_range",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_delete_limit_range"
    manager.delete_limit_range.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_governance_delete_limit_range_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="delete_limit_range",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for delete_limit_range"
    manager.delete_limit_range.assert_not_called()


def test_cm_k8s_governance_list_priority_classes_dispatches_to_list_priority_classes():
    manager, tool = _make_tool()
    manager.list_priority_classes.return_value = "SENTINEL_RESULT_list_priority_classes"
    result = asyncio.run(tool(
        action="list_priority_classes",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_priority_classes"
    manager.list_priority_classes.assert_called_once_with()


def test_cm_k8s_governance_describe_priority_class_dispatches_to_describe_priority_class():
    manager, tool = _make_tool()
    manager.describe_priority_class.return_value = "SENTINEL_RESULT_describe_priority_class"
    result = asyncio.run(tool(
        action="describe_priority_class",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_describe_priority_class"
    manager.describe_priority_class.assert_called_once_with("name_val")


def test_cm_k8s_governance_describe_priority_class_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="describe_priority_class",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for describe_priority_class"
    manager.describe_priority_class.assert_not_called()


def test_cm_k8s_governance_create_priority_class_dispatches_to_create_priority_class():
    manager, tool = _make_tool()
    manager.create_priority_class.return_value = "SENTINEL_RESULT_create_priority_class"
    result = asyncio.run(tool(
        action="create_priority_class",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_priority_class"
    manager.create_priority_class.assert_called_once_with("name_val", {"spec_k": "spec_v"})


def test_cm_k8s_governance_create_priority_class_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_priority_class",
        name=None, spec=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'spec' are required for create_priority_class"
    manager.create_priority_class.assert_not_called()


def test_cm_k8s_governance_delete_priority_class_dispatches_to_delete_priority_class():
    manager, tool = _make_tool()
    manager.delete_priority_class.return_value = "SENTINEL_RESULT_delete_priority_class"
    result = asyncio.run(tool(
        action="delete_priority_class",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_delete_priority_class"
    manager.delete_priority_class.assert_called_once_with("name_val")


def test_cm_k8s_governance_delete_priority_class_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="delete_priority_class",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for delete_priority_class"
    manager.delete_priority_class.assert_not_called()


def test_cm_k8s_governance_list_pod_disruption_budgets_dispatches_to_list_pod_disruption_budgets():
    manager, tool = _make_tool()
    manager.list_pod_disruption_budgets.return_value = "SENTINEL_RESULT_list_pod_disruption_budgets"
    result = asyncio.run(tool(
        action="list_pod_disruption_budgets",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_pod_disruption_budgets"
    manager.list_pod_disruption_budgets.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_governance_describe_pod_disruption_budget_dispatches_to_describe_pod_disruption_budget():
    manager, tool = _make_tool()
    manager.describe_pod_disruption_budget.return_value = "SENTINEL_RESULT_describe_pod_disruption_budget"
    result = asyncio.run(tool(
        action="describe_pod_disruption_budget",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_describe_pod_disruption_budget"
    manager.describe_pod_disruption_budget.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_governance_describe_pod_disruption_budget_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="describe_pod_disruption_budget",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for describe_pod_disruption_budget"
    manager.describe_pod_disruption_budget.assert_not_called()


def test_cm_k8s_governance_create_pod_disruption_budget_dispatches_to_create_pod_disruption_budget():
    manager, tool = _make_tool()
    manager.create_pod_disruption_budget.return_value = "SENTINEL_RESULT_create_pod_disruption_budget"
    result = asyncio.run(tool(
        action="create_pod_disruption_budget",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_pod_disruption_budget"
    manager.create_pod_disruption_budget.assert_called_once_with("name_val", "namespace_val", {"spec_k": "spec_v"})


def test_cm_k8s_governance_create_pod_disruption_budget_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_pod_disruption_budget",
        name=None, spec=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'spec' are required for create_pod_disruption_budget"
    manager.create_pod_disruption_budget.assert_not_called()


def test_cm_k8s_governance_delete_pod_disruption_budget_dispatches_to_delete_pod_disruption_budget():
    manager, tool = _make_tool()
    manager.delete_pod_disruption_budget.return_value = "SENTINEL_RESULT_delete_pod_disruption_budget"
    result = asyncio.run(tool(
        action="delete_pod_disruption_budget",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_delete_pod_disruption_budget"
    manager.delete_pod_disruption_budget.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_governance_delete_pod_disruption_budget_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="delete_pod_disruption_budget",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for delete_pod_disruption_budget"
    manager.delete_pod_disruption_budget.assert_not_called()


def test_cm_k8s_governance_list_horizontal_pod_autoscalers_dispatches_to_list_horizontal_pod_autoscalers():
    manager, tool = _make_tool()
    manager.list_horizontal_pod_autoscalers.return_value = "SENTINEL_RESULT_list_horizontal_pod_autoscalers"
    result = asyncio.run(tool(
        action="list_horizontal_pod_autoscalers",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_horizontal_pod_autoscalers"
    manager.list_horizontal_pod_autoscalers.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_governance_describe_horizontal_pod_autoscaler_dispatches_to_describe_horizontal_pod_autoscaler():
    manager, tool = _make_tool()
    manager.describe_horizontal_pod_autoscaler.return_value = "SENTINEL_RESULT_describe_horizontal_pod_autoscaler"
    result = asyncio.run(tool(
        action="describe_horizontal_pod_autoscaler",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_describe_horizontal_pod_autoscaler"
    manager.describe_horizontal_pod_autoscaler.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_governance_describe_horizontal_pod_autoscaler_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="describe_horizontal_pod_autoscaler",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for describe_horizontal_pod_autoscaler"
    manager.describe_horizontal_pod_autoscaler.assert_not_called()


def test_cm_k8s_governance_create_horizontal_pod_autoscaler_dispatches_to_create_horizontal_pod_autoscaler():
    manager, tool = _make_tool()
    manager.create_horizontal_pod_autoscaler.return_value = "SENTINEL_RESULT_create_horizontal_pod_autoscaler"
    result = asyncio.run(tool(
        action="create_horizontal_pod_autoscaler",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_horizontal_pod_autoscaler"
    manager.create_horizontal_pod_autoscaler.assert_called_once_with("name_val", "namespace_val", {"spec_k": "spec_v"})


def test_cm_k8s_governance_create_horizontal_pod_autoscaler_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_horizontal_pod_autoscaler",
        name=None, spec=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'spec' are required for create_horizontal_pod_autoscaler"
    manager.create_horizontal_pod_autoscaler.assert_not_called()


def test_cm_k8s_governance_update_horizontal_pod_autoscaler_dispatches_to_update_horizontal_pod_autoscaler():
    manager, tool = _make_tool()
    manager.update_horizontal_pod_autoscaler.return_value = "SENTINEL_RESULT_update_horizontal_pod_autoscaler"
    result = asyncio.run(tool(
        action="update_horizontal_pod_autoscaler",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_update_horizontal_pod_autoscaler"
    manager.update_horizontal_pod_autoscaler.assert_called_once_with("name_val", "namespace_val", {"spec_k": "spec_v"})


def test_cm_k8s_governance_update_horizontal_pod_autoscaler_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="update_horizontal_pod_autoscaler",
        name=None, spec=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'spec' are required for update_horizontal_pod_autoscaler"
    manager.update_horizontal_pod_autoscaler.assert_not_called()


def test_cm_k8s_governance_delete_horizontal_pod_autoscaler_dispatches_to_delete_horizontal_pod_autoscaler():
    manager, tool = _make_tool()
    manager.delete_horizontal_pod_autoscaler.return_value = "SENTINEL_RESULT_delete_horizontal_pod_autoscaler"
    result = asyncio.run(tool(
        action="delete_horizontal_pod_autoscaler",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_delete_horizontal_pod_autoscaler"
    manager.delete_horizontal_pod_autoscaler.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_governance_delete_horizontal_pod_autoscaler_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="delete_horizontal_pod_autoscaler",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for delete_horizontal_pod_autoscaler"
    manager.delete_horizontal_pod_autoscaler.assert_not_called()


def test_cm_k8s_governance_unknown_action_returns_error():
    manager, tool = _make_tool()
    result = asyncio.run(tool(action="bogus_action_xyz", ctx=None))
    assert result == "Error: Unknown action 'bogus_action_xyz'"


def test_cm_k8s_governance_manager_exception_is_caught_and_formatted():
    manager, tool = _make_tool()
    manager.list_resource_quotas.side_effect = RuntimeError("boom")
    result = asyncio.run(tool(
        action="list_resource_quotas",
        name="name_val", namespace="namespace_val", spec={"spec_k": "spec_v"},
        ctx=None,
    ))
    assert result == "Error executing list_resource_quotas: RuntimeError"


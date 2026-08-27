"""Characterization tests for CXA-FL-CONTAINERMANAGERMCP-01's
register_k8sworkloads_tools.cm_k8s_workloads (CCN 93), before decomposition.

cm_k8s_workloads is a 35-branch action dispatcher: for each `action` literal
it validates required params (a guard `if not X: return "Error: ..."`), then
calls exactly one `manager.<method>(...)` via `run_blocking` and returns its
result. This pins, per action: which manager method it calls, with exactly
which positional/keyword arguments, each guard's exact error string and that
the manager is not called when a guard fires, the unknown-action fallback,
and that a manager exception is caught and formatted as
`"Error executing {action}: {exc_type_name}"`.

Harness pattern (`_capture_tool`) matches the repo's own precedent in
tests/test_multi_context_manager.py: bypass the @mcp.tool decorator to get
the plain coroutine function, and monkeypatch module-level `create_manager`
to return a MagicMock so no real Kubernetes client is touched.

Guard tests pass the guarded parameter(s) explicitly as `None`: the raw
(undecorated) tool function's real Python defaults are FastMCP
`Field(default=None, ...)` sentinel objects, not `None` itself (FastMCP's
own decorator normally resolves those via pydantic before the call), so an
omitted guarded param would be a truthy FieldInfo object and the guard would
never fire when calling the bare function this way.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from container_manager_mcp.mcp import mcp_k8s_workloads


@pytest.fixture(autouse=True)
def _restore_create_manager():
    original = mcp_k8s_workloads.create_manager
    yield
    mcp_k8s_workloads.create_manager = original


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
    mcp_k8s_workloads.create_manager = lambda manager_type=None: manager
    tool = _capture_tool(mcp_k8s_workloads.register_k8sworkloads_tools)
    return manager, tool


def test_cm_k8s_workloads_list_pods_dispatches_to_list_pods():
    manager, tool = _make_tool()
    manager.list_pods.return_value = "SENTINEL_RESULT_list_pods"
    result = asyncio.run(
        tool(
            action="list_pods",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_pods"
    manager.list_pods.assert_called_once_with(
        namespace="namespace_val", label_selector="label_selector_val"
    )


def test_cm_k8s_workloads_describe_pod_dispatches_to_describe_pod():
    manager, tool = _make_tool()
    manager.describe_pod.return_value = "SENTINEL_RESULT_describe_pod"
    result = asyncio.run(
        tool(
            action="describe_pod",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_describe_pod"
    manager.describe_pod.assert_called_once_with(
        pod_name="pod_name_val", namespace="namespace_val"
    )


def test_cm_k8s_workloads_describe_pod_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="describe_pod",
            pod_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'pod_name' is required for describe_pod"
    manager.describe_pod.assert_not_called()


def test_cm_k8s_workloads_exec_pod_dispatches_to_exec_pod():
    manager, tool = _make_tool()
    manager.exec_pod.return_value = "SENTINEL_RESULT_exec_pod"
    result = asyncio.run(
        tool(
            action="exec_pod",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_exec_pod"
    manager.exec_pod.assert_called_once_with(
        pod_name="pod_name_val",
        namespace="namespace_val",
        command=["command_item"],
        container="exec_container_val",
    )


def test_cm_k8s_workloads_exec_pod_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="exec_pod",
            pod_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'pod_name' is required for exec_pod"
    manager.exec_pod.assert_not_called()


def test_cm_k8s_workloads_port_forward_pod_dispatches_to_port_forward_pod():
    manager, tool = _make_tool()
    manager.port_forward_pod.return_value = "SENTINEL_RESULT_port_forward_pod"
    result = asyncio.run(
        tool(
            action="port_forward_pod",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_port_forward_pod"
    manager.port_forward_pod.assert_called_once_with(
        pod_name="pod_name_val", namespace="namespace_val", local_port=7, remote_port=7
    )


def test_cm_k8s_workloads_port_forward_pod_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="port_forward_pod",
            local_port=None,
            pod_name=None,
            remote_port=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'pod_name', 'local_port', and 'remote_port' are required for port_forward_pod"
    )
    manager.port_forward_pod.assert_not_called()


def test_cm_k8s_workloads_attach_pod_dispatches_to_attach_pod():
    manager, tool = _make_tool()
    manager.attach_pod.return_value = "SENTINEL_RESULT_attach_pod"
    result = asyncio.run(
        tool(
            action="attach_pod",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_attach_pod"
    manager.attach_pod.assert_called_once_with(
        pod_name="pod_name_val",
        namespace="namespace_val",
        container="attach_container_val",
    )


def test_cm_k8s_workloads_attach_pod_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="attach_pod",
            pod_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'pod_name' is required for attach_pod"
    manager.attach_pod.assert_not_called()


def test_cm_k8s_workloads_copy_to_pod_dispatches_to_copy_to_pod():
    manager, tool = _make_tool()
    manager.copy_to_pod.return_value = "SENTINEL_RESULT_copy_to_pod"
    result = asyncio.run(
        tool(
            action="copy_to_pod",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_copy_to_pod"
    manager.copy_to_pod.assert_called_once_with(
        "pod_name_val", "namespace_val", "source_val", "destination_val"
    )


def test_cm_k8s_workloads_copy_to_pod_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="copy_to_pod",
            destination=None,
            pod_name=None,
            source=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'pod_name', 'source', and 'destination' are required for copy_to_pod"
    )
    manager.copy_to_pod.assert_not_called()


def test_cm_k8s_workloads_copy_from_pod_dispatches_to_copy_from_pod():
    manager, tool = _make_tool()
    manager.copy_from_pod.return_value = "SENTINEL_RESULT_copy_from_pod"
    result = asyncio.run(
        tool(
            action="copy_from_pod",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_copy_from_pod"
    manager.copy_from_pod.assert_called_once_with(
        "pod_name_val", "namespace_val", "source_val", "destination_val"
    )


def test_cm_k8s_workloads_copy_from_pod_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="copy_from_pod",
            destination=None,
            pod_name=None,
            source=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'pod_name', 'source', and 'destination' are required for copy_from_pod"
    )
    manager.copy_from_pod.assert_not_called()


def test_cm_k8s_workloads_rollout_status_dispatches_to_rollout_status():
    manager, tool = _make_tool()
    manager.rollout_status.return_value = "SENTINEL_RESULT_rollout_status"
    result = asyncio.run(
        tool(
            action="rollout_status",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_rollout_status"
    manager.rollout_status.assert_called_once_with(
        resource_type="resource_type_val",
        name="resource_name_val",
        namespace="namespace_val",
    )


def test_cm_k8s_workloads_rollout_status_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="rollout_status",
            resource_name=None,
            resource_type=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'resource_type' and 'resource_name' are required for rollout_status"
    )
    manager.rollout_status.assert_not_called()


def test_cm_k8s_workloads_rollout_history_dispatches_to_rollout_history():
    manager, tool = _make_tool()
    manager.rollout_history.return_value = "SENTINEL_RESULT_rollout_history"
    result = asyncio.run(
        tool(
            action="rollout_history",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_rollout_history"
    manager.rollout_history.assert_called_once_with(
        resource_type="resource_type_val",
        name="resource_name_val",
        namespace="namespace_val",
    )


def test_cm_k8s_workloads_rollout_history_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="rollout_history",
            resource_name=None,
            resource_type=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'resource_type' and 'resource_name' are required for rollout_history"
    )
    manager.rollout_history.assert_not_called()


def test_cm_k8s_workloads_rollout_restart_dispatches_to_rollout_restart():
    manager, tool = _make_tool()
    manager.rollout_restart.return_value = "SENTINEL_RESULT_rollout_restart"
    result = asyncio.run(
        tool(
            action="rollout_restart",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_rollout_restart"
    manager.rollout_restart.assert_called_once_with(
        resource_type="resource_type_val",
        name="resource_name_val",
        namespace="namespace_val",
    )


def test_cm_k8s_workloads_rollout_restart_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="rollout_restart",
            resource_name=None,
            resource_type=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'resource_type' and 'resource_name' are required for rollout_restart"
    )
    manager.rollout_restart.assert_not_called()


def test_cm_k8s_workloads_rollout_undo_dispatches_to_rollout_undo():
    manager, tool = _make_tool()
    manager.rollout_undo.return_value = "SENTINEL_RESULT_rollout_undo"
    result = asyncio.run(
        tool(
            action="rollout_undo",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_rollout_undo"
    manager.rollout_undo.assert_called_once_with(
        resource_type="resource_type_val",
        name="resource_name_val",
        namespace="namespace_val",
        revision=7,
    )


def test_cm_k8s_workloads_rollout_undo_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="rollout_undo",
            resource_name=None,
            resource_type=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'resource_type' and 'resource_name' are required for rollout_undo"
    )
    manager.rollout_undo.assert_not_called()


def test_cm_k8s_workloads_rollout_pause_dispatches_to_rollout_pause():
    manager, tool = _make_tool()
    manager.rollout_pause.return_value = "SENTINEL_RESULT_rollout_pause"
    result = asyncio.run(
        tool(
            action="rollout_pause",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_rollout_pause"
    manager.rollout_pause.assert_called_once_with(
        resource_type="resource_type_val",
        name="resource_name_val",
        namespace="namespace_val",
    )


def test_cm_k8s_workloads_rollout_pause_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="rollout_pause",
            resource_name=None,
            resource_type=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'resource_type' and 'resource_name' are required for rollout_pause"
    )
    manager.rollout_pause.assert_not_called()


def test_cm_k8s_workloads_rollout_resume_dispatches_to_rollout_resume():
    manager, tool = _make_tool()
    manager.rollout_resume.return_value = "SENTINEL_RESULT_rollout_resume"
    result = asyncio.run(
        tool(
            action="rollout_resume",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_rollout_resume"
    manager.rollout_resume.assert_called_once_with(
        resource_type="resource_type_val",
        name="resource_name_val",
        namespace="namespace_val",
    )


def test_cm_k8s_workloads_rollout_resume_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="rollout_resume",
            resource_name=None,
            resource_type=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'resource_type' and 'resource_name' are required for rollout_resume"
    )
    manager.rollout_resume.assert_not_called()


def test_cm_k8s_workloads_set_deployment_strategy_dispatches_to_set_deployment_strategy():
    manager, tool = _make_tool()
    manager.set_deployment_strategy.return_value = (
        "SENTINEL_RESULT_set_deployment_strategy"
    )
    result = asyncio.run(
        tool(
            action="set_deployment_strategy",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_set_deployment_strategy"
    manager.set_deployment_strategy.assert_called_once_with(
        "name_val", "namespace_val", {"spec_k": "spec_v"}
    )


def test_cm_k8s_workloads_set_deployment_strategy_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="set_deployment_strategy",
            name=None,
            spec=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' and 'spec' are required for set_deployment_strategy"
    manager.set_deployment_strategy.assert_not_called()


def test_cm_k8s_workloads_get_deployment_strategy_dispatches_to_get_deployment_strategy():
    manager, tool = _make_tool()
    manager.get_deployment_strategy.return_value = (
        "SENTINEL_RESULT_get_deployment_strategy"
    )
    result = asyncio.run(
        tool(
            action="get_deployment_strategy",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_get_deployment_strategy"
    manager.get_deployment_strategy.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_workloads_get_deployment_strategy_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="get_deployment_strategy",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for get_deployment_strategy"
    manager.get_deployment_strategy.assert_not_called()


def test_cm_k8s_workloads_set_daemonset_update_strategy_dispatches_to_set_daemonset_update_strategy():
    manager, tool = _make_tool()
    manager.set_daemonset_update_strategy.return_value = (
        "SENTINEL_RESULT_set_daemonset_update_strategy"
    )
    result = asyncio.run(
        tool(
            action="set_daemonset_update_strategy",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_set_daemonset_update_strategy"
    manager.set_daemonset_update_strategy.assert_called_once_with(
        "name_val", "namespace_val", {"spec_k": "spec_v"}
    )


def test_cm_k8s_workloads_set_daemonset_update_strategy_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="set_daemonset_update_strategy",
            name=None,
            spec=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'name' and 'spec' are required for set_daemonset_update_strategy"
    )
    manager.set_daemonset_update_strategy.assert_not_called()


def test_cm_k8s_workloads_get_daemonset_update_strategy_dispatches_to_get_daemonset_update_strategy():
    manager, tool = _make_tool()
    manager.get_daemonset_update_strategy.return_value = (
        "SENTINEL_RESULT_get_daemonset_update_strategy"
    )
    result = asyncio.run(
        tool(
            action="get_daemonset_update_strategy",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_get_daemonset_update_strategy"
    manager.get_daemonset_update_strategy.assert_called_once_with(
        "name_val", "namespace_val"
    )


def test_cm_k8s_workloads_get_daemonset_update_strategy_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="get_daemonset_update_strategy",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for get_daemonset_update_strategy"
    manager.get_daemonset_update_strategy.assert_not_called()


def test_cm_k8s_workloads_set_statefulset_update_strategy_dispatches_to_set_statefulset_update_strategy():
    manager, tool = _make_tool()
    manager.set_statefulset_update_strategy.return_value = (
        "SENTINEL_RESULT_set_statefulset_update_strategy"
    )
    result = asyncio.run(
        tool(
            action="set_statefulset_update_strategy",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_set_statefulset_update_strategy"
    manager.set_statefulset_update_strategy.assert_called_once_with(
        "name_val", "namespace_val", {"spec_k": "spec_v"}
    )


def test_cm_k8s_workloads_set_statefulset_update_strategy_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="set_statefulset_update_strategy",
            name=None,
            spec=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'name' and 'spec' are required for set_statefulset_update_strategy"
    )
    manager.set_statefulset_update_strategy.assert_not_called()


def test_cm_k8s_workloads_get_statefulset_update_strategy_dispatches_to_get_statefulset_update_strategy():
    manager, tool = _make_tool()
    manager.get_statefulset_update_strategy.return_value = (
        "SENTINEL_RESULT_get_statefulset_update_strategy"
    )
    result = asyncio.run(
        tool(
            action="get_statefulset_update_strategy",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_get_statefulset_update_strategy"
    manager.get_statefulset_update_strategy.assert_called_once_with(
        "name_val", "namespace_val"
    )


def test_cm_k8s_workloads_get_statefulset_update_strategy_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="get_statefulset_update_strategy",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for get_statefulset_update_strategy"
    manager.get_statefulset_update_strategy.assert_not_called()


def test_cm_k8s_workloads_list_statefulsets_dispatches_to_list_statefulsets():
    manager, tool = _make_tool()
    manager.list_statefulsets.return_value = "SENTINEL_RESULT_list_statefulsets"
    result = asyncio.run(
        tool(
            action="list_statefulsets",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_statefulsets"
    manager.list_statefulsets.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_workloads_create_stateful_set_dispatches_to_create_stateful_set():
    manager, tool = _make_tool()
    manager.create_stateful_set.return_value = "SENTINEL_RESULT_create_stateful_set"
    result = asyncio.run(
        tool(
            action="create_stateful_set",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_create_stateful_set"
    manager.create_stateful_set.assert_called_once_with(
        "name_val", "namespace_val", {"spec_k": "spec_v"}
    )


def test_cm_k8s_workloads_create_stateful_set_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="create_stateful_set",
            name=None,
            spec=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' and 'spec' are required for create_stateful_set"
    manager.create_stateful_set.assert_not_called()


def test_cm_k8s_workloads_scale_statefulset_dispatches_to_scale_statefulset():
    manager, tool = _make_tool()
    manager.scale_statefulset.return_value = "SENTINEL_RESULT_scale_statefulset"
    result = asyncio.run(
        tool(
            action="scale_statefulset",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_scale_statefulset"
    manager.scale_statefulset.assert_called_once_with(
        name="name_val", namespace="namespace_val", replicas=7
    )


def test_cm_k8s_workloads_scale_statefulset_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="scale_statefulset",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for scale_statefulset"
    manager.scale_statefulset.assert_not_called()


def test_cm_k8s_workloads_list_daemonsets_dispatches_to_list_daemonsets():
    manager, tool = _make_tool()
    manager.list_daemonsets.return_value = "SENTINEL_RESULT_list_daemonsets"
    result = asyncio.run(
        tool(
            action="list_daemonsets",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_daemonsets"
    manager.list_daemonsets.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_workloads_create_daemon_set_dispatches_to_create_daemon_set():
    manager, tool = _make_tool()
    manager.create_daemon_set.return_value = "SENTINEL_RESULT_create_daemon_set"
    result = asyncio.run(
        tool(
            action="create_daemon_set",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_create_daemon_set"
    manager.create_daemon_set.assert_called_once_with(
        "name_val", "namespace_val", {"spec_k": "spec_v"}
    )


def test_cm_k8s_workloads_create_daemon_set_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="create_daemon_set",
            name=None,
            spec=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' and 'spec' are required for create_daemon_set"
    manager.create_daemon_set.assert_not_called()


def test_cm_k8s_workloads_list_replicasets_dispatches_to_list_replica_sets():
    manager, tool = _make_tool()
    manager.list_replica_sets.return_value = "SENTINEL_RESULT_list_replicasets"
    result = asyncio.run(
        tool(
            action="list_replicasets",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_replicasets"
    manager.list_replica_sets.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_workloads_describe_replicaset_dispatches_to_describe_replica_set():
    manager, tool = _make_tool()
    manager.describe_replica_set.return_value = "SENTINEL_RESULT_describe_replicaset"
    result = asyncio.run(
        tool(
            action="describe_replicaset",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_describe_replicaset"
    manager.describe_replica_set.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_workloads_describe_replicaset_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="describe_replicaset",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for describe_replicaset"
    manager.describe_replica_set.assert_not_called()


def test_cm_k8s_workloads_scale_replicaset_dispatches_to_scale_replica_set():
    manager, tool = _make_tool()
    manager.scale_replica_set.return_value = "SENTINEL_RESULT_scale_replicaset"
    result = asyncio.run(
        tool(
            action="scale_replicaset",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_scale_replicaset"
    manager.scale_replica_set.assert_called_once_with("name_val", "namespace_val", 7)


def test_cm_k8s_workloads_scale_replicaset_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="scale_replicaset",
            name=None,
            replicas=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' and 'replicas' are required for scale_replicaset"
    manager.scale_replica_set.assert_not_called()


def test_cm_k8s_workloads_list_jobs_dispatches_to_list_jobs():
    manager, tool = _make_tool()
    manager.list_jobs.return_value = "SENTINEL_RESULT_list_jobs"
    result = asyncio.run(
        tool(
            action="list_jobs",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_jobs"
    manager.list_jobs.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_workloads_describe_job_dispatches_to_describe_job():
    manager, tool = _make_tool()
    manager.describe_job.return_value = "SENTINEL_RESULT_describe_job"
    result = asyncio.run(
        tool(
            action="describe_job",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_describe_job"
    manager.describe_job.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_workloads_describe_job_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="describe_job",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for describe_job"
    manager.describe_job.assert_not_called()


def test_cm_k8s_workloads_create_job_dispatches_to_create_job():
    manager, tool = _make_tool()
    manager.create_job.return_value = "SENTINEL_RESULT_create_job"
    result = asyncio.run(
        tool(
            action="create_job",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_create_job"
    manager.create_job.assert_called_once_with(
        "name_val", "namespace_val", {"spec_k": "spec_v"}
    )


def test_cm_k8s_workloads_create_job_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="create_job",
            name=None,
            spec=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' and 'spec' are required for create_job"
    manager.create_job.assert_not_called()


def test_cm_k8s_workloads_delete_job_dispatches_to_delete_job():
    manager, tool = _make_tool()
    manager.delete_job.return_value = "SENTINEL_RESULT_delete_job"
    result = asyncio.run(
        tool(
            action="delete_job",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_delete_job"
    manager.delete_job.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_workloads_delete_job_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="delete_job",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for delete_job"
    manager.delete_job.assert_not_called()


def test_cm_k8s_workloads_list_cron_jobs_dispatches_to_list_cron_jobs():
    manager, tool = _make_tool()
    manager.list_cron_jobs.return_value = "SENTINEL_RESULT_list_cron_jobs"
    result = asyncio.run(
        tool(
            action="list_cron_jobs",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_cron_jobs"
    manager.list_cron_jobs.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_workloads_describe_cron_job_dispatches_to_describe_cron_job():
    manager, tool = _make_tool()
    manager.describe_cron_job.return_value = "SENTINEL_RESULT_describe_cron_job"
    result = asyncio.run(
        tool(
            action="describe_cron_job",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_describe_cron_job"
    manager.describe_cron_job.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_workloads_describe_cron_job_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="describe_cron_job",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for describe_cron_job"
    manager.describe_cron_job.assert_not_called()


def test_cm_k8s_workloads_create_cron_job_dispatches_to_create_cron_job():
    manager, tool = _make_tool()
    manager.create_cron_job.return_value = "SENTINEL_RESULT_create_cron_job"
    result = asyncio.run(
        tool(
            action="create_cron_job",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_create_cron_job"
    manager.create_cron_job.assert_called_once_with(
        "name_val", "namespace_val", {"spec_k": "spec_v"}
    )


def test_cm_k8s_workloads_create_cron_job_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="create_cron_job",
            name=None,
            spec=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' and 'spec' are required for create_cron_job"
    manager.create_cron_job.assert_not_called()


def test_cm_k8s_workloads_delete_cron_job_dispatches_to_delete_cron_job():
    manager, tool = _make_tool()
    manager.delete_cron_job.return_value = "SENTINEL_RESULT_delete_cron_job"
    result = asyncio.run(
        tool(
            action="delete_cron_job",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_delete_cron_job"
    manager.delete_cron_job.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_workloads_delete_cron_job_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="delete_cron_job",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for delete_cron_job"
    manager.delete_cron_job.assert_not_called()


def test_cm_k8s_workloads_unknown_action_returns_error():
    manager, tool = _make_tool()
    result = asyncio.run(tool(action="bogus_action_xyz", ctx=None))
    assert result == "Error: Unknown action 'bogus_action_xyz'"


def test_cm_k8s_workloads_manager_exception_is_caught_and_formatted():
    manager, tool = _make_tool()
    manager.list_pods.side_effect = RuntimeError("boom")
    result = asyncio.run(
        tool(
            action="list_pods",
            pod_name="pod_name_val",
            namespace="namespace_val",
            label_selector="label_selector_val",
            exec_command="exec_command_val",
            command=["command_item"],
            exec_container="exec_container_val",
            local_port=7,
            remote_port=7,
            attach_container="attach_container_val",
            source="source_val",
            destination="destination_val",
            resource_type="resource_type_val",
            resource_name="resource_name_val",
            rollout_revision=7,
            name="name_val",
            spec={"spec_k": "spec_v"},
            replicas=7,
            ctx=None,
        )
    )
    assert result == "Error executing list_pods: RuntimeError"

"""Characterization tests for CXA-FL-CONTAINERMANAGERMCP-02's
register_k8sobservability_tools.cm_k8s_observability (CCN 52), before
decomposition.

cm_k8s_observability is an 18-branch action dispatcher (Metrics,
Autoscaler metrics, Watch/stream/events, Debug helpers). For every action:
which manager.<method> it calls and with exactly which args (including
stream_pod_logs's `tail_lines or 100` default fallback), each
required-param guard's exact error string with the manager left uncalled,
the unknown-action fallback, and manager-exception formatting.

Harness pattern (`_capture_tool`) matches the repo's own precedent in
tests/test_multi_context_manager.py. Guard tests pass the guarded
parameter(s) explicitly as `None` (the bare function's real defaults are
FastMCP FieldInfo objects, not None).
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from container_manager_mcp.mcp import mcp_k8s_observability


@pytest.fixture(autouse=True)
def _restore_create_manager():
    original = mcp_k8s_observability.create_manager
    yield
    mcp_k8s_observability.create_manager = original


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
    mcp_k8s_observability.create_manager = lambda manager_type=None: manager
    tool = _capture_tool(mcp_k8s_observability.register_k8sobservability_tools)
    return manager, tool


def test_cm_k8s_observability_top_pods_dispatches_to_top_pods():
    manager, tool = _make_tool()
    manager.top_pods.return_value = "SENTINEL_RESULT_top_pods"
    result = asyncio.run(tool(
        action="top_pods",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_top_pods"
    manager.top_pods.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_observability_top_nodes_dispatches_to_top_nodes():
    manager, tool = _make_tool()
    manager.top_nodes.return_value = "SENTINEL_RESULT_top_nodes"
    result = asyncio.run(tool(
        action="top_nodes",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_top_nodes"
    manager.top_nodes.assert_called_once_with()


def test_cm_k8s_observability_get_pod_metrics_dispatches_to_get_pod_metrics():
    manager, tool = _make_tool()
    manager.get_pod_metrics.return_value = "SENTINEL_RESULT_get_pod_metrics"
    result = asyncio.run(tool(
        action="get_pod_metrics",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_get_pod_metrics"
    manager.get_pod_metrics.assert_called_once_with("namespace_val")


def test_cm_k8s_observability_get_node_metrics_dispatches_to_get_node_metrics():
    manager, tool = _make_tool()
    manager.get_node_metrics.return_value = "SENTINEL_RESULT_get_node_metrics"
    result = asyncio.run(tool(
        action="get_node_metrics",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_get_node_metrics"
    manager.get_node_metrics.assert_called_once_with()


def test_cm_k8s_observability_get_pod_resource_usage_dispatches_to_get_pod_resource_usage():
    manager, tool = _make_tool()
    manager.get_pod_resource_usage.return_value = "SENTINEL_RESULT_get_pod_resource_usage"
    result = asyncio.run(tool(
        action="get_pod_resource_usage",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_get_pod_resource_usage"
    manager.get_pod_resource_usage.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_observability_get_pod_resource_usage_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="get_pod_resource_usage",
        name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'namespace' are required for get_pod_resource_usage"
    manager.get_pod_resource_usage.assert_not_called()


def test_cm_k8s_observability_get_cluster_resource_summary_dispatches_to_get_cluster_resource_summary():
    manager, tool = _make_tool()
    manager.get_cluster_resource_summary.return_value = "SENTINEL_RESULT_get_cluster_resource_summary"
    result = asyncio.run(tool(
        action="get_cluster_resource_summary",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_get_cluster_resource_summary"
    manager.get_cluster_resource_summary.assert_called_once_with()


def test_cm_k8s_observability_get_autoscaler_metrics_dispatches_to_get_autoscaler_metrics():
    manager, tool = _make_tool()
    manager.get_autoscaler_metrics.return_value = "SENTINEL_RESULT_get_autoscaler_metrics"
    result = asyncio.run(tool(
        action="get_autoscaler_metrics",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_get_autoscaler_metrics"
    manager.get_autoscaler_metrics.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_observability_get_autoscaler_metrics_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="get_autoscaler_metrics",
        name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'namespace' are required for get_autoscaler_metrics"
    manager.get_autoscaler_metrics.assert_not_called()


def test_cm_k8s_observability_set_autoscaler_metrics_dispatches_to_set_autoscaler_metrics():
    manager, tool = _make_tool()
    manager.set_autoscaler_metrics.return_value = "SENTINEL_RESULT_set_autoscaler_metrics"
    result = asyncio.run(tool(
        action="set_autoscaler_metrics",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_set_autoscaler_metrics"
    manager.set_autoscaler_metrics.assert_called_once_with("name_val", "namespace_val", ["metrics_item"])


def test_cm_k8s_observability_set_autoscaler_metrics_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="set_autoscaler_metrics",
        metrics=None, name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name', 'namespace', and 'metrics' are required for set_autoscaler_metrics"
    manager.set_autoscaler_metrics.assert_not_called()


def test_cm_k8s_observability_scale_deployment_autoscaler_dispatches_to_scale_deployment_autoscaler():
    manager, tool = _make_tool()
    manager.scale_deployment_autoscaler.return_value = "SENTINEL_RESULT_scale_deployment_autoscaler"
    result = asyncio.run(tool(
        action="scale_deployment_autoscaler",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_scale_deployment_autoscaler"
    manager.scale_deployment_autoscaler.assert_called_once_with("name_val", "namespace_val", 7, 7)


def test_cm_k8s_observability_scale_deployment_autoscaler_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="scale_deployment_autoscaler",
        max_replicas=None, min_replicas=None, name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name', 'namespace', 'min_replicas', and 'max_replicas' are required for scale_deployment_autoscaler"
    manager.scale_deployment_autoscaler.assert_not_called()


def test_cm_k8s_observability_get_autoscaler_history_dispatches_to_get_autoscaler_history():
    manager, tool = _make_tool()
    manager.get_autoscaler_history.return_value = "SENTINEL_RESULT_get_autoscaler_history"
    result = asyncio.run(tool(
        action="get_autoscaler_history",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_get_autoscaler_history"
    manager.get_autoscaler_history.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_observability_get_autoscaler_history_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="get_autoscaler_history",
        name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'namespace' are required for get_autoscaler_history"
    manager.get_autoscaler_history.assert_not_called()


def test_cm_k8s_observability_watch_resource_dispatches_to_watch_resource():
    manager, tool = _make_tool()
    manager.watch_resource.return_value = "SENTINEL_RESULT_watch_resource"
    result = asyncio.run(tool(
        action="watch_resource",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_watch_resource"
    manager.watch_resource.assert_called_once_with("resource_type_val", "name_val", "namespace_val")


def test_cm_k8s_observability_watch_resource_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="watch_resource",
        name=None, resource_type=None,
        ctx=None,
    ))
    assert result == "Error: 'resource_type' and 'name' are required for watch_resource"
    manager.watch_resource.assert_not_called()


def test_cm_k8s_observability_stream_pod_logs_dispatches_to_stream_pod_logs():
    manager, tool = _make_tool()
    manager.stream_pod_logs.return_value = "SENTINEL_RESULT_stream_pod_logs"
    result = asyncio.run(tool(
        action="stream_pod_logs",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_stream_pod_logs"
    manager.stream_pod_logs.assert_called_once_with("name_val", "namespace_val", 7)


def test_cm_k8s_observability_stream_pod_logs_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="stream_pod_logs",
        name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'namespace' are required for stream_pod_logs"
    manager.stream_pod_logs.assert_not_called()


def test_cm_k8s_observability_get_resource_events_dispatches_to_get_resource_events():
    manager, tool = _make_tool()
    manager.get_resource_events.return_value = "SENTINEL_RESULT_get_resource_events"
    result = asyncio.run(tool(
        action="get_resource_events",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_get_resource_events"
    manager.get_resource_events.assert_called_once_with("resource_type_val", "name_val", "namespace_val")


def test_cm_k8s_observability_get_resource_events_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="get_resource_events",
        name=None, resource_type=None,
        ctx=None,
    ))
    assert result == "Error: 'resource_type' and 'name' are required for get_resource_events"
    manager.get_resource_events.assert_not_called()


def test_cm_k8s_observability_list_field_selector_dispatches_to_list_field_selector():
    manager, tool = _make_tool()
    manager.list_field_selector.return_value = "SENTINEL_RESULT_list_field_selector"
    result = asyncio.run(tool(
        action="list_field_selector",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_field_selector"
    manager.list_field_selector.assert_called_once_with("resource_type_val", "field_selector_val", "namespace_val")


def test_cm_k8s_observability_list_field_selector_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="list_field_selector",
        field_selector=None, resource_type=None,
        ctx=None,
    ))
    assert result == "Error: 'resource_type' and 'field_selector' are required for list_field_selector"
    manager.list_field_selector.assert_not_called()


def test_cm_k8s_observability_debug_pod_dispatches_to_debug_pod():
    manager, tool = _make_tool()
    manager.debug_pod.return_value = "SENTINEL_RESULT_debug_pod"
    result = asyncio.run(tool(
        action="debug_pod",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_debug_pod"
    manager.debug_pod.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_observability_debug_pod_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="debug_pod",
        name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'namespace' are required for debug_pod"
    manager.debug_pod.assert_not_called()


def test_cm_k8s_observability_debug_node_dispatches_to_debug_node():
    manager, tool = _make_tool()
    manager.debug_node.return_value = "SENTINEL_RESULT_debug_node"
    result = asyncio.run(tool(
        action="debug_node",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_debug_node"
    manager.debug_node.assert_called_once_with("name_val")


def test_cm_k8s_observability_debug_node_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="debug_node",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for debug_node"
    manager.debug_node.assert_not_called()


def test_cm_k8s_observability_debug_service_dispatches_to_debug_service():
    manager, tool = _make_tool()
    manager.debug_service.return_value = "SENTINEL_RESULT_debug_service"
    result = asyncio.run(tool(
        action="debug_service",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_debug_service"
    manager.debug_service.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_observability_debug_service_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="debug_service",
        name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'namespace' are required for debug_service"
    manager.debug_service.assert_not_called()


def test_cm_k8s_observability_debug_deployment_dispatches_to_debug_deployment():
    manager, tool = _make_tool()
    manager.debug_deployment.return_value = "SENTINEL_RESULT_debug_deployment"
    result = asyncio.run(tool(
        action="debug_deployment",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_debug_deployment"
    manager.debug_deployment.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_observability_debug_deployment_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="debug_deployment",
        name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'namespace' are required for debug_deployment"
    manager.debug_deployment.assert_not_called()


def test_cm_k8s_observability_unknown_action_returns_error():
    manager, tool = _make_tool()
    result = asyncio.run(tool(action="bogus_action_xyz", ctx=None))
    assert result == "Error: Unknown action 'bogus_action_xyz'"


def test_cm_k8s_observability_manager_exception_is_caught_and_formatted():
    manager, tool = _make_tool()
    manager.top_pods.side_effect = RuntimeError("boom")
    result = asyncio.run(tool(
        action="top_pods",
        name="name_val", namespace="namespace_val", resource_type="resource_type_val", field_selector="field_selector_val", tail_lines=7, metrics=["metrics_item"], min_replicas=7, max_replicas=7,
        ctx=None,
    ))
    assert result == "Error executing top_pods: RuntimeError"


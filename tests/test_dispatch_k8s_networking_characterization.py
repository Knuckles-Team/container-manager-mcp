"""Characterization tests for CXA-FL-CONTAINERMANAGERMCP-01's
register_k8snetworking_tools.cm_k8s_networking (CCN 54), before
decomposition.

cm_k8s_networking is a 22-branch action dispatcher (Ingress, Ingress
classes, NetworkPolicies, Endpoints, DNS debugging, native core/v1
Services). For every action: which manager.<method> it calls and with
exactly which args (including the two `json.loads(...)`-parsed variants
for create_ingress/create_networkpolicy), each required-param guard's
exact error string with the manager left uncalled, the unknown-action
fallback, and manager-exception formatting.

Cross-check note: the initial literal-vs-dispatch action audit for this
file used a `[a-z_]+` regex that silently excludes digits, which made
the 4 native-service actions (list_k8s_services/get_k8s_service/
create_k8s_service/delete_k8s_service -- containing "k8s") look
dispatch-only/missing from the Literal[...] type. Re-run with
`[a-z0-9_]+` shows a clean 22/22 match -- not a real defect, a regex bug
in the audit itself. Recorded here so the same false positive isn't
re-discovered by a future lane.

Harness pattern (`_capture_tool`) matches the repo's own precedent in
tests/test_multi_context_manager.py. Guard tests pass the guarded
parameter(s) explicitly as `None` (the bare function's real defaults are
FastMCP FieldInfo objects, not None).
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from container_manager_mcp.mcp import mcp_k8s_networking


@pytest.fixture(autouse=True)
def _restore_create_manager():
    original = mcp_k8s_networking.create_manager
    yield
    mcp_k8s_networking.create_manager = original


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
    mcp_k8s_networking.create_manager = lambda manager_type=None: manager
    tool = _capture_tool(mcp_k8s_networking.register_k8snetworking_tools)
    return manager, tool


def test_cm_k8s_networking_list_ingress_dispatches_to_list_ingress():
    manager, tool = _make_tool()
    manager.list_ingress.return_value = "SENTINEL_RESULT_list_ingress"
    result = asyncio.run(tool(
        action="list_ingress",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_ingress"
    manager.list_ingress.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_networking_create_ingress_dispatches_to_create_ingress():
    manager, tool = _make_tool()
    manager.create_ingress.return_value = "SENTINEL_RESULT_create_ingress"
    result = asyncio.run(tool(
        action="create_ingress",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_ingress"
    manager.create_ingress.assert_called_once_with(name="ingress_name_val", namespace="namespace_val", spec={"rules": []})


def test_cm_k8s_networking_create_ingress_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_ingress",
        ingress_name=None,
        ctx=None,
    ))
    assert result == "Error: 'ingress_name' is required for create_ingress"
    manager.create_ingress.assert_not_called()


def test_cm_k8s_networking_delete_ingress_dispatches_to_delete_ingress():
    manager, tool = _make_tool()
    manager.delete_ingress.return_value = "SENTINEL_RESULT_delete_ingress"
    result = asyncio.run(tool(
        action="delete_ingress",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_delete_ingress"
    manager.delete_ingress.assert_called_once_with(name="ingress_name_val", namespace="namespace_val")


def test_cm_k8s_networking_delete_ingress_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="delete_ingress",
        ingress_name=None,
        ctx=None,
    ))
    assert result == "Error: 'ingress_name' is required for delete_ingress"
    manager.delete_ingress.assert_not_called()


def test_cm_k8s_networking_list_ingress_classes_dispatches_to_list_ingress_classes():
    manager, tool = _make_tool()
    manager.list_ingress_classes.return_value = "SENTINEL_RESULT_list_ingress_classes"
    result = asyncio.run(tool(
        action="list_ingress_classes",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_ingress_classes"
    manager.list_ingress_classes.assert_called_once_with()


def test_cm_k8s_networking_describe_ingress_class_dispatches_to_describe_ingress_class():
    manager, tool = _make_tool()
    manager.describe_ingress_class.return_value = "SENTINEL_RESULT_describe_ingress_class"
    result = asyncio.run(tool(
        action="describe_ingress_class",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_describe_ingress_class"
    manager.describe_ingress_class.assert_called_once_with("name_val")


def test_cm_k8s_networking_describe_ingress_class_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="describe_ingress_class",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for describe_ingress_class"
    manager.describe_ingress_class.assert_not_called()


def test_cm_k8s_networking_create_ingress_class_dispatches_to_create_ingress_class():
    manager, tool = _make_tool()
    manager.create_ingress_class.return_value = "SENTINEL_RESULT_create_ingress_class"
    result = asyncio.run(tool(
        action="create_ingress_class",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_ingress_class"
    manager.create_ingress_class.assert_called_once_with("name_val", {"spec_k": "spec_v"})


def test_cm_k8s_networking_create_ingress_class_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_ingress_class",
        name=None, spec=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'spec' are required for create_ingress_class"
    manager.create_ingress_class.assert_not_called()


def test_cm_k8s_networking_set_default_ingress_class_dispatches_to_set_default_ingress_class():
    manager, tool = _make_tool()
    manager.set_default_ingress_class.return_value = "SENTINEL_RESULT_set_default_ingress_class"
    result = asyncio.run(tool(
        action="set_default_ingress_class",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_set_default_ingress_class"
    manager.set_default_ingress_class.assert_called_once_with("name_val")


def test_cm_k8s_networking_set_default_ingress_class_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="set_default_ingress_class",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for set_default_ingress_class"
    manager.set_default_ingress_class.assert_not_called()


def test_cm_k8s_networking_list_networkpolicies_dispatches_to_list_networkpolicies():
    manager, tool = _make_tool()
    manager.list_networkpolicies.return_value = "SENTINEL_RESULT_list_networkpolicies"
    result = asyncio.run(tool(
        action="list_networkpolicies",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_networkpolicies"
    manager.list_networkpolicies.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_networking_create_networkpolicy_dispatches_to_create_networkpolicy():
    manager, tool = _make_tool()
    manager.create_networkpolicy.return_value = "SENTINEL_RESULT_create_networkpolicy"
    result = asyncio.run(tool(
        action="create_networkpolicy",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_networkpolicy"
    manager.create_networkpolicy.assert_called_once_with(name="netpol_name_val", namespace="namespace_val", spec={"podSelector": {}})


def test_cm_k8s_networking_create_networkpolicy_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_networkpolicy",
        netpol_name=None,
        ctx=None,
    ))
    assert result == "Error: 'netpol_name' is required for create_networkpolicy"
    manager.create_networkpolicy.assert_not_called()


def test_cm_k8s_networking_delete_networkpolicy_dispatches_to_delete_networkpolicy():
    manager, tool = _make_tool()
    manager.delete_networkpolicy.return_value = "SENTINEL_RESULT_delete_networkpolicy"
    result = asyncio.run(tool(
        action="delete_networkpolicy",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_delete_networkpolicy"
    manager.delete_networkpolicy.assert_called_once_with(name="netpol_name_val", namespace="namespace_val")


def test_cm_k8s_networking_delete_networkpolicy_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="delete_networkpolicy",
        netpol_name=None,
        ctx=None,
    ))
    assert result == "Error: 'netpol_name' is required for delete_networkpolicy"
    manager.delete_networkpolicy.assert_not_called()


def test_cm_k8s_networking_create_network_policy_with_cidr_dispatches_to_create_network_policy_with_cidr():
    manager, tool = _make_tool()
    manager.create_network_policy_with_cidr.return_value = "SENTINEL_RESULT_create_network_policy_with_cidr"
    result = asyncio.run(tool(
        action="create_network_policy_with_cidr",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_network_policy_with_cidr"
    manager.create_network_policy_with_cidr.assert_called_once_with("name_val", "namespace_val", {"spec_k": "spec_v"})


def test_cm_k8s_networking_create_network_policy_with_cidr_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_network_policy_with_cidr",
        name=None, spec=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'spec' are required for create_network_policy_with_cidr"
    manager.create_network_policy_with_cidr.assert_not_called()


def test_cm_k8s_networking_update_network_policy_rules_dispatches_to_update_network_policy_rules():
    manager, tool = _make_tool()
    manager.update_network_policy_rules.return_value = "SENTINEL_RESULT_update_network_policy_rules"
    result = asyncio.run(tool(
        action="update_network_policy_rules",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_update_network_policy_rules"
    manager.update_network_policy_rules.assert_called_once_with("name_val", "namespace_val", ["rules_item"])


def test_cm_k8s_networking_update_network_policy_rules_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="update_network_policy_rules",
        name=None, rules=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'rules' are required for update_network_policy_rules"
    manager.update_network_policy_rules.assert_not_called()


def test_cm_k8s_networking_test_network_policy_connectivity_dispatches_to_test_network_policy_connectivity():
    manager, tool = _make_tool()
    manager.test_network_policy_connectivity.return_value = "SENTINEL_RESULT_test_network_policy_connectivity"
    result = asyncio.run(tool(
        action="test_network_policy_connectivity",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_test_network_policy_connectivity"
    manager.test_network_policy_connectivity.assert_called_once_with("namespace_val", "name_val")


def test_cm_k8s_networking_test_network_policy_connectivity_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="test_network_policy_connectivity",
        name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'namespace' and 'name' (policy_name) are required for test_network_policy_connectivity"
    manager.test_network_policy_connectivity.assert_not_called()


def test_cm_k8s_networking_list_endpoints_dispatches_to_list_endpoints():
    manager, tool = _make_tool()
    manager.list_endpoints.return_value = "SENTINEL_RESULT_list_endpoints"
    result = asyncio.run(tool(
        action="list_endpoints",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_endpoints"
    manager.list_endpoints.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_networking_list_endpointslices_dispatches_to_list_endpointslices():
    manager, tool = _make_tool()
    manager.list_endpointslices.return_value = "SENTINEL_RESULT_list_endpointslices"
    result = asyncio.run(tool(
        action="list_endpointslices",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_endpointslices"
    manager.list_endpointslices.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_networking_check_dns_resolution_dispatches_to_check_dns_resolution():
    manager, tool = _make_tool()
    manager.check_dns_resolution.return_value = "SENTINEL_RESULT_check_dns_resolution"
    result = asyncio.run(tool(
        action="check_dns_resolution",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_check_dns_resolution"
    manager.check_dns_resolution.assert_called_once_with("namespace_val", "pod_name_val", "hostname_val")


def test_cm_k8s_networking_check_dns_resolution_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="check_dns_resolution",
        hostname=None, namespace=None, pod_name=None,
        ctx=None,
    ))
    assert result == "Error: 'namespace', 'pod_name', and 'hostname' are required for check_dns_resolution"
    manager.check_dns_resolution.assert_not_called()


def test_cm_k8s_networking_list_dns_endpoints_dispatches_to_list_dns_endpoints():
    manager, tool = _make_tool()
    manager.list_dns_endpoints.return_value = "SENTINEL_RESULT_list_dns_endpoints"
    result = asyncio.run(tool(
        action="list_dns_endpoints",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_dns_endpoints"
    manager.list_dns_endpoints.assert_called_once_with("namespace_val", "service_name_val")


def test_cm_k8s_networking_list_dns_endpoints_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="list_dns_endpoints",
        namespace=None, service_name=None,
        ctx=None,
    ))
    assert result == "Error: 'namespace' and 'service_name' are required for list_dns_endpoints"
    manager.list_dns_endpoints.assert_not_called()


def test_cm_k8s_networking_test_dns_connectivity_dispatches_to_test_dns_connectivity():
    manager, tool = _make_tool()
    manager.test_dns_connectivity.return_value = "SENTINEL_RESULT_test_dns_connectivity"
    result = asyncio.run(tool(
        action="test_dns_connectivity",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_test_dns_connectivity"
    manager.test_dns_connectivity.assert_called_once_with("namespace_val", "target_val")


def test_cm_k8s_networking_test_dns_connectivity_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="test_dns_connectivity",
        namespace=None, target=None,
        ctx=None,
    ))
    assert result == "Error: 'namespace' and 'target' are required for test_dns_connectivity"
    manager.test_dns_connectivity.assert_not_called()


def test_cm_k8s_networking_list_k8s_services_dispatches_to_list_native_services():
    manager, tool = _make_tool()
    manager.list_native_services.return_value = "SENTINEL_RESULT_list_k8s_services"
    result = asyncio.run(tool(
        action="list_k8s_services",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_k8s_services"
    manager.list_native_services.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_networking_get_k8s_service_dispatches_to_get_native_service():
    manager, tool = _make_tool()
    manager.get_native_service.return_value = "SENTINEL_RESULT_get_k8s_service"
    result = asyncio.run(tool(
        action="get_k8s_service",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_get_k8s_service"
    manager.get_native_service.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_networking_get_k8s_service_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="get_k8s_service",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for get_k8s_service"
    manager.get_native_service.assert_not_called()


def test_cm_k8s_networking_create_k8s_service_dispatches_to_create_native_service():
    manager, tool = _make_tool()
    manager.create_native_service.return_value = "SENTINEL_RESULT_create_k8s_service"
    result = asyncio.run(tool(
        action="create_k8s_service",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_k8s_service"
    manager.create_native_service.assert_called_once_with(name="name_val", namespace="namespace_val", spec={"service_spec_k": "service_spec_v"}, ports=["service_ports_item"], selector={"service_selector_k": "service_selector_v"}, type="service_type_val")


def test_cm_k8s_networking_create_k8s_service_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_k8s_service",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for create_k8s_service"
    manager.create_native_service.assert_not_called()


def test_cm_k8s_networking_delete_k8s_service_dispatches_to_delete_native_service():
    manager, tool = _make_tool()
    manager.delete_native_service.return_value = "SENTINEL_RESULT_delete_k8s_service"
    result = asyncio.run(tool(
        action="delete_k8s_service",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_delete_k8s_service"
    manager.delete_native_service.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_networking_delete_k8s_service_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="delete_k8s_service",
        name=None,
        ctx=None,
    ))
    assert result == "Error: 'name' is required for delete_k8s_service"
    manager.delete_native_service.assert_not_called()


def test_cm_k8s_networking_unknown_action_returns_error():
    manager, tool = _make_tool()
    result = asyncio.run(tool(action="bogus_action_xyz", ctx=None))
    assert result == "Error: Unknown action 'bogus_action_xyz'"


def test_cm_k8s_networking_manager_exception_is_caught_and_formatted():
    manager, tool = _make_tool()
    manager.list_ingress.side_effect = RuntimeError("boom")
    result = asyncio.run(tool(
        action="list_ingress",
        namespace="namespace_val", ingress_name="ingress_name_val", ingress_spec='{"rules": []}', netpol_name="netpol_name_val", netpol_spec='{"podSelector": {}}', name="name_val", spec={"spec_k": "spec_v"}, rules=["rules_item"], pod_name="pod_name_val", hostname="hostname_val", service_name="service_name_val", target="target_val", service_spec={"service_spec_k": "service_spec_v"}, service_ports=["service_ports_item"], service_selector={"service_selector_k": "service_selector_v"}, service_type="service_type_val",
        ctx=None,
    ))
    assert result == "Error executing list_ingress: RuntimeError"


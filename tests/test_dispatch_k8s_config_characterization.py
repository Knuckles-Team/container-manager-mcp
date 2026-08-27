"""Characterization tests for CXA-FL-CONTAINERMANAGERMCP-01's
register_k8sconfig_tools.cm_k8s_config (CCN 58), before decomposition.

cm_k8s_config is a 19-branch action dispatcher (ConfigMaps, Secrets,
Namespaces, Events, CRDs/custom resources, labels/annotations/patch,
config/secret state tracking). For every action: which manager.<method> it
calls and with exactly which args (including the `json.loads(...)`-parsed
variants for create_configmap/create_secret/label_resource/
annotate_resource/patch_resource), each required-param guard's exact error
string with the manager left uncalled, the unknown-action fallback, and
manager-exception formatting.

Harness pattern (`_capture_tool`) matches the repo's own precedent in
tests/test_multi_context_manager.py. Guard tests pass the guarded
parameter(s) explicitly as `None` (the bare function's real defaults are
FastMCP FieldInfo objects, not None).
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from container_manager_mcp.mcp import mcp_k8s_config


@pytest.fixture(autouse=True)
def _restore_create_manager():
    original = mcp_k8s_config.create_manager
    yield
    mcp_k8s_config.create_manager = original


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
    mcp_k8s_config.create_manager = lambda manager_type=None: manager
    tool = _capture_tool(mcp_k8s_config.register_k8sconfig_tools)
    return manager, tool


def test_cm_k8s_config_list_configmaps_dispatches_to_list_configmaps():
    manager, tool = _make_tool()
    manager.list_configmaps.return_value = "SENTINEL_RESULT_list_configmaps"
    result = asyncio.run(tool(
        action="list_configmaps",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_configmaps"
    manager.list_configmaps.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_config_create_configmap_dispatches_to_create_configmap():
    manager, tool = _make_tool()
    manager.create_configmap.return_value = "SENTINEL_RESULT_create_configmap"
    result = asyncio.run(tool(
        action="create_configmap",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_configmap"
    manager.create_configmap.assert_called_once_with(name="configmap_name_val", namespace="namespace_val", data={"key1": "value1"}, from_file="configmap_from_file_val")


def test_cm_k8s_config_create_configmap_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_configmap",
        configmap_name=None,
        ctx=None,
    ))
    assert result == "Error: 'configmap_name' is required for create_configmap"
    manager.create_configmap.assert_not_called()


def test_cm_k8s_config_list_secrets_dispatches_to_list_secrets():
    manager, tool = _make_tool()
    manager.list_secrets.return_value = "SENTINEL_RESULT_list_secrets"
    result = asyncio.run(tool(
        action="list_secrets",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_secrets"
    manager.list_secrets.assert_called_once_with(namespace="namespace_val")


def test_cm_k8s_config_create_secret_dispatches_to_create_secret():
    manager, tool = _make_tool()
    manager.create_secret.return_value = "SENTINEL_RESULT_create_secret"
    result = asyncio.run(tool(
        action="create_secret",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_secret"
    manager.create_secret.assert_called_once_with(name="secret_name_val", namespace="namespace_val", secret_type="secret_type_val", data={"key1": "dmFsdWUx"})


def test_cm_k8s_config_create_secret_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_secret",
        secret_name=None,
        ctx=None,
    ))
    assert result == "Error: 'secret_name' is required for create_secret"
    manager.create_secret.assert_not_called()


def test_cm_k8s_config_list_namespaces_dispatches_to_list_namespaces():
    manager, tool = _make_tool()
    manager.list_namespaces.return_value = "SENTINEL_RESULT_list_namespaces"
    result = asyncio.run(tool(
        action="list_namespaces",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_namespaces"
    manager.list_namespaces.assert_called_once_with()


def test_cm_k8s_config_create_namespace_dispatches_to_create_namespace():
    manager, tool = _make_tool()
    manager.create_namespace.return_value = "SENTINEL_RESULT_create_namespace"
    result = asyncio.run(tool(
        action="create_namespace",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_create_namespace"
    manager.create_namespace.assert_called_once_with(name="namespace_name_val")


def test_cm_k8s_config_create_namespace_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="create_namespace",
        namespace_name=None,
        ctx=None,
    ))
    assert result == "Error: 'namespace_name' is required for create_namespace"
    manager.create_namespace.assert_not_called()


def test_cm_k8s_config_delete_namespace_dispatches_to_delete_namespace():
    manager, tool = _make_tool()
    manager.delete_namespace.return_value = "SENTINEL_RESULT_delete_namespace"
    result = asyncio.run(tool(
        action="delete_namespace",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_delete_namespace"
    manager.delete_namespace.assert_called_once_with(name="namespace_name_val")


def test_cm_k8s_config_delete_namespace_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="delete_namespace",
        namespace_name=None,
        ctx=None,
    ))
    assert result == "Error: 'namespace_name' is required for delete_namespace"
    manager.delete_namespace.assert_not_called()


def test_cm_k8s_config_list_events_dispatches_to_list_events():
    manager, tool = _make_tool()
    manager.list_events.return_value = "SENTINEL_RESULT_list_events"
    result = asyncio.run(tool(
        action="list_events",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_events"
    manager.list_events.assert_called_once_with(namespace="namespace_val", field_selector="field_selector_val")


def test_cm_k8s_config_list_crds_dispatches_to_list_crds():
    manager, tool = _make_tool()
    manager.list_crds.return_value = "SENTINEL_RESULT_list_crds"
    result = asyncio.run(tool(
        action="list_crds",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_crds"
    manager.list_crds.assert_called_once_with()


def test_cm_k8s_config_describe_crd_dispatches_to_describe_crd():
    manager, tool = _make_tool()
    manager.describe_crd.return_value = "SENTINEL_RESULT_describe_crd"
    result = asyncio.run(tool(
        action="describe_crd",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_describe_crd"
    manager.describe_crd.assert_called_once_with(crd_name="crd_name_val")


def test_cm_k8s_config_describe_crd_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="describe_crd",
        crd_name=None,
        ctx=None,
    ))
    assert result == "Error: 'crd_name' is required for describe_crd"
    manager.describe_crd.assert_not_called()


def test_cm_k8s_config_list_custom_resources_dispatches_to_list_custom_resources():
    manager, tool = _make_tool()
    manager.list_custom_resources.return_value = "SENTINEL_RESULT_list_custom_resources"
    result = asyncio.run(tool(
        action="list_custom_resources",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_list_custom_resources"
    manager.list_custom_resources.assert_called_once_with(group="crd_group_val", version="crd_version_val", plural="crd_plural_val", namespace="namespace_val")


def test_cm_k8s_config_list_custom_resources_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="list_custom_resources",
        crd_group=None, crd_plural=None, crd_version=None,
        ctx=None,
    ))
    assert result == "Error: 'crd_group', 'crd_version', and 'crd_plural' are required for list_custom_resources"
    manager.list_custom_resources.assert_not_called()


def test_cm_k8s_config_label_resource_dispatches_to_label_resource():
    manager, tool = _make_tool()
    manager.label_resource.return_value = "SENTINEL_RESULT_label_resource"
    result = asyncio.run(tool(
        action="label_resource",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_label_resource"
    manager.label_resource.assert_called_once_with(resource_type="resource_type_val", name="resource_name_val", namespace="namespace_val", labels={"env": "prod"})


def test_cm_k8s_config_label_resource_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="label_resource",
        resource_name=None, resource_type=None,
        ctx=None,
    ))
    assert result == "Error: 'resource_type' and 'resource_name' are required for label_resource"
    manager.label_resource.assert_not_called()


def test_cm_k8s_config_annotate_resource_dispatches_to_annotate_resource():
    manager, tool = _make_tool()
    manager.annotate_resource.return_value = "SENTINEL_RESULT_annotate_resource"
    result = asyncio.run(tool(
        action="annotate_resource",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_annotate_resource"
    manager.annotate_resource.assert_called_once_with(resource_type="resource_type_val", name="resource_name_val", namespace="namespace_val", annotations={"note": "hi"})


def test_cm_k8s_config_annotate_resource_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="annotate_resource",
        resource_name=None, resource_type=None,
        ctx=None,
    ))
    assert result == "Error: 'resource_type' and 'resource_name' are required for annotate_resource"
    manager.annotate_resource.assert_not_called()


def test_cm_k8s_config_patch_resource_dispatches_to_patch_resource():
    manager, tool = _make_tool()
    manager.patch_resource.return_value = "SENTINEL_RESULT_patch_resource"
    result = asyncio.run(tool(
        action="patch_resource",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_patch_resource"
    manager.patch_resource.assert_called_once_with(resource_type="resource_type_val", name="name_val", namespace="namespace_val", patch_body={"spec": {"replicas": 3}}, patch_type="patch_type_val")


def test_cm_k8s_config_patch_resource_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="patch_resource",
        name=None, resource_type=None,
        ctx=None,
    ))
    assert result == "Error: 'resource_type' and 'name' are required for patch_resource"
    manager.patch_resource.assert_not_called()


def test_cm_k8s_config_compare_configmap_state_dispatches_to_compare_configmap_state():
    manager, tool = _make_tool()
    manager.compare_configmap_state.return_value = "SENTINEL_RESULT_compare_configmap_state"
    result = asyncio.run(tool(
        action="compare_configmap_state",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_compare_configmap_state"
    manager.compare_configmap_state.assert_called_once_with("name_val", "namespace_val", {"expected_data_k": "expected_data_v"})


def test_cm_k8s_config_compare_configmap_state_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="compare_configmap_state",
        expected_data=None, name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name', 'namespace', and 'expected_data' are required for compare_configmap_state"
    manager.compare_configmap_state.assert_not_called()


def test_cm_k8s_config_sync_configmap_from_file_dispatches_to_sync_configmap_from_file():
    manager, tool = _make_tool()
    manager.sync_configmap_from_file.return_value = "SENTINEL_RESULT_sync_configmap_from_file"
    result = asyncio.run(tool(
        action="sync_configmap_from_file",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_sync_configmap_from_file"
    manager.sync_configmap_from_file.assert_called_once_with("name_val", "namespace_val", "file_path_val")


def test_cm_k8s_config_sync_configmap_from_file_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="sync_configmap_from_file",
        file_path=None, name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name', 'namespace', and 'file_path' are required for sync_configmap_from_file"
    manager.sync_configmap_from_file.assert_not_called()


def test_cm_k8s_config_get_secret_state_hash_dispatches_to_get_secret_state_hash():
    manager, tool = _make_tool()
    manager.get_secret_state_hash.return_value = "SENTINEL_RESULT_get_secret_state_hash"
    result = asyncio.run(tool(
        action="get_secret_state_hash",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_get_secret_state_hash"
    manager.get_secret_state_hash.assert_called_once_with("name_val", "namespace_val")


def test_cm_k8s_config_get_secret_state_hash_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="get_secret_state_hash",
        name=None, namespace=None,
        ctx=None,
    ))
    assert result == "Error: 'name' and 'namespace' are required for get_secret_state_hash"
    manager.get_secret_state_hash.assert_not_called()


def test_cm_k8s_config_track_resource_version_dispatches_to_track_resource_version():
    manager, tool = _make_tool()
    manager.track_resource_version.return_value = "SENTINEL_RESULT_track_resource_version"
    result = asyncio.run(tool(
        action="track_resource_version",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_track_resource_version"
    manager.track_resource_version.assert_called_once_with("resource_type_val", "name_val", "namespace_val")


def test_cm_k8s_config_track_resource_version_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="track_resource_version",
        name=None, resource_type=None,
        ctx=None,
    ))
    assert result == "Error: 'resource_type' and 'name' are required for track_resource_version"
    manager.track_resource_version.assert_not_called()


def test_cm_k8s_config_wait_for_resource_version_dispatches_to_wait_for_resource_version():
    manager, tool = _make_tool()
    manager.wait_for_resource_version.return_value = "SENTINEL_RESULT_wait_for_resource_version"
    result = asyncio.run(tool(
        action="wait_for_resource_version",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "SENTINEL_RESULT_wait_for_resource_version"
    manager.wait_for_resource_version.assert_called_once_with("resource_type_val", "name_val", "namespace_val", "target_version_val", 7)


def test_cm_k8s_config_wait_for_resource_version_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(tool(
        action="wait_for_resource_version",
        name=None, namespace=None, resource_type=None, target_version=None,
        ctx=None,
    ))
    assert result == "Error: 'resource_type', 'name', 'namespace', and 'target_version' are required for wait_for_resource_version"
    manager.wait_for_resource_version.assert_not_called()


def test_cm_k8s_config_unknown_action_returns_error():
    manager, tool = _make_tool()
    result = asyncio.run(tool(action="bogus_action_xyz", ctx=None))
    assert result == "Error: Unknown action 'bogus_action_xyz'"


def test_cm_k8s_config_manager_exception_is_caught_and_formatted():
    manager, tool = _make_tool()
    manager.list_configmaps.side_effect = RuntimeError("boom")
    result = asyncio.run(tool(
        action="list_configmaps",
        namespace="namespace_val", configmap_name="configmap_name_val", configmap_data='{"key1": "value1"}', configmap_from_file="configmap_from_file_val", secret_name="secret_name_val", secret_type="secret_type_val", secret_data='{"key1": "dmFsdWUx"}', namespace_name="namespace_name_val", field_selector="field_selector_val", crd_name="crd_name_val", crd_group="crd_group_val", crd_version="crd_version_val", crd_plural="crd_plural_val", resource_type="resource_type_val", resource_name="resource_name_val", labels='{"env": "prod"}', annotations='{"note": "hi"}', name="name_val", patch_body='{"spec": {"replicas": 3}}', patch_type="patch_type_val", expected_data={"expected_data_k": "expected_data_v"}, file_path="file_path_val", target_version="target_version_val", timeout=7,
        ctx=None,
    ))
    assert result == "Error executing list_configmaps: RuntimeError"


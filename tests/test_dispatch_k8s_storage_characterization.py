"""Characterization tests for cm_k8s_storage (CCN 40), before decomposition
into per-domain sub-dispatchers (mirrors the pattern already used by
mcp_k8s_config.py / mcp_k8s_governance.py / mcp_k8s_networking.py in this
package: `_ACTION_GROUPS` + `_GROUP_FUNCS` + `_GROUP_PARAM_NAMES`).

Pins: which `manager.<method>` each of the 16 actions calls and with exactly
which args, each required-param guard's exact error string with the manager
left uncalled, the unknown-action fallback, and manager-exception
formatting.
"""

import asyncio
from unittest.mock import MagicMock

import pytest


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


@pytest.fixture
def tool_and_manager(monkeypatch):
    from container_manager_mcp.mcp import mcp_k8s_storage

    fake_manager = MagicMock()
    monkeypatch.setattr(
        mcp_k8s_storage, "create_manager", lambda backend: fake_manager
    )
    tool = _capture_tool(mcp_k8s_storage.register_k8sstorage_tools)
    return tool, fake_manager


def _run(tool, **kwargs):
    return asyncio.run(tool(**kwargs))


def test_list_persistent_volumes_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    manager.list_persistent_volumes.return_value = []
    assert _run(tool, action="list_persistent_volumes") == []
    manager.list_persistent_volumes.assert_called_once_with()


def test_create_persistent_volume_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="create_persistent_volume", name="pv1", spec={"a": 1})
    manager.create_persistent_volume.assert_called_once_with("pv1", {"a": 1})


def test_create_persistent_volume_guard(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(tool, action="create_persistent_volume", name=None, spec=None)
    assert result == "Error: 'name' and 'spec' are required for create_persistent_volume"
    manager.create_persistent_volume.assert_not_called()


def test_list_persistent_volume_claims_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="list_persistent_volume_claims", namespace="ns1")
    manager.list_persistent_volume_claims.assert_called_once_with(namespace="ns1")


def test_create_persistent_volume_claim_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(
        tool,
        action="create_persistent_volume_claim",
        pvc_name="pvc1",
        namespace="ns1",
        pvc_spec='{"x": 1}',
    )
    manager.create_persistent_volume_claim.assert_called_once_with(
        name="pvc1", namespace="ns1", spec={"x": 1}
    )


def test_create_persistent_volume_claim_guard(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(tool, action="create_persistent_volume_claim", pvc_name=None)
    assert result == "Error: 'pvc_name' is required for create_persistent_volume_claim"
    manager.create_persistent_volume_claim.assert_not_called()


def test_delete_persistent_volume_claim_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="delete_persistent_volume_claim", pvc_name="pvc1", namespace="ns1")
    manager.delete_persistent_volume_claim.assert_called_once_with(
        name="pvc1", namespace="ns1"
    )


def test_delete_persistent_volume_claim_guard(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(tool, action="delete_persistent_volume_claim", pvc_name=None)
    assert result == "Error: 'pvc_name' is required for delete_persistent_volume_claim"


def test_expand_pvc_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="expand_pvc", pvc_name="pvc1", namespace="ns1", pvc_size="10Gi")
    manager.expand_pvc.assert_called_once_with(name="pvc1", namespace="ns1", size="10Gi")


def test_expand_pvc_guard(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(tool, action="expand_pvc", pvc_name=None, pvc_size=None)
    assert result == "Error: 'pvc_name' and 'pvc_size' are required for expand_pvc"


def test_expand_persistent_volume_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="expand_persistent_volume", name="pv1", namespace="ns1", size="10Gi")
    manager.expand_persistent_volume.assert_called_once_with("pv1", "ns1", "10Gi")


def test_expand_persistent_volume_guard(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(
        tool, action="expand_persistent_volume", name=None, namespace=None, size=None
    )
    assert (
        result
        == "Error: 'name', 'namespace', and 'size' are required for expand_persistent_volume"
    )


def test_list_storage_classes_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="list_storage_classes")
    manager.list_storage_classes.assert_called_once_with()


def test_create_storage_class_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="create_storage_class", name="sc1", provisioner="csi.driver", parameters={"p": 1})
    manager.create_storage_class.assert_called_once_with("sc1", "csi.driver", {"p": 1})


def test_create_storage_class_guard(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(tool, action="create_storage_class", name=None, provisioner=None)
    assert result == "Error: 'name' and 'provisioner' are required for create_storage_class"


def test_set_default_storage_class_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="set_default_storage_class", name="sc1")
    manager.set_default_storage_class.assert_called_once_with("sc1")


def test_set_default_storage_class_guard(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(tool, action="set_default_storage_class", name=None)
    assert result == "Error: 'name' is required for set_default_storage_class"


def test_get_storage_class_provisioner_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="get_storage_class_provisioner", name="sc1")
    manager.get_storage_class_provisioner.assert_called_once_with("sc1")


def test_get_storage_class_provisioner_guard(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(tool, action="get_storage_class_provisioner", name=None)
    assert result == "Error: 'name' is required for get_storage_class_provisioner"


def test_list_volume_snapshots_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="list_volume_snapshots", namespace="ns1")
    manager.list_volume_snapshots.assert_called_once_with(namespace="ns1")


def test_create_volume_snapshot_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="create_volume_snapshot", name="snap1", namespace="ns1", spec={"s": 1})
    manager.create_volume_snapshot.assert_called_once_with("snap1", "ns1", {"s": 1})


def test_create_volume_snapshot_guard(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(
        tool, action="create_volume_snapshot", name=None, namespace=None, spec=None
    )
    assert (
        result
        == "Error: 'name', 'namespace', and 'spec' are required for create_volume_snapshot"
    )


def test_list_csi_drivers_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="list_csi_drivers")
    manager.list_csi_drivers.assert_called_once_with()


def test_describe_csi_driver_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="describe_csi_driver", name="driver1")
    manager.describe_csi_driver.assert_called_once_with("driver1")


def test_describe_csi_driver_guard(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(tool, action="describe_csi_driver", name=None)
    assert result == "Error: 'name' is required for describe_csi_driver"


def test_get_csi_driver_capacity_dispatches(tool_and_manager):
    tool, manager = tool_and_manager
    _run(tool, action="get_csi_driver_capacity", driver_name="driver1")
    manager.get_csi_driver_capacity.assert_called_once_with("driver1")


def test_get_csi_driver_capacity_guard(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(tool, action="get_csi_driver_capacity", driver_name=None)
    assert result == "Error: 'driver_name' is required for get_csi_driver_capacity"


def test_unknown_action_returns_error(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(tool, action="bogus_action")
    assert result == "Error: Unknown action 'bogus_action'"


def test_manager_exception_is_caught_and_formatted(tool_and_manager):
    tool, manager = tool_and_manager
    manager.list_csi_drivers.side_effect = RuntimeError("boom")
    result = _run(tool, action="list_csi_drivers")
    assert result == "Error executing list_csi_drivers: RuntimeError"

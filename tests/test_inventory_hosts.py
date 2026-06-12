"""Discoverable remote-host targeting (cm_list_hosts + clear errors)."""

from unittest.mock import MagicMock, patch

import pytest

from container_manager_mcp.container_manager import (
    list_inventory_hosts,
    resolve_host_from_inventory,
)


def _fake_hm(hosts):
    hm = MagicMock()
    hm.hosts = hosts
    hm.config_file = "/cfg/inventory.yaml"
    return hm


def test_list_inventory_hosts_shapes_aliases():
    fake = _fake_hm(
        {
            "R820": {"hostname": "10.0.0.20", "user": "genius", "port": 22},
            "R710": {"hostname": "10.0.0.10"},  # user/port default
        }
    )
    with patch("tunnel_manager.tunnel_manager.HostManager", return_value=fake):
        out = list_inventory_hosts()
    assert out["count"] == 2
    assert out["inventory_path"] == "/cfg/inventory.yaml"
    assert out["hosts"]["R820"]["hostname"] == "10.0.0.20"
    assert out["hosts"]["R710"]["user"] == "genius"  # default
    assert out["hosts"]["R710"]["port"] == 22  # default


def test_list_inventory_hosts_empty():
    with patch("tunnel_manager.tunnel_manager.HostManager", return_value=_fake_hm({})):
        out = list_inventory_hosts()
    assert out == {
        "inventory_path": "/cfg/inventory.yaml",
        "count": 0,
        "hosts": {},
    }


def test_resolve_unknown_host_lists_available():
    fake = _fake_hm({"R820": {"hostname": "10.0.0.20"}})
    with patch("tunnel_manager.tunnel_manager.HostManager", return_value=fake):
        with pytest.raises(ValueError) as exc:
            resolve_host_from_inventory("does-not-exist")
    msg = str(exc.value)
    assert "R820" in msg  # tells you what IS available
    assert "cm_list_hosts" in msg  # points at the discovery tool

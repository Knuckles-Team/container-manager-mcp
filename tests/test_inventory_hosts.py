"""Discoverable remote-host targeting (cm_list_hosts + clear errors)."""

from unittest.mock import MagicMock, patch

import pytest

from container_manager_mcp.container_manager import (
    DockerManager,
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
            "R820": {"hostname": "10.0.0.20", "user": "fixture-user", "port": 22},
            "R710": {"hostname": "10.0.0.10"},  # user/port default
        }
    )
    with patch("tunnel_manager.tunnel_manager.HostManager", return_value=fake):
        out = list_inventory_hosts()
    assert out["count"] == 2
    assert out["inventory_path"] == "/cfg/inventory.yaml"
    assert out["hosts"]["R820"]["hostname"] == "10.0.0.20"
    assert out["hosts"]["R710"]["user"] is None  # no implicit account
    assert out["hosts"]["R710"]["port"] == 22  # default


def test_list_inventory_hosts_empty():
    with patch("tunnel_manager.tunnel_manager.HostManager", return_value=_fake_hm({})):
        out = list_inventory_hosts()
    assert out == {
        "inventory_path": "/cfg/inventory.yaml",
        "count": 0,
        "hosts": {},
    }


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R003")
@pytest.mark.parametrize(
    ("configured_user", "expected_authority"),
    [("fixture-user", "fixture-user@"), (None, "")],
)
def test_docker_host_preserves_only_configured_user(
    configured_user, expected_authority
):
    item = {"hostname": "example.invalid", "port": 22}
    if configured_user is not None:
        item["user"] = configured_user
    with (
        patch(
            "tunnel_manager.tunnel_manager.HostManager",
            return_value=_fake_hm({"example-host": item}),
        ),
        patch("container_manager_mcp.container_manager.docker") as docker,
    ):
        DockerManager(host="example-host")
    docker.DockerClient.assert_called_once_with(
        base_url=f"ssh://{expected_authority}example.invalid:22"
    )


def test_resolve_unknown_host_lists_available():
    fake = _fake_hm({"R820": {"hostname": "10.0.0.20"}})
    with patch("tunnel_manager.tunnel_manager.HostManager", return_value=fake):
        with pytest.raises(ValueError) as exc:
            resolve_host_from_inventory("does-not-exist")
    msg = str(exc.value)
    assert "R820" in msg  # tells you what IS available
    assert "cm_list_hosts" in msg  # points at the discovery tool

"""Discoverable remote-host targeting (cm_list_hosts + clear errors)."""

from unittest.mock import patch

import pytest
import yaml
from agent_connector_sdk.identity import SYSTEM_ACTOR, use_actor

from container_manager_mcp.container_manager import (
    DockerManager,
    list_inventory_hosts,
    resolve_host_from_inventory,
)


@pytest.fixture
def inventory(request, tmp_path, monkeypatch):
    """Point the real tunnel-manager HostManager at a flat inventory file."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    local = use_actor(SYSTEM_ACTOR)
    local.__enter__()
    request.addfinalizer(lambda: local.__exit__(None, None, None))
    path = tmp_path / "agent-connectors" / "inventory.yml"
    path.parent.mkdir()

    def write(hosts):
        path.write_text(yaml.safe_dump(hosts))
        return str(path)

    return write


def test_list_inventory_hosts_shapes_aliases(inventory):
    path = inventory(
        {
            "R820": {"hostname": "10.0.0.20", "user": "fixture-user", "port": 22},
            "R710": {"hostname": "10.0.0.10"},  # user/port default
        }
    )
    out = list_inventory_hosts()
    assert out["count"] == 2
    assert out["inventory_path"] == path
    assert out["hosts"]["R820"]["hostname"] == "10.0.0.20"
    assert out["hosts"]["R710"]["user"] is None  # no implicit account
    assert out["hosts"]["R710"]["port"] == 22  # default


def test_list_inventory_hosts_empty(inventory):
    path = inventory({})
    assert list_inventory_hosts() == {
        "inventory_path": path,
        "count": 0,
        "hosts": {},
    }


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R003")
@pytest.mark.parametrize(
    ("configured_user", "expected_authority"),
    [("fixture-user", "fixture-user@"), (None, "")],
)
def test_docker_host_preserves_only_configured_user(
    inventory, configured_user, expected_authority
):
    item = {"hostname": "example.invalid", "port": 22}
    if configured_user is not None:
        item["user"] = configured_user
    inventory({"example-host": item})
    with patch("container_manager_mcp.container_manager.docker") as docker:
        DockerManager(host="example-host")
    docker.DockerClient.assert_called_once_with(
        base_url=f"ssh://{expected_authority}example.invalid:22"
    )


def test_resolve_unknown_host_lists_available(inventory):
    inventory({"R820": {"hostname": "10.0.0.20"}})
    with pytest.raises(ValueError) as exc:
        resolve_host_from_inventory("does-not-exist")
    msg = str(exc.value)
    assert "R820" in msg  # tells you what IS available
    assert "cm_list_hosts" in msg  # points at the discovery tool


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R004")
def test_tunnel_inventory_env_selects_the_shared_inventory(
    inventory, tmp_path, monkeypatch
):
    """A deployment mounts the shared inventory at ``$TUNNEL_INVENTORY`` (as
    tunnel-manager does); cm_list_hosts and host resolution must read it rather
    than the empty XDG default."""
    inventory({})  # the XDG default exists but is empty
    shared = tmp_path / "mounted" / "inventory.yml"
    shared.parent.mkdir()
    shared.write_text(yaml.safe_dump({"R820": {"hostname": "10.0.0.20"}}))
    monkeypatch.setenv("TUNNEL_INVENTORY", str(shared))
    out = list_inventory_hosts()
    assert out["inventory_path"] == str(shared)
    assert out["count"] == 1
    assert resolve_host_from_inventory("R820")["hostname"] == "10.0.0.20"

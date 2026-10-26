"""The generic docker_*/k8s_* tools accept a per-call ``context`` argument.

They replace the retired ``cm_multi_context__*`` tools, which took ``backend``
and ``context``; the backend is in the tool prefix, the context is an argument.
"""

import sys

sys.argv = ["mcp_server.py"]

import asyncio
from unittest.mock import patch

import pytest

from container_manager_mcp.call_context import backend_for_tool
from container_manager_mcp.mcp_server import get_mcp_instance


@pytest.fixture(scope="module")
def server():
    from agent_connector_sdk.mcp.tool_surface import registered_tools

    mcp, _, _ = get_mcp_instance()
    return mcp, registered_tools(mcp)


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R001.1")
def test_generic_tools_advertise_context(server):
    _, tools = server
    generic = {n: t for n, t in tools.items() if backend_for_tool(n)}
    assert len(generic) > 100
    for name in ("k8s_list_pods", "k8s_describe_pod", "docker_list_containers"):
        assert "context" in tools[name].parameters["properties"], name
        assert "context" not in tools[name].parameters.get("required", [])
    missing = [
        n for n, t in generic.items() if "context" not in t.parameters["properties"]
    ]
    assert not missing
    required = [
        n for n, t in generic.items() if "context" in t.parameters.get("required", [])
    ]
    assert not required, f"context is required for generic tools: {required}"


class _FakeManager:
    def list_containers(self, all=False):
        return []

    def list_pods(self, **kwargs):
        return []


def _call(mcp, name, arguments):
    from fastmcp import Client

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments, raise_on_error=False)

    return asyncio.run(run())


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R001.2")
@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R001.4")
@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R001.5")
def test_context_argument_selects_the_named_context(server):
    mcp, _ = server
    seen = []

    def fake_get_manager(self, backend="kubernetes", context=None):
        seen.append((backend, context))
        return _FakeManager()

    with (
        patch(
            "container_manager_mcp.multi_context_manager.MultiContextManager.get_manager",
            fake_get_manager,
        ),
        patch(
            "container_manager_mcp.multi_context_manager.MultiContextManager.__init__",
            lambda self, *a, **k: None,
        ),
    ):
        result = _call(mcp, "k8s_list_pods", {"context": "prod-east"})
        assert not result.is_error, result
        result = _call(mcp, "docker_list_containers", {"context": "edge-host"})
        assert not result.is_error, result
    assert seen == [("kubernetes", "prod-east"), ("docker", "edge-host")]


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R001.3")
def test_without_context_the_default_manager_is_used(server):
    mcp, _ = server
    created = []

    def fake_create_manager(backend=None, *a, **k):
        created.append(backend)
        return _FakeManager()

    with patch(
        "container_manager_mcp.container_manager.create_manager", fake_create_manager
    ):
        result = _call(mcp, "k8s_list_pods", {})
    assert not result.is_error, result
    assert created == ["kubernetes"]

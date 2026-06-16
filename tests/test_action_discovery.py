"""Tests for standardized action discovery on action-routed tools.

Each action-routed cm_* tool should:
  * return a {"service", "actions"} payload when called with
    action='list_actions' (discovery), and
  * raise a ValueError mentioning 'list_actions' on an unknown action.

The tools are closures registered via register_*_tools(mcp); we capture the
decorated function with a MagicMock FastMCP whose .tool decorator is identity.
"""

import asyncio
import importlib
from unittest.mock import MagicMock

import pytest

mcp_server = importlib.import_module("container_manager_mcp.mcp_server")


def _capture_tool(register_fn):
    """Run a register_*_tools function against a fake mcp and return the tool."""
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


# (register function, kwargs needed for a real call besides action)
ACTION_TOOLS = [
    (mcp_server.register_info_tools, {}),
    (mcp_server.register_image_tools, {}),
    (mcp_server.register_container_tools, {}),
    (mcp_server.register_volume_tools, {}),
    (mcp_server.register_network_tools, {}),
    (mcp_server.register_swarm_tools, {}),
    (mcp_server.register_system_tools, {}),
    (mcp_server.register_compose_tools, {}),
]


@pytest.mark.parametrize("register_fn,extra", ACTION_TOOLS)
def test_list_actions_returns_names(register_fn, extra):
    tool = _capture_tool(register_fn)
    result = asyncio.run(tool(action="list_actions", **extra))
    assert isinstance(result, dict)
    assert result["service"] == "container-manager-mcp"
    assert isinstance(result["actions"], list)
    assert len(result["actions"]) > 0


@pytest.mark.parametrize("register_fn,extra", ACTION_TOOLS)
def test_bogus_action_raises_with_discovery_hint(register_fn, extra):
    tool = _capture_tool(register_fn)
    with pytest.raises(ValueError) as exc:
        asyncio.run(tool(action="definitely_not_a_real_action", **extra))
    assert "list_actions" in str(exc.value)


def test_plural_alias_is_canonicalized():
    """A plural form should resolve to the singular canonical action.

    'remove_images' (plural) must canonicalize to 'remove_image' instead of
    raising the unknown-action ValueError. We assert solely that resolve_action
    does NOT raise (it would for a truly unknown action), proving the alias was
    accepted; the downstream manager call is irrelevant to discovery.
    """
    tool = _capture_tool(mcp_server.register_image_tools)
    try:
        asyncio.run(tool(action="remove_images", manager_type="docker"))
    except ValueError as exc:
        assert "Unknown action" not in str(exc)
    except Exception:
        # Any non-ValueError (e.g. a real Docker/manager error) means the
        # action was accepted and dispatch proceeded past resolve_action.
        pass

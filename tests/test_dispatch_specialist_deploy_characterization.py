"""Characterization tests for deploy_specialist_container (CCN 11), before
decomposition.

Pins: successful deploy (pull_image + run_container called with the parsed
port/env/label maps, inverted port bindings, and the auto-generated
container name when none is given), invalid-JSON parameters return a
success=False error dict without touching the manager, and a manager
exception is caught and reported generically.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


def _capture_tool(register_fn):
    captured = {}

    def tool_decorator(*args, **kwargs):
        def wrapper(fn):
            captured[fn.__name__] = fn
            return fn

        return wrapper

    fake_mcp = MagicMock()
    fake_mcp.tool = tool_decorator
    register_fn(fake_mcp)
    return captured


def _run(tool, **kwargs):
    return asyncio.run(tool(**kwargs))


@pytest.fixture
def tool_and_manager():
    from container_manager_mcp import specialist_tools

    fake_manager = MagicMock()
    fake_manager.pull_image.return_value = None
    fake_manager.run_container.return_value = {"id": "abc123"}
    with patch(
        "container_manager_mcp.container_manager.create_manager",
        return_value=fake_manager,
    ):
        tools = _capture_tool(specialist_tools.register_specialist_deployment_tools)
        yield tools["deploy_specialist_container"], fake_manager


def test_deploy_success_with_explicit_name(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(
        tool,
        image="registry.example.invalid/svc@sha256:deadbeef",
        name="svc1",
        ports='{"8004": "8004"}',
        env='{"X": "1"}',
        labels='{"managed-by": "agent-os"}',
    )
    manager.pull_image.assert_called_once_with("registry.example.invalid/svc@sha256:deadbeef")
    manager.run_container.assert_called_once_with(
        image="registry.example.invalid/svc@sha256:deadbeef",
        name="svc1",
        detach=True,
        ports={"8004": "8004"},
        environment=["X=1"],
        labels={"managed-by": "agent-os"},
    )
    assert result == {
        "success": True,
        "container_id": "abc123",
        "name": "svc1",
        "image": "registry.example.invalid/svc@sha256:deadbeef",
        "ports": {"8004": "8004"},
        "labels": {"managed-by": "agent-os"},
        "status": "running",
    }


def test_deploy_auto_generates_name_from_image(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(
        tool,
        image="registry.example.invalid/my-svc:latest",
        name="",
        ports="{}",
        env="{}",
        labels="{}",
    )
    assert result["name"] == "specialist-my-svc"
    manager.run_container.assert_called_once_with(
        image="registry.example.invalid/my-svc:latest",
        name="specialist-my-svc",
        detach=True,
        ports=None,
        environment=None,
        labels={},
    )


def test_deploy_invalid_json_returns_error_without_touching_manager(tool_and_manager):
    tool, manager = tool_and_manager
    result = _run(
        tool,
        image="registry.example.invalid/svc:latest",
        ports="not json",
        env="{}",
        labels="{}",
    )
    assert result["success"] is False
    assert "Invalid JSON parameter" in result["error"]
    manager.pull_image.assert_not_called()
    manager.run_container.assert_not_called()


def test_deploy_manager_exception_is_caught(tool_and_manager):
    tool, manager = tool_and_manager
    manager.pull_image.side_effect = RuntimeError("boom")
    result = _run(
        tool,
        image="registry.example.invalid/svc:latest",
        ports="{}",
        env="{}",
        labels="{}",
    )
    assert result == {"success": False, "error": "Operation failed"}

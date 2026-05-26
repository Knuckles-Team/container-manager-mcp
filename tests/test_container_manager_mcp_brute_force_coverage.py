import asyncio
import inspect
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure 'docker' and 'podman' modules exist in sys.modules so patch() can resolve them
# even when docker-py or podman-py is not installed. The production code handles this
# with try/except ImportError, but unittest.mock.patch needs the parent module to be importable.
if "docker" not in sys.modules or not hasattr(sys.modules.get("docker"), "from_env"):
    _fake_docker = types.ModuleType("docker")
    _fake_docker.from_env = MagicMock  # type: ignore[attr-defined]
    _fake_docker_errors = types.ModuleType("docker.errors")
    _fake_docker_errors.DockerException = Exception  # type: ignore[attr-defined]
    _fake_docker.errors = _fake_docker_errors  # type: ignore[attr-defined]
    sys.modules["docker"] = _fake_docker
    sys.modules["docker.errors"] = _fake_docker_errors

if "podman" not in sys.modules:
    _fake_podman = types.ModuleType("podman")
    _fake_podman.PodmanClient = MagicMock  # type: ignore[attr-defined]
    _fake_errors = types.ModuleType("podman.errors")
    _fake_errors.PodmanError = Exception  # type: ignore[attr-defined]
    _fake_podman.errors = _fake_errors  # type: ignore[attr-defined]
    sys.modules["podman"] = _fake_podman
    sys.modules["podman.errors"] = _fake_errors


@pytest.fixture
def mock_container_deps():
    mock_docker = MagicMock()
    mock_podman = MagicMock()
    with (
        patch(
            "container_manager_mcp.container_manager.docker", mock_docker, create=True
        ),
        patch(
            "container_manager_mcp.container_manager.podman", mock_podman, create=True
        ),
        patch("shutil.which", return_value="/usr/bin/docker"),
    ):
        # Mock Docker Client
        docker_client = mock_docker.from_env.return_value
        docker_client.containers.list.return_value = []
        docker_client.images.list.return_value = []
        docker_client.volumes.list.return_value = []
        docker_client.networks.list.return_value = []

        # Mock Podman Client
        podman_client = mock_podman.PodmanClient.return_value
        podman_client.containers.list.return_value = []

        yield mock_docker.from_env, mock_podman.PodmanClient


def test_container_manager_brute_force(mock_container_deps):
    from container_manager_mcp.container_manager import DockerManager, PodmanManager

    # PodmanClient is None at module level when podman-py isn't installed.
    # Patch it to a MagicMock so PodmanManager.__init__'s guard check passes.
    with (
        patch(
            "container_manager_mcp.container_manager.PodmanClient",
            MagicMock(),
        ),
        patch.object(
            PodmanManager,
            "_autodetect_podman_url",
            return_value="unix:///tmp/dummy.sock",
        ),
    ):
        managers = [DockerManager(silent=True), PodmanManager(silent=True)]

    common_kwargs = {
        "container_id": "test_id",
        "image": "nginx:latest",
        "command": ["ls"],
        "name": "test_name",
        "force": True,
        "all": True,
        "tail": "100",
        "timeout": 10,
        "detach": True,
        "network_id": "test_net",
        "driver": "bridge",
    }

    for manager in managers:
        manager_name = manager.__class__.__name__
        for name, method in inspect.getmembers(manager, predicate=inspect.ismethod):
            if name.startswith("_") or name == "setup_logging":
                continue
            print(f"Calling {manager_name}.{name}...")
            sig = inspect.signature(method)
            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if has_kwargs:
                kwargs = common_kwargs.copy()
            else:
                kwargs = {k: v for k, v in common_kwargs.items() if k in sig.parameters}
                for p_name, p in sig.parameters.items():
                    if p.default == inspect.Parameter.empty and p_name not in kwargs:
                        kwargs[p_name] = "test" if p.annotation is str else 1
            try:
                method(**kwargs)
            except Exception:
                pass


def test_mcp_server_coverage(mock_container_deps):
    _ = mock_container_deps
    from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware

    from container_manager_mcp.mcp_server import get_mcp_instance

    # Patch RateLimitingMiddleware to do nothing
    async def mock_on_request(self, context, call_next):
        return await call_next(context)

    with patch.object(RateLimitingMiddleware, "on_request", mock_on_request):
        # Patch create_manager in mcp_server to return a mock manager
        with patch("container_manager_mcp.mcp_server.create_manager") as mock_cm:
            mock_manager = MagicMock()
            mock_cm.return_value = mock_manager

            mcp_data = get_mcp_instance()
            mcp = mcp_data[1] if isinstance(mcp_data, tuple) else mcp_data

            async def run_tools():
                tool_objs = (
                    await mcp.list_tools()
                    if inspect.iscoroutinefunction(mcp.list_tools)
                    else mcp.list_tools()
                )
                for tool in tool_objs:
                    try:
                        target_params: dict[str, Any] = {
                            "container_id": "test_id",
                            "image": "nginx",
                            "tag": "latest",
                            "command": ["ls"],
                            "manager_type": "docker",
                            "network_id": "test_net",
                            "volume_name": "test_vol",
                        }
                        sig = inspect.signature(tool.fn)
                        for p_name, p in sig.parameters.items():
                            if p.default == inspect.Parameter.empty and p_name not in [
                                "_client",
                                "context",
                            ]:
                                if p_name not in target_params:
                                    target_params[p_name] = (
                                        "test" if p.annotation is str else 1
                                    )

                        has_kwargs = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in sig.parameters.values()
                        )
                        if not has_kwargs:
                            target_params = {
                                k: v
                                for k, v in target_params.items()
                                if k in sig.parameters
                            }

                        await mcp.call_tool(tool.name, target_params)
                    except Exception:
                        pass

            asyncio.run(run_tools())


@patch("container_manager_mcp.agent_server.create_agent_server")
@patch("container_manager_mcp.agent_server.create_agent_parser")
@patch("container_manager_mcp.agent_server.load_identity")
@patch("container_manager_mcp.agent_server.initialize_workspace")
def test_agent_server_coverage(
    _mock_init_workspace,
    mock_load_identity,
    mock_create_parser,
    mock_create_server,
):
    from container_manager_mcp.agent_server import agent_server

    mock_load_identity.return_value = {
        "name": "Test Agent",
        "description": "Test Description",
        "content": "Test system prompt",
    }
    mock_parser = MagicMock()
    mock_args = MagicMock()
    mock_args.debug = False
    mock_args.mcp_url = "http://localhost:8000/mcp"
    mock_args.mcp_config = None
    mock_args.host = "0.0.0.0"
    mock_args.port = 9000
    mock_args.provider = "openai"
    mock_args.model_id = "gpt-4"
    mock_args.base_url = None
    mock_args.api_key = None
    mock_args.custom_skills_directory = None
    mock_args.web = False
    mock_args.otel = False
    mock_args.otel_endpoint = None
    mock_args.otel_headers = None
    mock_args.otel_public_key = None
    mock_args.otel_secret_key = None
    mock_args.otel_protocol = None
    mock_parser.parse_args.return_value = mock_args
    mock_create_parser.return_value = mock_parser

    with patch("sys.argv", ["agent_server.py"]):
        agent_server()
        assert mock_create_server.called


def test_main_coverage():
    from container_manager_mcp.container_manager import container_manager

    with (
        patch("sys.argv", ["container_manager.py", "--get-version"]),
        patch("container_manager_mcp.container_manager.create_manager") as mock_cm,
    ):
        mock_manager = mock_cm.return_value
        mock_manager.get_version.return_value = {"version": "test"}
        try:
            container_manager()
        except SystemExit:
            pass

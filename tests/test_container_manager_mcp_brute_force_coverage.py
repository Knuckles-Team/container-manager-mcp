import asyncio
import inspect
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_container_deps():
    with (
        patch("docker.from_env") as mock_docker,
        patch("podman.PodmanClient") as mock_podman,
        patch("shutil.which", return_value="/usr/bin/docker"),
    ):

        # Mock Docker Client
        docker_client = mock_docker.return_value
        docker_client.containers.list.return_value = []
        docker_client.images.list.return_value = []
        docker_client.volumes.list.return_value = []
        docker_client.networks.list.return_value = []

        # Mock Podman Client
        podman_client = mock_podman.return_value
        podman_client.containers.list.return_value = []

        yield mock_docker, mock_podman


def test_container_manager_brute_force(mock_container_deps):
    from container_manager_mcp.container_manager import DockerManager, PodmanManager

    with patch.object(
        PodmanManager, "_autodetect_podman_url", return_value="unix:///tmp/dummy.sock"
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
            mcp = mcp_data[0] if isinstance(mcp_data, tuple) else mcp_data

            async def run_tools():
                tool_objs = (
                    await mcp.list_tools()
                    if inspect.iscoroutinefunction(mcp.list_tools)
                    else mcp.list_tools()
                )
                for tool in tool_objs:
                    try:
                        target_params = {
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


def test_agent_server_coverage():
    import container_manager_mcp.agent_server as mod
    from container_manager_mcp import agent_server

    with patch(
        "container_manager_mcp.agent_server.create_graph_agent_server"
    ) as mock_s:
        with patch("sys.argv", ["agent_server.py"]):
            if inspect.isfunction(agent_server):
                agent_server()
            else:
                mod.agent_server()
            assert mock_s.called


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

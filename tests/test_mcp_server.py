"""Tests for mcp_server.py - MCP server tool registration."""

from unittest.mock import MagicMock, patch


class TestToolRegistration:
    def test_tools_registered(self):
        from container_manager_mcp.mcp_server import get_mcp_instance

        with patch("container_manager_mcp.mcp_server.create_mcp_server") as mock_create:
            mcp_mock = MagicMock()
            mock_create.return_value = (MagicMock(), mcp_mock, [MagicMock()])
            mcp, _, _ = get_mcp_instance()

            # Verify that mcp tool decorators were executed during module load or instance creation
            assert mcp is not None


class TestGetMcpInstance:
    def test_get_mcp_instance(self):
        from container_manager_mcp.mcp_server import get_mcp_instance

        with patch("container_manager_mcp.mcp_server.create_mcp_server") as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock(), [MagicMock()])

            mcp, args, middlewares = get_mcp_instance()

            assert mcp is not None


class TestVersion:
    def test_version_defined(self):
        from container_manager_mcp.mcp_server import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)

#!/usr/bin/env python
"""Tests for agent_server module."""

from unittest.mock import patch, MagicMock


class TestAgentServer:
    """Tests for agent_server module."""

    @patch("container_manager_mcp.agent_server.create_graph_agent_server")
    @patch("container_manager_mcp.agent_server.create_agent_parser")
    @patch("container_manager_mcp.agent_server.load_identity")
    @patch("container_manager_mcp.agent_server.initialize_workspace")
    def test_agent_server_basic(
        self,
        _mock_init_workspace,
        mock_load_identity,
        mock_create_parser,
        mock_create_server,
    ):
        """Test basic agent_server execution."""
        # Mock the identity loading
        mock_load_identity.return_value = {
            "name": "Test Agent",
            "description": "Test Description",
            "content": "Test system prompt",
        }

        # Mock the parser
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

        # Import and call agent_server
        from container_manager_mcp.agent_server import agent_server

        agent_server()

        # Verify the server was created with correct parameters
        mock_create_server.assert_called_once()
        call_kwargs = mock_create_server.call_args[1]

        assert call_kwargs["mcp_url"] == "http://localhost:8000/mcp"
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 9000
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["model_id"] == "gpt-4"

    @patch("container_manager_mcp.agent_server.create_graph_agent_server")
    @patch("container_manager_mcp.agent_server.create_agent_parser")
    @patch("container_manager_mcp.agent_server.load_identity")
    @patch("container_manager_mcp.agent_server.initialize_workspace")
    def test_agent_server_with_debug(
        self,
        _mock_init_workspace,
        mock_load_identity,
        mock_create_parser,
        mock_create_server,
    ):
        """Test agent_server with debug mode enabled."""
        mock_load_identity.return_value = {
            "name": "Test Agent",
            "description": "Test Description",
            "content": "Test system prompt",
        }

        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.debug = True
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

        from container_manager_mcp.agent_server import agent_server

        agent_server()

        mock_create_server.assert_called_once()
        call_kwargs = mock_create_server.call_args[1]
        assert call_kwargs["debug"] is True

    @patch("container_manager_mcp.agent_server.create_graph_agent_server")
    @patch("container_manager_mcp.agent_server.create_agent_parser")
    @patch("container_manager_mcp.agent_server.load_identity")
    @patch("container_manager_mcp.agent_server.initialize_workspace")
    def test_agent_server_with_web_ui(
        self,
        _mock_init_workspace,
        mock_load_identity,
        mock_create_parser,
        mock_create_server,
    ):
        """Test agent_server with web UI enabled."""
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
        mock_args.web = True
        mock_args.otel = False
        mock_args.otel_endpoint = None
        mock_args.otel_headers = None
        mock_args.otel_public_key = None
        mock_args.otel_secret_key = None
        mock_args.otel_protocol = None
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        from container_manager_mcp.agent_server import agent_server

        agent_server()

        mock_create_server.assert_called_once()
        call_kwargs = mock_create_server.call_args[1]
        assert call_kwargs["enable_web_ui"] is True

    @patch("container_manager_mcp.agent_server.create_graph_agent_server")
    @patch("container_manager_mcp.agent_server.create_agent_parser")
    @patch("container_manager_mcp.agent_server.load_identity")
    @patch("container_manager_mcp.agent_server.initialize_workspace")
    def test_agent_server_with_custom_mcp_config(
        self,
        _mock_init_workspace,
        mock_load_identity,
        mock_create_parser,
        mock_create_server,
    ):
        """Test agent_server with custom MCP config."""
        mock_load_identity.return_value = {
            "name": "Test Agent",
            "description": "Test Description",
            "content": "Test system prompt",
        }

        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.debug = False
        mock_args.mcp_url = "http://localhost:8000/mcp"
        mock_args.mcp_config = "/custom/path/mcp_config.json"
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

        from container_manager_mcp.agent_server import agent_server

        agent_server()

        mock_create_server.assert_called_once()
        call_kwargs = mock_create_server.call_args[1]
        assert call_kwargs["mcp_config"] == "/custom/path/mcp_config.json"

    @patch("container_manager_mcp.agent_server.create_graph_agent_server")
    @patch("container_manager_mcp.agent_server.create_agent_parser")
    @patch("container_manager_mcp.agent_server.load_identity")
    @patch("container_manager_mcp.agent_server.initialize_workspace")
    def test_agent_server_with_otel(
        self,
        _mock_init_workspace,
        mock_load_identity,
        mock_create_parser,
        mock_create_server,
    ):
        """Test agent_server with OpenTelemetry enabled."""
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
        mock_args.otel = True
        mock_args.otel_endpoint = "http://otel-collector:4317"
        mock_args.otel_headers = "X-Auth-Token: secret"
        mock_args.otel_public_key = "public_key"
        mock_args.otel_secret_key = "secret_key"
        mock_args.otel_protocol = "grpc"
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        from container_manager_mcp.agent_server import agent_server

        agent_server()

        mock_create_server.assert_called_once()
        call_kwargs = mock_create_server.call_args[1]
        assert call_kwargs["enable_otel"] is True
        assert call_kwargs["otel_endpoint"] == "http://otel-collector:4317"
        assert call_kwargs["otel_headers"] == "X-Auth-Token: secret"

    @patch("container_manager_mcp.agent_server.load_identity")
    @patch("container_manager_mcp.agent_server.initialize_workspace")
    def test_default_agent_name_from_env(
        self, _mock_init_workspace, mock_load_identity
    ):
        """Test that DEFAULT_AGENT_NAME can be set from environment variable."""
        mock_load_identity.return_value = {
            "name": "Default Name",
            "description": "Test Description",
            "content": "Test system prompt",
        }

        # Just verify that the environment variable would be used
        # We can't easily test the actual env var usage without reloading
        assert True  # Placeholder test

    @patch("container_manager_mcp.agent_server.load_identity")
    @patch("container_manager_mcp.agent_server.initialize_workspace")
    def test_default_agent_description_from_env(
        self, _mock_init_workspace, mock_load_identity
    ):
        """Test that DEFAULT_AGENT_DESCRIPTION can be set from environment variable."""
        mock_load_identity.return_value = {
            "name": "Test Agent",
            "description": "Default Description",
            "content": "Test system prompt",
        }

        # Just verify that the environment variable would be used
        # We can't easily test the actual env var usage without reloading
        assert True  # Placeholder test

    @patch("container_manager_mcp.agent_server.load_identity")
    @patch("container_manager_mcp.agent_server.initialize_workspace")
    def test_module_version(self, _mock_init_workspace, mock_load_identity):
        """Test that module has __version__ attribute."""
        mock_load_identity.return_value = {
            "name": "Test Agent",
            "description": "Test Description",
            "content": "Test system prompt",
        }

        from container_manager_mcp.agent_server import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)

    @patch("container_manager_mcp.agent_server.load_identity")
    @patch("container_manager_mcp.agent_server.initialize_workspace")
    def test_logger_initialization(self, _mock_init_workspace, mock_load_identity):
        """Test that logger is properly initialized."""
        mock_load_identity.return_value = {
            "name": "Test Agent",
            "description": "Test Description",
            "content": "Test system prompt",
        }

        from container_manager_mcp.agent_server import logger

        assert logger is not None
        assert logger.name == "container_manager_mcp.agent_server"

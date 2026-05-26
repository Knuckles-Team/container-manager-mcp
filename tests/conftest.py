"""Shared test fixtures for Container Manager Mcp."""

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("CONTAINER_URL", "https://test.example.com")
    monkeypatch.setenv("CONTAINER_TOKEN", "test-token-12345")
    monkeypatch.setenv("CONTAINER_SSL_VERIFY", "False")

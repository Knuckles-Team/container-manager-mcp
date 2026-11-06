"""Shared test fixtures for Container Manager Mcp."""

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("CONTAINER_URL", "https://test.example.com")
    monkeypatch.setenv("CONTAINER_TOKEN", "test-token-12345")
    monkeypatch.setenv("CONTAINER_SSL_VERIFY", "False")


@pytest.fixture(autouse=True)
def _fresh_context_pool():
    """Each test builds its own process-wide context pool."""
    from container_manager_mcp import multi_context_manager

    multi_context_manager._POOLS.clear()
    yield
    multi_context_manager._POOLS.clear()

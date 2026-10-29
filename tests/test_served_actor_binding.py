"""The served MCP stack binds one actor per request, by transport.

Each test builds the real container-manager server and calls the host tools
through a real fastmcp client: in memory (the local transport, bound to the
SYSTEM actor) and over streamable HTTP served by uvicorn, with no token (the
ANONYMOUS actor) or with a JWT (an authenticated actor whose roles come from the
token). Mirrors scion-sdk ``tests/test_actor_binding_served.py``.
"""

from __future__ import annotations

import functools
import importlib
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest
import uvicorn
import yaml
from agent_connector_sdk.mcp.served_security import ServedSecurityOverride
from fastmcp import Client
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from mcp.shared.exceptions import MCPError

mcp_server = importlib.import_module("container_manager_mcp.mcp_server")

ALIASES = ("prod-a", "prod-b", "lab", "edge")
ISSUER = "https://idp.example.invalid/realms/demo"
AUDIENCE = "container-manager-hosts"


@pytest.fixture(autouse=True)
def _inventory(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "agent-connectors" / "inventory.yml"
    path.parent.mkdir()
    path.write_text(
        yaml.safe_dump(
            {a: {"hostname": f"{a}.example.invalid", "user": "ops"} for a in ALIASES}
        )
    )


def _server(
    monkeypatch: pytest.MonkeyPatch, *argv: str, anonymous: bool = False
) -> Any:
    monkeypatch.setattr(sys, "argv", ["container-manager-mcp", *argv])
    if anonymous:
        monkeypatch.setattr(
            mcp_server,
            "create_mcp_server",
            functools.partial(
                mcp_server.create_mcp_server,
                served_security_override=ServedSecurityOverride(
                    "test serves HTTP without auth"
                ),
            ),
        )
    mcp, _args, _middlewares = mcp_server.get_mcp_instance()
    return mcp


@contextmanager
def _serving(mcp: Any) -> Iterator[str]:
    ready = threading.Event()
    server = uvicorn.Server(
        uvicorn.Config(
            mcp.http_app(),
            host="127.0.0.1",
            port=0,
            log_level="error",
            lifespan="on",
            callback_notify=AsyncMock(side_effect=ready.set),
            timeout_notify=1,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        assert ready.wait(30) and server.started
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(10)


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R003")
async def test_local_transport_lists_every_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp = _server(monkeypatch, "--transport", "stdio")
    async with Client(mcp) as client:
        listed = await client.call_tool("cm_list_hosts", {})
    assert set(listed.data["hosts"]) == set(ALIASES)
    assert listed.data["count"] == len(ALIASES)


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R003")
async def test_anonymous_http_lists_nothing_and_a_single_alias_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp = _server(monkeypatch, "--transport", "streamable-http", anonymous=True)
    with _serving(mcp) as endpoint:
        async with Client(endpoint) as client:
            listed = await client.call_tool("cm_list_hosts", {})
            assert listed.data["hosts"] == {}
            with pytest.raises(MCPError) as denied:
                await client.call_tool(
                    "cm_ingest_inventory", {"host": "lab"}
                )
    assert denied.value.code == -32000
    assert "not entitled" in denied.value.message
    assert "actor context is required" not in denied.value.message
    assert "Internal error" not in denied.value.message


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R003")
async def test_authenticated_http_lists_the_entitled_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = RSAKeyPair.generate()
    mcp = _server(
        monkeypatch,
        "--transport",
        "streamable-http",
        "--auth-type",
        "jwt",
        "--token-issuer",
        ISSUER,
        "--token-audience",
        AUDIENCE,
        "--token-public-key",
        keys.public_key,
    )
    token = keys.create_token(
        subject="operator-1",
        issuer=ISSUER,
        audience=AUDIENCE,
        additional_claims={
            "tenant_id": "tenant-1",
            "realm_access": {"roles": ["ssh:prod-a"]},
            "resource_access": {"hosts": {"roles": ["ssh:edge"]}},
            "groups": ["/lab", "/prod-b-operators"],
        },
    )
    with _serving(mcp) as endpoint:
        async with Client(endpoint, auth=token) as client:
            listed = await client.call_tool("cm_list_hosts", {})
            assert set(listed.data["hosts"]) == {"prod-a", "lab", "edge"}
            with pytest.raises(MCPError) as denied:
                await client.call_tool(
                    "cm_ingest_inventory", {"host": "prod-b"}
                )
    assert denied.value.code == -32000
    assert "not entitled" in denied.value.message

"""Destructive confirmation over real transports (CONN-CONTAINER-MANAGER-MCP-R007).

The real MCP server runs as a subprocess three ways: stdio, streamable HTTP
with ``--auth-type none`` on 127.0.0.1, and streamable HTTP with JWT auth
against a local JWKS endpoint. A real fastmcp client calls ``cm_volume_operations``
``remove_volume`` with and without elicitation. A tiny in-test HTTP server stands
in for the Docker daemon (``DOCKER_HOST``) and records every request, so the tests
can prove that a refused call never reaches the manager and that ``confirm=true``
does.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from fastmcp.server.auth.providers.jwt import RSAKeyPair

SPEC = "CONN-CONTAINER-MANAGER-MCP-R007"
ISSUER = "https://idp.example.invalid/realms/homelab"
AUDIENCE = "agent-services"
CALL_TIMEOUT = 90
VOLUME = "scratch-volume"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeDocker:
    """A stand-in Docker daemon that records the requests it receives."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                outer.requests.append((self.command, self.path))
                if self.command == "DELETE":
                    self.send_response(204)
                    self.end_headers()
                    return
                body = json.dumps(
                    {"Name": VOLUME, "ApiVersion": "1.43", "Version": "25.0.0"}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = do_DELETE = do_HEAD = _answer  # noqa: N815

            def log_message(self, *_args: Any) -> None:
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"tcp://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def deletes(self) -> list[str]:
        return [path for method, path in self.requests if method == "DELETE"]


def _jwks(keys: RSAKeyPair) -> dict[str, Any]:
    from cryptography.hazmat.primitives import serialization

    public = serialization.load_pem_public_key(keys.public_key.encode())
    numbers = public.public_numbers()  # type: ignore[union-attr]

    def b64(value: int) -> str:
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": "test-key",
                "use": "sig",
                "alg": "RS256",
                "n": b64(numbers.n),
                "e": b64(numbers.e),
            }
        ]
    }


def _serve_jwks(document: dict[str, Any]) -> tuple[HTTPServer, str]:
    body = json.dumps(document).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/jwks"


class _World:
    def __init__(self) -> None:
        self.keys = RSAKeyPair.generate()
        self.jwks_server, self.jwks_uri = _serve_jwks(_jwks(self.keys))

    def token(self) -> str:
        return self.keys.create_token(
            subject="arbora-service-account-id",
            issuer=ISSUER,
            audience=AUDIENCE,
            additional_claims={
                "azp": "arbora",
                "client_id": "arbora",
                "preferred_username": "service-account-arbora",
                "typ": "Bearer",
                "tenant_id": "homelab",
            },
            kid="test-key",
        )


@pytest.fixture(scope="module")
def world() -> Iterator[_World]:
    built = _World()
    yield built
    built.jwks_server.shutdown()


@pytest.fixture
def docker() -> Iterator[_FakeDocker]:
    fake = _FakeDocker()
    yield fake
    fake.server.shutdown()


def _executable() -> str:
    found = Path(sys.executable).parent / "container-manager-mcp"
    return str(found) if found.exists() else shutil.which("container-manager-mcp") or ""


def _server_env(tmp: Path, docker: _FakeDocker) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("CONTAINER_", "AUTH_", "FASTMCP_", "CONNECTOR_", "MCP_", "DOCKER_")
        )
    }
    home = tmp / "home"
    home.mkdir(parents=True, exist_ok=True)
    env.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(tmp / "config"),
        DOCKER_HOST=docker.host,
        MCP_FLEET_REGISTRATION="false",
        CONNECTOR_PRINCIPAL_GRANTS="arbora=*",
        PYTHONUNBUFFERED="1",
    )
    return env


class _HttpServer:
    def __init__(self, argv: list[str], env: dict[str, str], port: int) -> None:
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.log_path = Path(env["HOME"]).parent / f"server-{port}.log"
        self.log = self.log_path.open("w+", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            [_executable(), *argv, "--port", str(port)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=self.log,
            text=True,
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(f"server exited: {self._read_log()}")
            try:
                httpx.get(f"http://127.0.0.1:{port}/health", timeout=1)
                return
            except httpx.HTTPError:
                time.sleep(0.2)
        raise AssertionError("server did not start")

    def _read_log(self) -> str:
        if not self.log.closed:
            self.log.flush()
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()
        print(f"--- server stderr ---\n{self._read_log()[-4000:]}")


async def _accept(message: str, response_type: Any, params: Any, context: Any) -> Any:
    return response_type(confirmed=True) if response_type else None


_CALL = {"name": VOLUME}


async def _destructive_flow(
    open_client: Any, docker: _FakeDocker, *, elicits: bool, label: str
) -> None:
    kwargs: dict[str, Any] = {"timeout": CALL_TIMEOUT}
    if elicits:
        kwargs["elicitation_handler"] = _accept
    async with open_client(**kwargs) as client:
        if elicits:
            done = await client.call_tool(
                "docker_remove_volume", _CALL, raise_on_error=False
            )
            assert not done.is_error, f"{label}: {done.content}"
            assert docker.deletes(), f"{label}: an accepted elicitation must proceed"
            return
        refused = await client.call_tool(
            "docker_remove_volume", _CALL, raise_on_error=False
        )
        assert not refused.is_error, f"{label}: {refused.content}"
        assert refused.structured_content["cancelled"] is True
        assert "confirm=true" in refused.structured_content["reason"]
        assert not docker.deletes(), "an unconfirmed call must not reach the manager"
        confirmed = await client.call_tool(
            "docker_remove_volume", {**_CALL, "confirm": True}, raise_on_error=False
        )
        assert not confirmed.is_error, f"{label}: {confirmed.content}"
        assert not (confirmed.structured_content or {}).get("cancelled")
        assert docker.deletes(), f"{label}: confirm=true must perform the removal"


@pytest.mark.spec(SPEC)
@pytest.mark.timeout(300)
@pytest.mark.asyncio
@pytest.mark.parametrize("elicits", [False, True])
async def test_stdio_no_auth_destructive_call(
    tmp_path: Path, docker: _FakeDocker, elicits: bool
) -> None:
    env = _server_env(tmp_path, docker)

    def open_client(**kwargs: Any) -> Client[Any]:
        transport = StdioTransport(
            command=_executable(), args=["--transport", "stdio"], env=env
        )
        return Client(transport, **kwargs)

    await _destructive_flow(open_client, docker, elicits=elicits, label="stdio")


@pytest.mark.spec(SPEC)
@pytest.mark.timeout(300)
@pytest.mark.asyncio
@pytest.mark.parametrize("elicits", [False, True])
async def test_http_no_auth_destructive_call(
    tmp_path: Path, docker: _FakeDocker, elicits: bool
) -> None:
    port = _free_port()
    server = _HttpServer(
        [
            "--transport",
            "streamable-http",
            "--auth-type",
            "none",
            "--host",
            "127.0.0.1",
            "--allowed-hosts",
            f"127.0.0.1:{port},localhost:{port}",
        ],
        _server_env(tmp_path, docker),
        port,
    )
    try:

        def open_client(**kwargs: Any) -> Client[Any]:
            return Client(StreamableHttpTransport(server.url), **kwargs)

        await _destructive_flow(open_client, docker, elicits=elicits, label="http")
    finally:
        server.stop()


@pytest.mark.spec(SPEC)
@pytest.mark.timeout(300)
@pytest.mark.asyncio
@pytest.mark.parametrize("elicits", [False, True])
async def test_http_jwt_destructive_call(
    tmp_path: Path, docker: _FakeDocker, world: _World, elicits: bool
) -> None:
    port = _free_port()
    server = _HttpServer(
        [
            "--transport",
            "streamable-http",
            "--auth-type",
            "jwt",
            "--host",
            "127.0.0.1",
            "--token-issuer",
            ISSUER,
            "--token-audience",
            AUDIENCE,
            "--token-jwks-uri",
            world.jwks_uri,
            "--allowed-hosts",
            f"127.0.0.1:{port},localhost:{port}",
        ],
        _server_env(tmp_path, docker),
        port,
    )
    try:

        def open_client(**kwargs: Any) -> Client[Any]:
            return Client(
                StreamableHttpTransport(server.url, auth=world.token()), **kwargs
            )

        await _destructive_flow(open_client, docker, elicits=elicits, label="jwt")
    finally:
        server.stop()


_PRUNE_AND_SWARM_TOOLS = [
    f"{backend}_{name}"
    for backend in ("docker", "podman", "k8s")
    for name in (
        "prune_containers",
        "prune_images",
        "prune_networks",
        "prune_system",
        "prune_volumes",
        "init_swarm",
        "leave_swarm",
    )
]


@pytest.mark.spec(SPEC)
@pytest.mark.asyncio
async def test_prune_and_swarm_tools_require_confirmation(
    monkeypatch: pytest.MonkeyPatch, docker: _FakeDocker
) -> None:
    """Prune, swarm init and leave are destructive: each tool takes ``confirm``
    and a client that cannot elicit is refused before the manager is reached."""
    monkeypatch.setenv("DOCKER_HOST", docker.host)
    from container_manager_mcp.mcp_server import get_mcp_instance

    mcp, _, _ = get_mcp_instance()
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        for name in _PRUNE_AND_SWARM_TOOLS:
            if name.startswith("k8s_") and name.endswith("_swarm"):
                continue  # Kubernetes serves no swarm tools
            assert "confirm" in tools[name].inputSchema["properties"], name
        refused = await client.call_tool(
            "docker_prune_volumes", {}, raise_on_error=False
        )
        assert refused.structured_content["cancelled"] is True
        assert "confirm=true" in refused.structured_content["reason"]
        assert not [r for r in docker.requests if r[0] in ("POST", "DELETE")]

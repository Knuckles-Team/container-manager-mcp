"""Characterization tests for scripts/validate_a2a_agent.py's main (CCN 23,
COG 122 -- the highest cognitive complexity in this lane's whole target
list), before decomposition.

This is a manual CLI validation script with no existing tests. It is
loaded directly from its file path (scripts/ is not a package) and driven
against an httpx.MockTransport so no real network call is made. Pins,
via captured stdout: the immediate-completion happy path (agent text
response printed), an in-progress-then-completed poll sequence, a
non-200 initial response, a JSON-decode failure, a polling HTTP failure,
a JSON-RPC error key on both the initial and polling responses, an empty
task history, and a network-level RequestError -- one test per real
branch of the original function.
"""

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "validate_a2a_agent.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_a2a_agent_under_test", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module()


def _run_main(mod, transport, monkeypatch, query="ping"):
    monkeypatch.setenv("A2A_VALIDATION_QUERY", query)

    real_client_cls = httpx.AsyncClient

    def _patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(mod.httpx, "AsyncClient", _patched_client)
    asyncio.run(mod.main())


def test_immediate_completion_prints_agent_response(mod, monkeypatch, capsys):
    """Pins the real (surprising) behavior: the initial message/send response
    is never itself checked for a terminal state -- any response carrying
    result.id always goes to at least one poll, even if that poll
    immediately reports 'completed'."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "message/send":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": 1, "result": {"id": "task-1"}}
            )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "status": {"state": "completed"},
                    "history": [
                        {"role": "user", "parts": [{"kind": "text", "text": "ping"}]},
                        {"role": "agent", "parts": [{"text": "pong"}]},
                    ],
                },
            },
        )

    _run_main(mod, httpx.MockTransport(handler), monkeypatch)
    out = capsys.readouterr().out
    assert "--- Agent Response ---" in out
    assert "Validation result received; body omitted." in out


def test_polling_sequence_running_then_completed(mod, monkeypatch, capsys):
    calls = {"n": 0}

    async def _no_sleep(_seconds):
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "message/send":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": 1, "result": {"id": "task-1"}},
            )
        calls["n"] += 1
        state = "running" if calls["n"] == 1 else "completed"
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"status": {"state": state}, "history": []},
            },
        )

    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)
    _run_main(mod, httpx.MockTransport(handler), monkeypatch)
    out = capsys.readouterr().out
    assert "Task State: running" in out
    assert "Task State: completed" in out
    assert calls["n"] == 2


def test_non_200_initial_response(mod, monkeypatch, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _run_main(mod, httpx.MockTransport(handler), monkeypatch)
    out = capsys.readouterr().out
    assert "Error: 500" in out
    assert "Response body omitted (HTTP 500)." in out


def test_json_decode_error_on_initial_response(mod, monkeypatch, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    _run_main(mod, httpx.MockTransport(handler), monkeypatch)
    out = capsys.readouterr().out
    assert "Response body omitted (HTTP 200)." in out


def test_polling_http_failure_stops_loop(mod, monkeypatch, capsys):
    async def _no_sleep(_seconds):
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "message/send":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": 1, "result": {"id": "task-1"}}
            )
        return httpx.Response(503, text="unavailable")

    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)
    _run_main(mod, httpx.MockTransport(handler), monkeypatch)
    out = capsys.readouterr().out
    assert "Polling Failed: 503" in out
    assert "Polling failed with HTTP 503." in out


def test_polling_jsonrpc_error_key_stops_loop(mod, monkeypatch, capsys):
    async def _no_sleep(_seconds):
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "message/send":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": 1, "result": {"id": "task-1"}}
            )
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": 2, "error": {"code": -32000}}
        )

    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)
    _run_main(mod, httpx.MockTransport(handler), monkeypatch)
    out = capsys.readouterr().out
    assert "Starting polling error key check..." in out
    assert "Polling JSON-RPC error code: -32000" in out


def test_initial_jsonrpc_error_key_is_reported(mod, monkeypatch, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32601}}
        )

    _run_main(mod, httpx.MockTransport(handler), monkeypatch)
    out = capsys.readouterr().out
    assert "JSON-RPC error code: -32601" in out


def test_completed_task_with_empty_history(mod, monkeypatch, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "message/send":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": 1, "result": {"id": "task-1"}}
            )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"status": {"state": "completed"}, "history": []},
            },
        )

    _run_main(mod, httpx.MockTransport(handler), monkeypatch)
    out = capsys.readouterr().out
    assert "--- Agent Response ---" not in out
    assert "Validation result received; body omitted." in out


def test_completed_task_with_only_user_history_reports_no_response_found(
    mod, monkeypatch, capsys
):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "message/send":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": 1, "result": {"id": "task-1"}}
            )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "status": {"state": "completed"},
                    "history": [{"role": "user", "parts": [{"kind": "text", "text": "ping"}]}],
                },
            },
        )

    _run_main(mod, httpx.MockTransport(handler), monkeypatch)
    out = capsys.readouterr().out
    assert "--- No Agent Response Found in History ---" in out


def test_request_error_is_caught(mod, monkeypatch, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _run_main(mod, httpx.MockTransport(handler), monkeypatch)
    out = capsys.readouterr().out
    assert "Operation failed: ConnectError" in out

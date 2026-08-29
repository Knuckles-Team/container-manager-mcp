"""Characterization tests for mcp_server.ctx_log (CCN 23), before
decomposition.

ctx_log has three call shapes: 2-arg (level, message) where an int `level`
is mapped through a logging-level->name table (unknown ints fall back to
"info"); 3-arg (server_logger, level, message) where `level_str` is always
``str(level).lower()`` regardless of type -- a real, pre-existing asymmetry
with the 2-arg form when `level` is an int (e.g. ``logging.INFO`` becomes
the string ``"20"``, not ``"info"``) -- this test pins that asymmetry
rather than "fixing" it; and any other arg count, which falls through to
``agent_utilities.mcp.context_helpers.ctx_log`` if importable. In every
shape, if `ctx` is truthy, the same client-side logging is attempted:
``getattr(ctx, level_str, None) or getattr(ctx, "info", None)``, called
with `message`; a coroutine result is scheduled onto the running loop if
one exists, silently dropped otherwise; any exception from the client call
is swallowed.
"""

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

import container_manager_mcp.mcp_server as mcp_server


@pytest.fixture(autouse=True)
def _restore_logger(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(mcp_server, "logger", fake_logger)
    return fake_logger


def test_two_arg_int_level_maps_to_name_and_logs_server_side(_restore_logger):
    mcp_server.ctx_log(None, logging.WARNING, "hello")
    _restore_logger.warning.assert_called_once_with("hello")


def test_two_arg_unknown_int_level_falls_back_to_info(_restore_logger):
    mcp_server.ctx_log(None, 99, "hello")
    _restore_logger.info.assert_called_once_with("hello")


def test_two_arg_string_level_lowercased(_restore_logger):
    mcp_server.ctx_log(None, "ERROR", "boom")
    _restore_logger.error.assert_called_once_with("boom")


def test_two_arg_with_ctx_calls_matching_client_method(_restore_logger):
    ctx = MagicMock()
    ctx.warning = MagicMock(return_value=None)
    mcp_server.ctx_log(ctx, logging.WARNING, "hello")
    ctx.warning.assert_called_once_with("hello")


def test_two_arg_ctx_missing_level_method_falls_back_to_info(_restore_logger):
    ctx = MagicMock(spec=["info"])
    mcp_server.ctx_log(ctx, logging.WARNING, "hello")
    ctx.info.assert_called_once_with("hello")


def test_two_arg_ctx_client_exception_is_swallowed(_restore_logger):
    ctx = MagicMock()
    ctx.info.side_effect = RuntimeError("client boom")
    # Must not raise.
    mcp_server.ctx_log(ctx, logging.INFO, "hello")


def test_two_arg_ctx_coroutine_result_scheduled_when_loop_running(_restore_logger):
    ctx = MagicMock()
    scheduled = {}

    async def fake_client_info(message):
        return message

    ctx.info = fake_client_info

    async def driver():
        mcp_server.ctx_log(ctx, logging.INFO, "hello")
        # give the scheduled task a chance to run so it doesn't warn
        await asyncio.sleep(0)

    asyncio.run(driver())


def test_two_arg_ctx_coroutine_result_dropped_without_running_loop(_restore_logger):
    ctx = MagicMock()

    async def fake_client_info(message):
        return message

    ctx.info = fake_client_info
    # No running loop here (sync test) -- must not raise.
    mcp_server.ctx_log(ctx, logging.INFO, "hello")


def test_three_arg_form_logs_via_server_logger(_restore_logger):
    server_logger = MagicMock()
    mcp_server.ctx_log(None, server_logger, "warning", "hi")
    server_logger.warning.assert_called_once_with("hi")


def test_three_arg_form_int_level_is_not_name_mapped(_restore_logger):
    """Pins the pre-existing asymmetry: the 3-arg form does str(level).lower()
    unconditionally, so an int level like logging.INFO (20) becomes "20",
    not "info" -- unlike the 2-arg form.
    """
    server_logger = MagicMock()
    mcp_server.ctx_log(None, server_logger, logging.INFO, "hi")
    server_logger.info.assert_not_called()
    # getattr(server_logger, "20", None) is falsy on a MagicMock's default
    # attribute access for a numeric-named attr, so nothing logs server-side.


def test_three_arg_form_with_ctx_calls_matching_client_method(_restore_logger):
    server_logger = MagicMock()
    ctx = MagicMock()
    mcp_server.ctx_log(ctx, server_logger, "error", "hi")
    ctx.error.assert_called_once_with("hi")


def test_other_arg_count_falls_through_to_real_ctx_log(monkeypatch, _restore_logger):
    fake_real = MagicMock()
    monkeypatch.setattr(
        "agent_utilities.mcp.context_helpers.ctx_log", fake_real, raising=False
    )
    mcp_server.ctx_log(None, "one_positional")
    fake_real.assert_called_once_with(None, "one_positional")


def test_zero_extra_args_falls_through_to_real_ctx_log(monkeypatch, _restore_logger):
    fake_real = MagicMock()
    monkeypatch.setattr(
        "agent_utilities.mcp.context_helpers.ctx_log", fake_real, raising=False
    )
    mcp_server.ctx_log(None)
    fake_real.assert_called_once_with(None)


def test_fallback_swallows_real_ctx_log_exception(monkeypatch, _restore_logger):
    fake_real = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(
        "agent_utilities.mcp.context_helpers.ctx_log", fake_real, raising=False
    )
    # Must not raise.
    mcp_server.ctx_log(None, "one_positional")

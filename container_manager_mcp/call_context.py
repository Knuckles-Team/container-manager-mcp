"""Per-call context selection for the generic ``docker_*`` and ``k8s_*`` operation tools.

The retired ``cm_multi_context__*`` tools took ``backend`` and ``context``. The
generic operation tools name the backend in their prefix, so only ``context``
is added: an optional string argument on every ``docker_*``/``k8s_*`` tool.
When set, the call runs against that named context of the multi-context pool;
when omitted the tool behaves exactly as before.
"""

from __future__ import annotations

import contextvars
from typing import Any

from fastmcp.server.middleware import Middleware, MiddlewareContext

CONTEXT_ARGUMENT = "context"
_BACKEND_BY_PREFIX = {"docker_": "docker", "k8s_": "kubernetes"}
_CALL_CONTEXT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "container_manager_call_context", default=None
)


def backend_for_tool(name: str) -> str | None:
    """The backend a generic operation tool targets, from its name prefix."""
    for prefix, backend in _BACKEND_BY_PREFIX.items():
        if name.startswith(prefix):
            return backend
    return None


def current_call_context() -> str | None:
    """The context named by the current tool call, if any."""
    return _CALL_CONTEXT.get()


def manager_for(backend: str) -> Any:
    """The manager for ``backend``: the named context's manager, else the default."""
    from container_manager_mcp.container_manager import create_manager

    context = current_call_context()
    if not context:
        return create_manager(backend)
    from container_manager_mcp.multi_context_manager import MultiContextManager

    return MultiContextManager(silent=True).get_manager(backend, context)


class CallContextMiddleware(Middleware):
    """Move the ``context`` argument of a generic tool call into the call scope."""

    async def on_call_tool(self, context: MiddlewareContext, call_next: Any) -> Any:
        message = context.message
        arguments = getattr(message, "arguments", None)
        if (
            backend_for_tool(getattr(message, "name", "")) is None
            or not isinstance(arguments, dict)
            or CONTEXT_ARGUMENT not in arguments
        ):
            return await call_next(context)
        value = arguments.pop(CONTEXT_ARGUMENT)
        token = _CALL_CONTEXT.set(str(value) if value else None)
        try:
            return await call_next(context)
        finally:
            _CALL_CONTEXT.reset(token)


def advertise_context_argument(tools: dict[str, Any]) -> list[str]:
    """Add the optional ``context`` property to each generic tool; return their names."""
    patched: list[str] = []
    for name, tool in tools.items():
        if backend_for_tool(name) is None:
            continue
        schema = tool.parameters
        properties = schema.setdefault("properties", {})
        if CONTEXT_ARGUMENT in properties:
            continue
        properties[CONTEXT_ARGUMENT] = {
            "type": "string",
            "description": (
                "Named context of the multi-context pool to run against "
                "(see cm_multi_context list_contexts); the default context if omitted."
            ),
        }
        patched.append(name)
    return patched

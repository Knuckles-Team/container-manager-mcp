# CONN-CONTAINER-MANAGER-MCP — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Existing system and reuse

`container_manager_mcp/multi_context_manager.py` holds the pool of named contexts. `container_manager_mcp/call_context.py` scopes one call to a context.
The generic tools replace the retired `cm_multi_context__*` tools. The backend now sits in the tool prefix, and the context is an argument.

## Architecture

`advertise_context_argument` adds the optional `context` property to each generic tool schema.
`CallContextMiddleware` moves the argument into a context variable for the duration of the call.
`manager_for(backend)` reads that variable and returns the named context manager, or the default manager when it is unset.

## Interfaces and data model

Tools with a `docker_` prefix use the `docker` backend. Tools with a `k8s_` prefix use the `kubernetes` backend.
The `context` value is a string. An empty value selects the default manager.

## Live integration path

Entry point: `container_manager_mcp.mcp_server:get_mcp_instance`. It calls `register_tool_surface` at `container_manager_mcp/mcp_server.py:1602`.
The `get_client` hooks at `:1609` and `:1628` return `manager_for("docker")` and `manager_for("kubernetes")`.
Lines `:1634` and `:1635` advertise the argument and add the middleware.

## Quality and release gates

Run `pytest tests/test_call_context.py` and `pipelines-hook spec-standard`.

## Risks, alternatives, and decisions

A tool without the `context` property would ignore the named context and run on the default manager. `R001.1` guards that case.

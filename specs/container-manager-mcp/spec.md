# CONN-CONTAINER-MANAGER-MCP — Container manager call context

## Outcome

The generic `docker_*` and `k8s_*` tools must accept an optional `context` argument.
The argument names the context of the multi-context pool that runs the call.

## Scope and owner

The `container-manager-mcp` repository owns this behavior. The shared connector SDK owns the tool-surface mechanism.
The shared pipelines repository owns quality gates and delivery rendering.

Component: Connector integration
References: [Connector SDK](https://github.com/Knuckles-Team/scion-sdk/blob/main/README.md)
Code: ../../container_manager_mcp/call_context.py ../../container_manager_mcp/mcp_server.py:1602 ../../container_manager_mcp/mcp_server.py:1634 ../../tests/test_call_context.py

## Non-goals

This specification grants no service access, credential creation, deployment, or approval exception.
The connector must not implement a graph engine, a policy authority, or a duplicate SDK runtime.

## Why a new spec

The search `git grep -n context -- specs` found no spec in this repository. The `specs/` tree holds only the template. No other repository owns this behavior.

## Requirements

See [requirements.md](requirements.md), [plan.md](plan.md), and [test-spec.md](test-spec.md).

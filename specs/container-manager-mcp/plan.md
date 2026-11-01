# CONN-CONTAINER-MANAGER-MCP — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

`container_manager_mcp/multi_context_manager.py` holds the pool of named contexts. `container_manager_mcp/call_context.py` scopes one call to a context. The generic tools replace the retired `cm_multi_context__*` tools. The backend now sits in the tool prefix, and the context is an argument. `advertise_context_argument` adds the optional `context` property to each generic tool schema. `CallContextMiddleware` moves the argument into a context variable for the duration of the call. `manager_for(backend)` reads that variable and returns the named context manager, or the default manager when it is unset.

## Technical Context

- **Language/Version**: Python 3.14 (floor 3.12)
- **Primary Dependencies**: scion-sdk, fastmcp
- **Storage**: N/A; the connector stores nothing
- **Testing**: pytest; the proof level each row names
- **Target Platform**: Linux containers
- **Project Type**: connector (MCP server)
- **Performance Goals**: N/A
- **Constraints**: one tool per API operation; shared capabilities come from the SDK
- **Scale/Scope**: 13 rows

## Constitution Check

- One owner per concept: pass. The connector owns one external system's calls, mappings and tools only.
- Connectors depend on the SDK and are called through arbora: pass.
- No ratchets: pass. The rows add no suppression.

## Project Structure

```text
No file is added; the rows edit existing files.
```

## Phase 0: Research

No unknowns. Each row names the code it touches.

## Phase 1: Design

### Interfaces and data model

Tools with a `docker_` prefix use the `docker` backend. Tools with a `k8s_` prefix use the `kubernetes` backend.
The `context` value is a string. An empty value selects the default manager.

### Live integration path

Entry point: `container_manager_mcp.mcp_server:get_mcp_instance`. It calls `register_tool_surface` at `container_manager_mcp/mcp_server.py:1602`.
The `get_client` hooks at `:1609` and `:1628` return `manager_for("docker")` and `manager_for("kubernetes")`.
Lines `:1634` and `:1635` advertise the argument and add the middleware.

### Quality and release gates

Run `pytest tests/test_call_context.py` and `pipelines-hook spec-standard`.

### Risks, alternatives, and decisions

A tool without the `context` property would ignore the named context and run on the default manager. `R001.1` guards that case.

### Amendment: inventory identities

Inventory entries retain a configured SSH user. If no user is configured,
the connector omits the username from the SSH authority and lets SSH choose
its normal identity. Public examples use synthetic aliases and accounts.

### Amendment: deployed workload access

The existing `k8s_list_services` operation reads Kubernetes deployments.
Its summary carries the provider's `Available` condition as status, or
`unknown` when that condition is absent. Publish the SDK access contract and
test the listed field bindings against a served response from a synthetic
Kubernetes provider. It is not a recording from a live cluster. This
implements the connector side of SCION-CONNECTOR-CONTROL-R039.

## Phase 2: Task approach

`tasks.md` holds one task line per row, producer first.

## Complexity Tracking

None. The constitution check passed.

## Build contract

The repository pre-commit hooks apply: `ruff-format`, `connector-standard`, `secrets`, `tests-fast`. Each row names its proving test. No suppression and no duplicate implementation.

## Production caller chain

1. `CONN-CONTAINER-MANAGER-MCP-R001.1`: console_script container-manager-mcp -> container_manager_mcp/mcp_server.py::mcp_server (L1663) -> get_mcp_instance (L1630), plus the scion_sdk.connector entry point container_manager_mcp::connector (__init__.py L101) -> container_manager_mcp/surface.py::wire_call_context (L65) -> advertise_context_argument (call_context.py L69) over registered_tools.
2. `CONN-CONTAINER-MANAGER-MCP-R001.2`: console_script container-manager-mcp -> container_manager_mcp/mcp_server.py::mcp_server (L1663) -> get_mcp_instance (L1630), plus the scion_sdk.connector entry point container_manager_mcp::connector (__init__.py L101) -> container_manager_mcp/surface.py::wire_call_context (L65) -> manager_for (call_context.py L37) -> MultiContextManager.get_manager.
3. `CONN-CONTAINER-MANAGER-MCP-R001.3`: console_script container-manager-mcp -> container_manager_mcp/mcp_server.py::mcp_server (L1663) -> get_mcp_instance (L1630), plus the scion_sdk.connector entry point container_manager_mcp::connector (__init__.py L101) -> container_manager_mcp/surface.py::wire_call_context (L65) -> manager_for (call_context.py L37) -> container_manager.create_manager.
4. `CONN-CONTAINER-MANAGER-MCP-R001.4`: console_script container-manager-mcp -> container_manager_mcp/mcp_server.py::mcp_server (L1663) -> get_mcp_instance (L1630), plus the scion_sdk.connector entry point container_manager_mcp::connector (__init__.py L101) -> container_manager_mcp/surface.py::wire_call_context (L65) -> advertise_context_argument and CallContextMiddleware (call_context.py L49), now wired from surface.py::wire_call_context L65 and mcp_server.py L1655-1656.
5. `CONN-CONTAINER-MANAGER-MCP-R001.5`: console_script container-manager-mcp -> container_manager_mcp/mcp_server.py::mcp_server (L1663) -> get_mcp_instance (L1630), plus the scion_sdk.connector entry point container_manager_mcp::connector (__init__.py L101) -> container_manager_mcp/surface.py::wire_call_context (L65) -> surface.py::VERBOSE_TARGETS get_client lambdas calling manager_for.
6. `CONN-CONTAINER-MANAGER-MCP-R002`: arbora fleet.call -> connector tools/call verbose-tier k8s_list_services (surface.py::VERBOSE_TARGETS) -> container_manager_mcp/k8s/workloads.py::list_services (L22) -> _K8sBase._deployment_summary; contract at container_manager_mcp/ontology/access_contracts.ttl (cm:deploymentAccess).
7. `CONN-CONTAINER-MANAGER-MCP-R003`: MCP tool cm_list_hosts (mcp_server.py L197, register_inventory_tools) -> container_manager.py::list_inventory_hosts (L438) -> tunnel_manager HostManager.list_hosts; host aliases are resolved by resolve_host_from_inventory (L462).
8. `CONN-CONTAINER-MANAGER-MCP-R004`: MCP tool cm_list_hosts (mcp_server.py L197) and container-manager-doctor console_script (doctor.py::doctor) -> container_manager.py::inventory_path (L420) reading TUNNEL_INVENTORY via scion_sdk.config.setting.
9. `CONN-CONTAINER-MANAGER-MCP-R005`: arbora fleet.call -> connector tools/call verbose-tier k8s_list_services (surface.py::VERBOSE_TARGETS) -> container_manager_mcp/k8s/workloads.py::list_services (L22); the fixture is tests/fixtures/deployment_response.json read through the contract in container_manager_mcp/ontology/access_contracts.ttl.
10. `CONN-CONTAINER-MANAGER-MCP-R008`: arbora fleet.call -> connector tools/call verbose-tier k8s_list_services (surface.py::VERBOSE_TARGETS) -> container_manager_mcp/k8s/workloads.py::list_services (L22) -> _K8sBase._deployment_summary; contract is cm:deploymentAccess in container_manager_mcp/ontology/access_contracts.ttl.
11. `CONN-CONTAINER-MANAGER-MCP-R006`: console_script container-manager-mcp -> mcp_server.py::mcp_server -> container_manager.py manager creation -> container_manager_mcp/platform_compat.py (default_docker_endpoint, current_uid, default_kubeconfig, restrict_to_owner, write_private_text).
12. `CONN-CONTAINER-MANAGER-MCP-R007`: MCP tools cm_image_operations (mcp_server.py L316), cm_container_operations, cm_volume_operations, cm_network_operations, cm_swarm_operations, cm_system_operations -> mcp_server.py::_confirm_or_refuse (L75) -> scion_sdk.mcp.context.ctx_confirm_destructive and confirmation_refusal.

## Cross-repo seams

none. The cited owner rows are library capabilities the connector calls in process; spec.md names them.

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

## Scope and interfaces

In scope: the rows of [requirements.md](requirements.md). Out of scope: the SDK modules and the engine, which own the shared capabilities.

## User Scenarios & Testing

### User Story 1 - Container manager call context (Priority: P1)

The generic `docker_*` and `k8s_*` tools must accept an optional `context` argument.

**Acceptance Scenarios**:

1. **Given** the connector is configured, **When** the behaviour of row `CONN-CONTAINER-MANAGER-MCP-R001` (per-call context) is exercised, **Then** `tests/test_call_context.py` rows R001.1 to R001.5 pass.
2. **Given** the connector is configured, **When** the behaviour of row `CONN-CONTAINER-MANAGER-MCP-R002` (deployed workload access contract) is exercised, **Then** `tests/test_access_contracts.py` parses the contract and compares its fields with a controlled MCP tool response. Test: `tests/test_access_contracts.py::test_served_deployment_access_contract_matches_fixture`.
3. **Given** the connector is configured, **When** the behaviour of row `CONN-CONTAINER-MANAGER-MCP-R003` (inventory privacy) is exercised, **Then** `tests/test_inventory_hosts.py` covers configured and missing users and the generated SSH authority. `tests/test_served_actor_binding.py` serves the real connector in memory and over uvicorn with and without a JWT. Test: `tests/test_inventory_hosts.py::test_docker_host_preserves_only_configured_user`.

### Edge Cases

- If `CONN-CONTAINER-MANAGER-MCP-R001` is not met, its proving test fails: per-call context.
- If `CONN-CONTAINER-MANAGER-MCP-R002` is not met, its proving test fails: deployed workload access contract.
- If `CONN-CONTAINER-MANAGER-MCP-R003` is not met, its proving test fails: inventory privacy.

## Requirements

- **FR-001**: Per-call context. Rows: `CONN-CONTAINER-MANAGER-MCP-R001`.
- **FR-002**: Deployed workload access contract. Rows: `CONN-CONTAINER-MANAGER-MCP-R002`.
- **FR-003**: Inventory privacy. Rows: `CONN-CONTAINER-MANAGER-MCP-R003`.
- **FR-004**: Shared inventory path. Rows: `CONN-CONTAINER-MANAGER-MCP-R004`.
- **FR-005**: Workload recorded response. Rows: `CONN-CONTAINER-MANAGER-MCP-R005`.
- **FR-006**: k8s_list_services matches the declared Deployment contract. Rows: `CONN-CONTAINER-MANAGER-MCP-R008`.
- **FR-007**: Windows portability. Rows: `CONN-CONTAINER-MANAGER-MCP-R006`.
- **FR-008**: Destructive confirmation fails closed. Rows: `CONN-CONTAINER-MANAGER-MCP-R007`.

## Success Criteria

- **SC-001**: `tests/test_call_context.py` rows R001.1 to R001.5 pass. Rows: `CONN-CONTAINER-MANAGER-MCP-R001`.
- **SC-002**: `tests/test_access_contracts.py` parses the contract and compares its fields with a controlled MCP tool response. Rows: `CONN-CONTAINER-MANAGER-MCP-R002`.
- **SC-003**: `tests/test_inventory_hosts.py` covers configured and missing users and the generated SSH authority. `tests/test_served_actor_binding.py` serves the real connector in memory and over uvicorn with and without a JWT. Rows: `CONN-CONTAINER-MANAGER-MCP-R003`.
- **SC-004**: `tests/test_inventory_hosts.py::test_tunnel_inventory_env_selects_the_shared_inventory` passes with an empty XDG default and a populated `$TUNNEL_INVENTORY`. Rows: `CONN-CONTAINER-MANAGER-MCP-R004`.
- **SC-005**: `tests/test_access_contracts.py` reads the recorded response through the declared contract and compares the rows. Rows: `CONN-CONTAINER-MANAGER-MCP-R005`.
- **SC-006**: K8s_list_services matches the declared Deployment contract. Rows: `CONN-CONTAINER-MANAGER-MCP-R008`.
- **SC-007**: `tests/test_windows_portability.py` passes, and `.github/workflows/windows-ci.yml` runs the suite on `windows-latest`. Rows: `CONN-CONTAINER-MANAGER-MCP-R006`.
- **SC-008**: `tests/test_destructive_confirmation_transports.py` proves it over stdio, HTTP without auth and JWT HTTP. Rows: `CONN-CONTAINER-MANAGER-MCP-R007`.

## Assumptions

- The external system is reachable with operator-supplied credentials.
- The SDK and engine contracts the rows cite are at the floors declared in `pyproject.toml`.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `CONN-CONTAINER-MANAGER-MCP-R001` | [plan.md](plan.md) | T-001 | [test-spec.md](test-spec.md) |
| FR-002 | `CONN-CONTAINER-MANAGER-MCP-R002` | [plan.md](plan.md) | T-002 | [test-spec.md](test-spec.md) |
| FR-003 | `CONN-CONTAINER-MANAGER-MCP-R003` | [plan.md](plan.md) | T-003 | [test-spec.md](test-spec.md) |
| FR-004 | `CONN-CONTAINER-MANAGER-MCP-R004` | [plan.md](plan.md) | T-004 | [test-spec.md](test-spec.md) |
| FR-005 | `CONN-CONTAINER-MANAGER-MCP-R005` | [plan.md](plan.md) | T-005 | [test-spec.md](test-spec.md) |
| FR-006 | `CONN-CONTAINER-MANAGER-MCP-R008` | [plan.md](plan.md) | T-006 | [test-spec.md](test-spec.md) |
| FR-007 | `CONN-CONTAINER-MANAGER-MCP-R006` | [plan.md](plan.md) | T-007 | [test-spec.md](test-spec.md) |
| FR-008 | `CONN-CONTAINER-MANAGER-MCP-R007` | [plan.md](plan.md) | T-008 | [test-spec.md](test-spec.md) |

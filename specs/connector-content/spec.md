# CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT — The connector serves its packaged content

Owner: container-manager-mcp. Cross-repo IDs: `SCION-CONNECTOR-CONTROL-R071`, `SCION-CONNECTOR-CONTROL-R073`.

## Purpose and user stories

The `container-manager-mcp` server serves its packaged skills, prompts and ontology to any client, with no wiring of its own.

## Why a new spec

The specs of this repository were read; none owns this capability. This spec states it for this connector only.

## Scope and interfaces

In scope: the rows of [requirements.md](requirements.md). Out of scope: the SDK modules and the engine, which own the shared capabilities.

## User Scenarios & Testing

### User Story 1 - The connector serves its packaged content (Priority: P1)

The `container-manager-mcp` server serves its packaged skills, prompts and ontology to any client, with no wiring of its own.

**Acceptance Scenarios**:

1. **Given** the connector is configured, **When** the behaviour of row `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R001` (default content) is exercised, **Then** `resources/list` includes `skill://` and `ontology://`; `prompts/list` includes each packaged prompt. Test: `tests/test_default_content_surfaces.py::test_default_content_surfaces`.
2. **Given** the connector is configured, **When** the behaviour of row `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R002` (declare access contracts for the competency reads of container-manager-mcp) is exercised, **Then** declare access contracts for the competency reads of container-manager-mcp.
3. **Given** the connector is configured, **When** the behaviour of row `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R003` (add access contracts for pods, containers and images) is exercised, **Then** add access contracts for pods, containers and images.

### Edge Cases

- If `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R001` is not met, its proving test fails: default content.
- If `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R002` is not met, its proving test fails: declare access contracts for the competency reads of container-manager-mcp.
- If `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R003` is not met, its proving test fails: add access contracts for pods, containers and images.

## Requirements

- **FR-001**: Default content. Rows: `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R001`.
- **FR-002**: Declare access contracts for the competency reads of container-manager-mcp. Rows: `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R002`.
- **FR-003**: Add access contracts for pods, containers and images. Rows: `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R003`.
- **FR-004**: Serve sandbox_provision from the container runtime. Rows: `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R004`.
- **FR-005**: Serve desktop actions from the gui-sandbox container. Rows: `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R005`.

## Success Criteria

- **SC-001**: `resources/list` includes `skill://` and `ontology://`; `prompts/list` includes each packaged prompt. Rows: `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R001`.
- **SC-002**: Declare access contracts for the competency reads of container-manager-mcp. Rows: `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R002`.
- **SC-003**: Add access contracts for pods, containers and images. Rows: `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R003`.
- **SC-004**: Serve sandbox_provision from the container runtime. Rows: `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R004`.
- **SC-005**: Serve desktop actions from the gui-sandbox container. Rows: `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R005`.

## Assumptions

- The external system is reachable with operator-supplied credentials.
- The SDK and engine contracts the rows cite are at the floors declared in `pyproject.toml`.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R001` | [plan.md](plan.md) | T-001 | [test-spec.md](test-spec.md) |
| FR-002 | `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R002` | [plan.md](plan.md) | T-002 | [test-spec.md](test-spec.md) |
| FR-003 | `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R003` | [plan.md](plan.md) | T-003 | [test-spec.md](test-spec.md) |
| FR-004 | `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R004` | [plan.md](plan.md) | T-004 | [test-spec.md](test-spec.md) |
| FR-005 | `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R005` | [plan.md](plan.md) | T-005 | [test-spec.md](test-spec.md) |

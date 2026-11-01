# CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION — Transport through the SDK

Owner: container-manager-mcp. Cross-repo IDs: `SCION-CENTRALIZED-CAPABILITIES-R001`, `SCION-CENTRALIZED-CAPABILITIES-R008`, `SCION-CENTRALIZED-CAPABILITIES-R012`, `SCION-CENTRALIZED-CAPABILITIES-R040`.

Extends: [SCION-CENTRALIZED-CAPABILITIES](https://github.com/Knuckles-Team/scion-sdk/blob/main/specs/centralized-capabilities/spec.md)

## Purpose and user stories

The connector must reach the cluster identity provider through the SDK. The SDK owns the governed HTTP client, retry and TLS profiles. Docker, Podman and Kubernetes calls use their native clients and are out of scope.

An operator imports a kube context with an OIDC login. Transport security comes from one TLS profile, not from a connector flag.

## Why a new spec

The search covered `specs/` of this repository and the SDK centralized-capabilities spec. This repository had no spec for transport. The SDK owns the capability. This spec states how this connector uses it.

## Scope and interfaces

In scope: the connector's HTTP client, TLS handling and dependency floor.

Out of scope: the tool surface, which stays one tool per API operation, and the SDK modules themselves.

## User Scenarios & Testing

### User Story 1 - Transport through the SDK (Priority: P1)

The connector must reach the cluster identity provider through the SDK. The SDK owns the governed HTTP client, retry and TLS profiles. Docker, Podman and Kubernetes calls use their native clients and are out of scope.

**Acceptance Scenarios**:

1. **Given** the connector is configured, **When** the behaviour of row `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001` (transport comes from the sdk (rollup)) is exercised, **Then** see each child.

### Edge Cases

- If `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001` is not met, its proving test fails: transport comes from the sdk (rollup).

## Requirements

- **FR-001**: Transport comes from the SDK (rollup). Rows: `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001`.

## Success Criteria

- **SC-001**: See each child. Rows: `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001`.

## Assumptions

- The external system is reachable with operator-supplied credentials.
- The SDK and engine contracts the rows cite are at the floors declared in `pyproject.toml`.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001` | [plan.md](plan.md) | T-001 | [test-spec.md](test-spec.md) |

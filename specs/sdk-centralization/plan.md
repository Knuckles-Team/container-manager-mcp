# CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

The SDK owns the capability rows cited in [requirements.md](requirements.md). The connector adds no transport code of its own. The connector client builds on the SDK client. TLS comes from `resolve_tls_profile` for the connector's settings prefix.

## Technical Context

- **Language/Version**: Python 3.14 (floor 3.12)
- **Primary Dependencies**: scion-sdk, fastmcp
- **Storage**: N/A; the connector stores nothing
- **Testing**: pytest; the proof level each row names
- **Target Platform**: Linux containers
- **Project Type**: connector (MCP server)
- **Performance Goals**: N/A
- **Constraints**: one tool per API operation; shared capabilities come from the SDK
- **Scale/Scope**: 4 rows

## Constitution Check

- One owner per concept: pass. The connector owns one external system's calls, mappings and tools only.
- Connectors depend on the SDK and are called through arbora: pass.
- No ratchets: pass. The rows add no suppression.

## Project Structure

```text
container_manager_mcp/k8s/kubeconfig_import.py  existing
```

## Phase 0: Research

No unknowns. Each row names the code it touches.

## Phase 1: Design

### Interfaces and data model

No tool, argument or result changes. Operators who relied on a disabled-verification flag must configure a TLS profile instead.

### Live integration path

Every tool reaches the cluster identity provider through the connector client, which sends each request through the SDK client.

### Quality and release gates

The focused tests in [test-spec.md](test-spec.md) must pass. The SDK floor must be the latest published release.

### Risks, alternatives, and decisions

A private certificate authority must be supplied through the TLS profile. A flag that disables verification must not return.

## Phase 2: Task approach

`tasks.md` holds one task line per row, producer first.

## Complexity Tracking

None. The constitution check passed.

## Build contract

The repository pre-commit hooks apply: `ruff-format`, `connector-standard`, `secrets`, `tests-fast`. Each row names its proving test. No suppression and no duplicate implementation.

## Production caller chain

1. `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001.1`: console_script container-manager-save-context -> container_manager_mcp/k8s/kubeconfig_import.py::main (L864) -> save_kube_context (L674) -> _resolve_oidc_mode (L570) -> _oidc_password_grant (L213) -> _password_grant_with (L252) -> discover_token_endpoint.
2. `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001.2`: console_script container-manager-save-context -> kubeconfig_import.py::main (L864) -> save_kube_context (L674) -> _resolve_oidc_mode (L570) -> _oidc_tls_profile (L199) -> resolve_tls_profile.
3. `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001.3`: package install -> pyproject.toml dependencies (scion-sdk[mcp]) -> scion_sdk transport used by container_manager_mcp/k8s/kubeconfig_import.py and the console_script container-manager-mcp.

## Cross-repo seams

none. The cited owner rows are library capabilities the connector calls in process; spec.md names them.

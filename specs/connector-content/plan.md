# CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

`pyproject.toml` declares `container-manager-mcp` as the connector identity. `create_mcp_server` resolves the packaged content of `container_manager_mcp` from it.

## Technical Context

- **Language/Version**: Python 3.14 (floor 3.12)
- **Primary Dependencies**: scion-sdk, fastmcp
- **Storage**: N/A; the connector stores nothing
- **Testing**: pytest; the proof level each row names
- **Target Platform**: Linux containers
- **Project Type**: connector (MCP server)
- **Performance Goals**: N/A
- **Constraints**: one tool per API operation; shared capabilities come from the SDK
- **Scale/Scope**: 5 rows

## Constitution Check

- One owner per concept: pass. The connector owns one external system's calls, mappings and tools only.
- Connectors depend on the SDK and are called through arbora: pass.
- No ratchets: pass. The rows add no suppression.

## Project Structure

```text
pyproject.toml  existing
tests/test_default_content_surfaces.py  existing
container_manager_mcp/ontology/access_contracts.ttl  existing
```

## Phase 0: Research

No unknowns. Each row names the code it touches.

## Phase 1: Design

The rows are delivered in place; the design is the caller chain below.

## Phase 2: Task approach

`tasks.md` holds one task line per row, producer first.

## Complexity Tracking

None. The constitution check passed.

## Build contract

The repository pre-commit hooks apply: `ruff-format`, `connector-standard`, `secrets`, `tests-fast`. Each row names its proving test. No suppression and no duplicate implementation.

## Production caller chain

1. `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R001`: pyproject scion_sdk.connector entry point container-manager-mcp -> container_manager_mcp/__init__.py::connector (L101) -> scion_sdk.connector::build_server, which serves the packaged skill://, ontology:// resources and prompts from container_manager_mcp/skills, ontology and prompts.
2. `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R002`: arbora fleet.call -> connector tools/call -> verbose-tier k8s_list_services (surface.py::VERBOSE_TARGETS, prefix k8s over KubernetesManager) -> container_manager_mcp/k8s/workloads.py::list_services (L22); the contract file is container_manager_mcp/ontology/access_contracts.ttl.

## Cross-repo seams

none. The cited owner rows are library capabilities the connector calls in process; spec.md names them.

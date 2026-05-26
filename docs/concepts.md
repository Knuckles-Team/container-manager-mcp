# Concept Registry — container-manager-mcp

> **Prefix**: `CONCEPT:CMGR-*`
> **Version**: 1.15.0
> **Bridge**: [`CONCEPT:ECO-4.0`](../../agent-utilities/docs/concepts.md) (Unified Toolkit Ingestion)

---

## Project-Specific Concepts

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:CMGR-001` | Compose Operations | MCP tool domain `compose` — Action-routed dynamic tool registration |
| `CONCEPT:CMGR-002` | Container Operations | MCP tool domain `container` — Action-routed dynamic tool registration |
| `CONCEPT:CMGR-003` | Image Operations | MCP tool domain `image` — Action-routed dynamic tool registration |
| `CONCEPT:CMGR-004` | Info Operations | MCP tool domain `info` — Action-routed dynamic tool registration |
| `CONCEPT:CMGR-005` | Misc Operations | MCP tool domain `misc` — Action-routed dynamic tool registration |
| `CONCEPT:CMGR-006` | Network Operations | MCP tool domain `network` — Action-routed dynamic tool registration |
| `CONCEPT:CMGR-007` | Swarm Operations | MCP tool domain `swarm` — Action-routed dynamic tool registration |
| `CONCEPT:CMGR-008` | System Information & Health | MCP tool domain `system` — Action-routed dynamic tool registration |
| `CONCEPT:CMGR-009` | Volume Operations | MCP tool domain `volume` — Action-routed dynamic tool registration |

## Cross-Project References (from agent-utilities)

| Concept ID | Name | Origin |
|------------|------|--------|
| `CONCEPT:ECO-4.0` | Unified Toolkit Ingestion | agent-utilities |
| `CONCEPT:ORCH-1.2` | Confidence-Gated Router | agent-utilities |
| `CONCEPT:OS-5.1` | Prompt Injection Defense | agent-utilities |
| `CONCEPT:OS-5.2` | Cognitive Scheduler | agent-utilities |
| `CONCEPT:OS-5.3` | Guardrail Engine | agent-utilities |
| `CONCEPT:OS-5.4` | Audit Logging | agent-utilities |
| `CONCEPT:KG-2.0` | Knowledge Graph Core | agent-utilities |

## Synergy with agent-utilities

This project integrates with `agent-utilities` via `CONCEPT:ECO-4.0` (Unified Toolkit Ingestion). The `container_manager_mcp` MCP server registers its tools with the agent-utilities FastMCP middleware, enabling automatic discovery, telemetry, and Knowledge Graph ingestion of all CMGR-* concepts.

# Concept Registry — container-manager-mcp

> **Prefix**: `CONCEPT:CMGR-*`
> **Version**: 1.15.0
> **Bridge**: [`CONCEPT:AU-ECO.messaging.native-backend-abstraction`](https://github.com/Knuckles-Team/agent-utilities/blob/main/docs/concepts.md) (Unified Toolkit Ingestion)

---

## Project-Specific Concepts

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:CN-OS.governance.cmgr` | Compose Operations | MCP tool domain `compose` — Action-routed dynamic tool registration |
| `CONCEPT:CN-OS.governance.cmgr-2` | Container Operations | MCP tool domain `container` — Action-routed dynamic tool registration |
| `CONCEPT:CN-OS.governance.cmgr-3` | Image Operations | MCP tool domain `image` — Action-routed dynamic tool registration |
| `CONCEPT:CN-OS.governance.cmgr-4` | Info Operations | MCP tool domain `info` — Action-routed dynamic tool registration |
| `CONCEPT:CN-OS.governance.cmgr-5` | Misc Operations | MCP tool domain `misc` — Action-routed dynamic tool registration |
| `CONCEPT:CN-OS.governance.cmgr-6` | Network Operations | MCP tool domain `network` — Action-routed dynamic tool registration |
| `CONCEPT:CN-OS.governance.cmgr-7` | Swarm Operations | MCP tool domain `swarm` — Action-routed dynamic tool registration |
| `CONCEPT:CN-OS.governance.cmgr-8` | System Information & Health | MCP tool domain `system` — Action-routed dynamic tool registration |
| `CONCEPT:CN-OS.governance.cmgr-9` | Volume Operations | MCP tool domain `volume` — Action-routed dynamic tool registration |

## Cross-Project References (from agent-utilities)

| Concept ID | Name | Origin |
|------------|------|--------|
| `CONCEPT:AU-ECO.messaging.native-backend-abstraction` | Unified Toolkit Ingestion | agent-utilities |
| `CONCEPT:AU-ORCH.adapter.hot-cache-invalidation` | Confidence-Gated Router | agent-utilities |
| `CONCEPT:AU-OS.config.secrets-authentication` | Prompt Injection Defense | agent-utilities |
| `CONCEPT:AU-OS.state.cognitive-scheduler-preemption` | Cognitive Scheduler | agent-utilities |
| `CONCEPT:AU-OS.governance.reactive-multi-axis-budget` | Guardrail Engine | agent-utilities |
| `CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox` | Audit Logging | agent-utilities |
| `CONCEPT:AU-KG.query.object-graph-mapper` | Knowledge Graph Core | agent-utilities |

## Synergy with agent-utilities

This project integrates with `agent-utilities` via `CONCEPT:AU-ECO.messaging.native-backend-abstraction` (Unified Toolkit Ingestion). The `container_manager_mcp` MCP server registers its tools with the agent-utilities FastMCP middleware, enabling automatic discovery, telemetry, and Knowledge Graph ingestion of all CMGR-* concepts.

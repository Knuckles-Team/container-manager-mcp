# container-manager-mcp — Concept Overview

> **Category**: Infrastructure | **Ecosystem Role**: MCP Server + A2A Agent
> Built on [`agent-utilities`](https://github.com/Knuckles-Team/agent-utilities) — the unified AGI Harness.

## Description

Container Manager - manage Docker, Docker Swarm, Podman, and Kubernetes containers and workloads.
MCP+A2A Servers Out of the Box!

## Capability Overview

`container-manager-mcp` provides **full coverage across all three engines**:

- **Docker** — the base `cm_container_operations` / `cm_image_operations` / `cm_network_operations` /
  `cm_volume_operations` / `cm_system_operations` / `cm_compose_operations` surface, plus
  Swarm/service/stack/config/secret/node operations via `cm_docker_swarm`.
- **Podman** — the same base surface (rootless-compatible) plus pod-level operations
  (pods, `generate`/`play kube`, checkpoint/restore, health) via `cm_podman`.
- **Kubernetes** (RKE2 / k3s / vanilla) — a full operational surface built on the official `kubernetes`
  Python client, exposed as 8 themed tools: `cm_k8s_workloads`, `cm_k8s_config`, `cm_k8s_networking`,
  `cm_k8s_storage`, `cm_k8s_rbac`, `cm_k8s_cluster`, `cm_k8s_governance`, and `cm_k8s_observability`.
- **Multi-context** — `cm_multi_context` operates several Docker/Podman/Swarm/Kubernetes contexts
  simultaneously with parallel fan-out, health checks, and lazy reconnect.
- **Knowledge Graph ingestion** — `cm_ingest_inventory` maps live inventory (containers, images, volumes,
  networks, Swarm services/nodes, and — on a Kubernetes manager — pods, deployments, namespaces, and native
  k8s Services) into typed OWL/RDF nodes for cross-source reasoning.

## Enterprise Readiness

All agents in the ecosystem inherit enterprise-grade infrastructure from `agent-utilities`:

| Feature | Status | Source |
|:--------|:-------|:-------|
| **JWT/OIDC Authentication** | ✅ Built-in | `agent-utilities[auth]` — Authlib JWKS + API key middleware |
| **OpenTelemetry Instrumentation** | ✅ Built-in | `agent-utilities[logfire]` — OTLP export, FastAPI auto-instrumentation |
| **HashiCorp Vault Integration** | ✅ Built-in | `agent-utilities[vault]` — `secret://`, `env://`, `vault://` URI schemes |
| **Audit Logging** | ✅ Built-in | Append-only compliance trail with 30+ action types (CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox) |
| **Token Usage Analytics** | ✅ Built-in | 4-bucket tracking with budget alerting (CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox) |
| **Prompt Injection Defense** | ✅ Built-in | 25+ pattern scanner + jailbreak taxonomy (CONCEPT:AU-OS.config.secrets-authentication) |
| **Guardrail Engine** | ✅ Built-in | Input/output interception with block/redact/warn (CONCEPT:AU-OS.governance.reactive-multi-axis-budget) |
| **Action Execution Pipeline** | ✅ Built-in | Token, cost, duration, and node transition limits Dry-run / commit / rollback phases (CONCEPT:AU-ORCH.adapter.kg-graph-materialization) |
| **Resource Scheduling** | ✅ Built-in | Priority queuing + preemption limits (CONCEPT:AU-OS.state.cognitive-scheduler-preemption) |
| **Session Concurrency** | ✅ Built-in | Enqueue/reject/interrupt/rollback (CONCEPT:AU-OS.governance.reactive-multi-axis-budget) |

## Concept Registry

This project implements or inherits the following ecosystem concepts:

| Concept ID | Description | Source |
|:-----------|:------------|:-------|
| ECO-4.1 | MCP & Universal Skills | `agent-utilities` (inherited) |
| OS-5.0 | Agent OS Kernel | `agent-utilities` (inherited) |

> 📖 **Full Registry**: See [`agent-utilities/docs/overview.md`](https://github.com/Knuckles-Team/agent-utilities/blob/main/docs/overview.md) for the complete 5-Pillar concept index.

## Architecture

This project follows the standardized agent-package pattern:

```
container-manager-mcp/
├── container_manager_mcp/        # Source code
│   ├── __init__.py
│   ├── agent_server.py      # Entry point (create_graph_agent_server)
│   ├── api_client.py        # REST/GraphQL API wrapper
│   └── mcp_server.py        # FastMCP tool definitions
├── tests/                   # Test suite
├── docs/                    # Documentation
├── pyproject.toml           # Package metadata
├── mcp_config.json          # MCP server configuration
├── main_agent.json          # Agent identity & system prompt
└── Dockerfile               # Container deployment
```

## MCP Configuration

### stdio Mode
```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "command": "uv",
      "args": ["run", "--with", "container-manager-mcp", "container-mcp"],
      "env": {}
    }
  }
}
```

### Streamable HTTP Mode
```bash
container-mcp --transport streamable-http --port 8001
```

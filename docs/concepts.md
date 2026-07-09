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
| `CONCEPT:CN-OS.governance.cmgr-10` | Docker Advanced Operations | MCP tool domain `cm_docker_advanced` — Swarm, services, stacks, configs, secrets, nodes |
| `CONCEPT:CN-OS.governance.cmgr-11` | Podman Advanced Operations | MCP tool domain `cm_podman_advanced` — pods, `generate`/`play kube`, checkpoint/restore, system |
| `CONCEPT:CN-OS.governance.cmgr-12` | Kubernetes Workloads | MCP tool domain `cm_k8s_workloads` — pods, rollouts, StatefulSets, DaemonSets, ReplicaSets, Jobs, CronJobs |
| `CONCEPT:CN-OS.governance.cmgr-13` | Kubernetes Config | MCP tool domain `cm_k8s_config` — ConfigMaps, Secrets, Namespaces, CRDs, `patch_resource`, state tracking |
| `CONCEPT:CN-OS.governance.cmgr-14` | Kubernetes Networking | MCP tool domain `cm_k8s_networking` — Ingress, NetworkPolicies, DNS, native Services |
| `CONCEPT:CN-OS.governance.cmgr-15` | Kubernetes Storage | MCP tool domain `cm_k8s_storage` — PV, PVC, StorageClasses, VolumeSnapshots, CSI |
| `CONCEPT:CN-OS.governance.cmgr-16` | Kubernetes RBAC | MCP tool domain `cm_k8s_rbac` — roles, bindings, service accounts, tokens, access reviews, pod security |
| `CONCEPT:CN-OS.governance.cmgr-17` | Kubernetes Cluster | MCP tool domain `cm_k8s_cluster` — nodes, contexts, CSRs, API resources, cluster info |
| `CONCEPT:CN-OS.governance.cmgr-18` | Kubernetes Governance | MCP tool domain `cm_k8s_governance` — ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs |
| `CONCEPT:CN-OS.governance.cmgr-19` | Kubernetes Observability | MCP tool domain `cm_k8s_observability` — metrics, watch/stream/events, debug helpers |
| `CONCEPT:CN-OS.governance.cmgr-20` | Multi-Context Operations | MCP tool domain `cm_multi_context` — parallel Docker/Podman/Swarm/Kubernetes fan-out |

## Kubernetes Ontology Classes

The Kubernetes tool surface is backed by net-new OWL classes in `container_manager_mcp/ontology/container.ttl`
(reusing the hub's `:Container` / `:ContainerImage` / `:Host` where applicable):

| Class | Label | Populated by |
|-------|-------|--------------|
| `:Pod` | Kubernetes Pod | `cm_k8s_workloads` `list_pods` / `describe_pod`; `cm_ingest_inventory` modality `pods` |
| `:Deployment` | Kubernetes Deployment | `cm_k8s_workloads` rollout actions; `cm_ingest_inventory` modality `deployments` |
| `:Namespace` | Kubernetes Namespace | `cm_k8s_config` `list_namespaces`; `cm_ingest_inventory` modality `namespaces` |
| `:K8sService` | Kubernetes Service | `cm_k8s_networking` `list_k8s_services`; `cm_ingest_inventory` modality `k8s_services` |
| `:K8sNode` | Kubernetes Node | `cm_k8s_cluster` `list_nodes` |

Linking properties include `:runsInNamespace` (Pod/Deployment/K8sService → Namespace), `:managedByDeployment`
(Pod → Deployment), and `:scheduledOnK8sNode` (Pod → K8sNode), plus fields `:podPhase`,
`:deploymentReplicas` / `:deploymentReadyReplicas`, and `:namespaceStatus`.

## Knowledge Graph Ingestion Modalities

`cm_ingest_inventory` accepts a `modality` of `all`, `containers`, `images`, `volumes`, `networks`,
`services`, `nodes`, and — when the active manager is Kubernetes — `pods`, `deployments`, `namespaces`, and
`k8s_services`. `modality="all"` always sweeps the Docker/Swarm-shaped modalities and additionally sweeps the
four Kubernetes modalities when `CONTAINER_MANAGER_TYPE=kubernetes` (or the resolved manager is a
`KubernetesManager`). See [Kubernetes](kubernetes.md) for the tool surface these modalities are sourced from.

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

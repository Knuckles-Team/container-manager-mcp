# Container Manager Mcp
## CLI or API | MCP | Agent

![PyPI - Version](https://img.shields.io/pypi/v/container-manager-mcp)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
![PyPI - Downloads](https://img.shields.io/pypi/dd/container-manager-mcp)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/container-manager-mcp)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/container-manager-mcp)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/container-manager-mcp)
![PyPI - License](https://img.shields.io/pypi/l/container-manager-mcp)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/container-manager-mcp)
![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/container-manager-mcp)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/container-manager-mcp)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/container-manager-mcp)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/container-manager-mcp)
![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/container-manager-mcp)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/container-manager-mcp)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/container-manager-mcp)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/container-manager-mcp)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/container-manager-mcp)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/container-manager-mcp)

*Version: 2.1.4*

> **Documentation** — Installation, deployment, usage across the API, CLI, MCP, and
> A2A agent interfaces, and the multi-host control plane are maintained in the
> [official documentation](https://knuckles-team.github.io/container-manager-mcp/).

---

## Overview

**Container Manager Mcp** is a production-grade Agent and Model Context Protocol (MCP) server designed to interface directly with Container Manager - manage Docker, Docker Swarm, Podman, **and Kubernetes** containers and workloads. MCP+A2A Servers Out of the Box!.

**Full coverage across all three engines:** Docker (incl. Swarm/service/stack/config/secret/node
operations via `cm_docker_swarm`), Podman (incl. pod/kube operations, `generate`/`play kube`, checkpoint/restore
via `cm_podman`), and a **complete Kubernetes surface** (RKE2 / k3s / vanilla) spanning workloads,
config, networking, storage, RBAC, cluster admin, governance, and observability through 8 themed `cm_k8s_*`
tools built on the official `kubernetes` Python client. A `cm_multi_context` tool lets an agent fan out
operations across several Docker/Podman/Swarm/Kubernetes contexts in parallel, and `cm_ingest_inventory` feeds
Docker, Swarm, **and Kubernetes** resources into the ontology-driven Knowledge Graph as typed nodes.

---

## Key Features

- **Consolidated Action-Routed MCP Tools:** Minimizes token overhead and eliminates tool bloat in LLM contexts by grouping methods into optimized, togglable tool modules.
- **Full Docker + Swarm + Podman + Kubernetes Coverage:** First-class support for Docker (incl. Swarm/service/stack/config/secret/node ops), rootless Podman (incl. pods, `generate`/`play kube`, checkpoint/restore), and a full Kubernetes surface (RKE2 / k3s / vanilla) across workloads, config, networking, storage, RBAC, cluster admin, governance, and observability. See [Kubernetes](#kubernetes) below.
- **Multi-Context Parallel Operation:** `cm_multi_context` fans operations out across several Docker, Podman, Swarm, and/or Kubernetes contexts at once (`ThreadPoolExecutor`-backed), with health checks and lazy reconnect.
- **Enterprise-Grade Security:** Comprehensive support for Eunomia policies, OIDC token delegation, and granular execution context tracking.
- **Integrated Graph Agent:** Built-in Pydantic AI agent supporting the Agent Control Protocol (ACP) and standard Web interfaces (AG-UI).
- **Ontology-Driven KG Ingestion:** `cm_ingest_inventory` maps live Docker/Swarm/Kubernetes inventory (containers, images, volumes, networks, services, nodes, pods, deployments, namespaces, native k8s Services) into typed OWL/RDF nodes for cross-source reasoning.
- **Native Telemetry & Tracing:** Out-of-the-box OpenTelemetry exports and native Langfuse tracing.

---

## Multi-Host & Zero-Script Remote Docker Orchestration

`container-manager-mcp` allows a single master instance of the MCP server on your controller to route container and volume operations securely to remote hosts over SSH standard tunneling.

- **Unified Inventory**: Connection endpoints are loaded dynamically from the XDG shared inventory at `~/.config/agent-utilities/inventory.yml` (`.yml` preferred; a legacy `inventory.yaml` is still read when no `.yml` exists).
- **Zero TCP Socket Exposure**: Operations route directly over the standard SSH channel securely, removing the need to expose Docker socket TCP ports.

> **Shared inventory:** the `cm_*` host aliases you pass as `host` come from the **same**
> `inventory.yml` used by **tunnel-manager** — define your fleet once. Create and validate
> it with `tunnel-manager inventory init` / `tunnel-manager inventory doctor`. See
> tunnel-manager's [Inventory guide](https://knuckles-team.github.io/tunnel-manager/inventory/)
> for the full schema, template, and override options.

To configure and utilize the multi-host remote routing, see the detailed [Multi-Host Architecture Guide](docs/multi_host.md).

---

## CLI or API

This agent wraps the Container Manager - manage Docker, Docker Swarm, and Podman containers. MCP+A2A Servers Out of the Box! API. You can interact with it programmatically or via its integrated execution entrypoints.

Detailed instructions on how to use the underlying API wrappers, extended schema bindings, and developer SDK references are maintained in [docs/index.md](docs/index.md).

### Environment doctor

Not sure your environment is wired up? Run the guided **doctor** first. It probes
every surface with real checks — python client libs + CLIs, `CONTAINER_MANAGER_TYPE`
/ toggles / `K8S_CONTEXTS` parsing, the tunnel-manager SSH **inventory** (and per-host
reachability), the **docker** / **podman** daemons, and each **kubernetes** context —
and prints concrete remediation for anything that is not OK, so you are walked through
connecting to your environments. Available as the `container-manager-doctor` CLI and
the `cm_doctor` MCP tool.

```bash
container-manager-doctor                       # diagnose everything (exit 1 if any FAIL)
container-manager-doctor --guided              # + probe every inventory host
container-manager-doctor --backend inventory --host <alias>
container-manager-doctor --backend kubernetes --context <ctx>
container-manager-doctor --backend docker --host <alias>
container-manager-doctor --json                # machine-readable report
```

The `cm_doctor` MCP tool mirrors it: `action=run` for a full sweep, or a focused
`check_backends` / `check_inventory` / `check_docker` / `check_podman` /
`check_kubernetes`. Each check returns `{name, category, status: ok|warn|fail, detail,
remediation}` plus a summary. Start here, then follow the
[container-manager-config-walkthrough](container_manager_mcp/skills/container-manager-config-walkthrough/SKILL.md)
skill.

---

## MCP

This server utilizes dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools

_Auto-generated — do not edit (synced by the `mcp-readme-table` pre-commit hook)._

<!-- MCP-TOOLS-TABLE:START -->

#### Condensed action-routed tools (default — `MCP_TOOL_MODE=condensed`)

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `cm_compose_operations` | `COMPOSETOOL` | Manage docker-compose or podman-compose operations. |
| `cm_container_operations` | `CONTAINERTOOL` | Manage container operations. |
| `cm_docker_swarm` | `DOCKERSWARMTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_doctor` | `DOCTORTOOL` | Diagnose + get remediation for the inventory / kubernetes / docker / podman environment. |
| `cm_image_operations` | `IMAGETOOL` | Manage container images. |
| `cm_info_operations` | `INFOTOOL` | Manage container manager info operations. |
| `cm_ingest_inventory` | `MISCTOOL` | Natively ingest the container inventory into epistemic-graph as typed nodes. |
| `cm_k8s_cluster` | `K8SCLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_config` | `K8SCONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_governance` | `K8SGOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_networking` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_observability` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_rbac` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_storage` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_workloads` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_list_hosts` | `INVENTORYTOOL` | List the host aliases you can pass as ``host`` to any cm_* operation |
| `cm_multi_context` | `MULTICONTEXTTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_network_operations` | `NETWORKTOOL` | Manage network operations. |
| `cm_podman` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_swarm_operations` | `SWARMTOOL` | Manage swarm operations. |
| `cm_system_operations` | `SYSTEMTOOL` | Manage container manager system operations. |
| `cm_volume_operations` | `VOLUMETOOL` | Manage volume operations. |
| `trace_port_namespace` | `MISCTOOL` | Locate the container actively using/mapping the specified port on the target host. |

#### Verbose 1:1 API-mapped tools (`MCP_TOOL_MODE=verbose` or `both`)

<details>
<summary>291 per-operation tools — one per public API method (click to expand)</summary>

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `cm_container_operations__exec_in_container` | `CONTAINERTOOL` | Manage container operations. |
| `cm_container_operations__get_container_logs` | `CONTAINERTOOL` | Manage container operations. |
| `cm_container_operations__list_containers` | `CONTAINERTOOL` | Manage container operations. |
| `cm_container_operations__prune_containers` | `CONTAINERTOOL` | Manage container operations. |
| `cm_container_operations__remove_container` | `CONTAINERTOOL` | Manage container operations. |
| `cm_container_operations__stop_container` | `CONTAINERTOOL` | Manage container operations. |
| `cm_docker_swarm__docker_config_create` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_config_list` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_node_inspect` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_node_ls` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_node_update` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_secret_create` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_secret_list` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_service_create` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_service_list` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_service_logs` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_service_ps` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_service_rm` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_service_update` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_stack_deploy` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_stack_rm` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_stack_services` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_swarm_init` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_swarm_join` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_docker_swarm__docker_swarm_leave` | `DOCKERTOOL` | Manage Docker Swarm operations (Swarm, services, stacks, configs, secrets, nodes). |
| `cm_doctor__check_backends` | `DOCTORTOOL` | Diagnose + get remediation for the inventory / kubernetes / docker / podman environment. |
| `cm_doctor__check_docker` | `DOCTORTOOL` | Diagnose + get remediation for the inventory / kubernetes / docker / podman environment. |
| `cm_doctor__check_inventory` | `DOCTORTOOL` | Diagnose + get remediation for the inventory / kubernetes / docker / podman environment. |
| `cm_doctor__check_kubernetes` | `DOCTORTOOL` | Diagnose + get remediation for the inventory / kubernetes / docker / podman environment. |
| `cm_doctor__check_podman` | `DOCTORTOOL` | Diagnose + get remediation for the inventory / kubernetes / docker / podman environment. |
| `cm_doctor__run` | `DOCTORTOOL` | Diagnose + get remediation for the inventory / kubernetes / docker / podman environment. |
| `cm_image_operations__list_images` | `IMAGETOOL` | Manage container images. |
| `cm_image_operations__prune_images` | `IMAGETOOL` | Manage container images. |
| `cm_image_operations__pull_image` | `IMAGETOOL` | Manage container images. |
| `cm_image_operations__remove_image` | `IMAGETOOL` | Manage container images. |
| `cm_k8s_cluster__approve_csr` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__cluster_info_dump` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__cordon_node` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__deny_csr` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__describe_api_resource` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__describe_cluster_plugin` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__drain_node` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__get_api_server_info` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__get_cluster_info` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__get_config` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__get_node_affinity` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__get_node_conditions` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__inspect_node` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__list_api_resources` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__list_cluster_plugins` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__list_contexts` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__list_csr` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__list_node_taints` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__list_nodes` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__rename_context` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__set_node_affinity` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__set_pod_anti_affinity` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__taint_node` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__test_cluster_plugin` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__uncordon_node` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__untaint_node` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__use_context` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_cluster__validate_kubeconfig` | `CLUSTERTOOL` | Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins). |
| `cm_k8s_config__annotate_resource` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__compare_configmap_state` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__create_configmap` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__create_namespace` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__create_secret` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__delete_namespace` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__describe_crd` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__get_secret_state_hash` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__label_resource` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__list_configmaps` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__list_crds` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__list_custom_resources` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__list_events` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__list_namespaces` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__list_secrets` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__patch_resource` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__sync_configmap_from_file` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__track_resource_version` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_config__wait_for_resource_version` | `CONFIGTOOL` | Manage Kubernetes configuration (configmaps, secrets, namespaces, events, CRDs, labels/annotations/patch, state tracking). |
| `cm_k8s_governance__create_horizontal_pod_autoscaler` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__create_limit_range` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__create_pod_disruption_budget` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__create_priority_class` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__create_resource_quota` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__delete_horizontal_pod_autoscaler` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__delete_limit_range` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__delete_pod_disruption_budget` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__delete_priority_class` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__delete_resource_quota` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__describe_horizontal_pod_autoscaler` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__describe_limit_range` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__describe_pod_disruption_budget` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__describe_priority_class` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__describe_resource_quota` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__list_horizontal_pod_autoscalers` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__list_limit_ranges` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__list_pod_disruption_budgets` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__list_priority_classes` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__list_resource_quotas` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__update_horizontal_pod_autoscaler` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_governance__update_resource_quota` | `GOVERNANCETOOL` | Manage Kubernetes governance resources (ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs). |
| `cm_k8s_networking__check_dns_resolution` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__create_ingress` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__create_ingress_class` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__create_k8s_service` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__create_network_policy_with_cidr` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__create_networkpolicy` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__delete_ingress` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__delete_k8s_service` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__delete_networkpolicy` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__describe_ingress_class` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__get_k8s_service` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__list_dns_endpoints` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__list_endpoints` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__list_endpointslices` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__list_ingress` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__list_ingress_classes` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__list_k8s_services` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__list_networkpolicies` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__set_default_ingress_class` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__test_dns_connectivity` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__test_network_policy_connectivity` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_networking__update_network_policy_rules` | `K8SNETWORKINGTOOL` | Manage Kubernetes networking (ingress, ingress classes, network policies, endpoints, DNS, native services). |
| `cm_k8s_observability__debug_deployment` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__debug_node` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__debug_pod` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__debug_service` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__get_autoscaler_history` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__get_autoscaler_metrics` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__get_cluster_resource_summary` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__get_node_metrics` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__get_pod_metrics` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__get_pod_resource_usage` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__get_resource_events` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__list_field_selector` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__scale_deployment_autoscaler` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__set_autoscaler_metrics` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__stream_pod_logs` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__top_nodes` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__top_pods` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_observability__watch_resource` | `K8SOBSERVABILITYTOOL` | Observe Kubernetes resources (metrics, autoscaler metrics, watch/stream/events, debug helpers). |
| `cm_k8s_rbac__auth_can_i` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__create_aggregated_cluster_role` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__create_cluster_rolebinding` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__create_pod_security_policy` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__create_role` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__create_rolebinding` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__create_service_account_token` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__create_serviceaccount` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__delete_cluster_rolebinding` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__delete_pod_security_policy` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__delete_role` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__delete_rolebinding` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__delete_service_account_token` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__delete_serviceaccount` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__describe_pod_security_policy` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__evaluate_pod_security` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__list_cluster_rolebindings` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__list_cluster_roles` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__list_pod_security_policies` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__list_rolebindings` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__list_roles` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__list_service_account_mapped_secrets` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__list_service_account_tokens` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__list_serviceaccounts` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__local_subject_access_review` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__map_secret_to_service_account` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__subject_access_review` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__unmap_secret_from_service_account` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_rbac__update_aggregated_cluster_role` | `K8SRBACTOOL` | Manage Kubernetes RBAC and security (roles, bindings, service accounts, tokens, access reviews, pod security, secret mapping). |
| `cm_k8s_storage__create_persistent_volume` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__create_persistent_volume_claim` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__create_storage_class` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__create_volume_snapshot` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__delete_persistent_volume_claim` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__describe_csi_driver` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__expand_persistent_volume` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__expand_pvc` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__get_csi_driver_capacity` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__get_storage_class_provisioner` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__list_csi_drivers` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__list_persistent_volume_claims` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__list_persistent_volumes` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__list_storage_classes` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__list_volume_snapshots` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_storage__set_default_storage_class` | `K8SSTORAGETOOL` | Manage Kubernetes storage (PV, PVC, storage classes, volume snapshots, CSI drivers). |
| `cm_k8s_workloads__attach_pod` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__copy_from_pod` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__copy_to_pod` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__create_cron_job` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__create_daemon_set` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__create_job` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__create_stateful_set` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__delete_cron_job` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__delete_job` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__describe_cron_job` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__describe_job` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__describe_pod` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__describe_replicaset` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__exec_pod` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__get_daemonset_update_strategy` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__get_deployment_strategy` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__get_statefulset_update_strategy` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__list_cron_jobs` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__list_daemonsets` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__list_jobs` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__list_pods` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__list_replicasets` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__list_statefulsets` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__port_forward_pod` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__rollout_history` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__rollout_pause` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__rollout_restart` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__rollout_resume` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__rollout_status` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__rollout_undo` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__scale_replicaset` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__scale_statefulset` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__set_daemonset_update_strategy` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__set_deployment_strategy` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_k8s_workloads__set_statefulset_update_strategy` | `K8SWORKLOADSTOOL` | Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs). |
| `cm_multi_context__create_network` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__create_service` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__create_volume` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__describe_pod` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__inspect_container` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__list_containers` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__list_contexts` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__list_deployments` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__list_images` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__list_networks` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__list_pods` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__list_services` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__list_volumes` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__pull_image` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__remove_container` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__remove_image` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__remove_network` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__remove_service` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__remove_volume` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__run_container` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__scale_deployment` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_multi_context__stop_container` | `DOCKERTOOL` | Manage containers across multiple backends (Kubernetes, Docker, Podman, Swarm) with context selection. |
| `cm_network_operations__create_network` | `NETWORKTOOL` | Manage network operations. |
| `cm_network_operations__list_networks` | `NETWORKTOOL` | Manage network operations. |
| `cm_network_operations__prune_networks` | `NETWORKTOOL` | Manage network operations. |
| `cm_network_operations__remove_network` | `NETWORKTOOL` | Manage network operations. |
| `cm_podman__podman_checkpoint` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_generate_kube_yaml` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_health_check` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_network_create` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_network_inspect` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_network_list` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_play_kube_yaml` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_pod_create` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_pod_inspect` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_pod_list` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_pod_logs` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_pod_rm` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_pod_stats` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_pod_stop` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_pod_top` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_restore` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_system_prune` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_volume_create` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_volume_inspect` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_podman__podman_volume_list` | `PODMANTOOL` | Manage Podman operations (pods, networks, volumes, checkpoint/restore, kube interop, system). |
| `cm_swarm_operations__create_service` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__init_swarm` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__inspect_node` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__inspect_service` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__leave_swarm` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__list_nodes` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__list_services` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__remove_node` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__remove_service` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__scale_service` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__service_logs` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__service_ps` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__update_node` | `SWARMTOOL` | Manage swarm operations. |
| `cm_swarm_operations__update_service` | `SWARMTOOL` | Manage swarm operations. |
| `cm_system_operations__get_info` | `SYSTEMTOOL` | Manage container manager system operations. |
| `cm_system_operations__get_version` | `SYSTEMTOOL` | Manage container manager system operations. |
| `cm_system_operations__prune_system` | `SYSTEMTOOL` | Manage container manager system operations. |
| `cm_volume_operations__create_volume` | `VOLUMETOOL` | Manage volume operations. |
| `cm_volume_operations__list_volumes` | `VOLUMETOOL` | Manage volume operations. |
| `cm_volume_operations__prune_volumes` | `VOLUMETOOL` | Manage volume operations. |
| `cm_volume_operations__remove_volume` | `VOLUMETOOL` | Manage volume operations. |

</details>

_23 action-routed tool(s) (default) · 291 verbose 1:1 tool(s). Each is enabled unless its `<DOMAIN>TOOL` toggle is set false; `MCP_TOOL_MODE` selects the surface (`condensed` default · `verbose` 1:1 · `both`). Auto-generated — do not edit._
<!-- MCP-TOOLS-TABLE:END -->

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/mcp.md](docs/mcp.md).

### Kubernetes

`container-manager-mcp` ships **full Kubernetes coverage** (RKE2 / k3s / vanilla) through the **official
`kubernetes` Python client**, exposed as 8 themed, action-routed MCP tools (replacing an earlier, messier
tool sprawl):

| Tool | Covers |
|------|--------|
| `cm_k8s_workloads` | Pods, Deployments, StatefulSets, DaemonSets, Jobs, CronJobs, ReplicaSets, rollouts (status/history/restart/undo/pause/resume), exec/logs/attach/copy |
| `cm_k8s_config` | ConfigMaps, Secrets, Namespaces, CRDs, Events, label/annotate, generic `patch_resource`, and ConfigMap/Secret config-state tracking |
| `cm_k8s_networking` | Ingress, IngressClasses, NetworkPolicies, Endpoints/EndpointSlices, DNS checks, and **true core Services** (`list_k8s_services` / `get_k8s_service` / `create_k8s_service` / `delete_k8s_service`) |
| `cm_k8s_storage` | PersistentVolumes, PersistentVolumeClaims, StorageClasses, VolumeSnapshots, CSI drivers, `expand_pvc` |
| `cm_k8s_rbac` | Roles, ClusterRoles, RoleBindings, ClusterRoleBindings, ServiceAccounts, tokens, `auth_can_i`, `subject_access_review`, pod-security |
| `cm_k8s_cluster` | Nodes (cordon/drain/taint/affinity), kubeconfig contexts, CertificateSigningRequests, API resources, cluster-info |
| `cm_k8s_governance` | ResourceQuotas, LimitRanges, PriorityClasses, PodDisruptionBudgets, HorizontalPodAutoscalers (full CRUD) |
| `cm_k8s_observability` | `top`/metrics, autoscaler metrics, watch/stream/events, and `debug_pod`/`debug_node`/`debug_service`/`debug_deployment` helpers |

Two details worth knowing:

- **`patch_resource`** (on `cm_k8s_config`) is a generic patch action supporting `strategic` / `merge` / `json`
  patch types against any resource kind — use it when there's no dedicated action for a field-level update.
- **True core Services vs. Swarm-parity services** — `cm_k8s_networking`'s `list_k8s_services` /
  `get_k8s_service` / `create_k8s_service` / `delete_k8s_service` operate on real Kubernetes `Service`
  objects. This is distinct from `cm_multi_context`'s Swarm-parity `list_services`, which is
  Deployment-shaped for cross-backend comparability.

Kubernetes access is configured via `CONTAINER_MANAGER_TYPE=kubernetes` and `CONTAINER_MANAGER_KUBECONTEXT`
(see [Environment Variables](#environment-variables)); for operating several clusters at once, see
`K8S_CONTEXTS` / `cm_multi_context` below. For a full worked walkthrough of the tool surface, see the
[`container-manager-kubernetes-operations`](container_manager_mcp/skills/container-manager-kubernetes-operations)
skill and [docs/usage.md](docs/usage.md#kubernetes).

### Multi-Context Operation

`cm_multi_context` (toggle `MULTICONTEXTTOOL`) lets a single MCP call operate across **several Docker,
Podman, Swarm, and/or Kubernetes contexts at once** — configured via `K8S_CONTEXTS`, `DOCKER_CONTEXTS`,
`SWARM_CONTEXTS`, and their `DEFAULT_*_CONTEXT` defaults (`MULTI_CONTEXT_MODE=True` routes every call through
it). Fan-out is parallel (`ThreadPoolExecutor`-backed) with per-backend health checks and lazy reconnect, so
an agent can compare, migrate between, or simultaneously act on multiple clusters/engines/hosts in one call.
See the [`container-manager-multi-context`](container_manager_mcp/skills/container-manager-multi-context)
skill and [docs/usage.md](docs/usage.md#multi-context) for examples.

### Dynamic Tool Selection & Visibility

This MCP server supports dynamic toolset selection and visibility filtering at runtime. This allows you to restrict the set of exposed tools in order to prevent blowing up the LLM's context window.

You can configure tool filtering via multiple input channels:

- **CLI Arguments:** Pass `--tools` or `--toolsets` (or their disabled counterparts `--disabled-tools` and `--disabled-toolsets`) during startup.
- **Environment Variables:** Define standard environment variables:
  - `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS`
  - `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS`
- **HTTP SSE Request Headers:** Pass custom headers during transport initialization:
  - `x-mcp-enabled-tools` / `x-mcp-disabled-tools`
  - `x-mcp-enabled-tags` / `x-mcp-disabled-tags`
- **HTTP SSE Request Query Parameters:** Append query parameters directly to your transport connection URL:
  - `?tools=tool1,tool2`
  - `?tags=tag1`

When query strings or parameters are supplied, an LLM-free **Knowledge Graph resolution layer** (using `DynamicToolOrchestrator`) matches query intents against known tool tags, names, or descriptions, with safe fallback and automated 24-hour background cache refreshing.

---

### MCP Configuration Examples

<!-- MCP-CONFIG-EXAMPLES:START -->

> **Install the slim `[mcp]` extra.** All examples install `container-manager-mcp[mcp]` — the
> MCP-server extra that pulls only the FastMCP / FastAPI tooling (`agent-utilities[mcp]`).
> It deliberately **excludes** the heavy agent runtime (`pydantic-ai`, the epistemic-graph
> engine, `dspy`, `llama-index`), so `uvx` / container installs are far smaller. Use the
> full `[agent]` extra only when you need the integrated Pydantic AI agent.

#### stdio Transport (local IDEs — Cursor, Claude Desktop, VS Code)

```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "container-manager-mcp[mcp]",
        "container-manager-mcp"
      ],
      "env": {
        "MCP_TOOL_MODE": "condensed",
        "COMPOSETOOL": "True",
        "CONTAINERTOOL": "True",
        "CONTAINER_MANAGER_HOST": "",
        "CONTAINER_MANAGER_KUBECONTEXT": "",
        "CONTAINER_MANAGER_PODMAN_BASE_URL": "",
        "CONTAINER_MANAGER_TYPE": "docker",
        "DEFAULT_DOCKER_CONTEXT": "",
        "DEFAULT_K8S_CONTEXT": "",
        "DEFAULT_SWARM_CONTEXT": "",
        "DOCKERSWARMTOOL": "True",
        "DOCKER_CONTEXTS": "",
        "DOCTORTOOL": "True",
        "HEALTH_CHECK_TTL_SECONDS": "30",
        "IMAGETOOL": "True",
        "INFOTOOL": "True",
        "INVENTORYTOOL": "True",
        "K8SCLUSTERTOOL": "True",
        "K8SCONFIGTOOL": "True",
        "K8SGOVERNANCETOOL": "True",
        "K8SNETWORKINGTOOL": "True",
        "K8SOBSERVABILITYTOOL": "True",
        "K8SRBACTOOL": "True",
        "K8SSTORAGETOOL": "True",
        "K8SWORKLOADSTOOL": "True",
        "K8S_CONTEXTS": "",
        "KUBECONFIG": "",
        "KUBERNETES_SERVICE_HOST": "",
        "MISCTOOL": "True",
        "MULTICONTEXTTOOL": "True",
        "MULTI_CONTEXT_MAX_WORKERS": "",
        "MULTI_CONTEXT_MODE": "True",
        "NETWORKTOOL": "True",
        "PODMANTOOL": "True",
        "PODMAN_ENABLED": "true",
        "SPECIALIST_DEPLOYMENTTOOL": "True",
        "SWARMTOOL": "True",
        "SWARM_CONTEXTS": "",
        "SYSTEMTOOL": "True",
        "VOLUMETOOL": "True"
      }
    }
  }
}
```

#### Streamable-HTTP Transport (networked / production)

```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "container-manager-mcp[mcp]",
        "container-manager-mcp",
        "--transport",
        "streamable-http",
        "--port",
        "8000"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "MCP_TOOL_MODE": "condensed",
        "COMPOSETOOL": "True",
        "CONTAINERTOOL": "True",
        "CONTAINER_MANAGER_HOST": "",
        "CONTAINER_MANAGER_KUBECONTEXT": "",
        "CONTAINER_MANAGER_PODMAN_BASE_URL": "",
        "CONTAINER_MANAGER_TYPE": "docker",
        "DEFAULT_DOCKER_CONTEXT": "",
        "DEFAULT_K8S_CONTEXT": "",
        "DEFAULT_SWARM_CONTEXT": "",
        "DOCKERSWARMTOOL": "True",
        "DOCKER_CONTEXTS": "",
        "DOCTORTOOL": "True",
        "HEALTH_CHECK_TTL_SECONDS": "30",
        "IMAGETOOL": "True",
        "INFOTOOL": "True",
        "INVENTORYTOOL": "True",
        "K8SCLUSTERTOOL": "True",
        "K8SCONFIGTOOL": "True",
        "K8SGOVERNANCETOOL": "True",
        "K8SNETWORKINGTOOL": "True",
        "K8SOBSERVABILITYTOOL": "True",
        "K8SRBACTOOL": "True",
        "K8SSTORAGETOOL": "True",
        "K8SWORKLOADSTOOL": "True",
        "K8S_CONTEXTS": "",
        "KUBECONFIG": "",
        "KUBERNETES_SERVICE_HOST": "",
        "MISCTOOL": "True",
        "MULTICONTEXTTOOL": "True",
        "MULTI_CONTEXT_MAX_WORKERS": "",
        "MULTI_CONTEXT_MODE": "True",
        "NETWORKTOOL": "True",
        "PODMANTOOL": "True",
        "PODMAN_ENABLED": "true",
        "SPECIALIST_DEPLOYMENTTOOL": "True",
        "SWARMTOOL": "True",
        "SWARM_CONTEXTS": "",
        "SYSTEMTOOL": "True",
        "VOLUMETOOL": "True"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed Streamable-HTTP instance by `url`:

```json
{
  "mcpServers": {
    "container-manager-mcp": {
      "url": "http://localhost:8000/container-manager-mcp/mcp"
    }
  }
}
```

Deploying the Streamable-HTTP server via Docker:

```bash
docker run -d \
  --name container-manager-mcp-mcp \
  -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -e MCP_TOOL_MODE=condensed \
  -e COMPOSETOOL=True \
  -e CONTAINERTOOL=True \
  -e CONTAINER_MANAGER_HOST="" \
  -e CONTAINER_MANAGER_KUBECONTEXT="" \
  -e CONTAINER_MANAGER_PODMAN_BASE_URL="" \
  -e CONTAINER_MANAGER_TYPE=docker \
  -e DEFAULT_DOCKER_CONTEXT="" \
  -e DEFAULT_K8S_CONTEXT="" \
  -e DEFAULT_SWARM_CONTEXT="" \
  -e DOCKERSWARMTOOL=True \
  -e DOCKER_CONTEXTS="" \
  -e DOCTORTOOL=True \
  -e HEALTH_CHECK_TTL_SECONDS=30 \
  -e IMAGETOOL=True \
  -e INFOTOOL=True \
  -e INVENTORYTOOL=True \
  -e K8SCLUSTERTOOL=True \
  -e K8SCONFIGTOOL=True \
  -e K8SGOVERNANCETOOL=True \
  -e K8SNETWORKINGTOOL=True \
  -e K8SOBSERVABILITYTOOL=True \
  -e K8SRBACTOOL=True \
  -e K8SSTORAGETOOL=True \
  -e K8SWORKLOADSTOOL=True \
  -e K8S_CONTEXTS="" \
  -e KUBECONFIG="" \
  -e KUBERNETES_SERVICE_HOST="" \
  -e MISCTOOL=True \
  -e MULTICONTEXTTOOL=True \
  -e MULTI_CONTEXT_MAX_WORKERS="" \
  -e MULTI_CONTEXT_MODE=True \
  -e NETWORKTOOL=True \
  -e PODMANTOOL=True \
  -e PODMAN_ENABLED=true \
  -e SPECIALIST_DEPLOYMENTTOOL=True \
  -e SWARMTOOL=True \
  -e SWARM_CONTEXTS="" \
  -e SYSTEMTOOL=True \
  -e VOLUMETOOL=True \
  knucklessg1/container-manager-mcp:mcp
```

_Auto-generated from the code-read env surface (`MCP_TOOL_MODE` + package vars) — do not edit._
<!-- MCP-CONFIG-EXAMPLES:END -->

<!-- BEGIN GENERATED: additional-deployment-options -->
### Additional Deployment Options

`container-manager-mcp` can also run as a **local container** (Docker / Podman / `uv`) or be
consumed from a **remote deployment**. The
[Deployment guide](https://knuckles-team.github.io/container-manager-mcp/deployment/) has full, copy-paste
`mcp_config.json` for all four transports — **stdio**, **streamable-http**,
**local container / uv**, and **remote URL**:

- **Local container / uv** — launch the server from `mcp_config.json` via `uvx`,
  `docker run`, or `podman run`, or point at a local streamable-http container by `url`.
- **Remote URL** — connect to a server deployed behind Caddy at
  `http://container-manager-mcp.arpa/mcp` using the `"url"` key.
<!-- END GENERATED: additional-deployment-options -->

---

## Environment Variables

<!-- ENV-VARS-TABLE:START -->

#### Package environment variables

| Variable | Example | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` |  |
| `PORT` | `8000` |  |
| `TRANSPORT` | `stdio` | options: stdio, streamable-http, sse |
| `ENABLE_OTEL` | `True` |  |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:8080/api/public/otel` |  |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` | `pk-...` |  |
| `OTEL_EXPORTER_OTLP_SECRET_KEY` | `sk-...` |  |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` |  |
| `EUNOMIA_TYPE` | `none` | options: none, embedded, remote |
| `EUNOMIA_POLICY_FILE` | `mcp_policies.json` |  |
| `EUNOMIA_REMOTE_URL` | `http://eunomia-server:8000` |  |
| `CONTAINER_MANAGER_TYPE` | `docker` | options: docker, podman, swarm, kubernetes |
| `CONTAINER_MANAGER_HOST` | — | remote docker daemon host (e.g. tcp://host:2375); empty = local |
| `CONTAINER_MANAGER_PODMAN_BASE_URL` | — | podman service base URL (e.g. unix:///run/podman/podman.sock) |
| `CONTAINER_MANAGER_KUBECONTEXT` | — | kubeconfig context name; empty = current-context |
| `KUBECONFIG` | — | path(s) to kubeconfig file(s); empty = ~/.kube/config |
| `KUBERNETES_SERVICE_HOST` | — | injected by the cluster when running in-pod; leave empty |
| `INVENTORYTOOL` | `True` |  |
| `INFOTOOL` | `True` |  |
| `IMAGETOOL` | `True` |  |
| `CONTAINERTOOL` | `True` |  |
| `VOLUMETOOL` | `True` |  |
| `NETWORKTOOL` | `True` |  |
| `SWARMTOOL` | `True` |  |
| `SYSTEMTOOL` | `True` |  |
| `COMPOSETOOL` | `True` |  |
| `MISCTOOL` | `True` |  |
| `DOCTORTOOL` | `True` |  |
| `SPECIALIST_DEPLOYMENTTOOL` | `True` |  |
| `K8SWORKLOADSTOOL` | `True` |  |
| `K8SCONFIGTOOL` | `True` |  |
| `K8SNETWORKINGTOOL` | `True` |  |
| `K8SSTORAGETOOL` | `True` |  |
| `K8SRBACTOOL` | `True` |  |
| `K8SCLUSTERTOOL` | `True` |  |
| `K8SGOVERNANCETOOL` | `True` |  |
| `K8SOBSERVABILITYTOOL` | `True` |  |
| `PODMANTOOL` | `True` |  |
| `DOCKERSWARMTOOL` | `True` |  |
| `MULTICONTEXTTOOL` | `True` |  |
| `MULTI_CONTEXT_MODE` | `True` | Multi-Context Configuration |
| `HEALTH_CHECK_TTL_SECONDS` | `30` | Multi-context health-check cache TTL (seconds) and parallel worker cap |
| `MULTI_CONTEXT_MAX_WORKERS` | — |  |
| `K8S_CONTEXTS` | — | Example: "dev=dev-cluster;prod=prod-cluster;staging=staging-cluster" |
| `DEFAULT_K8S_CONTEXT` | — | Default Kubernetes context name (must match a key in K8S_CONTEXTS) |
| `DOCKER_CONTEXTS` | — | Example: "local=unix:///var/run/docker.sock;remote=tcp://192.168.1.100:2375" |
| `DEFAULT_DOCKER_CONTEXT` | — | Default Docker context name (must match a key in DOCKER_CONTEXTS) |
| `SWARM_CONTEXTS` | — | Example: "swarm1=tcp://swarm1:2375;swarm2=tcp://swarm2:2375" |
| `DEFAULT_SWARM_CONTEXT` | — | Default Swarm context name (must match a key in SWARM_CONTEXTS) |
| `PODMAN_ENABLED` | `true` | Enable Podman (local only) |

#### Inherited agent-utilities variables (apply to every connector)

| Variable | Example | Description |
|----------|---------|-------------|
| `MCP_TOOL_MODE` | `condensed` | Tool surface: `condensed` | `verbose` | `both` |
| `MCP_ENABLED_TOOLS` | — | Comma-separated tool allow-list |
| `MCP_DISABLED_TOOLS` | — | Comma-separated tool deny-list |
| `MCP_ENABLED_TAGS` | — | Comma-separated tag allow-list |
| `MCP_DISABLED_TAGS` | — | Comma-separated tag deny-list |
| `MCP_CLIENT_AUTH` | — | Outbound MCP auth (`oidc-client-credentials` for fleet calls) |
| `OIDC_CLIENT_ID` | — | OIDC client id (service-account auth) |
| `OIDC_CLIENT_SECRET` | — | OIDC client secret (service-account auth) |
| `DEBUG` | `False` | Verbose logging |
| `PYTHONUNBUFFERED` | `1` | Unbuffered stdout (recommended in containers) |
| `MCP_URL` | `http://localhost:8000/mcp` | URL of the MCP server the agent connects to |
| `PROVIDER` | `openai` | LLM provider for the agent |
| `MODEL_ID` | `gpt-4o` | Model id for the agent |
| `ENABLE_WEB_UI` | `True` | Serve the AG-UI web interface |

_50 package + 14 inherited variable(s). Auto-generated from `.env.example` + the shared agent-utilities set — do not edit._
<!-- ENV-VARS-TABLE:END -->


Every variable is listed in the auto-generated table above (package vars from
`.env.example` + the inherited agent-utilities surface). A few pointers:

- **Tool toggles** — each action-routed tool can be disabled via its `<DOMAIN>TOOL`
  toggle; the tool ↔ toggle mapping is in the [Available MCP Tools](#available-mcp-tools)
  table above.
- **Multi-host control plane** — remote host endpoints load from the XDG shared inventory
  `~/.config/agent-utilities/inventory.yml` (`.yml` preferred, `.yaml` legacy fallback),
  managed via `tunnel-manager inventory init|doctor` (see [Multi-Host guide](docs/multi_host.md)).

See [`.env.example`](.env.example) for a copy-paste starting point.

## Agent

This repository features a fully integrated Pydantic AI Graph Agent. It communicates over the **Agent Control Protocol (ACP)** and interacts seamlessly with the **Agent Web UI (AG-UI)** and Terminal interface.

### Running the Agent CLI
To start the interactive command-line agent:

```bash
# Run the agent server
container-manager-agent --provider openai --model-id gpt-4o
```

### Docker Compose Orchestration
The following `docker/agent.compose.yml` configures the Agent, Web UI, and Terminal Interface together:

```yaml
version: '3.8'

services:
  container-manager-mcp-mcp:
    image: knucklessg1/container-manager-mcp:mcp
    container_name: container-manager-mcp-mcp
    hostname: container-manager-mcp-mcp
    restart: always
    env_file:
      - ../.env
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  container-manager-mcp-agent:
    image: knucklessg1/container-manager-mcp:latest
    container_name: container-manager-mcp-agent
    hostname: container-manager-mcp-agent
    restart: always
    depends_on:
      - container-manager-mcp-mcp
    env_file:
      - ../.env
    command: [ "container-manager-agent" ]
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=9019
      - MCP_URL=http://container-manager-mcp-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
      - ENABLE_OTEL=True
    ports:
      - "9019:9019"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9019/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

```

Detailed graph node architecture explanations, custom skill configurations, and agentic trace guides are available in [docs/agent.md](docs/agent.md).

---

## Security & Governance

Built directly upon the enterprise-ready [`agent-utilities`](https://github.com/Knuckles-Team/agent-utilities) core, standard security parameters are fully supported:

### Access Control & Policy Enforcement
- **Eunomia Policies:** Fine-grained, policy-driven tool authorization. Supports `none`, local `embedded` (`mcp_policies.json`), or centralized `remote` modes.
- **OIDC Token Delegation:** Compliant with RFC 8693 token exchange for flowing authenticating user credentials from Web UI / ACP → Agent → MCP.
- **Scoped Credentials:** Execution context runs restricted to the specific caller identity.

### Runtime Security Grid
| Feature | Functionality | Enablement |
|---------|---------------|------------|
| **Tool Guard** | Sensitivity inspection with human-in-the-loop validation | Enabled by default |
| **Prompt Injection Defense** | Input scanning, repetition monitoring, and recursive loop blocks | Enabled by default |
| **Context Safety Guard** | Stuck-loop detectors and contextual overflow preemptive alerts | Enabled by default |

---

## Installation

Pick the extra that matches what you want to run:

| Extra | Installs | Use when |
|-------|----------|----------|
| `container-manager-mcp[mcp]` | MCP server + the Docker/Podman/Kubernetes client libraries bundled by default (`agent-utilities[mcp]` — FastMCP/FastAPI; `docker` + `podman` + `kubernetes`) | You run the **MCP server** with full Docker/Podman/Kubernetes support (no separate extras needed) |
| `container-manager-mcp[agent]` | Full agent runtime (`agent-utilities[agent,logfire]` — Pydantic AI + the epistemic-graph engine) | You run the **integrated agent** |
| `container-manager-mcp[all]` | Everything (`mcp` + `agent` + the `docker` / `podman` / `kubernetes` backends) | Development / both surfaces |

```bash
# MCP server only (recommended for tool hosting — slim deps)
uv pip install "container-manager-mcp[mcp]"

# Full agent runtime (Pydantic AI + epistemic-graph engine)
uv pip install "container-manager-mcp[agent]"

# Everything (development)
uv pip install "container-manager-mcp[all]"      # or: python -m pip install "container-manager-mcp[all]"
```

### Container images (`:mcp` vs `:agent`)

One multi-stage `docker/Dockerfile` builds two right-sized images, selected by `--target`:

| Image tag | Build target | Contents | Entrypoint |
|-----------|--------------|----------|------------|
| `knucklessg1/container-manager-mcp:mcp` | `--target mcp` | `container-manager-mcp[mcp]` — **slim**, no engine/`pydantic-ai`/`dspy`/`llama-index`/`tree-sitter` | `container-manager-mcp` |
| `knucklessg1/container-manager-mcp:latest` | `--target agent` (default) | `container-manager-mcp[agent]` — **full** agent runtime + epistemic-graph engine | `container-manager-agent` |

```bash
docker build --target mcp   -t knucklessg1/container-manager-mcp:mcp    docker/   # slim MCP server
docker build --target agent -t knucklessg1/container-manager-mcp:latest docker/   # full agent
```

`docker/mcp.compose.yml` runs the slim `:mcp` server; `docker/agent.compose.yml` runs the
agent (`:latest`) with a co-located `:mcp` sidecar.

### Knowledge-graph database (`epistemic-graph`)

The **full agent** (`[agent]` / `:latest`) embeds the **epistemic-graph** engine (pulled in
transitively via `agent-utilities[agent]`). For production — or to share one knowledge graph
across multiple agents — run **epistemic-graph as its own database container** and point the
agent at it instead of embedding it. Deployment recipes (single-node + Raft HA), connection
config, and the full database architecture (with diagrams) are documented in the
[epistemic-graph deployment guide](https://knuckles-team.github.io/epistemic-graph/deployment/).
The slim `[mcp]` server does **not** require the database.

---

## Documentation

The complete documentation is published as the
[official documentation site](https://knuckles-team.github.io/container-manager-mcp/)
and is the recommended reference for installation, deployment, and day-to-day
operation.

| Page | Contents |
|---|---|
| [Installation](https://knuckles-team.github.io/container-manager-mcp/installation/) | pip, source, extras, prebuilt Docker image |
| [Deployment](https://knuckles-team.github.io/container-manager-mcp/deployment/) | run the MCP and agent servers, Compose, Caddy + Technitium, env config |
| [Usage](https://knuckles-team.github.io/container-manager-mcp/usage/) | the MCP tools, the `DockerManager` API, the CLI |
| [Overview](https://knuckles-team.github.io/container-manager-mcp/overview/) | ecosystem role, enterprise readiness, architecture |
| [Multi-Host](https://knuckles-team.github.io/container-manager-mcp/multi_host/) | zero-script Docker-over-SSH control plane |
| [Kubernetes](https://knuckles-team.github.io/container-manager-mcp/kubernetes/) | the 8 `cm_k8s_*` tools, `patch_resource`, true core Services, multi-context |
| [Concepts](https://knuckles-team.github.io/container-manager-mcp/concepts/) | concept registry (`CONCEPT:CMGR-*`) |

`AGENTS.md` is the canonical contributor/agent guidance.

### Skills

Alongside the MCP tools, `container_manager_mcp/skills/` ships Universal Skills that guide an agent through
common operational flows:

| Skill | Covers |
|-------|--------|
| `container-manager-config-walkthrough` | Onboarding — choosing `CONTAINER_MANAGER_TYPE`, wiring `.env`/`mcp_config` toggles, remote Docker/Podman hosts (inventory) vs. remote Kubernetes clusters (kubeconfig contexts), first-run verification |
| `container-manager-lifecycle` | Docker/Podman container and image lifecycle on local or remote hosts (list/inspect/stop/remove/exec, logs, port tracing, image pull/prune, Compose) |
| `container-manager-swarm` | Docker Swarm cluster orchestration (init/leave, nodes, services) |
| `container-manager-podman-operations` | Rootless Podman pod/kube operations — pods, `generate`/`play kube`, checkpoint/restore, pod-scoped networks/volumes, health, system prune |
| `container-manager-kubernetes-operations` | The full Kubernetes operational surface — workloads, config, networking, storage, RBAC, cluster, governance, observability |
| `container-manager-multi-context` | Operating several Docker/Podman/Swarm/Kubernetes backends and contexts at once via `cm_multi_context` |
| `container-manager-kg-ingestion` | Snapshotting Docker/Podman/Swarm **and Kubernetes** inventory into the epistemic-graph Knowledge Graph via `cm_ingest_inventory` |

`container-manager-kubernetes-operations`, `container-manager-podman-operations`,
`container-manager-multi-context`, and `container-manager-config-walkthrough` are new; `lifecycle`, `swarm`,
and `kg-ingestion` were updated for the expanded tool surface.

---

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)

---

## Contribute

Contributions are welcome! Please ensure code quality by executing local checks before submitting pull requests:
- Format code using `ruff format .`
- Lint code using `ruff check .`
- Validate type-safety with `mypy .`
- Execute test suites using `pytest`


<!-- BEGIN agent-os-genesis-deploy (generated; do not edit between markers) -->

## Deploy with `agent-os-genesis`

This package can be provisioned for you — skill-guided — by the **`agent-os-genesis`**
universal skill (its *single-package deploy mode*): it picks your install method, seeds
secrets to OpenBao/Vault (or `.env`), trusts your enterprise CA, registers the MCP
server, and verifies it — the same machinery that stands up the whole Agent OS, narrowed
to just this package. Ask your agent to **"deploy `container-manager-mcp` with agent-os-genesis"**.

| Install mode | Command |
|------|---------|
| Bare-metal, prod (PyPI) | `uvx container-manager-mcp` · or `uv tool install container-manager-mcp` |
| Bare-metal, dev (editable) | `uv pip install -e ".[all]"` · or `pip install -e ".[all]"` |
| Container, prod | deploy `knucklessg1/container-manager-mcp:latest` via docker-compose / swarm / podman / podman-compose / kubernetes |
| Container, dev (editable) | deploy `docker/compose.dev.yml` (source-mounted at `/src`; edits live on restart) |

Secrets are read-existing + seeded via `vault_sync` — you are only prompted for what's missing.

<!-- END agent-os-genesis-deploy -->

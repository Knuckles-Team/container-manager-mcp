# Kubernetes

`container-manager-mcp` provides **full Kubernetes coverage** (RKE2 / k3s / vanilla) built on the
**official `kubernetes` Python client**, exposed as 8 themed, action-routed MCP tools. This replaces an
earlier, messier tool sprawl with one consolidated module per resource domain — consistent with the
condensed, action-routed pattern used for Docker/Podman (`cm_container_operations`,
`cm_image_operations`, etc.).

## The 8 tool modules

| Tool | Toggle | Covers |
|------|--------|--------|
| `cm_k8s_workloads` | `K8SWORKLOADSTOOL` | Pods, Deployments, StatefulSets, DaemonSets, ReplicaSets, Jobs, CronJobs; rollout status/history/restart/undo/pause/resume; deployment/daemonset/statefulset update strategies; `exec`/`logs`/`attach`/`copy` into pods |
| `cm_k8s_config` | `K8SCONFIGTOOL` | ConfigMaps, Secrets, Namespaces, Events, CRDs and custom resources, `label_resource`/`annotate_resource`, generic `patch_resource`, and config/secret state tracking (`compare_configmap_state`, `get_secret_state_hash`, `track_resource_version`, `wait_for_resource_version`) |
| `cm_k8s_networking` | `K8SNETWORKINGTOOL` | Ingress, IngressClasses, NetworkPolicies (incl. CIDR helpers and connectivity tests), Endpoints/EndpointSlices, DNS resolution/connectivity checks, and true core Kubernetes Services |
| `cm_k8s_storage` | `K8SSTORAGETOOL` | PersistentVolumes, PersistentVolumeClaims (incl. `expand_pvc`), StorageClasses, VolumeSnapshots, CSI driver inspection |
| `cm_k8s_rbac` | `K8SRBACTOOL` | Roles, ClusterRoles, RoleBindings, ClusterRoleBindings, ServiceAccounts and tokens, `auth_can_i`, `subject_access_review`, pod-security evaluation, service-account-to-secret mapping |
| `cm_k8s_cluster` | `K8SCLUSTERTOOL` | Nodes (`cordon_node`/`uncordon_node`/`drain_node`/`taint_node`/affinity), kubeconfig contexts (`list_contexts`/`use_context`/`validate_kubeconfig`), CertificateSigningRequests, API resources, cluster-info, admission plugins |
| `cm_k8s_governance` | `K8SGOVERNANCETOOL` | ResourceQuotas, LimitRanges, PriorityClasses, PodDisruptionBudgets, HorizontalPodAutoscalers — full CRUD |
| `cm_k8s_observability` | `K8SOBSERVABILITYTOOL` | `top_pods`/`top_nodes`, pod/node/cluster metrics, autoscaler metrics and history, `watch_resource`, `stream_pod_logs`, `get_resource_events`, and `debug_pod`/`debug_node`/`debug_service`/`debug_deployment` helpers |

Every action is routed through a single `action` parameter per tool (the same condensed pattern as the
Docker/Podman tools), keeping the exposed tool count small regardless of how many operations a domain covers.

## `patch_resource`: the generic escalation hatch

`cm_k8s_config`'s `patch_resource` action applies a `strategic`, `merge`, or `json` patch to **any** resource
kind and name. Use it whenever a domain tool doesn't have a dedicated action for the field you need to change
(for example, an ad-hoc field on a CRD instance, or a spec field not covered by a themed helper action).

## True core Services vs. Swarm-parity services

`cm_k8s_networking`'s `list_k8s_services` / `get_k8s_service` / `create_k8s_service` / `delete_k8s_service`
operate on **real Kubernetes `Service` objects**. This is intentionally distinct from the Swarm-parity
`list_services` exposed by `cm_multi_context` (and `cm_docker_advanced`/`cm_swarm_operations`), which is
**Deployment-shaped** so it can be compared uniformly across Docker Swarm and Kubernetes backends in a single
multi-context call. If you need the literal `Service` resource (`ClusterIP`/`NodePort`/`LoadBalancer`,
selectors, ports), use the `cm_k8s_networking` actions; if you need a cross-backend "what's running" view,
use `cm_multi_context`.

## Configuration

| Variable | Purpose |
|----------|---------|
| `CONTAINER_MANAGER_TYPE=kubernetes` | Route `cm_k8s_*` calls at a Kubernetes cluster |
| `CONTAINER_MANAGER_KUBECONTEXT` | kubeconfig context name; empty = current-context |
| `K8S_CONTEXTS` / `DEFAULT_K8S_CONTEXT` | Multi-cluster context map for `cm_multi_context` |

See [Multi-Host → Kubernetes Kubeconfig Contexts](multi_host.md#kubernetes-kubeconfig-contexts) for how this
differs from the Docker/Podman remote-host inventory model, and [Usage → Kubernetes](usage.md#kubernetes) for
worked examples of each tool.

## Multi-context operation

`cm_multi_context` (toggle `MULTICONTEXTTOOL`) lets one call fan out across several Docker, Podman, Swarm,
and/or Kubernetes contexts in parallel (`ThreadPoolExecutor`-backed), with per-backend health checks and lazy
reconnect. Configure the pool via `K8S_CONTEXTS`, `DOCKER_CONTEXTS`, `SWARM_CONTEXTS`, and their
`DEFAULT_*_CONTEXT` values, or set `MULTI_CONTEXT_MODE=True` to route every call through it. See
[Usage → Multi-context](usage.md#multi-context) for examples.

## Knowledge Graph ingestion

`cm_ingest_inventory` supports Kubernetes modalities `pods`, `deployments`, `namespaces`, and `k8s_services`
when the active manager is Kubernetes (`manager_type=kubernetes` or `CONTAINER_MANAGER_TYPE=kubernetes`),
mapping them to typed `:Pod` / `:Deployment` / `:Namespace` / `:K8sService` nodes in the ontology
(`container_manager_mcp/ontology/container.ttl`). See [Concepts](concepts.md#kubernetes-ontology-classes) and
[Concepts → KG ingestion modalities](concepts.md#knowledge-graph-ingestion-modalities) for the full class and
modality reference.

## Skill

The [`container-manager-kubernetes-operations`](https://github.com/Knuckles-Team/container-manager-mcp/tree/main/container_manager_mcp/skills/container-manager-kubernetes-operations)
Universal Skill walks an agent through driving real cluster operations across all 8 tool modules. For
onboarding a new cluster/context, start with
[`container-manager-config-walkthrough`](https://github.com/Knuckles-Team/container-manager-mcp/tree/main/container_manager_mcp/skills/container-manager-config-walkthrough);
for parallel multi-cluster work, see
[`container-manager-multi-context`](https://github.com/Knuckles-Team/container-manager-mcp/tree/main/container_manager_mcp/skills/container-manager-multi-context).

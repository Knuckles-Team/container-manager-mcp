# Container Manager Kubernetes Operations

Full operational Kubernetes surface via the container-manager-mcp MCP server — workloads (pods/rollouts/StatefulSets/DaemonSets/ReplicaSets/Jobs/CronJobs), config (ConfigMaps/Secrets/Namespaces/CRDs/patch), networking (Ingress/native Services/NetworkPolicy/DNS), storage (PV/PVC/StorageClass/snapshots/CSI), RBAC (roles/bindings/tokens/access reviews/pod security), cluster (nodes/contexts/ CSRs/admission plugins), governance (quotas/limits/PDB/HPA), and observability (metrics/watch/debug). Use when the agent must drive real k8s migrations, sustain, or ops work on a cluster. Do NOT use for Docker/Podman single-host lifecycle (container-manager-lifecycle), Docker Swarm (container-manager-swarm), or rootless Podman pods (container-manager-podman-operations).

# Kubernetes Operations

Domain-typed control of a **Kubernetes** cluster through the eight themed `cm_k8s_*`
tools on the **container-manager-mcp** server. These wrap the Kubernetes Python client
with typed records and drive real clusters end to end — deploys, rollouts, networking,
storage, RBAC, node maintenance, governance, and metrics — without ever shelling out to
`kubectl`. Prefer these `cm_*` tools over raw `kubectl`/`docker`/`podman` shell.

## When to use
- Operate workloads: list/describe/exec/port-forward pods, drive rollouts, manage
  StatefulSets/DaemonSets/ReplicaSets/Jobs/CronJobs.
- Manage cluster configuration: ConfigMaps, Secrets, Namespaces, CRDs, generic
  label/annotate/patch, and config/secret drift tracking.
- Manage networking: Ingress + IngressClasses, NetworkPolicy (incl. CIDR rules and
  connectivity tests), DNS debugging, and **true core/v1 Services**.
- Manage storage: PersistentVolumes/Claims (incl. expansion), StorageClasses,
  VolumeSnapshots, CSI drivers.
- Audit and manage RBAC: roles/bindings/service accounts, `auth_can_i`,
  SubjectAccessReviews, ServiceAccount tokens, pod security evaluation.
- Manage the cluster itself: node cordon/drain/taint/affinity, kubeconfig contexts,
  CSR approval, API resource discovery, cluster info, admission plugins.
- Enforce governance: ResourceQuotas, LimitRanges, PriorityClasses,
  PodDisruptionBudgets, HorizontalPodAutoscalers.
- Observe: `top` pods/nodes, pod/node/cluster resource summaries, autoscaler metrics
  and history, watch/stream/events, and the `debug_*` helpers.

## When NOT to use
- Single-host Docker/Podman container/image lifecycle → `container-manager-lifecycle`.
- Docker Swarm cluster services/nodes → `container-manager-swarm`.
- Rootless Podman pods, `generate/play kube`, checkpoint/restore →
  `container-manager-podman-operations`.
- Operating several clusters/backends in one call, or fanning an action out across
  contexts → `container-manager-multi-context`.
- First-time environment setup / choosing `CONTAINER_MANAGER_TYPE` →
  `container-manager-config-walkthrough`.
- Snapshotting inventory into the KG → `container-manager-kg-ingestion`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`container-manager-mcp`** MCP server.
A reachable Kubernetes cluster and valid kubeconfig are required.

| Variable | Required | Notes |
|----------|----------|-------|
| `CONTAINER_MANAGER_TYPE` | yes | Set to `kubernetes` to target a cluster |
| `CONTAINER_MANAGER_KUBECONTEXT` | optional | kubeconfig context to use; else the kubeconfig's current-context |
| `K8S_CONTEXTS` / `DEFAULT_K8S_CONTEXT` | optional | Multi-context mode only — see `container-manager-multi-context` |
| `K8SWORKLOADSTOOL` / `K8SCONFIGTOOL` / `K8SNETWORKINGTOOL` / `K8SSTORAGETOOL` / `K8SRBACTOOL` / `K8SCLUSTERTOOL` / `K8SGOVERNANCETOOL` / `K8SOBSERVABILITYTOOL` | optional | Per-theme tool toggles, default `true`; set to `false` to hide a theme |

`MCP_TOOL_MODE` (`condensed`\|`verbose`\|`both`) selects the condensed `action`-based
tools used below. Most `namespace` parameters default to the manager's configured
namespace when omitted.

## Walkthrough — first run
1. **Select the backend and context.**
   ```
   # env: CONTAINER_MANAGER_TYPE=kubernetes
   cm_k8s_cluster action=list_contexts
   cm_k8s_cluster action=use_context context_name=<name>
   ```
2. **Verify connectivity.**
   ```
   cm_k8s_cluster action=get_cluster_info
   cm_k8s_cluster action=list_nodes
   ```
3. **Confirm RBAC before writing anything.**
   ```
   cm_k8s_rbac action=auth_can_i auth_verb=create auth_resource=deployments namespace=<ns>
   ```
4. Proceed to the themed recipes below.

## Tools & actions
| Condensed tool | Theme | Actions |
|----------------|-------|---------|
| `cm_k8s_workloads` | Pods, rollouts, strategies, StatefulSets, DaemonSets, ReplicaSets, Jobs, CronJobs | `list_pods`, `describe_pod`, `exec_pod`, `port_forward_pod`, `attach_pod`, `copy_to_pod`, `copy_from_pod`, `rollout_status`, `rollout_history`, `rollout_restart`, `rollout_undo`, `rollout_pause`, `rollout_resume`, `set_deployment_strategy`, `get_deployment_strategy`, `set_daemonset_update_strategy`, `get_daemonset_update_strategy`, `set_statefulset_update_strategy`, `get_statefulset_update_strategy`, `list_statefulsets`, `create_stateful_set`, `scale_statefulset`, `list_daemonsets`, `create_daemon_set`, `list_replicasets`, `describe_replicaset`, `scale_replicaset`, `list_jobs`, `describe_job`, `create_job`, `delete_job`, `list_cron_jobs`, `describe_cron_job`, `create_cron_job`, `delete_cron_job` |
| `cm_k8s_config` | ConfigMaps, Secrets, Namespaces, Events, CRDs, label/annotate/patch, state tracking | `list_configmaps`, `create_configmap`, `list_secrets`, `create_secret`, `list_namespaces`, `create_namespace`, `delete_namespace`, `list_events`, `list_crds`, `describe_crd`, `list_custom_resources`, `label_resource`, `annotate_resource`, `patch_resource`, `compare_configmap_state`, `sync_configmap_from_file`, `get_secret_state_hash`, `track_resource_version`, `wait_for_resource_version` |
| `cm_k8s_networking` | Ingress, IngressClasses, NetworkPolicy, endpoints, DNS, native Services | `list_ingress`, `create_ingress`, `delete_ingress`, `list_ingress_classes`, `describe_ingress_class`, `create_ingress_class`, `set_default_ingress_class`, `list_networkpolicies`, `create_networkpolicy`, `delete_networkpolicy`, `create_network_policy_with_cidr`, `update_network_policy_rules`, `test_network_policy_connectivity`, `list_endpoints`, `list_endpointslices`, `check_dns_resolution`, `list_dns_endpoints`, `test_dns_connectivity`, `list_k8s_services`, `get_k8s_service`, `create_k8s_service`, `delete_k8s_service` |
| `cm_k8s_storage` | PV/PVC, StorageClasses, VolumeSnapshots, CSI drivers | `list_persistent_volumes`, `create_persistent_volume`, `list_persistent_volume_claims`, `create_persistent_volume_claim`, `delete_persistent_volume_claim`, `expand_pvc`, `expand_persistent_volume`, `list_storage_classes`, `create_storage_class`, `set_default_storage_class`, `get_storage_class_provisioner`, `list_volume_snapshots`, `create_volume_snapshot`, `list_csi_drivers`, `describe_csi_driver`, `get_csi_driver_capacity` |
| `cm_k8s_rbac` | Roles/bindings/ServiceAccounts, tokens, access reviews, pod security | `list_roles`, `create_role`, `delete_role`, `list_cluster_roles`, `list_rolebindings`, `create_rolebinding`, `delete_rolebinding`, `list_cluster_rolebindings`, `create_cluster_rolebinding`, `delete_cluster_rolebinding`, `list_serviceaccounts`, `create_serviceaccount`, `delete_serviceaccount`, `auth_can_i`, `create_service_account_token`, `list_service_account_tokens`, `delete_service_account_token`, `subject_access_review`, `local_subject_access_review`, `create_aggregated_cluster_role`, `update_aggregated_cluster_role`, `list_pod_security_policies`, `describe_pod_security_policy`, `create_pod_security_policy`, `delete_pod_security_policy`, `evaluate_pod_security`, `list_service_account_mapped_secrets`, `map_secret_to_service_account`, `unmap_secret_from_service_account` |
| `cm_k8s_cluster` | Nodes, kubeconfig contexts, CSRs, API resources, cluster info, admission plugins | `list_nodes`, `inspect_node`, `cordon_node`, `uncordon_node`, `drain_node`, `get_node_conditions`, `taint_node`, `untaint_node`, `list_node_taints`, `set_node_affinity`, `get_node_affinity`, `set_pod_anti_affinity`, `list_contexts`, `use_context`, `get_config`, `rename_context`, `validate_kubeconfig`, `list_csr`, `approve_csr`, `deny_csr`, `list_api_resources`, `describe_api_resource`, `cluster_info_dump`, `get_cluster_info`, `get_api_server_info`, `list_cluster_plugins`, `describe_cluster_plugin`, `test_cluster_plugin` |
| `cm_k8s_governance` | ResourceQuotas, LimitRanges, PriorityClasses, PDBs, HPAs | full CRUD (`list_*`, `describe_*`, `create_*`, `update_*` where applicable, `delete_*`) for `resource_quota`, `limit_range`, `priority_class`, `pod_disruption_budget`, `horizontal_pod_autoscaler` |
| `cm_k8s_observability` | Metrics, autoscaler metrics/history, watch/stream/events, debug helpers | `top_pods`, `top_nodes`, `get_pod_metrics`, `get_node_metrics`, `get_pod_resource_usage`, `get_cluster_resource_summary`, `get_autoscaler_metrics`, `set_autoscaler_metrics`, `scale_deployment_autoscaler`, `get_autoscaler_history`, `watch_resource`, `stream_pod_logs`, `get_resource_events`, `list_field_selector`, `debug_pod`, `debug_node`, `debug_service`, `debug_deployment` |

### Key parameters
- `namespace` — target namespace across nearly every action; omit to use the
  manager's configured/default namespace.
- `name` / `pod_name` / `node_name` / `context_name` / etc. — the theme-specific
  resource identifier; each tool's action list above pins which name field it uses
  (e.g. `resource_type` + `resource_name` for rollouts, `name` for most CRUD).
- `spec` — a `dict` resource spec for most `create_*`/`update_*` calls
  (StatefulSets, Jobs, CronJobs, PV/PVC, RBAC objects, governance objects, HPAs).
- `configmap_data` / `secret_data` / `ingress_spec` / `netpol_spec` / `role_rules` /
  `role_ref` / `subjects` / `labels` / `annotations` / `patch_body` — **JSON strings**
  (not dicts) on `cm_k8s_config`, `cm_k8s_networking`, and `cm_k8s_rbac`; the server
  `json.loads()`s them.
- `patch_type` — `strategic` (default), `merge`, or `json` for `patch_resource`.
- `grace_period_seconds` — for `drain_node` (default `120`).

## Recipes — deploy + rollout (workloads)
Roll out a change and watch it land, then roll back if needed:
```
cm_k8s_workloads action=rollout_restart resource_type=deployment resource_name=web namespace=prod
cm_k8s_workloads action=rollout_status resource_type=deployment resource_name=web namespace=prod
cm_k8s_workloads action=rollout_undo resource_type=deployment resource_name=web namespace=prod rollout_revision=<n>
```
Create a StatefulSet and scale it:
```
cm_k8s_workloads action=create_stateful_set name=db namespace=prod spec={...}
cm_k8s_workloads action=scale_statefulset name=db namespace=prod replicas=3
```
Run an ad-hoc Job:
```
cm_k8s_workloads action=create_job name=migrate-2026-07 namespace=prod spec={...}
cm_k8s_workloads action=describe_job name=migrate-2026-07 namespace=prod
```

## Recipes — config
Create a Secret and a ConfigMap, then patch a Deployment's annotation:
```
cm_k8s_config action=create_secret secret_name=db-creds namespace=prod secret_type=Opaque secret_data='{"password":"<base64>"}'
cm_k8s_config action=create_configmap configmap_name=app-config namespace=prod configmap_data='{"LOG_LEVEL":"info"}'
cm_k8s_config action=annotate_resource resource_type=deployment resource_name=web namespace=prod annotations='{"kubernetes.io/change-cause":"2026-07-09 config bump"}'
cm_k8s_config action=patch_resource resource_type=deployment name=web namespace=prod patch_body='{"spec":{"template":{"metadata":{"labels":{"rev":"12"}}}}}' patch_type=strategic
```
Detect drift before a sync:
```
cm_k8s_config action=compare_configmap_state name=app-config namespace=prod expected_data={...}
cm_k8s_config action=sync_configmap_from_file name=app-config namespace=prod file_path=[REDACTED_POSIX_LOCAL_PATH]
```

## Recipes — networking
Ingress + a **true core/v1 Service** + NetworkPolicy:
```
cm_k8s_networking action=create_k8s_service name=web namespace=prod service_selector={"app":"web"} service_ports=[{"port":80,"targetPort":8080}] service_type=ClusterIP
cm_k8s_networking action=create_ingress ingress_name=web namespace=prod ingress_spec='{...}'
cm_k8s_networking action=create_network_policy_with_cidr name=deny-external namespace=prod spec={...}
cm_k8s_networking action=test_network_policy_connectivity namespace=prod name=deny-external
```
> **`list_k8s_services` vs the Deployment-shaped `list_services`:** `cm_k8s_networking
> action=list_k8s_services` returns **true core/v1 Services** (ClusterIP / NodePort /
> LoadBalancer) — distinct from `cm_swarm_operations action=list_services`
> (`container-manager-swarm`), which is Swarm's Deployment-shaped replicated-service
> model. Never conflate the two when the target cluster is Kubernetes.

## Recipes — storage
Provision and later expand a PVC:
```
cm_k8s_storage action=create_storage_class name=fast-ssd provisioner=csi.example.com/ssd
cm_k8s_storage action=create_persistent_volume_claim pvc_name=data namespace=prod pvc_spec='{"storageClassName":"fast-ssd","resources":{"requests":{"storage":"10Gi"}}}'
cm_k8s_storage action=expand_pvc pvc_name=data namespace=prod pvc_size=50Gi
cm_k8s_storage action=create_volume_snapshot name=data-snap-2026-07-09 namespace=prod spec={...}
```

## Recipes — RBAC audit
Audit before granting, then grant narrowly:
```
cm_k8s_rbac action=auth_can_i auth_verb=delete auth_resource=pods namespace=prod
cm_k8s_rbac action=subject_access_review spec={"resourceAttributes":{"namespace":"prod","verb":"delete","resource":"pods"},"user":"jane"}
cm_k8s_rbac action=create_role role_name=pod-reader namespace=prod role_rules='[{"apiGroups":[""],"resources":["pods"],"verbs":["get","list"]}]'
cm_k8s_rbac action=create_rolebinding rolebinding_name=jane-pod-reader namespace=prod role_ref='{"kind":"Role","name":"pod-reader"}' subjects='[{"kind":"User","name":"jane"}]'
cm_k8s_rbac action=create_service_account_token name=deploy-bot namespace=prod spec={"expirationSeconds":3600}
```

## Recipes — cluster maintenance
Safe node drain for a patch cycle:
```
cm_k8s_cluster action=cordon_node node_name=worker-3
cm_k8s_cluster action=drain_node node_name=worker-3 grace_period_seconds=300
# ... patch/reboot ...
cm_k8s_cluster action=uncordon_node node_name=worker-3
```
Approve a pending CSR:
```
cm_k8s_cluster action=list_csr
cm_k8s_cluster action=approve_csr csr_name=<name>
```

## Recipes — governance
Cap a namespace and protect availability during voluntary disruption:
```
cm_k8s_governance action=create_resource_quota name=prod-quota namespace=prod spec={"hard":{"requests.cpu":"20","requests.memory":"40Gi"}}
cm_k8s_governance action=create_pod_disruption_budget name=web-pdb namespace=prod spec={"minAvailable":2,"selector":{"matchLabels":{"app":"web"}}}
cm_k8s_governance action=create_horizontal_pod_autoscaler name=web-hpa namespace=prod spec={"minReplicas":2,"maxReplicas":10,"targetCPUUtilizationPercentage":70}
```

## Recipes — observability
Everyday health checks and a live debug:
```
cm_k8s_observability action=top_pods namespace=prod
cm_k8s_observability action=get_cluster_resource_summary
cm_k8s_observability action=debug_deployment name=web namespace=prod
cm_k8s_observability action=stream_pod_logs name=web-7d9f-abcde namespace=prod tail_lines=200
cm_k8s_observability action=watch_resource resource_type=deployment name=web namespace=prod
```

## Gotchas
- Every `cm_k8s_*` tool auto-selects the `kubernetes` manager; set
  `CONTAINER_MANAGER_TYPE=kubernetes` (or pass `manager_type=kubernetes`) or calls
  will target whatever the default backend resolves to.
- `cm_k8s_config`, `cm_k8s_networking`, and `cm_k8s_rbac` take several fields as
  **JSON strings** (`configmap_data`, `secret_data`, `ingress_spec`, `netpol_spec`,
  `role_rules`, `role_ref`, `subjects`, `labels`, `annotations`, `patch_body`) while
  most other themes take native `dict`/`list` `spec` — check the action's field
  before calling.
- `list_k8s_services` (native Services) and `cm_swarm_operations action=list_services`
  (Swarm replicated services) are **not the same shape** — see the networking note
  above.
- A per-theme toggle env var (e.g. `K8SRBACTOOL=false`) hides that entire tool; if a
  `cm_k8s_*` tool is missing from the server, check the toggle before assuming a bug.
- Destructive actions (`delete_namespace`, `delete_persistent_volume_claim`,
  `drain_node`, `delete_*` RBAC/governance objects) are irreversible or
  cluster-disruptive — confirm intent and prefer `cordon_node` before `drain_node`.
- `namespace` is frequently required even when the field-level validation doesn't
  say so at a glance (e.g. `evaluate_pod_security`, `local_subject_access_review`,
  `stream_pod_logs`) — pass it explicitly rather than relying on defaults.

## Related
- **`container-manager-config-walkthrough`** — first-time `CONTAINER_MANAGER_TYPE` /
  kubeconfig setup before using this skill.
- **`container-manager-multi-context`** — operate several k8s contexts (or mixed
  k8s/Docker/Podman/Swarm backends) at once.
- **`container-manager-lifecycle`** / **`container-manager-swarm`** — the
  Docker/Podman-native surfaces for non-k8s workloads.
- **`container-manager-kg-ingestion`** — push pod/deployment/namespace/service
  inventory into the knowledge graph.

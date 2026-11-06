---
name: container-manager-kubernetes-operations
skill_type: skill
description: >-
  Full operational Kubernetes surface via the container-manager-mcp MCP server —
  workloads (pods/rollouts/StatefulSets/DaemonSets/ReplicaSets/Jobs/CronJobs),
  config (ConfigMaps/Secrets/Namespaces/CRDs/patch), networking (Ingress/native
  Services/NetworkPolicy/DNS), storage (PV/PVC/StorageClass/snapshots/CSI), RBAC
  (roles/bindings/tokens/access reviews/pod security), cluster (nodes/contexts/
  CSRs/admission plugins), governance (quotas/limits/PDB/HPA), and observability
  (metrics/watch/debug). Use when the agent must drive real k8s migrations,
  sustain, or ops work on a cluster. Do NOT use for Docker/Podman single-host
  lifecycle (container-manager-lifecycle), Docker Swarm (container-manager-swarm),
  or rootless Podman pods (container-manager-podman-operations).
license: MIT
tags: [container-manager-mcp, kubernetes, k8s, cluster, ops, mcp]
metadata:
  author: Repository Maintainers
  version: '0.1.0'
---
# Kubernetes Operations

Whole-cluster Kubernetes management through the **container-manager-mcp** server, one
typed tool per operation, named `k8s_<operation>`. Reach it through arbora:
`find(intent='kubernetes ...')`, then `act(op='fleet.call', params={'server':
'container-manager-mcp', 'tool': 'k8s_list_pods', 'arguments': {...}})`. Never shell out to
`kubectl`.

## When to use
- Read the cluster: nodes, pods and events in every namespace, logs, any object of any
  kind (CRDs included).
- Change workloads: scale, patch, apply manifests, rollout restart/status/history/undo.
- Config, networking, storage, RBAC, governance and node maintenance.
- Secrets and ExternalSecrets (metadata by default; values only on request).

## When NOT to use
- Persisting a change to `services/<name>/`: `container-manager-gitops-persist`.
- Refreshing a Secret from the vault: `container-manager-externalsecret-sync`.
- The tested change-and-rollback recipe: `container-manager-k8s-change-validation`.
- Docker/Podman hosts: `container-manager-lifecycle`; Swarm: `container-manager-swarm`;
  several clusters at once: `container-manager-multi-context`.

## Access
The server runs in-cluster with a cluster-admin-equivalent ServiceAccount. A caller must
be entitled to the k8s context (`k8s:*` or `k8s:<context>`); arbora's service identity is
granted `k8s:*`. Every tool takes an optional `context` (default: the in-cluster context)
and most take `namespace` (default: the server's configured namespace).

## Tools by theme
| Theme | Tools |
|---|---|
| Read anything | `k8s_get_resource(api_version, kind, name, namespace, reveal)`, `k8s_list_resources(api_version, kind, namespace, label_selector, all_namespaces)`, `k8s_list_api_resources`, `k8s_list_crds`, `k8s_describe_crd`, `k8s_list_custom_resources` |
| Apply / delete | `k8s_apply_manifest(manifest_yaml, namespace, dry_run, force_conflicts)` (server-side apply), `k8s_delete_resource(api_version, kind, name, namespace, confirm)`, `k8s_patch_resource(resource_type, name, namespace, patch_body, patch_type)`, `k8s_label_resource`, `k8s_annotate_resource` |
| GitOps | `k8s_export_manifest` (clean YAML for services/), `k8s_diff_manifest` (drift: paths that would change) |
| Pods | `k8s_list_pods(namespace, label_selector, all_namespaces)`, `k8s_describe_pod`, `k8s_read_pod_logs(pod_name, namespace, container, tail_lines, previous, since_seconds)`, `k8s_exec_pod`, `k8s_port_forward_pod`, `k8s_copy_to_pod`, `k8s_copy_from_pod` |
| Deployments | `k8s_describe_deployment`, `k8s_scale_deployment`, `k8s_list_services` (deployment summaries, all namespaces) |
| Rollouts (deployment, statefulset, daemonset) | `k8s_rollout_status` (complete + message), `k8s_rollout_history` (revisions with images), `k8s_rollout_restart` (stamps the pod template), `k8s_rollout_undo(revision)` (previous by default), `k8s_rollout_pause`, `k8s_rollout_resume` |
| Other workloads | `k8s_list_statefulsets`, `k8s_scale_statefulset`, `k8s_list_daemonsets`, `k8s_list_replicasets`, `k8s_list_jobs`, `k8s_create_job`, `k8s_list_cron_jobs`, `k8s_create_cron_job` |
| Config | `k8s_list_configmaps`, `k8s_create_configmap`, `k8s_list_namespaces`, `k8s_create_namespace`, `k8s_delete_namespace`, `k8s_list_events(namespace, field_selector, all_namespaces, limit)` |
| Secrets | `k8s_list_secrets` (metadata), `k8s_get_secret(name, namespace, reveal=false)`, `k8s_list_external_secrets`, `k8s_force_sync_external_secret` |
| Networking | `k8s_list_native_services`, `k8s_create_native_service`, `k8s_list_ingresses`, `k8s_create_ingress`, `k8s_list_networkpolicies`, `k8s_create_networkpolicy`, `k8s_list_endpoints`, `k8s_check_dns_resolution` |
| Storage | `k8s_list_persistent_volumes`, `k8s_list_persistent_volume_claims`, `k8s_expand_pvc`, `k8s_list_storage_classes`, `k8s_list_volume_snapshots`, `k8s_create_volume_snapshot`, `k8s_list_csi_drivers` |
| RBAC | `k8s_auth_can_i`, `k8s_list_roles`, `k8s_create_role`, `k8s_list_cluster_roles`, `k8s_list_rolebindings`, `k8s_create_rolebinding`, `k8s_list_serviceaccounts`, `k8s_create_service_account_token`, `k8s_subject_access_review` |
| Nodes / cluster | `k8s_list_nodes`, `k8s_inspect_node`, `k8s_cordon_node`, `k8s_uncordon_node`, `k8s_drain_node(confirm)`, `k8s_taint_node`, `k8s_get_node_conditions`, `k8s_list_contexts`, `k8s_get_cluster_info`, `k8s_list_certificate_signing_requests`, `k8s_approve_csr` |
| Governance | `k8s_list_resource_quotas`, `k8s_create_resource_quota`, `k8s_list_limit_ranges`, `k8s_list_pod_disruption_budgets`, `k8s_list_horizontal_pod_autoscalers`, `k8s_create_horizontal_pod_autoscaler`, `k8s_list_priority_classes` |
| Observability | `k8s_top_pods`, `k8s_top_nodes`, `k8s_get_cluster_resource_summary`, `k8s_get_resource_events`, `k8s_watch_resource`, `k8s_debug_deployment`, `k8s_debug_pod` |

## Recipes
Health sweep:
```
k8s_list_nodes
k8s_list_pods all_namespaces=true
k8s_list_events all_namespaces=true limit=50
k8s_list_external_secrets all_namespaces=true
```
Troubleshoot a deployment:
```
k8s_describe_deployment name=web namespace=apps
k8s_list_pods namespace=apps label_selector=app=web
k8s_read_pod_logs pod_name=<pod> namespace=apps previous=true
```
Change and roll back:
```
k8s_patch_resource resource_type=deployment name=web namespace=apps patch_body={"spec":{"template":{"spec":{"containers":[{"name":"web","env":[{"name":"LOG_LEVEL","value":"debug"}]}]}}}}
k8s_rollout_status resource_type=deployment name=web namespace=apps
k8s_rollout_history resource_type=deployment name=web namespace=apps
k8s_rollout_undo resource_type=deployment name=web namespace=apps
```
Apply a manifest and check drift:
```
k8s_diff_manifest manifest_yaml=<yaml>
k8s_apply_manifest manifest_yaml=<yaml>
k8s_diff_manifest manifest_yaml=<yaml>      # drift=false everywhere
```

## Gotchas
- Secret bodies from `k8s_get_resource` are redacted (key names, `dataDigest`) and so is
  their last-applied annotation; values need `reveal=true` on an explicit request.
- `k8s_delete_resource`, `k8s_delete_namespace`, `k8s_drain_node` and the prunes need
  `confirm=true`.
- `k8s_list_services` returns Deployments (the cross-backend workload shape);
  `k8s_list_native_services` returns core/v1 Services.
- A rollout restart stamps `kubectl.kubernetes.io/restartedAt` on the pod template; it is
  bookkeeping and is stripped by `k8s_export_manifest`.
- Every live change must end in a `services/<name>/` commit
  (`container-manager-gitops-persist`).

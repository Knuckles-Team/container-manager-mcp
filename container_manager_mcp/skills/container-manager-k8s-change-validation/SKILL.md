---
name: container-manager-k8s-change-validation
skill_type: skill
description: >-
  The proven end-to-end Kubernetes workload change workflow via container-manager-mcp: create a test object, scale, patch, restart, check status and history, undo, apply a manifest, clean up. Use to validate the connector or as a safe-change template. Do NOT use for Docker/Swarm hosts or to persist the change.
license: MIT
tags: [container-manager-mcp, kubernetes, rollout, validation]
metadata:
  author: Repository Maintainers
  version: '0.1.0'
---
# Validate a Kubernetes change end to end

Scope: The proven end-to-end change workflow for a Kubernetes workload through container-manager-mcp: create a test object, scale, patch, rollout restart, status, history, undo, apply a manifest and clean up. Use to validate the connector against a cluster, or as the template for any safe workload change with rollback. Do NOT use for Docker/Swarm hosts or for persisting the change (container-manager-gitops-persist).

Use a disposable namespace `zz-validation` and objects named `zz-validation-<date>`.

## Steps

1. Read the cluster with `k8s_list_nodes` and verify every node is Ready before creating anything.
2. Create the test objects with `k8s_apply_manifest` for the `zz-validation` Namespace and a small Deployment, and verify `k8s_rollout_status(resource_type=deployment)` returns `complete=true`.
3. Scale with `k8s_scale_deployment(replicas=2)` and verify `k8s_rollout_status` completes at two replicas.
4. Patch the pod template with `k8s_patch_resource` and verify `k8s_rollout_history` lists a new revision.
5. Revert with `k8s_rollout_undo` and verify `k8s_describe_deployment` shows the restored template.
6. Clean up with `k8s_delete_resource(api_version=v1, kind=Namespace, name=zz-validation, confirm=true)` and verify `k8s_list_namespaces` no longer lists it.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| Delete refused | `confirm=true` was not passed | Pass `confirm=true` in the `k8s_delete_resource` call |
| Rollout never reports `complete=true` | The pod cannot start or schedule | Read `k8s_list_events(namespace=zz-validation)` and `k8s_read_pod_logs`, fix the manifest and re-apply |
| Apply reports a field conflict | Another manager owns a field | Re-apply with `force_conflicts=true` in `k8s_apply_manifest` after confirming the owner |
| Patch rejected | `patch_body` does not match the object schema | Rebuild `patch_body` from the `containers` name shown by `k8s_describe_deployment` |

## Validation sequence

1. Read: `k8s_list_nodes`, `k8s_list_pods(all_namespaces=true)`,
   `k8s_list_events(all_namespaces=true, limit=50)`.
2. Create: `k8s_apply_manifest` with a Namespace and a small Deployment (one container,
   requests set). Poll `k8s_rollout_status(resource_type=deployment)` until
   `complete=true`.
3. Describe and logs: `k8s_describe_deployment`, `k8s_list_pods(label_selector=app=...)`,
   `k8s_read_pod_logs`.
4. Scale: `k8s_scale_deployment(replicas=2)` then back to 1; check `k8s_rollout_status`.
5. Patch: `k8s_patch_resource(resource_type=deployment, patch_body={spec: {template:
   {spec: {containers: [{name, env: [...]}]}}}})` creates a new revision.
6. Restart: `k8s_rollout_restart` stamps the pod template; another revision.
7. History and undo: `k8s_rollout_history` lists revisions with images;
   `k8s_rollout_undo` (default previous revision) and `k8s_rollout_status` until
   complete; `k8s_describe_deployment` shows the restored template.
8. Drift: `k8s_diff_manifest` with the original manifest shows the paths that differ.
9. Clean up: `k8s_delete_resource(api_version=v1, kind=Namespace, name=zz-validation,
   confirm=true)`.

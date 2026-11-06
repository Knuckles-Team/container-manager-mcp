---
name: container-manager-k8s-change-validation
skill_type: skill
description: >-
  The proven end-to-end change workflow for a Kubernetes workload through
  container-manager-mcp: create a test object, scale, patch, rollout restart,
  status, history, undo, apply a manifest and clean up. Use to validate the
  connector against a cluster, or as the template for any safe workload change
  with rollback. Do NOT use for Docker/Swarm hosts or for persisting the change
  (container-manager-gitops-persist).
license: MIT
tags: [container-manager-mcp, kubernetes, rollout, validation]
metadata:
  author: Repository Maintainers
  version: '0.1.0'
---
# Validate a Kubernetes change end to end

Use a disposable namespace `zz-validation` and objects named `zz-validation-<date>`.

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

---
name: container-manager-gitops-persist
skill_type: skill
description: >-
  Change a Kubernetes setting and persist it to the owning services/<name>/
  repository so the cluster never drifts live-only. Uses container-manager-mcp
  (k8s_export_manifest, k8s_diff_manifest, k8s_apply_manifest) with the git
  repository connector that hosts services/ (gitlab-mcp gitlab_create_commit).
  Use when the agent must change replicas, env, image, resources, an annotation
  or any other k8s field, or must check that a live object still matches its
  manifest. Do NOT use to read secret values (container-manager-externalsecret-sync,
  k8s_get_secret) or for Docker/Swarm hosts (container-manager-swarm).
license: MIT
tags: [container-manager-mcp, kubernetes, gitops, drift, services]
metadata:
  author: Repository Maintainers
  version: '0.1.0'
---
# Persist a Kubernetes change to services/

Rule: every Kubernetes setting lives in `services/<name>/k8s/*.yaml` and is applied from
there. A live-only change is unfinished work. Two paths, same end state.

## Path A: manifest first (preferred)
1. Read the current manifest from the services repository (git connector file read, for
   example `gitlab_get_repository_file` on project `<group>/services/<name>`, path
   `k8s/manifests.yaml`).
2. Edit the YAML locally in your reasoning; keep comments and unrelated objects.
3. `k8s_diff_manifest(manifest_yaml=<edited>)`: every changed object reports
   `drift=true` with the JSON `paths` that will change. Confirm they are exactly the
   intended fields.
4. Commit the edited file to the repository's default branch
   (`gitlab_create_commit` with one `update` action, message
   `deploy(<name>): <what and why>`).
5. `k8s_apply_manifest(manifest_yaml=<committed>)` (server-side apply, field manager
   `container-manager-mcp`).
6. For workloads, poll `k8s_rollout_status` until `complete=true`.
7. `k8s_diff_manifest(manifest_yaml=<committed>)` must return `drift=false` for every
   document. Report the commit id.

## Path B: urgent live change, persisted straight after
1. Make the change: `k8s_patch_resource`, `k8s_scale_deployment`, `k8s_rollout_restart`
   (a restart stamps `restartedAt`, which is not persisted) or `k8s_apply_manifest`.
2. `k8s_export_manifest(api_version, kind, name, namespace)` returns `manifest_yaml`
   with status, managed fields, ids and rollout bookkeeping removed.
3. Replace that object's document in `services/<name>/k8s/<file>.yaml` with the export
   (keep the file's other documents and header comments), commit as in A.4.
4. `k8s_diff_manifest` on the committed file: `drift=false`.

## Refusals and edge cases
- `k8s_export_manifest` refuses a `Secret`: values belong in OpenBao. Persist the
  `ExternalSecret` instead and write the value with the secret-vault skill.
- `drift=true` with `live=absent`: the manifest declares an object that is not deployed;
  apply it or remove it from the manifest, never leave it.
- Cluster-wide objects (ClusterRole, StorageClass, CRDs) still belong to the services
  repository of the component that owns them.
- Never `git add -A`; one logical change per commit.

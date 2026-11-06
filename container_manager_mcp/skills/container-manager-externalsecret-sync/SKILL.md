---
name: container-manager-externalsecret-sync
skill_type: skill
description: >-
  Refresh a Kubernetes Secret from its secret store now and prove it refreshed,
  through container-manager-mcp (k8s_list_external_secrets,
  k8s_force_sync_external_secret, k8s_get_secret). Use after a value changed in
  the vault, when an ExternalSecret reports Ready=False, or when a workload still
  sees an old credential. Do NOT use to write vault values (secret-vault-manager)
  or to persist manifests (container-manager-gitops-persist).
license: MIT
tags: [container-manager-mcp, kubernetes, external-secrets, openbao, secrets]
metadata:
  author: Repository Maintainers
  version: '0.1.0'
---
# Force-sync an ExternalSecret

1. Find it: `k8s_list_external_secrets(all_namespaces=true)` returns name, namespace,
   target Secret, store, `ready`, `reason`, `refresh_time`.
2. Refresh: `k8s_force_sync_external_secret(name, namespace)`.
   - It stamps a new `force-sync` annotation and waits (default 60 s, max 300 s) for a
     newer `refresh_time` with Ready=True.
   - If the controller does not refresh in time, it deletes the target Secret and waits
     for the controller to recreate it from the store (`method:
     annotation+recreate-target`). Pass `recreate_target=true` to go straight there.
   - Result: `refreshed`, `method`, `target_present`, `target_changed` (data digest
     before vs after), `ready`, `message`. No values are returned.
3. `target_changed=false` with `refreshed=true` means the store still holds the old
   value: fix the value in the vault first, then repeat.
4. Workloads read env from Secrets only at start: run `k8s_rollout_restart` on the
   consumers and poll `k8s_rollout_status`.
5. `k8s_get_secret(name, namespace)` shows key names and a digest; `reveal=true` only on
   an explicit user request.

Why a dedicated tool: an annotation alone is not always acted on promptly (the
controller runs one reconcile at a time and retries failing objects first), and nothing
reports whether it happened. Deleting the target forces a rebuild; the tool does both and
reports the outcome.

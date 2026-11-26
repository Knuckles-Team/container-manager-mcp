---
name: container-manager-externalsecret-sync
skill_type: skill
description: >-
  Refresh a Kubernetes Secret from its secret store now and prove it refreshed, via k8s_list_external_secrets, k8s_force_sync_external_secret and k8s_get_secret. Use after a vault value changed, when an ExternalSecret reports Ready=False, or when a workload sees an old credential. Do NOT use to write vault values or persist manifests.
license: MIT
tags: [container-manager-mcp, kubernetes, external-secrets, openbao, secrets]
metadata:
  author: Repository Maintainers
  version: '0.1.0'
---
# Force-sync an ExternalSecret

Scope: Refresh a Kubernetes Secret from its secret store now and prove it refreshed, through container-manager-mcp (k8s_list_external_secrets, k8s_force_sync_external_secret, k8s_get_secret). Use after a value changed in the vault, when an ExternalSecret reports Ready=False, or when a workload still sees an old credential. Do NOT use to write vault values (secret-vault-manager) or to persist manifests (container-manager-gitops-persist).

## Steps

1. List the ExternalSecrets with `k8s_list_external_secrets(all_namespaces=true)` and verify the target shows its `ready`, `reason` and `refresh_time`.
2. Refresh it with `k8s_force_sync_external_secret(name, namespace)` and verify the result has `refreshed=true` and `ready=true`.
3. Check the outcome in the same result and verify `target_changed=true`, which proves the store held a new value.
4. Read the target with `k8s_get_secret(name, namespace)` and verify the expected key names are present (values stay hidden unless `reveal=true` is explicitly requested).
5. Restart the consumers with `k8s_rollout_restart` and verify `k8s_rollout_status` returns `complete=true`.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| `refreshed=false` | The controller did not refresh within the wait window | Call `k8s_force_sync_external_secret` with `recreate_target=true` to delete and rebuild the target Secret |
| `target_changed=false` with `refreshed=true` | The store still holds the old value | Correct the value in the vault first with the secret-vault-manager skill, then sync again |
| `ready=false` in the list | The store or the remote key is invalid | Read the `reason` field from `k8s_list_external_secrets` and fix the store reference or key it names |
| Workload still sees the old credential | Env from Secrets is read only at container start | Run `k8s_rollout_restart` on the consumers and watch `k8s_rollout_status` |

## Reference procedure

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

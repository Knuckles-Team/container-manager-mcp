---
name: container-manager-mcp-operations
skill_type: skill
description: >-
  Operate container-manager-mcp through its governed MCP and Arbora capabilities, including container manager config walkthrough, container manager kg ingestion, container manager kubernetes operations, container manager lifecycle, container manager multi context, container manager podman operations, container manager swarm, kubernetes mesh provisioner, and related workflows.
---

# Container Manager Mcp Operations

Scope: Operate container-manager-mcp through its governed MCP and Arbora capabilities, including container manager config walkthrough, container manager kg ingestion, container manager kubernetes operations, container manager lifecycle, container manager multi context, container manager podman operations, container manager swarm, kubernetes mesh provisioner, and related workflows. Use when a request requires this provider's read, change, automation, ingestion, troubleshooting, or evidence workflows.

Use the provider's governed MCP tools through Arbora delegation.

## Steps

1. Discover the current tool surface with `find(intent='container manager kubernetes pods')` and verify the returned tool names and schemas before building any call.
2. Open `references/catalog.md` with `sed -n 1,40p references/catalog.md` and verify the provider workflow you need is listed.
3. Run a read-only call with `act(op='fleet.call', params={'server': 'container-manager-mcp', 'tool': 'k8s_list_pods', 'arguments': {'all_namespaces': True}})` and verify pods are returned.
4. Preview any change with `act(op='fleet.call', params={'server': 'container-manager-mcp', 'tool': 'k8s_apply_manifest', 'arguments': {'manifest_yaml': '<yaml>', 'dry_run': True}})` and verify the reported impact matches the request.
5. Re-read the object with `k8s_list_pods` through the same `fleet.call` as step 3 and verify the durable result and its trace evidence before reporting completion.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| Tool or schema not found | A stale tool name was used | Repeat `find(intent='...')` and use the name and schema it returns |
| ACL, tenant or session unknown | The verified GraphSession was not established | Stop and report a hard failure, then establish the session before any retrieval |
| Destructive call blocked | Explicit approval was not given | Ask the user for approval of the exact action before executing it |
| Dry run shows unexpected changes | The manifest differs from the live object | Correct the manifest and repeat the preview until the diff shows only the intended fields |

## Workflow

1. Establish the verified GraphSession and tenant before discovery or retrieval.
2. Discover the current condensed tool surface; never assume a stale tool name or schema.
3. Prefer read-only inspection first. For changes, present impact and use the provider's
   dry-run or preview mode when available.
4. Execute mutations as fenced WorkItems so retries remain idempotent and auditable.
5. Ingest source data only through the signed connector preset and ChangeEnvelope path.
6. Verify the durable result and its trace/evidence before reporting completion.

## Safety contract

- Never persist credentials, endpoints, raw personal identifiers, hostnames, or local paths.
- Resolve TLS trust and verification from environment/configuration; never hardcode bypasses.
- Treat unknown ACL, tenant, schema, or tool-contract state as a hard failure.
- Require explicit approval for destructive, externally visible, or irreversible actions.
- Keep runtime traces policy-scoped and privacy-sanitized.

## Specialized workflows

Read [the workflow catalog](references/catalog.md) only when the request needs a
provider-specific procedure, parameter map, script, or reference asset.

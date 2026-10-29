# CONN-CONTAINER-MANAGER-MCP — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A row names the file it touches and the test that proves it.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-CONTAINER-MANAGER-MCP-R001` | **Per-call context.** Every generic `docker_*` and `k8s_*` tool must accept an optional `context` argument that selects the manager. | `tests/test_call_context.py` rows R001.1 to R001.5 pass. |
| `CONN-CONTAINER-MANAGER-MCP-R001.1` | **Advertised argument.** Each generic tool schema must list an optional `context` property, and the property must not be required. | `tests/test_call_context.py::test_generic_tools_advertise_context` must pass. |
| `CONN-CONTAINER-MANAGER-MCP-R001.2` | **Named context selects the manager.** A call with `context` must run against the manager of that context for the tool backend. | `tests/test_call_context.py::test_context_argument_selects_the_named_context` must pass. |
| `CONN-CONTAINER-MANAGER-MCP-R001.3` | **Default manager.** A call without `context` must use the default manager of the tool backend. | `tests/test_call_context.py::test_without_context_the_default_manager_is_used` must pass. |
| `CONN-CONTAINER-MANAGER-MCP-R001.4` | **Wiring.** `container_manager_mcp/mcp_server.py:1634` must advertise the argument and `:1635` must add `CallContextMiddleware`. `:1609` and `:1628` must resolve clients through `manager_for`. | `tests/test_call_context.py::test_context_argument_selects_the_named_context` must reach those call sites. |
| `CONN-CONTAINER-MANAGER-MCP-R001.5` | **Hot-path proof.** An MCP client calling `k8s_list_pods` and `docker_list_containers` with `context` must reach the named manager. | `tests/test_call_context.py::test_context_argument_selects_the_named_context` must pass. |
| `CONN-CONTAINER-MANAGER-MCP-R002` | **Deployed workload access contract.** Publish an SDK-readable access contract for `k8s_list_services`. Bind deployment name, source-reported availability condition and image reference to its returned fields. Missing availability is `unknown`. | `tests/test_access_contracts.py` parses the contract and compares its fields with a controlled MCP tool response. |
| `CONN-CONTAINER-MANAGER-MCP-R003` | **Inventory privacy.** Preserve explicitly configured SSH users. When no user is configured, leave the SSH username unset so SSH selects its normal identity. Examples use synthetic identities and host aliases. The inventory is scoped to the caller: `cm_list_hosts` returns every alias on the local transport, none for an anonymous HTTP caller and the entitled aliases for an authenticated one, and a tool given an alias the caller is not entitled to refuses with a typed permission error (JSON-RPC -32000), never an actor-context or internal error. | `tests/test_inventory_hosts.py` covers configured and missing users and the generated SSH authority. `tests/test_served_actor_binding.py` serves the real connector in memory and over uvicorn with and without a JWT. |

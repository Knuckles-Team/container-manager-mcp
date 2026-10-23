# CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A child row `.n` is one five-minute unit of work. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001` | **Transport comes from the SDK (rollup).** The connector must use the SDK for HTTP, retry and TLS. Children `.1` to `.3` deliver it. | See each child. |
| `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001.1` | **OIDC through the governed client.** The password-grant import in `container_manager_mcp/k8s/kubeconfig_import.py` must find the token endpoint with `discover_token_endpoint` and post the grant through the SDK governed client (`SDK-CENTRALIZED-CAPABILITIES-R001`, `SDK-CENTRALIZED-CAPABILITIES-R012`). The module must not import `requests`. Proving test: `tests/test_kubeconfig_import.py::test_oidc_password_grant_discovers_and_posts`. | Discovery and the grant reach the issuer through the SDK client, and a rejected grant raises a clear error. |
| `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001.2` | **TLS profile for OIDC.** The OIDC calls must take trust from `resolve_tls_profile` for the `CONTAINER_MANAGER_OIDC` prefix (`SDK-CENTRALIZED-CAPABILITIES-R008`). A readable CA file must join the trust anchors. The `insecure_skip_tls_verify` flag must shape only the kubeconfig cluster entry and must not weaken these calls. Proving test: `tests/test_kubeconfig_import.py::test_oidc_calls_resolve_a_tls_profile_and_never_skip_verification`. | The profile is resolved with the CA file, and no call passes a verification override. |
| `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001.3` | **SDK floor.** The package must depend on `agent-connector-sdk[mcp]` at the latest release floor and must not pin `requests` itself (`SDK-CENTRALIZED-CAPABILITIES-R040`). Proving test: `tests/test_sdk_transport.py::test_dependencies_come_from_the_sdk`. | `pyproject.toml` has no `requests` pin and the SDK floor is `0.3.11` or later. |

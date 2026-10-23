# CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Setup and input | Expected observation | Evidence |
|---|---|---|---|---|---|
| T-001 | `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001` | Unit | Every child row. | See each child. |  |
| T-002 | `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001.1` | Unit | `tests/test_kubeconfig_import.py::test_oidc_password_grant_discovers_and_posts` | Discovery and the grant reach the issuer through the SDK client, and a rejected grant raises a clear error. |  |
| T-003 | `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001.2` | Unit | `tests/test_kubeconfig_import.py::test_oidc_calls_resolve_a_tls_profile_and_never_skip_verification` | The profile is resolved with the CA file, and no call passes a verification override. |  |
| T-004 | `CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001.3` | Unit | `tests/test_sdk_transport.py::test_dependencies_come_from_the_sdk` | `pyproject.toml` has no `requests` pin and the SDK floor is `0.3.11` or later. |  |

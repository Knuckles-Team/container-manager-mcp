# CONN-CONTAINER-MANAGER-MCP — Test specification

Governing spec: [spec.md](spec.md).

| Requirement | Level | Proving test |
|---|---|---|
| `CONN-CONTAINER-MANAGER-MCP-R001` | Unit | `tests/test_call_context.py` rows R001.1 to R001.5 pass. |
| `CONN-CONTAINER-MANAGER-MCP-R001.1` | Unit | `tests/test_call_context.py::test_generic_tools_advertise_context` must pass. |
| `CONN-CONTAINER-MANAGER-MCP-R001.2` | Unit | `tests/test_call_context.py::test_context_argument_selects_the_named_context` must pass. |
| `CONN-CONTAINER-MANAGER-MCP-R001.3` | Unit | `tests/test_call_context.py::test_without_context_the_default_manager_is_used` must pass. |
| `CONN-CONTAINER-MANAGER-MCP-R001.4` | Integration | `tests/test_call_context.py::test_context_argument_selects_the_named_context` must reach those call sites. |
| `CONN-CONTAINER-MANAGER-MCP-R001.5` | Integration | `tests/test_call_context.py::test_context_argument_selects_the_named_context` must pass. |
| `CONN-CONTAINER-MANAGER-MCP-R002` | Integration | `tests/test_access_contracts.py::test_served_deployment_access_contract_matches_fixture` checks a synthetic provider through MCP. |
| `CONN-CONTAINER-MANAGER-MCP-R003` | Integration | `tests/test_inventory_hosts.py::test_docker_host_preserves_only_configured_user` checks both SSH authority forms; `tests/test_served_actor_binding.py` proves host listing and refusal per transport. |

The tests run against the production server object. They need no vendor requests.

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
| `CONN-CONTAINER-MANAGER-MCP-R004` | Unit | `tests/test_inventory_hosts.py::test_tunnel_inventory_env_selects_the_shared_inventory` proves `$TUNNEL_INVENTORY` wins over the XDG default for listing and resolution. |
| `CONN-CONTAINER-MANAGER-MCP-R006` | Unit | `tests/test_windows_portability.py::test_docker_default_endpoint_is_named_pipe_on_windows` and `tests/test_windows_portability.py::test_cluster_dump_on_windows_branch` prove the Windows branches by monkeypatching. |
| `CONN-CONTAINER-MANAGER-MCP-R007` | Integration | `tests/test_destructive_confirmation_transports.py` calls a destructive action through real stdio, HTTP and JWT HTTP clients, with and without elicitation. |
| `CONN-CONTAINER-MANAGER-MCP-R009` | Unit | `tests/test_k8s_management.py::test_pods_and_events_list_every_namespace` |
| `CONN-CONTAINER-MANAGER-MCP-R010` | Unit | `tests/test_k8s_management.py::test_secret_values_are_redacted_unless_revealed` |
| `CONN-CONTAINER-MANAGER-MCP-R011` | Unit | `tests/test_k8s_management.py::test_rollout_undo_restores_the_previous_replicaset_template` |
| `CONN-CONTAINER-MANAGER-MCP-R012` | Unit | `tests/test_k8s_management.py::test_force_sync_recreates_the_target_when_asked` |
| `CONN-CONTAINER-MANAGER-MCP-R013` | Unit | `tests/test_k8s_management.py::test_diff_reports_drift_paths_and_absent_objects` |
| `CONN-CONTAINER-MANAGER-MCP-R014` | Unit | `tests/test_k8s_management.py::test_apply_manifest_server_side_applies_each_document` |
| `CONN-CONTAINER-MANAGER-MCP-R015` | Unit | `tests/test_k8s_management.py::test_served_surface_has_the_management_tools_prompts_skills_and_ontology` |
| `CONN-CONTAINER-MANAGER-MCP-R016` | Unit | `tests/test_k8s_management.py::test_contracts_name_served_tools_prompts_and_skills_and_roll_up_to_core` |
| `CONN-CONTAINER-MANAGER-MCP-R017` | Unit | `tests/test_k8s_management.py::test_deployment_describe_scale_and_pod_logs_take_a_namespace` |
| `CONN-CONTAINER-MANAGER-MCP-R018` | Unit | `tests/test_k8s_management.py::test_context_pool_is_built_once_per_process` |

The tests run against the production server object. They need no vendor requests.

## Rows awaiting bound tests
- `CONN-CONTAINER-MANAGER-MCP-R005`: proof `tests/test_access_contracts.py::test_served_deployment_access_contract_matches_fixture`; the test carries the row's `spec` marker when it lands.
- `CONN-CONTAINER-MANAGER-MCP-R008`: proof `tests/test_access_contracts.py::test_deployment_contract_matches_served_fixture`; the test carries the row's `spec` marker when it lands.

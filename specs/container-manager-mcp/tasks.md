# CONN-CONTAINER-MANAGER-MCP — Implementation tasks

Governing spec: [spec.md](spec.md). Design: [plan.md](plan.md). Rows: [requirements.md](requirements.md).

- [x] `CONN-CONTAINER-MANAGER-MCP-R001` — Per-call context. `tests/test_call_context.py` rows R001.1 to R001.5 pass.
- [x] `CONN-CONTAINER-MANAGER-MCP-R001.1` — Advertised argument. `tests/test_call_context.py::test_generic_tools_advertise_context` must pass.
- [x] `CONN-CONTAINER-MANAGER-MCP-R001.2` — Named context selects the manager. `tests/test_call_context.py::test_context_argument_selects_the_named_context` must pass.
- [x] `CONN-CONTAINER-MANAGER-MCP-R001.3` — Default manager. `tests/test_call_context.py::test_without_context_the_default_manager_is_used` must pass.
- [x] `CONN-CONTAINER-MANAGER-MCP-R001.4` — Wiring. `tests/test_call_context.py::test_context_argument_selects_the_named_context` must reach those call sites.
- [x] `CONN-CONTAINER-MANAGER-MCP-R001.5` — Hot-path proof. `tests/test_call_context.py::test_context_argument_selects_the_named_context` must pass.
- [x] `CONN-CONTAINER-MANAGER-MCP-R002` — Serve the deployment access contract and verify its fields against a synthetic provider response.
- [x] `CONN-CONTAINER-MANAGER-MCP-R003` — Preserve configured users and omit unconfigured SSH usernames.
- [x] `CONN-CONTAINER-MANAGER-MCP-R004` — Read the shared inventory from `$TUNNEL_INVENTORY` like tunnel-manager.

# CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R001` | `tests/test_default_content_surfaces.py` | The listings include the packaged skills, ontology and prompts. |
| T-002 | `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R001` | `tests/test_semantic_core_mappings.py` | The approved connector classes reach their shared core parents and every ontology/shape resource is served with exact packaged bytes. |
| T-003 | `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R002` | connector ontology certification | Connector ontology certification passes. |

## Rows awaiting bound tests
- `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R003`: planned proof `tests/test_access_contracts.py::test_pod_container_image_contracts`. Expected: The ttl parses into four contracts and the three new ones name list_pods, list_containers and list_images with key and class set.. Refusal: A contract missing ac:keyField or ac:classIri, or naming an unregistered operation, fails certification..
- `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R004`: planned proof `tests/test_sandbox_provision_served.py::test_sandbox_provision_passes_conformance`. Expected: sandbox_provision creates a container on a disposable runtime and run_sandbox_conformance reports every check passed.. Refusal: A provision request with an unknown image or without confirmation is refused and no container is created..
- `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R005`: planned proof `tests/test_desktop_served.py::test_desktop_tools_pass_conformance`. Expected: desktop_click, desktop_type and desktop_verify act on the sandbox container and run_desktop_conformance passes.. Refusal: A call when no gui-sandbox container is running returns a typed not-found error and executes nothing..

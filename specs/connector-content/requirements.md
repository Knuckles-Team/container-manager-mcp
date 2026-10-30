# CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R001` | **Default content.** `pyproject.toml` declares `container-manager-mcp` as the single `agent_connector_sdk.connector` entry point. The server lists the packaged `skill://` and `ontology://` resources and the packaged prompts with no `content=` argument. Proving test: `tests/test_default_content_surfaces.py`. | `resources/list` includes `skill://` and `ontology://`; `prompts/list` includes each packaged prompt. |
| `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R002` | **Access contract for pods, containers and images.** Add `ac:AccessContract` rows for pods, containers and images under `container_manager_mcp/ontology/`, using `cm_k8s_workloads list_pods`, container `list_containers` and `cm_image_operations list_images`. | Connector ontology certification passes. |

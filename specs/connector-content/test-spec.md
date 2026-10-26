# CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R001` | `tests/test_default_content_surfaces.py` | The listings include the packaged skills, ontology and prompts. |
| T-002 | `CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R001` | `tests/test_semantic_core_mappings.py` | The approved connector classes reach their shared core parents and every ontology/shape resource is served with exact packaged bytes. |

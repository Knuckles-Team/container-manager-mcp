---
name: container-manager-logs
description: Manages logs. Use for container/compose logs. Triggers - debugging, monitoring.
---

### Overview
Log retrieval for troubleshooting.

### Key Tools
- `get_container_logs`: Get logs. Params: container_id (required), tail="all".
- `compose_logs`: Compose logs. Params: compose_file (required), service?.

### Usage Instructions
1. Tail for recent lines (e.g., "100").
2. Subset: Service-specific in compose.

### Examples
- Container: `get_container_logs` with container_id="nginx", tail="50".
- Compose: `compose_logs` with compose_file="docker-compose.yml".

### Error Handling
- No logs: Container not running.

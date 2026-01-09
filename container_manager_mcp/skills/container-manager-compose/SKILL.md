---
name: container-manager-compose
description: Manages Docker Compose. Use for up/down/ps/logs. Triggers - multi-container apps.
---

### Overview
Compose for app stacks.

### Key Tools
- `compose_up`: Start stack. Params: compose_file (required), detach=true, build?.
- `compose_down`: Stop/remove.
- `compose_ps`: List services.
- `compose_logs`: Get logs. Params: service?.

### Usage Instructions
1. Provide compose_file path.
2. Subset: Service-specific logs.

### Examples
- Up: `compose_up` with compose_file="docker-compose.yml", build=true.
- Logs: `compose_logs` with service="db".

### Error Handling
- Invalid YAML: Validate file.
- Conflicts: Down first.

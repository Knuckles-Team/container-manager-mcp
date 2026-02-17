---
name: container-manager-info
description: Retrieves Container Manager info. Use for version/system details before ops. Triggers - setup checks, compatibility.
---

### Overview
Basic info tools for Docker/Podman. Call first to verify environment.

### Key Tools
- `get_version`: Get manager version. Params: manager_type? (docker/podman), silent?, log_file?.
- `get_info`: Get system info (OS, drivers). Similar params.

### Usage Instructions
1. Optional manager_type; auto-detects.
2. Use for troubleshooting setup issues.

### Examples
- Version: `get_version` with manager_type="docker".
- Info: `get_info`.

### Error Handling
- No manager: Check installation.
- Logs: Use log_file for persistence.

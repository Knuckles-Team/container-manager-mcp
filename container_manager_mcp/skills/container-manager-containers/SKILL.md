---
name: container-manager-containers
description: Manages containers. Use for listing/running/stopping/removing/pruning. Triggers - runtime ops, scaling.
---

### Overview
Core container control. Supports troubleshooting subsets (list -> logs -> exec).

### Key Tools
- `list_containers`: List running/all. Params: all?, manager_type?.
- `run_container`: Run new. Params: image (required), name?, command?, detach?, ports?, volumes?, environment?.
- `stop_container`: Stop. Params: container_id (required), timeout=10.
- `remove_container`: Remove. Params: container_id (required), force?.
- `prune_containers`: Clean stopped.
- `exec_in_container`: Exec command. Params: container_id (required), command (list), detach?.

### Usage Instructions
1. Use ID/name for actions.
2. For troubleshooting: list -> get_logs (logs skill) -> exec.

### Examples
- Run: `run_container` with image="nginx", ports={"80/tcp": "8080"}.
- Exec: `exec_in_container` with container_id="mycont", command=["ls", "-l"].

### Error Handling
- Not running: Check status first.
- Conflicts: Force for removal.
  Reference `troubleshoot.md` for workflows.

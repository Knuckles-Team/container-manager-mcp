---
name: container-manager-volumes
description: Manages volumes. Use for listing/creating/removing/pruning. Triggers - data persistence.
---

### Overview
Volume ops for stateful containers.

### Key Tools
- `list_volumes`: List all.
- `create_volume`: Create. Params: name (required).
- `remove_volume`: Remove. Params: name (required), force?.
- `prune_volumes`: Clean unused. Params: all?.

### Usage Instructions
1. Use with run_container volumes param.

### Examples
- Create: `create_volume` with name="data-vol".
- Prune: `prune_volumes`.

### Error Handling
- In use: Stop containers first.

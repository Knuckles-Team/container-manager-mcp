---
name: container-manager-swarm
description: Manages Docker Swarm. Use for init/leave, nodes/services. Triggers - clustering, orchestration. Note - Docker only.
---

### Overview
Swarm for distributed ops. Restrict to Docker.

### Key Tools
- `init_swarm`: Init cluster. Params: advertise_addr?.
- `leave_swarm`: Leave. Params: force?.
- `list_nodes`: List nodes.
- `list_services`: List services.
- `create_service`: Create. Params: name, image (required), replicas=1, ports?, mounts?.
- `remove_service`: Remove. Params: service_id (required).

### Usage Instructions
1. Manager_type="docker" required.
2. For services: Similar to containers but replicated.

### Examples
- Init: `init_swarm`.
- Create service: `create_service` with name="web", image="nginx", replicas=3.

### Error Handling
- Not Swarm: Init first.
- Node down: Check status.
  Reference `orchestrate.md` for scaling workflows.

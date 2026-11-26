---
name: swarm-mesh-provisioner
skill_type: skill
description: >
  Swarm Mesh Provisioner atomic skill. Initializes Docker Swarm Mode,
  manages manager/worker nodes join processes, and provisions global Swarm Overlay networks using container-manager-mcp.
domain: infrastructure
tags:
  - swarm
  - docker
  - overlay-net
  - cluster
requires:
  - container-manager-mcp
  - tunnel-manager-mcp
---

# Swarm Mesh Provisioner Skill

Stateless atomic operation to establish multi-node container orchestration clusters and secure virtual networks.

## Prerequisites

- `container-manager-mcp` — for executing Swarm init, join, network, and node operations.
- `tunnel-manager-mcp` — for running command execution on worker nodes.

## Steps

1. Initialize the manager with `docker swarm init --advertise-addr <manager-ip>` and verify `docker info` reports `Swarm: active`.
2. Fetch the worker token with `docker swarm join-token worker` and verify the output contains a join command.
3. Join each worker through tunnel-manager-mcp with `docker swarm join --token <token> <manager-ip>:2377` and verify `docker info` on the worker reports `Swarm: active`.
4. Create the overlay with `docker network create --driver overlay --attachable --opt encrypted overlay-net` and verify `docker network ls` shows it with scope `swarm`.
5. Check membership with `docker node ls` on the manager and verify every node is `Ready` and `Active`.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| `swarm init` reports the node is already in a swarm | Swarm mode is already active | Read `docker info` and reuse the existing swarm, or leave it with `docker swarm leave --force` only after the user confirms |
| Worker join times out | The manager is unreachable on port 2377 | Check the route to `<manager-ip>:2377` from the worker and fix the firewall or advertise address |
| Join is rejected | The token is stale or for the wrong role | Read a fresh token with `docker swarm join-token worker` and join again with it |
| Overlay creation fails | The command ran on a worker | Run `docker network create` on the manager node |

## Procedure detail

### Step 1: Initialize Swarm Manager
On the target Manager node, verify Swarm status. If inactive:
- Run `docker swarm init` specifying advertise address.
- Retrieve manager and worker join tokens.

### Step 2: Join Swarm Workers
For each designated worker node listed in the inventory:
- Verify remote Docker engine state.
- Join the Swarm cluster using the worker join token.
- Validate active cluster membership.

### Step 3: Provision Global Overlay Network
Create a secure, encrypted overlay network (e.g., `overlay-net`) to bridge containers across different hardware hosts:
- Ensure the network is created with `--attachable` to support dynamic containers.
- Check MTU and driver settings.

### Step 4: Verify Cluster State
Confirm that all node statuses read `Ready` and state shows `Active` on the Swarm manager.

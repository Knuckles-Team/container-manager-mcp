---
name: podman-mesh-provisioner
skill_type: skill
description: >-
  Podman Mesh Provisioner atomic skill. Provisions a Podman container control plane — rootful OR rootless — enabling the Podman API socket, wiring podman-compose, and registering remote hosts as Podman system connections over SSH, using container-manager-mcp. The Podman peer of swarm-mesh-provisioner and kubernetes-mesh-provisioner.
domain: infrastructure
tags:
  - podman
  - rootless
  - podman-compose
  - cluster
  - overlay-net
requires:
  - container-manager-mcp
  - tunnel-manager-mcp
---

# Podman Mesh Provisioner Skill

Scope: Podman Mesh Provisioner atomic skill. Provisions a Podman container control plane — rootful OR rootless — enabling the Podman API socket, wiring podman-compose, and registering remote hosts as Podman system connections over SSH, using container-manager-mcp. The Podman peer of swarm-mesh-provisioner and kubernetes-mesh-provisioner. Triggers on "provision podman", "rootless podman", "podman-compose cluster", "set up podman hosts".

Stateless atomic operation to stand up a Podman-based container runtime — **rootful
or rootless** — across one or more hosts, as a first-class peer to Docker Swarm and
Kubernetes. Podman has no built-in clustering daemon; the "mesh" is the API socket +
`podman-compose` locally and **Podman system connections over SSH** for remote hosts
(the same Docker-over-SSH inventory model `container-manager-mcp` already uses).

## Prerequisites

- `container-manager-mcp` — set `CONTAINER_MANAGER_TYPE=podman` and (optionally)
  `COMPOSE_TOOL=podman-compose`; it drives Podman socket/compose/host operations.
- `tunnel-manager-mcp` — remote command execution + the shared
  `~/.config/scion-sdk/inventory.yaml`.

## Inputs

- `rootless` (bool, default **true**) — rootless runs the Podman socket under the
  invoking user (`$XDG_RUNTIME_DIR/podman/podman.sock`, no daemon, better isolation);
  rootful uses the system socket (`/run/podman/podman.sock`) for host-network / low
  ports / shared volumes.
- `hosts` — inventory aliases to provision (default: the controller only).

## Steps

1. For rootless mode enable lingering with `loginctl enable-linger <user>` and verify `loginctl show-user <user> --property=Linger` prints `Linger=yes`.
2. Enable the API socket with `systemctl --user enable --now podman.socket` (rootful: `sudo systemctl enable --now podman.socket`) and verify the socket path exists with `ls $XDG_RUNTIME_DIR/podman/podman.sock`.
3. Wire compose with `podman-compose --version` and verify it prints a version before setting `COMPOSE_TOOL=podman-compose`.
4. Register each remote host with `podman system connection add <alias> ssh://<user>@<host><socket>` and verify `podman system connection list` shows the alias.
5. Check each target with `podman --connection <alias> info` and verify it reports the requested rootless or rootful mode.

## Failure recovery

| Failure | Cause | Recovery |
|---|---|---|
| Socket unit not found | Podman is not installed or the user session has no systemd | Install Podman on the host and verify with `podman --version`, then enable the socket again |
| Rootless socket disappears after logout | Lingering is not enabled for the user | Run `loginctl enable-linger <user>` and confirm with `loginctl show-user` |
| `podman-compose` not found | The compose tool is not installed | Install `podman-compose` and confirm with `podman-compose --version` |
| Remote connection fails | SSH key or inventory entry is wrong | Correct the host, user and key in the shared tunnel-manager inventory and re-add the connection |

## Procedure detail

### Step 1: Select the Podman mode (rootful vs rootless)
Resolve the `rootless` input. For **rootless**, ensure lingering is enabled so the
user socket survives logout (`loginctl enable-linger <user>`) and that subuid/subgid
ranges exist for the user. For **rootful**, operations run against the system socket
and require privileged access on the host.

### Step 2: Enable the Podman API socket
Enable and start the Podman socket service so `container-manager-mcp` can speak the
Podman API:
- Rootless: `systemctl --user enable --now podman.socket` →
  `$XDG_RUNTIME_DIR/podman/podman.sock`.
- Rootful: `sudo systemctl enable --now podman.socket` → `/run/podman/podman.sock`.
Record the resolved socket path as `CONTAINER_MANAGER_PODMAN_BASE_URL`
(`unix://<socket>`).

### Step 3: Wire podman-compose
Confirm `podman-compose` (or `podman compose`) is available; set `COMPOSE_TOOL=podman-compose`
so stack deploys route through Podman rather than Docker.

### Step 4: Register remote hosts (multi-host mesh)
For each host in the inventory, register a Podman **system connection** over SSH so
the controller can target it by alias:
`podman system connection add <alias> ssh://<user>@<host><socket>`. This reuses the
shared inventory (host/user/key) — no remote daemon or exposed TCP socket.

### Step 5: Verify
Confirm `podman info` returns healthy on the controller and on each registered remote
connection (rootless vs rootful matches the requested mode), and that a trivial
`podman run` / `podman-compose` smoke deploy succeeds.

## Execution

This is an atomic skill — run the steps above in order via `container-manager-mcp`
(Podman socket/connection/info/run actions) and `tunnel-manager-mcp` (remote
`systemctl`/`loginctl`/`podman system connection` commands). It is the orchestrator's
Podman branch (peer to `swarm-mesh-provisioner` / `kubernetes-mesh-provisioner`); the
caller picks the branch from the operator's chosen orchestrator.

> **Execution:** If arbora is reachable, offload via
> `graph_orchestrate action=execute_agent` (or the `kg-delegate` skill) so
> the per-host steps fan out; otherwise execute the steps natively in order against
> the two MCP servers above.

# Changelog

All notable changes to `container-manager-mcp` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.1] - 2026-07-09

### Changed
- The `[mcp]` extra now bundles the Docker, Podman, and Kubernetes client libraries by default
  (`docker[ssh]` + `podman` + `kubernetes`), so a single `container-manager-mcp[mcp]` install (and the
  `:mcp`/`:latest` images built from it) has full Docker/Podman/Kubernetes support with no separate extras.
  Fixes deployments where the k8s/podman tools registered but their operations failed at runtime because the
  client libraries were absent (surfaced by `container-manager-doctor`).

## [2.1.0] - 2026-07-09

### Added
- **Full, real Kubernetes operational surface** via the official `kubernetes` python client,
  grouped by function into 8 themed MCP tools: `cm_k8s_workloads`, `cm_k8s_config`,
  `cm_k8s_networking`, `cm_k8s_storage`, `cm_k8s_rbac`, `cm_k8s_cluster`, `cm_k8s_governance`,
  `cm_k8s_observability` (pods/deployments/statefulsets/daemonsets/jobs/cronjobs/rollouts,
  configmaps/secrets/namespaces/CRDs, ingress/networkpolicy/endpoints/DNS + true core Services,
  PV/PVC/StorageClass/snapshots/CSI, RBAC + serviceaccounts + access reviews, nodes/contexts/CSR,
  quotas/limits/PDB/HPA, metrics/watch/debug).
- Generic `patch_resource` (strategic/merge/json) and true core-`Service` verbs
  (`list_k8s_services`/`get`/`create`/`delete`), distinct from the Swarm-parity Deployment view.
- **Real advanced Docker Swarm** (`cm_docker_swarm`: services/stacks/configs/secrets/nodes) and
  **real Podman** (`cm_podman`: pods, generate/play kube, checkpoint/restore, networks, volumes,
  health) operations via the docker/podman SDKs + CLI — replacing the previous simulated stubs.
- **Multi-context/multi-instance** operation (`cm_multi_context`) with real parallel fan-out
  (ThreadPoolExecutor), per-context health checks and lazy reconnect.
- **Environment doctor**: `container-manager-doctor` CLI + `cm_doctor` MCP tool that diagnose and
  guide the user through backend, inventory (SSH), and Kubernetes (kubeconfig/context) configuration
  with actionable remediation.
- Kubernetes ontology classes (`:Pod`/`:Deployment`/`:K8sService`/`:Namespace`/`:K8sNode`/…) and
  KG ingestion modalities (`pods`/`deployments`/`namespaces`/`k8s_services`) for `cm_ingest_inventory`.
- New skills: `container-manager-kubernetes-operations`, `container-manager-podman-operations`,
  `container-manager-multi-context`, `container-manager-config-walkthrough`.
- Capability-parity test suite asserting 100% docker/podman/kubernetes action coverage.

### Changed
- Split the monolithic `k8s_manager.py` into a `k8s/` mixin subpackage grouped by function; all
  Kubernetes API-group clients are constructed once in `__init__`.
- Renamed the previous "advanced" tools to function-based names: `cm_docker_advanced` →
  `cm_docker_swarm`, `cm_podman_advanced` → `cm_podman` (env toggles `DOCKERSWARMTOOL`/`PODMANTOOL`);
  collapsed 19 sprawling k8s dispatchers into the 8 themed tools with a single registration path.

### Fixed
- Repaired the entirely-broken RBAC path (was calling role APIs on `CoreV1Api`; now uses
  `RbacAuthorizationV1Api`/`AuthorizationV1Api`).
- `rollout_restart` `NameError` (missing `datetime` import); `list_ingress` host extraction bug;
  uninitialized batch client; removed ~20 duplicate/shadowed manager methods.
- Path-traversal hardening on pod file extraction (bandit B202); real `kubectl cp` for
  `copy_to_pod`/`copy_from_pod` (previously returned fake success).
- `MultiContextManager` no longer crashes on construction.
- Consolidated the Kubernetes tool env toggles (16 → 8) and drove env-var drift to zero.

## [1.15.0] - 2026-05-22

### Added
- Initial CHANGELOG.md creation
- docs/concepts.md with CONCEPT ID registry

### Changed
- Standardized project structure per agent-packages ecosystem conventions

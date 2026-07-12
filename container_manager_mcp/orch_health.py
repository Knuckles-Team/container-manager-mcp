"""Orchestration-layer health producer — Phase C of the unified infra-intelligence plan
(``reports/unified-infra-intelligence-plan.md``). Samples the local Kubernetes cluster's
per-node orchestration health (readiness/pressure conditions, pod restarts, pending/evicted
pods, image-pull failures, resource allocation), distills to lightweight per-window trends,
learns per-node baselines, and flags anomalies — the orchestration-layer twin of
``fan_manager.kg_control`` (hardware) and ``systems_manager.os_health`` (OS), generalized
via the shared fleet primitive.

CONCEPT:CM-OS.observability.orch-health-producer. This module is a thin **producer**: it
emits named numeric signals into the shared kernels
(:mod:`agent_utilities.observability.health` / ``health_ingest``) and gets
trend/baseline/anomaly for free — no bespoke statistics live here. Sampling reuses the
package's existing, already-authenticated Kubernetes client
(``container_manager_mcp.container_manager.create_manager("kubernetes")`` — the same
kubeconfig/in-cluster loading every ``cm_k8s_*`` tool uses) over a small injectable
:class:`ClusterClient` seam, mirroring ``fan_manager.fan_manager.CommandRunner`` /
``systems_manager.os_health.CommandRunner``.

**Guarded import:** container-manager-mcp installs ``agent-utilities`` from PyPI, which may
predate the shared ``agent_utilities.observability.health*`` modules (another session
publishes them separately). Every entry point below degrades to a clean no-op when the
shared kernels are absent — this package never crashes for it, mirroring
``fan_manager.kg_control``'s / ``systems_manager.os_health``'s ``_HAS_SHARED_HEALTH`` guard
exactly.

**Report-only by design.** This producer only observes and writes typed KG nodes; it never
mutates the cluster (no pod eviction, no cordon/drain, no scaling) — see the plan's
"report-only → approved → closed-loop" staging.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("container_manager_mcp.orch_health")

try:
    from agent_utilities.observability.health import (
        HealthTrendBuffer,
        compute_baseline,
        correlate,
        detect_anomaly,
    )
    from agent_utilities.observability.health_ingest import (
        ingest_health_anomaly,
        ingest_health_baseline,
        ingest_health_trend,
        read_health_trends,
    )

    _HAS_SHARED_HEALTH = True
except Exception as _e:  # noqa: BLE001 — older/absent agent-utilities: no-op producer
    logger.debug(
        "shared health kernels unavailable, orchestration-health producer degrades "
        "to no-op: %s",
        _e,
    )
    _HAS_SHARED_HEALTH = False

# The named orchestration signals this producer samples per Kubernetes node
# (CONCEPT:CM-OS.observability.orch-health-producer).
ORCH_SIGNALS: tuple[str, ...] = (
    "ready",
    "mem_pressure",
    "disk_pressure",
    "pid_pressure",
    "pod_restart_rate",
    "pods_pending",
    "pods_evicted",
    "image_pull_errors",
    "cpu_alloc_pct",
    "mem_alloc_pct",
)

_IMAGE_PULL_ERROR_REASONS = frozenset({"ErrImagePull", "ImagePullBackOff"})


# --------------------------------------------------------------------------- #
# cluster-client seam (mirrors fan_manager.fan_manager.CommandRunner /         #
# systems_manager.os_health.CommandRunner)                                     #
# --------------------------------------------------------------------------- #
@runtime_checkable
class ClusterClient(Protocol):
    """Seam for reading raw node/pod state from the local Kubernetes cluster.

    Injecting this client lets callers and tests substitute the cluster read
    without a live API server, matching the DI seam
    ``fan_manager.fan_manager.CommandRunner`` / ``systems_manager.os_health.CommandRunner``
    use for their shell-outs.
    """

    def list_nodes(self) -> list[dict[str, Any]]:
        """Raw per-node state: ``name``/``unschedulable``/``conditions``/``allocatable``."""
        ...

    def list_pods(self) -> list[dict[str, Any]]:
        """Raw per-pod state across all namespaces: ``name``/``namespace``/``node``/
        ``phase``/``reason``/``containers``/``requests_cpu``/``requests_mem``."""
        ...


class KubernetesClusterClient:
    """Default :class:`ClusterClient` backed by the package's existing Kubernetes manager.

    Reuses ``container_manager_mcp.container_manager.create_manager("kubernetes")`` for
    auth/client construction (the same kubeconfig/in-cluster loading every ``cm_k8s_*``
    tool goes through) and reads the raw ``CoreV1Api`` node/pod objects directly — the
    higher-level ``KubernetesManager.list_nodes``/``list_pods`` helpers strip fields
    (conditions, allocatable, container statuses, resource requests) this producer needs.
    """

    def __init__(self, manager: Any | None = None) -> None:
        self._manager = manager

    def _mgr(self) -> Any:
        if self._manager is None:
            from container_manager_mcp.container_manager import create_manager

            self._manager = create_manager("kubernetes")
        return self._manager

    def list_nodes(self) -> list[dict[str, Any]]:
        mgr = self._mgr()
        result: list[dict[str, Any]] = []
        for node in mgr.core.list_node().items:
            conditions = {
                c.type: c.status
                for c in ((node.status.conditions or []) if node.status else [])
            }
            allocatable = (
                dict(node.status.allocatable)
                if node.status and node.status.allocatable
                else {}
            )
            result.append(
                {
                    "name": node.metadata.name,
                    "unschedulable": (
                        bool(node.spec.unschedulable) if node.spec else False
                    ),
                    "conditions": conditions,
                    "allocatable": allocatable,
                }
            )
        return result

    def list_pods(self) -> list[dict[str, Any]]:
        mgr = self._mgr()
        result: list[dict[str, Any]] = []
        for pod in mgr.core.list_pod_for_all_namespaces().items:
            containers: list[dict[str, Any]] = []
            for cs in (pod.status.container_statuses or []) if pod.status else []:
                waiting_reason = None
                if cs.state and cs.state.waiting:
                    waiting_reason = cs.state.waiting.reason
                containers.append(
                    {
                        "restart_count": cs.restart_count or 0,
                        "waiting_reason": waiting_reason,
                    }
                )
            requests_cpu = 0.0
            requests_mem = 0.0
            for container in (pod.spec.containers or []) if pod.spec else []:
                requests = (
                    (container.resources.requests or {}) if container.resources else {}
                )
                requests_cpu += _parse_cpu(requests.get("cpu"))
                requests_mem += _parse_mem(requests.get("memory"))
            result.append(
                {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "node": (pod.spec.node_name or "") if pod.spec else "",
                    "phase": pod.status.phase if pod.status else "Unknown",
                    "reason": pod.status.reason if pod.status else None,
                    "containers": containers,
                    "requests_cpu": requests_cpu,
                    "requests_mem": requests_mem,
                }
            )
        return result


def _default_client() -> ClusterClient:
    return KubernetesClusterClient()


# --------------------------------------------------------------------------- #
# Kubernetes resource-quantity parsing (stdlib only)                           #
# --------------------------------------------------------------------------- #
def _parse_cpu(qty: Any) -> float:
    """Parse a Kubernetes CPU quantity (``"500m"``, ``"2"``, ``"250000n"``) into cores."""
    text = str(qty).strip() if qty is not None else ""
    if not text:
        return 0.0
    try:
        if text.endswith("n"):
            return float(text[:-1]) / 1_000_000_000.0
        if text.endswith("m"):
            return float(text[:-1]) / 1000.0
        return float(text)
    except ValueError:
        return 0.0


_MEM_UNITS: tuple[tuple[str, float], ...] = (
    ("Ei", 1024**6),
    ("Pi", 1024**5),
    ("Ti", 1024**4),
    ("Gi", 1024**3),
    ("Mi", 1024**2),
    ("Ki", 1024),
    ("E", 1000**6),
    ("P", 1000**5),
    ("T", 1000**4),
    ("G", 1000**3),
    ("M", 1000**2),
    ("K", 1000),
)


def _parse_mem(qty: Any) -> float:
    """Parse a Kubernetes memory quantity (``"16336696Ki"``, ``"512Mi"``) into bytes."""
    text = str(qty).strip() if qty is not None else ""
    if not text:
        return 0.0
    for suffix, mult in _MEM_UNITS:
        if text.endswith(suffix):
            try:
                return float(text[: -len(suffix)]) * mult
            except ValueError:
                return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------- #
# sampling — per-node orchestration signals from raw node/pod state            #
# --------------------------------------------------------------------------- #
def collect_orchestration_signals(
    client: ClusterClient | None = None,
) -> dict[str, dict[str, float]]:
    """Sample the LOCAL cluster's current per-node orchestration signals
    (CONCEPT:CM-OS.observability.orch-health-producer).

    Keyed by node name. Best-effort: an unreachable cluster yields ``{}`` rather than
    raising. Signals (see :data:`ORCH_SIGNALS`): ``ready``/``mem_pressure``/
    ``disk_pressure``/``pid_pressure`` (1/0 from node conditions), ``pod_restart_rate``
    (sum of container restart counts on that node), ``pods_pending``/``pods_evicted``,
    ``image_pull_errors`` (``ErrImagePull``/``ImagePullBackOff`` containers), and
    ``cpu_alloc_pct``/``mem_alloc_pct`` (summed pod requests ÷ node allocatable).
    """
    client = client or _default_client()
    try:
        nodes = client.list_nodes()
    except Exception as e:  # noqa: BLE001 — sampling is best-effort
        logger.debug("collect_orchestration_signals: list_nodes failed: %s", e)
        return {}
    try:
        pods = client.list_pods()
    except Exception as e:  # noqa: BLE001 — sampling is best-effort
        logger.debug("collect_orchestration_signals: list_pods failed: %s", e)
        pods = []

    by_node: dict[str, dict[str, float]] = {}
    allocatable_by_node: dict[str, tuple[float, float]] = {}
    for node in nodes:
        name = node.get("name")
        if not name:
            continue
        conditions = node.get("conditions") or {}
        by_node[name] = {
            "ready": 1.0 if conditions.get("Ready") == "True" else 0.0,
            "mem_pressure": 1.0 if conditions.get("MemoryPressure") == "True" else 0.0,
            "disk_pressure": 1.0 if conditions.get("DiskPressure") == "True" else 0.0,
            "pid_pressure": 1.0 if conditions.get("PIDPressure") == "True" else 0.0,
            "pod_restart_rate": 0.0,
            "pods_pending": 0.0,
            "pods_evicted": 0.0,
            "image_pull_errors": 0.0,
            "cpu_alloc_pct": 0.0,
            "mem_alloc_pct": 0.0,
        }
        allocatable = node.get("allocatable") or {}
        allocatable_by_node[name] = (
            _parse_cpu(allocatable.get("cpu")),
            _parse_mem(allocatable.get("memory")),
        )

    requested_by_node: dict[str, tuple[float, float]] = dict.fromkeys(
        by_node, (0.0, 0.0)
    )
    for pod in pods:
        node_name = pod.get("node") or ""
        signals = by_node.get(node_name)
        if not node_name or signals is None:
            continue
        if pod.get("phase") == "Pending":
            signals["pods_pending"] += 1.0
        if pod.get("phase") == "Failed" and (pod.get("reason") or "") == "Evicted":
            signals["pods_evicted"] += 1.0
        for container in pod.get("containers") or []:
            signals["pod_restart_rate"] += float(container.get("restart_count") or 0)
            if (container.get("waiting_reason") or "") in _IMAGE_PULL_ERROR_REASONS:
                signals["image_pull_errors"] += 1.0
        req_cpu, req_mem = requested_by_node[node_name]
        requested_by_node[node_name] = (
            req_cpu + float(pod.get("requests_cpu") or 0.0),
            req_mem + float(pod.get("requests_mem") or 0.0),
        )

    for name, signals in by_node.items():
        alloc_cpu, alloc_mem = allocatable_by_node.get(name, (0.0, 0.0))
        req_cpu, req_mem = requested_by_node.get(name, (0.0, 0.0))
        signals["cpu_alloc_pct"] = (
            round(100.0 * req_cpu / alloc_cpu, 3) if alloc_cpu else 0.0
        )
        signals["mem_alloc_pct"] = (
            round(100.0 * req_mem / alloc_mem, 3) if alloc_mem else 0.0
        )

    return by_node


# --------------------------------------------------------------------------- #
# distill-to-trend — one HealthTrendBuffer per (node, signal); bounded writes  #
# --------------------------------------------------------------------------- #
_BUFFERS: dict[tuple[str, str], Any] = {}


def _buffer_for(node: str, signal: str) -> Any | None:
    """Return the ``(node, signal)``'s rolling :class:`HealthTrendBuffer`, or
    ``None`` when the shared kernel is unavailable."""
    if not _HAS_SHARED_HEALTH:
        return None
    key = (node, signal)
    buf = _BUFFERS.get(key)
    if buf is None:
        window_s = int(os.getenv("CONTAINER_MANAGER_HEALTH_AGGREGATE_S", "3600"))
        buf = HealthTrendBuffer(window_s=window_s)
        _BUFFERS[key] = buf
    return buf


def _health_ingest_enabled() -> bool:
    return os.getenv("CONTAINER_MANAGER_HEALTH_INGEST", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def sample_and_ingest(client: ClusterClient | None = None) -> dict[str, Any]:
    """One collection pass: collect → feed the per-signal trend buffers → ingest
    any flushed trends (CONCEPT:CM-OS.observability.orch-health-producer).

    Idempotent and best-effort: a disabled toggle
    (``CONTAINER_MANAGER_HEALTH_INGEST=false``) or an absent shared-health kernel both
    degrade to collecting signals without writing to the KG — never raises. Bounded by
    design: a ``:HealthTrend`` node is written only when a buffer's aggregate window
    elapses, never per sample.
    """
    signals_by_node = collect_orchestration_signals(client)
    if not _health_ingest_enabled():
        return {
            "nodes": len(signals_by_node),
            "signals": signals_by_node,
            "ingested": False,
            "flushed": [],
        }
    if not _HAS_SHARED_HEALTH:
        return {
            "nodes": len(signals_by_node),
            "signals": signals_by_node,
            "ingested": False,
            "flushed": [],
            "reason": "shared health kernels unavailable",
        }

    flushed: list[dict[str, Any]] = []
    for node, signals in signals_by_node.items():
        entity_id = f"container:k8snode:{node}"
        for signal, value in signals.items():
            buf = _buffer_for(node, signal)
            if buf is None:
                continue
            trend = buf.add(value)
            if trend is None:
                continue
            result = ingest_health_trend(
                entity_id=entity_id,
                entity_type="Node",
                layer="orchestration",
                signal=signal,
                trend=trend,
                host=node,
            )
            flushed.append(
                {
                    "node": node,
                    "signal": signal,
                    "trend": trend,
                    "ingested": bool(result),
                }
            )
            logger.info(
                "orchestration-health trend[%s]: %s avg=%s min=%s max=%s over %d samples",
                node,
                signal,
                trend.get("avg"),
                trend.get("min"),
                trend.get("max"),
                trend.get("samples") or 0,
            )
    return {
        "nodes": len(signals_by_node),
        "signals": signals_by_node,
        "ingested": True,
        "flushed": flushed,
    }


# --------------------------------------------------------------------------- #
# orchestration — one report-only derivation pass over the cluster's nodes     #
# --------------------------------------------------------------------------- #
def _notify(message: str) -> None:
    """Best-effort push to the intelligent alert router
    (``CONTAINER_MANAGER_HEALTH_NOTIFY_URL``), mirroring
    ``fan_manager.kg_control._notify`` / ``systems_manager.os_health._notify``."""
    url = os.getenv("CONTAINER_MANAGER_HEALTH_NOTIFY_URL")
    logger.info(message)
    if not url:
        return
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(
                {"source": "container-manager-health", "message": message}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)  # noqa: S310  # nosec B310 — operator-configured URL
    except Exception as e:  # noqa: BLE001 — notification is best-effort
        logger.debug("notify skipped: %s", e)


def run_orch_derivation(
    days: int = 14, *, client: ClusterClient | None = None
) -> dict[str, Any]:
    """One learn→flag pass over the cluster's current nodes.

    For each node×signal: reads recent ``:HealthTrend`` history, learns a
    ``:HealthBaseline``, and checks the recent tail for a ``:HealthAnomaly`` off that
    baseline. Anomalies simultaneous across a majority of nodes for the same signal are
    collapsed into one ``systemic`` orchestration-level cause (e.g. every node's pod
    restart rate spiking together points at a shared root cause — an image registry
    outage, a bad rollout — not N independent faults). **Report-only** — no
    remediation, no cluster mutation; only writes typed KG nodes and a best-effort
    notification. All KG I/O is best-effort: with no reachable engine every node
    degrades to "no data". With no shared health kernel, or no reachable cluster, this
    is a clean no-op.
    """
    if not _HAS_SHARED_HEALTH:
        logger.info(
            "orchestration-health derivation skipped: shared health kernels unavailable"
        )
        return {"nodes": 0, "results": {}}

    client = client or _default_client()
    try:
        nodes = [n["name"] for n in client.list_nodes() if n.get("name")]
    except Exception as e:  # noqa: BLE001 — derivation is best-effort
        logger.debug("run_orch_derivation: list_nodes failed: %s", e)
        nodes = []
    if not nodes:
        return {"nodes": 0, "results": {}}

    results: dict[str, dict[str, Any]] = {node: {} for node in nodes}
    anomalies_by_signal: dict[str, dict[str, dict[str, Any] | None]] = {
        signal: {} for signal in ORCH_SIGNALS
    }

    for node in nodes:
        entity_id = f"container:k8snode:{node}"
        for signal in ORCH_SIGNALS:
            trends = read_health_trends(entity_id, signal, days=days) or []
            baseline = compute_baseline(trends, value_key="avg", peak_key="max")
            anomaly = detect_anomaly(trends[-3:], baseline, value_key="avg")
            anomalies_by_signal[signal][node] = anomaly
            results[node][signal] = {
                "trends": len(trends),
                "baseline": baseline,
                "anomaly": anomaly,
            }

    for _signal, anomalies in anomalies_by_signal.items():
        correlate(
            anomalies, len(nodes), kind="above-baseline", systemic_kind="systemic"
        )

    for node in nodes:
        entity_id = f"container:k8snode:{node}"
        seen_signals = 0
        for signal in ORCH_SIGNALS:
            data = results[node][signal]
            baseline = data["baseline"]
            if baseline:
                ingest_health_baseline(entity_id, signal, baseline, entity_type="Node")
            anomaly = anomalies_by_signal[signal][node]
            data["anomaly"] = anomaly
            if data["trends"]:
                seen_signals += 1
            if anomaly:
                ingest_health_anomaly(entity_id, signal, anomaly, entity_type="Node")
                _notify(
                    f"[container-manager-health] {node}: {signal} {anomaly['kind']} — "
                    f"observed={anomaly['observed']} expected={anomaly['expected']} "
                    f"(z={anomaly['zscore']})"
                )
        logger.info(
            "%s: %d/%d signals with history, %d anomal%s",
            node,
            seen_signals,
            len(ORCH_SIGNALS),
            sum(1 for s in ORCH_SIGNALS if anomalies_by_signal[s][node]),
            (
                "y"
                if sum(1 for s in ORCH_SIGNALS if anomalies_by_signal[s][node]) == 1
                else "ies"
            ),
        )

    return {"nodes": len(nodes), "results": results}


# --------------------------------------------------------------------------- #
# CLI entry points                                                              #
# --------------------------------------------------------------------------- #
def main_sample() -> None:
    """CLI (``container-manager-health``): one collect+ingest pass; prints a JSON summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    summary = sample_and_ingest()
    print(json.dumps(summary, default=str, indent=2))


def main_derive() -> None:
    """CLI (``container-manager-health-derive``): one derivation pass; prints a JSON summary."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    p = argparse.ArgumentParser(
        description="container-manager-mcp orchestration-health derivation pass."
    )
    p.add_argument(
        "--days", type=int, default=14, help="trend lookback window (default 14)"
    )
    args = p.parse_args()
    summary = run_orch_derivation(days=args.days)
    print(json.dumps(summary, default=str, indent=2))


def main() -> None:
    """CLI: ``python -m container_manager_mcp.orch_health {sample|derive} [options]``
    (mirrors ``systems_manager.os_health.main``); defaults to ``sample``."""
    import argparse

    p = argparse.ArgumentParser(
        description="container-manager-mcp orchestration-health producer."
    )
    sub = p.add_subparsers(dest="command")
    sub.add_parser("sample", help="one collect+ingest pass (default)")
    derive_p = sub.add_parser(
        "derive", help="learn baselines + flag anomalies (report-only)"
    )
    derive_p.add_argument("--days", type=int, default=14)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if args.command == "derive":
        summary = run_orch_derivation(days=args.days)
    else:
        summary = sample_and_ingest()
    print(json.dumps(summary, default=str, indent=2))


if __name__ == "__main__":
    main()

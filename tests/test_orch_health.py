"""Tests for the orchestration-layer health producer
(``container_manager_mcp.orch_health``) — Phase C of the unified infra-intelligence
plan (``reports/unified-infra-intelligence-plan.md``).

Mirrors ``fan_manager/tests/test_kg_control.py`` / ``systems_manager/tests/test_os_health.py``'s
shape: pure sampling via an injected ``ClusterClient`` seam, the distill-not-per-sample
buffer, an end-to-end derivation pass against a fake KG, and the guarded-import no-op
path. Because ``orch_health`` never reimplements the shared kernels (it only *consumes*
``agent_utilities.observability.health``/``health_ingest``), the shared symbols are
monkeypatched directly onto the module for deterministic, environment-independent
coverage of the orchestration (not the kernel maths, which is agent-utilities' own test
surface).
"""

from __future__ import annotations

from types import SimpleNamespace

import container_manager_mcp.orch_health as oh


# --- fakes for the ClusterClient seam ---------------------------------------- #
class _FakeClusterClient:
    def __init__(self, nodes=None, pods=None, *, nodes_error=None, pods_error=None):
        self._nodes = nodes or []
        self._pods = pods or []
        self._nodes_error = nodes_error
        self._pods_error = pods_error

    def list_nodes(self):
        if self._nodes_error:
            raise self._nodes_error
        return self._nodes

    def list_pods(self):
        if self._pods_error:
            raise self._pods_error
        return self._pods


def _node(name, conditions, allocatable=None):
    return {
        "name": name,
        "unschedulable": False,
        "conditions": conditions,
        "allocatable": allocatable or {},
    }


def _pod(
    name,
    node,
    phase,
    *,
    reason=None,
    containers=None,
    requests_cpu=0.0,
    requests_mem=0.0,
):
    return {
        "name": name,
        "namespace": "default",
        "node": node,
        "phase": phase,
        "reason": reason,
        "containers": containers or [],
        "requests_cpu": requests_cpu,
        "requests_mem": requests_mem,
    }


# --- collect_orchestration_signals ------------------------------------------- #
def test_collect_orchestration_signals_reads_injected_seam():
    nodes = [
        _node(
            "r510",
            {
                "Ready": "True",
                "MemoryPressure": "False",
                "DiskPressure": "False",
                "PIDPressure": "False",
            },
            allocatable={"cpu": "4", "memory": "8388608Ki"},
        )
    ]
    pods = [
        _pod(
            "p1",
            "r510",
            "Running",
            containers=[{"restart_count": 2, "waiting_reason": None}],
            requests_cpu=1.0,
            requests_mem=1073741824.0,
        ),
        _pod(
            "p2",
            "r510",
            "Pending",
            containers=[{"restart_count": 0, "waiting_reason": "ImagePullBackOff"}],
        ),
        _pod("p3", "r510", "Failed", reason="Evicted"),
    ]
    signals = oh.collect_orchestration_signals(_FakeClusterClient(nodes, pods))
    assert signals["r510"] == {
        "ready": 1.0,
        "mem_pressure": 0.0,
        "disk_pressure": 0.0,
        "pid_pressure": 0.0,
        "pod_restart_rate": 2.0,
        "pods_pending": 1.0,
        "pods_evicted": 1.0,
        "image_pull_errors": 1.0,
        "cpu_alloc_pct": 25.0,
        "mem_alloc_pct": 12.5,
    }


def test_collect_orchestration_signals_pressure_conditions():
    nodes = [
        _node(
            "r710",
            {
                "Ready": "False",
                "MemoryPressure": "True",
                "DiskPressure": "True",
                "PIDPressure": "True",
            },
        )
    ]
    signals = oh.collect_orchestration_signals(_FakeClusterClient(nodes, []))
    assert signals["r710"]["ready"] == 0.0
    assert signals["r710"]["mem_pressure"] == 1.0
    assert signals["r710"]["disk_pressure"] == 1.0
    assert signals["r710"]["pid_pressure"] == 1.0


def test_collect_orchestration_signals_pods_on_unknown_node_ignored():
    nodes = [_node("r510", {"Ready": "True"})]
    pods = [_pod("orphan", "not-a-node", "Pending")]
    signals = oh.collect_orchestration_signals(_FakeClusterClient(nodes, pods))
    assert signals["r510"]["pods_pending"] == 0.0


def test_collect_orchestration_signals_no_allocatable_yields_zero_pct():
    nodes = [_node("r820", {"Ready": "True"})]
    pods = [_pod("p1", "r820", "Running", requests_cpu=2.0, requests_mem=1024.0)]
    signals = oh.collect_orchestration_signals(_FakeClusterClient(nodes, pods))
    assert signals["r820"]["cpu_alloc_pct"] == 0.0
    assert signals["r820"]["mem_alloc_pct"] == 0.0


def test_collect_orchestration_signals_unreachable_cluster_returns_empty():
    client = _FakeClusterClient(nodes_error=OSError("no cluster"))
    assert oh.collect_orchestration_signals(client) == {}


def test_collect_orchestration_signals_pods_unreachable_still_reports_nodes():
    nodes = [_node("r510", {"Ready": "True"})]
    client = _FakeClusterClient(nodes, pods_error=OSError("metrics down"))
    signals = oh.collect_orchestration_signals(client)
    assert signals["r510"]["pod_restart_rate"] == 0.0


# --- Kubernetes resource-quantity parsing ------------------------------------ #
def test_parse_cpu():
    assert oh._parse_cpu("500m") == 0.5
    assert oh._parse_cpu("2") == 2.0
    assert oh._parse_cpu("250000000n") == 0.25
    assert oh._parse_cpu(None) == 0.0
    assert oh._parse_cpu("") == 0.0
    assert oh._parse_cpu("bogus") == 0.0


def test_parse_mem():
    assert oh._parse_mem("512Mi") == 512 * 1024**2
    assert oh._parse_mem("8388608Ki") == 8388608 * 1024
    assert oh._parse_mem("1G") == 1000**3
    assert oh._parse_mem("2048") == 2048.0
    assert oh._parse_mem(None) == 0.0
    assert oh._parse_mem("bogus") == 0.0


# --- KubernetesClusterClient — reuses the existing cm k8s client -------------- #
class _FakeCore:
    def __init__(self, nodes, pods):
        self._nodes = nodes
        self._pods = pods

    def list_node(self):
        return SimpleNamespace(items=self._nodes)

    def list_pod_for_all_namespaces(self):
        return SimpleNamespace(items=self._pods)


class _FakeManager:
    def __init__(self, nodes, pods):
        self.core = _FakeCore(nodes, pods)


def _raw_node(name, conditions, unschedulable=False, allocatable=None):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        spec=SimpleNamespace(unschedulable=unschedulable),
        status=SimpleNamespace(
            conditions=[
                SimpleNamespace(type=t, status=s) for t, s in conditions.items()
            ],
            allocatable=allocatable or {},
        ),
    )


def _raw_pod(
    name, namespace, node, phase, *, reason=None, containers=None, requests=None
):
    container_statuses = [
        SimpleNamespace(
            restart_count=c.get("restart_count", 0),
            state=SimpleNamespace(
                waiting=(
                    SimpleNamespace(reason=c["waiting_reason"])
                    if c.get("waiting_reason")
                    else None
                )
            ),
        )
        for c in (containers or [])
    ]
    spec_containers = [
        SimpleNamespace(resources=SimpleNamespace(requests=r)) for r in (requests or [])
    ]
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, namespace=namespace),
        status=SimpleNamespace(
            phase=phase, reason=reason, container_statuses=container_statuses
        ),
        spec=SimpleNamespace(node_name=node, containers=spec_containers),
    )


def test_kubernetes_cluster_client_list_nodes():
    nodes = [
        _raw_node(
            "r510",
            {"Ready": "True", "MemoryPressure": "False"},
            allocatable={"cpu": "4", "memory": "8388608Ki"},
        )
    ]
    client = oh.KubernetesClusterClient(manager=_FakeManager(nodes, []))
    assert client.list_nodes() == [
        {
            "name": "r510",
            "unschedulable": False,
            "conditions": {"Ready": "True", "MemoryPressure": "False"},
            "allocatable": {"cpu": "4", "memory": "8388608Ki"},
        }
    ]


def test_kubernetes_cluster_client_list_pods():
    pods = [
        _raw_pod(
            "p1",
            "default",
            "r510",
            "Running",
            containers=[{"restart_count": 2}],
            requests=[{"cpu": "500m", "memory": "256Mi"}],
        )
    ]
    client = oh.KubernetesClusterClient(manager=_FakeManager([], pods))
    result = client.list_pods()
    assert result[0]["node"] == "r510"
    assert result[0]["phase"] == "Running"
    assert result[0]["containers"] == [{"restart_count": 2, "waiting_reason": None}]
    assert result[0]["requests_cpu"] == 0.5
    assert result[0]["requests_mem"] == 256 * 1024**2


def test_kubernetes_cluster_client_list_pods_waiting_reason():
    pods = [
        _raw_pod(
            "p2",
            "default",
            "r710",
            "Pending",
            containers=[{"waiting_reason": "ErrImagePull"}],
        )
    ]
    client = oh.KubernetesClusterClient(manager=_FakeManager([], pods))
    result = client.list_pods()
    assert result[0]["containers"] == [
        {"restart_count": 0, "waiting_reason": "ErrImagePull"}
    ]


# --- distill-to-trend buffer: one trend per window, never per sample --------- #
class _FakeBuffer:
    instances = 0

    def __init__(self, **_kwargs) -> None:
        _FakeBuffer.instances += 1
        self.calls = 0

    def add(self, value, **_kwargs):
        self.calls += 1
        if self.calls < 3:
            return None
        return {
            "min": value,
            "max": value,
            "avg": value,
            "avg_control": None,
            "samples": self.calls,
            "window_s": 3600,
        }


def test_sample_and_ingest_distills_not_per_sample(monkeypatch):
    oh._BUFFERS.clear()
    _FakeBuffer.instances = 0
    monkeypatch.setattr(oh, "_HAS_SHARED_HEALTH", True)
    monkeypatch.setattr(oh, "HealthTrendBuffer", _FakeBuffer, raising=False)
    monkeypatch.setattr(
        oh,
        "collect_orchestration_signals",
        lambda client=None: {"r510": {"pods_pending": 1.0}},
    )
    ingested: list[dict] = []
    monkeypatch.setattr(
        oh,
        "ingest_health_trend",
        lambda **kw: ingested.append(kw) or {"nodes": 1, "edges": 1},
        raising=False,
    )

    for _ in range(3):
        result = oh.sample_and_ingest()

    # only the 3rd pass crossed the buffer's window -> exactly one :HealthTrend write
    assert len(ingested) == 1
    assert ingested[0]["signal"] == "pods_pending"
    assert ingested[0]["entity_id"] == "container:k8snode:r510"
    assert ingested[0]["entity_type"] == "Node"
    assert ingested[0]["layer"] == "orchestration"
    # the buffer is created ONCE per (node, signal) and reused across passes
    assert _FakeBuffer.instances == 1
    assert result["ingested"] is True
    assert result["flushed"][0]["signal"] == "pods_pending"


def test_sample_and_ingest_disabled_by_env(monkeypatch):
    monkeypatch.setenv("CONTAINER_MANAGER_HEALTH_INGEST", "false")
    monkeypatch.setattr(oh, "_HAS_SHARED_HEALTH", True)
    monkeypatch.setattr(
        oh,
        "collect_orchestration_signals",
        lambda client=None: {"r510": {"pods_pending": 1.0}},
    )
    called = []
    monkeypatch.setattr(
        oh, "ingest_health_trend", lambda **kw: called.append(kw), raising=False
    )
    result = oh.sample_and_ingest()
    assert result["ingested"] is False
    assert result["signals"] == {"r510": {"pods_pending": 1.0}}
    assert called == []


# --- guarded-import no-op path (shared kernels absent) ----------------------- #
def test_module_imports_cleanly_regardless_of_shared_health():
    # The guarded import must never raise, whether or not the shared kernels are
    # installed; _HAS_SHARED_HEALTH just reports which path is active.
    assert isinstance(oh._HAS_SHARED_HEALTH, bool)


def test_sample_and_ingest_noop_when_shared_health_absent(monkeypatch):
    monkeypatch.setattr(oh, "_HAS_SHARED_HEALTH", False)
    monkeypatch.setattr(
        oh,
        "collect_orchestration_signals",
        lambda client=None: {"r510": {"pods_pending": 1.0}},
    )
    result = oh.sample_and_ingest()
    assert result["ingested"] is False
    assert result["signals"] == {"r510": {"pods_pending": 1.0}}
    assert result["reason"] == "shared health kernels unavailable"


def test_run_orch_derivation_noop_when_shared_health_absent(monkeypatch):
    monkeypatch.setattr(oh, "_HAS_SHARED_HEALTH", False)
    assert oh.run_orch_derivation() == {"nodes": 0, "results": {}}


def test_run_orch_derivation_no_nodes_discovered(monkeypatch):
    monkeypatch.setattr(oh, "_HAS_SHARED_HEALTH", True)
    client = _FakeClusterClient([], [])
    assert oh.run_orch_derivation(client=client) == {"nodes": 0, "results": {}}


# --- run_orch_derivation: end-to-end against a fake KG ------------------------ #
def test_run_orch_derivation_end_to_end(monkeypatch):
    monkeypatch.setattr(oh, "_HAS_SHARED_HEALTH", True)

    n1_restart_trends = [{"avg": v} for v in (0, 1, 0, 2, 1, 20)]

    def fake_read_health_trends(entity_id, signal, *, days=14):
        node = entity_id.rsplit(":", 1)[-1]
        if node == "n1" and signal == "pod_restart_rate":
            return n1_restart_trends
        return []

    def fake_compute_baseline(trends, *, value_key, min_windows=6, **_kwargs):
        if len(trends) < min_windows:
            return None
        return {
            "p50": 1.0,
            "p95": 2.0,
            "min_env": 0.0,
            "max_env": 20.0,
            "avg_control": None,
            "inertia": None,
            "windows": len(trends),
        }

    def fake_detect_anomaly(recent, baseline, *, value_key, **_kw):
        if not baseline or not recent:
            return None
        observed = recent[-1][value_key]
        if observed > baseline["p95"]:
            return {
                "kind": "above-baseline",
                "zscore": 9.9,
                "observed": observed,
                "expected": baseline["p50"],
            }
        return None

    correlate_calls = []

    def fake_correlate(anomalies, total, **_kwargs):
        correlate_calls.append((dict(anomalies), total))
        return anomalies

    baselines_written: list[tuple] = []
    anomalies_written: list[tuple] = []
    notified: list[str] = []

    monkeypatch.setattr(
        oh, "read_health_trends", fake_read_health_trends, raising=False
    )
    monkeypatch.setattr(oh, "compute_baseline", fake_compute_baseline, raising=False)
    monkeypatch.setattr(oh, "detect_anomaly", fake_detect_anomaly, raising=False)
    monkeypatch.setattr(oh, "correlate", fake_correlate, raising=False)
    monkeypatch.setattr(
        oh,
        "ingest_health_baseline",
        lambda eid, sig, b, **kw: baselines_written.append((eid, sig, b)),
        raising=False,
    )
    monkeypatch.setattr(
        oh,
        "ingest_health_anomaly",
        lambda eid, sig, a, **kw: anomalies_written.append((eid, sig, a)),
        raising=False,
    )
    monkeypatch.setattr(oh, "_notify", lambda msg: notified.append(msg))

    client = _FakeClusterClient([_node("n1", {}), _node("n2", {})], [])
    out = oh.run_orch_derivation(days=14, client=client)

    assert out["nodes"] == 2
    n1_restarts = out["results"]["n1"]["pod_restart_rate"]
    assert n1_restarts["trends"] == 6
    assert n1_restarts["baseline"]["p95"] == 2.0
    assert n1_restarts["anomaly"]["kind"] == "above-baseline"

    n2_restarts = out["results"]["n2"]["pod_restart_rate"]
    assert n2_restarts["trends"] == 0
    assert n2_restarts["baseline"] is None
    assert n2_restarts["anomaly"] is None

    # baseline/anomaly writes only happen where there was real history
    assert (
        "container:k8snode:n1",
        "pod_restart_rate",
        n1_restarts["baseline"],
    ) in baselines_written
    assert not any(eid == "container:k8snode:n2" for eid, _, _ in baselines_written)
    assert (
        "container:k8snode:n1",
        "pod_restart_rate",
        n1_restarts["anomaly"],
    ) in anomalies_written

    # correlate ran once per signal (report-only — never touched the cluster)
    assert len(correlate_calls) == len(oh.ORCH_SIGNALS)
    assert any(msg for msg in notified if "n1" in msg and "pod_restart_rate" in msg)

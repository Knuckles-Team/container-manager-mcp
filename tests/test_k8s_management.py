"""Whole-cluster management tools (CONN-CONTAINER-MANAGER-MCP-R009..R016).

The kubernetes client is mocked; each test drives the KubernetesManager method
the verbose tier serves as ``k8s_<method>``.
"""

import asyncio
import base64
import json
import re
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import rdflib
from fastmcp import Client

import container_manager_mcp.k8s_manager as k8s_mod
from container_manager_mcp.k8s import management
from container_manager_mcp.k8s_manager import KubernetesManager

ROOT = Path(__file__).resolve().parents[1] / "container_manager_mcp"
NS = types.SimpleNamespace


class _ApiException(Exception):
    def __init__(self, status=500):
        super().__init__(status)
        self.status = status


@pytest.fixture(autouse=True)
def _mock_k8s():
    with (
        patch.object(k8s_mod, "k8s_client", MagicMock()),
        patch.object(k8s_mod, "k8s_config", MagicMock()),
        patch.object(k8s_mod, "ApiException", _ApiException),
    ):
        yield


def _manager() -> KubernetesManager:
    manager = KubernetesManager(namespace="default")
    manager.core = MagicMock()
    manager.apps = MagicMock()
    manager.custom_objects = MagicMock()
    return manager


def _obj(body: dict):
    return NS(to_dict=lambda: json.loads(json.dumps(body)))


SECRET = {
    "apiVersion": "v1",
    "kind": "Secret",
    "metadata": {
        "name": "db",
        "namespace": "apps",
        "annotations": {management.LAST_APPLIED: '{"data":{"password":"cGFzcw=="}}'},
    },
    "data": {"password": "cGFzcw=="},
}


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R010")
def test_secret_values_are_redacted_unless_revealed():
    manager = _manager()
    resource = MagicMock(namespaced=True)
    resource.get.return_value = _obj(SECRET)
    manager._resource = MagicMock(return_value=resource)
    body = manager.get_resource("v1", "Secret", "db", "apps")
    assert body["data"] == {"password": management.REDACTED}
    assert (
        body["metadata"]["annotations"][management.LAST_APPLIED] == management.REDACTED
    )
    assert "cGFzcw" not in json.dumps(body)
    assert (
        manager.get_resource("v1", "Secret", "db", "apps", reveal=True)["data"]
        == SECRET["data"]
    )

    manager.core.read_namespaced_secret.return_value = NS(
        data={"password": base64.b64encode(b"pass").decode()},
        type="Opaque",
        metadata=NS(labels={}, owner_references=None),
    )
    hidden = manager.get_secret("db", "apps")
    assert hidden["keys"] == ["password"] and "data" not in hidden
    assert manager.get_secret("db", "apps", reveal=True)["data"] == {"password": "pass"}


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R009")
def test_pods_and_events_list_every_namespace():
    manager = _manager()
    manager.core.list_pod_for_all_namespaces.return_value = NS(items=[])
    manager.core.list_event_for_all_namespaces.return_value = NS(items=[])
    assert manager.list_pods(all_namespaces=True) == []
    assert manager.list_events(all_namespaces=True) == []
    manager.core.list_pod_for_all_namespaces.assert_called_once()
    manager.core.list_event_for_all_namespaces.assert_called_once()
    manager.core.list_namespaced_pod.assert_not_called()


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R011")
def test_rollout_restart_stamps_the_pod_template():
    manager = _manager()
    manager.rollout_restart("deployment", "web", "apps")
    name, ns, body = manager.apps.patch_namespaced_deployment.call_args.args
    assert (name, ns) == ("web", "apps")
    stamp = body["spec"]["template"]["metadata"]["annotations"][
        "kubectl.kubernetes.io/restartedAt"
    ]
    assert re.match(r"\d{4}-\d\d-\d\dT", stamp)
    manager.rollout_restart("sts", "db", "apps")
    manager.apps.patch_namespaced_stateful_set.assert_called_once()


def _rs(revision: int, image: str, uid="dep-uid"):
    template = NS(spec=NS(containers=[NS(image=image)]))
    return NS(
        metadata=NS(
            name=f"web-{revision}",
            annotations={"deployment.kubernetes.io/revision": str(revision)},
            owner_references=[NS(uid=uid)],
            creation_timestamp=None,
        ),
        spec=NS(template=template),
    )


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R011")
def test_rollout_undo_restores_the_previous_replicaset_template():
    manager = _manager()
    dep = NS(
        metadata=NS(
            uid="dep-uid", annotations={"deployment.kubernetes.io/revision": "3"}
        )
    )
    manager.apps.read_namespaced_deployment.return_value = dep
    manager.apps.list_namespaced_replica_set.return_value = NS(
        items=[
            _rs(1, "img:1"),
            _rs(2, "img:2"),
            _rs(3, "img:3"),
            _rs(9, "other", uid="x"),
        ]
    )
    manager.core.api_client.sanitize_for_serialization.side_effect = lambda t: {
        "metadata": {"labels": {"app": "web", "pod-template-hash": "abc"}},
        "spec": {"containers": [{"image": t.spec.containers[0].image}]},
    }
    result = manager.rollout_undo("deployment", "web", "apps")
    assert (result["from_revision"], result["to_revision"]) == (3, 2)
    _, _, patch_body = manager.apps.patch_namespaced_deployment.call_args.args
    assert patch_body[0]["path"] == "/spec/template"
    assert patch_body[0]["value"]["spec"]["containers"][0]["image"] == "img:2"
    assert "pod-template-hash" not in patch_body[0]["value"]["metadata"]["labels"]
    history = manager.rollout_history("deployment", "web", "apps")
    assert [h["revision"] for h in history] == [1, 2, 3] and history[-1]["current"]


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R011")
def test_rollout_status_reports_completion():
    manager = _manager()
    dep = NS(
        metadata=NS(generation=2, annotations={}),
        spec=NS(replicas=2),
        status=NS(
            observed_generation=2,
            updated_replicas=2,
            replicas=3,
            available_replicas=2,
            ready_replicas=2,
        ),
    )
    manager.apps.read_namespaced_deployment.return_value = dep
    assert manager.rollout_status("deployment", "web", "apps")["complete"] is False
    dep.status.replicas = 2
    assert manager.rollout_status("deployment", "web", "apps")["complete"] is True


def _eso(refresh_time, ready="True"):
    return {
        "metadata": {"name": "app", "namespace": "apps"},
        "spec": {
            "target": {"name": "app-secret"},
            "secretStoreRef": {"name": "openbao"},
        },
        "status": {
            "refreshTime": refresh_time,
            "conditions": [{"type": "Ready", "status": ready}],
        },
    }


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R012")
def test_force_sync_waits_for_a_newer_refresh(monkeypatch):
    monkeypatch.setattr(management.time, "sleep", lambda _: None)
    manager = _manager()
    manager.custom_objects.get_namespaced_custom_object.side_effect = [
        _eso("t1"),
        _eso("t1"),
        _eso("t2"),
    ]
    manager.core.read_namespaced_secret.return_value = NS(data={"k": "dg=="})
    result = manager.force_sync_external_secret("app", "apps", timeout_seconds=30)
    assert result["refreshed"] and result["method"] == "annotation"
    body = manager.custom_objects.patch_namespaced_custom_object.call_args.args[-1]
    assert management.FORCE_SYNC in body["metadata"]["annotations"]
    manager.core.delete_namespaced_secret.assert_not_called()


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R012")
def test_force_sync_recreates_the_target_when_asked(monkeypatch):
    monkeypatch.setattr(management.time, "sleep", lambda _: None)
    manager = _manager()
    manager.custom_objects.get_namespaced_custom_object.side_effect = [
        _eso("t1"),
        _eso("t2"),
        _eso("t3"),
    ]
    manager.core.read_namespaced_secret.side_effect = [
        NS(data={"k": "b2xk"}),
        NS(data={"k": "bmV3"}),
    ]
    result = manager.force_sync_external_secret("app", "apps", recreate_target=True)
    manager.core.delete_namespaced_secret.assert_called_once_with("app-secret", "apps")
    assert result["method"] == "annotation+recreate-target"
    assert result["refreshed"] and result["target_changed"] and "data" not in result


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R013")
def test_export_strips_server_fields_and_refuses_secrets():
    import yaml

    manager = _manager()
    live = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "web",
            "namespace": "apps",
            "uid": "u",
            "resourceVersion": "9",
            "managedFields": [{}],
            "annotations": {"deployment.kubernetes.io/revision": "4", "team": "a"},
        },
        "spec": {"replicas": 2},
        "status": {"readyReplicas": 2},
    }
    resource = MagicMock(namespaced=True)
    resource.get.return_value = _obj(live)
    manager._resource = MagicMock(return_value=resource)
    clean = yaml.safe_load(
        manager.export_manifest("apps/v1", "Deployment", "web", "apps")["manifest_yaml"]
    )
    assert "status" not in clean and set(clean["metadata"]) == {
        "name",
        "namespace",
        "annotations",
    }
    assert clean["metadata"]["annotations"] == {"team": "a"}
    resource.get.return_value = _obj(SECRET)
    with pytest.raises(ValueError, match="OpenBao"):
        manager.export_manifest("v1", "Secret", "db", "apps")


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R013")
def test_diff_reports_drift_paths_and_absent_objects():
    manager = _manager()
    live = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "c", "namespace": "apps"},
        "data": {"A": "1"},
    }
    would_be = {**live, "data": {"A": "2"}}
    resource = MagicMock(namespaced=True)
    resource.get.return_value = _obj(live)
    resource.server_side_apply.return_value = _obj(would_be)
    manager._resource = MagicMock(return_value=resource)
    doc = "apiVersion: v1\nkind: ConfigMap\nmetadata: {name: c, namespace: apps}\ndata: {A: '2'}\n"
    resource.create.return_value = _obj(
        {**live, "metadata": {"name": "c-drift-probe", "namespace": "apps"}}
    )
    (entry,) = manager.diff_manifest(doc)
    assert entry["drift"] and entry["paths"] == [".data.A"]
    assert entry["live_only_paths"] == []
    assert (
        resource.create.call_args.kwargs["body"]["metadata"]["name"] == "c-drift-probe"
    )
    assert resource.server_side_apply.call_args.kwargs["dry_run"] == "All"
    resource.server_side_apply.return_value = _obj(live)
    assert manager.diff_manifest(doc)[0]["drift"] is False
    # A live-only field (set by another field manager) is drift too.
    resource.create.return_value = _obj({**live, "data": {}})
    entry = manager.diff_manifest(doc)[0]
    assert entry["drift"] and entry["live_only_paths"] == [".data.A"]
    resource.get.side_effect = _ApiException(404)
    assert manager.diff_manifest(doc)[0]["live"] == "absent"


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R014")
def test_apply_manifest_server_side_applies_each_document():
    manager = _manager()
    resource = MagicMock(namespaced=True)
    resource.server_side_apply.return_value = _obj(
        {"metadata": {"resourceVersion": "7"}}
    )
    manager._resource = MagicMock(return_value=resource)
    out = manager.apply_manifest(
        "apiVersion: v1\nkind: ConfigMap\nmetadata: {name: a}\n---\napiVersion: v1\nkind: ConfigMap\nmetadata: {name: b}\n",
        namespace="zz",
    )
    assert [o["name"] for o in out] == ["a", "b"] and all(
        o["namespace"] == "zz" for o in out
    )
    kwargs = resource.server_side_apply.call_args.kwargs
    assert (
        kwargs["field_manager"] == management.FIELD_MANAGER
        and kwargs["force_conflicts"]
    )


def _served():
    from container_manager_mcp.mcp_server import get_mcp_instance

    async def go():
        server, _, _ = get_mcp_instance()
        async with Client(server) as client:
            return (
                {t.name: t for t in await client.list_tools()},
                {p.name for p in await client.list_prompts()},
                {str(r.uri) for r in await client.list_resources()},
            )

    return asyncio.run(go())


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R015")
def test_served_surface_has_the_management_tools_prompts_skills_and_ontology():
    tools, prompts, resources = _served()
    for name in (
        "k8s_apply_manifest",
        "k8s_delete_resource",
        "k8s_get_resource",
        "k8s_list_resources",
        "k8s_export_manifest",
        "k8s_diff_manifest",
        "k8s_describe_deployment",
        "k8s_scale_deployment",
        "k8s_read_pod_logs",
        "k8s_get_secret",
        "k8s_list_external_secrets",
        "k8s_force_sync_external_secret",
        "k8s_cluster_topology",
    ):
        assert name in tools, name
    assert "reveal" in tools["k8s_get_secret"].inputSchema["properties"]
    assert "confirm" in tools["k8s_delete_resource"].inputSchema["properties"]
    assert {"kubernetes-operator", "platform-reliability-engineer"} <= prompts
    for uri in (
        "k8s://cluster/topology",
        "ontology://container-manager-mcp/surface_contracts.ttl",
        "skill://container-manager-gitops-persist/SKILL.md",
        "skill://container-manager-externalsecret-sync/SKILL.md",
        "skill://container-manager-k8s-change-validation/SKILL.md",
    ):
        assert uri in resources, uri


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R016")
def test_contracts_name_served_tools_prompts_and_skills_and_roll_up_to_core():
    tools, prompts, _ = _served()
    graph = rdflib.Graph()
    graph.parse(ROOT / "ontology/surface_contracts.ttl", format="turtle")
    label = rdflib.RDFS.label
    kg = rdflib.Namespace("http://knuckles.team/kg#")
    named = {
        kind: {
            str(graph.value(s, label))
            for s in graph.subjects(rdflib.RDF.type, kg[kind])
        }
        for kind in ("Tool", "Prompt", "Skill")
    }
    assert named["Tool"] and named["Tool"] <= set(tools)
    assert named["Prompt"] <= prompts
    skills = {p.name for p in (ROOT / "skills").iterdir() if (p / "SKILL.md").exists()}
    assert named["Skill"] <= skills
    for prompt in (ROOT / "prompts").glob("*.json"):
        agent = json.loads(prompt.read_text())["agent"]
        assert set(agent["tools"]) <= set(tools), prompt.name
        assert set(agent["skills"]) <= skills, prompt.name

    onto = rdflib.Graph()
    onto.parse(ROOT / "ontology/container.ttl", format="turtle")
    infra = rdflib.Namespace("http://knuckles.team/kg/infrastructure#")
    for local, parent in (
        ("K8sNode", infra.Node),
        ("Deployment", infra.Deployment),
        ("StatefulSet", infra.Workload),
        ("ExternalSecret", kg.ConfigurationItem),
        ("PersistentVolume", infra.Volume),
        ("CustomResourceDefinition", kg.Schema),
    ):
        assert (kg[local], rdflib.RDFS.subClassOf, parent) in onto, local
    assert not list(onto.objects(None, rdflib.OWL.imports))


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R017")
def test_deployment_describe_scale_and_pod_logs_take_a_namespace():
    manager = _manager()
    container = NS(name="web", image="img:1", resources=None)
    manager.apps.read_namespaced_deployment.return_value = NS(
        metadata=NS(
            annotations={"deployment.kubernetes.io/revision": "2"}, generation=3
        ),
        spec=NS(
            replicas=1,
            paused=False,
            strategy=NS(type="RollingUpdate"),
            selector=NS(match_labels={"app": "web"}),
            template=NS(spec=NS(containers=[container], node_selector=None)),
        ),
        status=NS(
            observed_generation=3,
            ready_replicas=1,
            updated_replicas=1,
            available_replicas=1,
            conditions=[],
        ),
    )
    manager.core.list_namespaced_event.return_value = NS(items=[])
    described = manager.describe_deployment("web", "zz")
    assert (
        described["namespace"] == "zz"
        and described["containers"][0]["image"] == "img:1"
    )
    manager.scale_deployment("web", 3, "zz")
    manager.apps.patch_namespaced_deployment_scale.assert_called_once_with(
        "web", "zz", {"spec": {"replicas": 3}}
    )
    manager.core.read_namespaced_pod_log.return_value = "line"
    logs = manager.read_pod_logs("web-1", "zz", container="web", previous=True)
    assert logs["logs"] == "line"
    assert manager.core.read_namespaced_pod_log.call_args.kwargs["container"] == "web"

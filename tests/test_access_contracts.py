"""The deployment contract binds fields returned by the served tool."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastmcp import Client
from scion_sdk.access_contract import parse_access_contracts

from container_manager_mcp.k8s.base import _K8sBase
from container_manager_mcp.mcp_server import get_mcp_instance

ROOT = Path(__file__).resolve().parents[1]


def _deployment(status="True"):
    container = SimpleNamespace(image="example.invalid/fixture:v1", ports=None)
    template = SimpleNamespace(spec=SimpleNamespace(containers=[container]))
    condition = SimpleNamespace(type="Available", status=status)
    return SimpleNamespace(
        spec=SimpleNamespace(template=template, replicas=2),
        status=SimpleNamespace(conditions=[condition]) if status else None,
        metadata=SimpleNamespace(
            uid="fixture-uid",
            name="fixture-deployment",
            namespace="fixture-namespace",
            creation_timestamp=None,
        ),
    )


class _FixtureManager:
    _ts = staticmethod(_K8sBase._ts)

    def list_services(self):
        return [_K8sBase._deployment_summary(self, _deployment())]


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-R002")
def test_served_deployment_access_contract_matches_fixture():
    contract_path = ROOT / "container_manager_mcp/ontology/access_contracts.ttl"
    (contract,) = [
        c
        for c in parse_access_contracts(contract_path.read_text())
        if c.operation == "k8s_list_services"
    ]
    fixture = json.loads((ROOT / "tests/fixtures/deployment_response.json").read_text())
    assert contract.operation == "k8s_list_services"
    assert contract.key_field == "id"
    assert {field for _, field in contract.predicates} == {"name", "status", "image"}

    async def call():
        server, _, _ = get_mcp_instance()
        with patch(
            "container_manager_mcp.container_manager.create_manager",
            return_value=_FixtureManager(),
        ):
            async with Client(server) as client:
                return await client.call_tool(
                    contract.operation, {}, raise_on_error=False
                )

    result = asyncio.run(call())
    assert not result.is_error
    records = json.loads(result.content[0].text)
    assert records == [fixture]
    assert all(field in records[0] for _, field in contract.predicates)


def test_missing_available_condition_is_unknown():
    assert (
        _K8sBase._deployment_summary(_FixtureManager(), _deployment(None))["status"]
        == "unknown"
    )


class _RecordedManager:
    """Serves one recorded response for every list read."""

    def __init__(self, records):
        self._records = records

    def list_images(self):
        return self._records

    def list_containers(self, all=False):
        return self._records

    def list_pods(self, namespace=None, label_selector=None, all_namespaces=False):
        return self._records


def _contracts():
    text = (ROOT / "container_manager_mcp/ontology/access_contracts.ttl").read_text()
    return {c.entity: c for c in parse_access_contracts(text)}


def _serve(contract, records):
    async def call():
        server, _, _ = get_mcp_instance()
        with patch(
            "container_manager_mcp.container_manager.create_manager",
            return_value=_RecordedManager(records),
        ):
            async with Client(server) as client:
                return await client.call_tool(
                    contract.operation, {}, raise_on_error=False
                )

    result = asyncio.run(call())
    assert not result.is_error
    return json.loads(result.content[0].text)


def _check(entity, class_local, key, fields):
    contract = _contracts()[entity]
    recording = json.loads((ROOT / f"tests/recordings/{entity}.json").read_text())
    assert contract.class_iri.endswith(class_local)
    assert contract.key_field == key
    assert contract.access_kind == "mcp_tool"
    assert contract.pagination
    assert {field for _, field in contract.predicates} == fields
    ontology = (ROOT / "container_manager_mcp/ontology/container.ttl").read_text()
    for prop, _ in contract.predicates:
        local = prop.rsplit("#", 1)[-1]
        assert local == "name" or f":{local} a owl:DatatypeProperty" in ontology
    records = _serve(contract, recording)
    assert records == recording
    for record in records:
        assert key in record
        assert all(field in record for _, field in contract.predicates)


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R002")
def test_competency_contracts_certified():
    _check("image", "ContainerImage", "id", {"repository", "tag"})


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-CONNECTOR-CONTENT-R003")
def test_pod_container_image_contracts():
    _check("pod", "Pod", "name", {"name", "status"})
    _check("container", "Container", "id", {"name", "status"})
    _check("image", "ContainerImage", "id", {"repository", "tag"})

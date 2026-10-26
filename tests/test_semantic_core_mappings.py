"""Approved semantic core mappings and SDK-served ontology payloads."""

from pathlib import Path

import pytest

PACKAGE = "container_manager_mcp"
CONNECTOR = "container-manager-mcp"
ROOT = Path(__file__).resolve().parent.parent / PACKAGE
ONTOLOGY = ROOT / "ontology"
TTL = ONTOLOGY / "container.ttl"
MAPPINGS = (("Ingress", "http://knuckles.team/kg/infrastructure#ProxyRoute"),)


def test_approved_classes_reach_shared_core_parents():
    rdflib = pytest.importorskip("rdflib")
    graph = rdflib.Graph()
    graph.parse(TTL, format="turtle")
    for local_name, parent_iri in MAPPINGS:
        assert (
            rdflib.URIRef(f"http://knuckles.team/kg#{local_name}"),
            rdflib.RDFS.subClassOf,
            rdflib.URIRef(parent_iri),
        ) in graph


def test_ontology_and_shapes_are_served_as_exact_sdk_payloads():
    content = pytest.importorskip("agent_connector_sdk.mcp.content")
    resources = content.ontology_resources(CONNECTOR, ONTOLOGY)
    expected = []
    for path in sorted(ONTOLOGY.glob("*.ttl")):
        expected.append((f"ontology://{CONNECTOR}/{path.name}", path))
    for path in sorted((ONTOLOGY / "shapes").glob("*.ttl")):
        expected.append((f"shapes://{CONNECTOR}/{path.name}", path))
    assert [str(resource.uri) for resource in resources] == [uri for uri, _ in expected]
    for resource, (_, path) in zip(resources, expected, strict=True):
        assert resource.path.read_bytes() == path.read_bytes()

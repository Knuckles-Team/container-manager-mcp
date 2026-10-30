"""Container Manager ontology contribution (CONCEPT:AU-KG.ontology.federation-provider-leg).

Data-only subpackage: it carries ``container.ttl`` (the ``owl:Ontology``
``http://knuckles.team/kg/container`` module) which the scion-sdk hub federates in via
the ``scion_sdk.ontology_providers`` entry-point. It holds no business logic
and no heavy imports so the hub can resolve it cheaply.
"""

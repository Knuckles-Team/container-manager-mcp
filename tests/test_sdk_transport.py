"""Transport through the SDK (CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001)."""

import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.spec("CONN-CONTAINER-MANAGER-MCP-SDK-CENTRALIZATION-R001.3")
def test_dependencies_come_from_the_sdk():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    deps = project["dependencies"]
    assert not any(d.startswith("requests") for d in deps)
    sdk = next(d for d in deps if d.startswith("scion-sdk"))
    assert ">=0.5.9" in sdk

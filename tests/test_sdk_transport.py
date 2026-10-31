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
    from packaging.requirements import Requirement
    from packaging.version import Version

    requirement = Requirement(sdk)
    floor = max(Version(s.version) for s in requirement.specifier if s.operator == ">=")
    assert floor >= Version("0.6.3")
    assert not requirement.specifier.contains("0.7.0")

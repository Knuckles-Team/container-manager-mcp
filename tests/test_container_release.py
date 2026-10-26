"""The released-wheel image refresh uses the shared public payload guard."""

from fnmatch import fnmatchcase
from pathlib import Path

import yaml


def test_public_wheel_preflight_precedes_container_publication():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load(
        (root / ".github/workflows/container-release.yml").read_text()
    )
    jobs = workflow["jobs"]
    steps = jobs["verify-public-wheel"]["steps"]
    guard = next(
        step
        for step in steps
        if step.get("uses", "").endswith("/verify-release-version")
    )
    assert guard["with"] == {
        "mode": "published-wheel",
        "distribution": "container-manager-mcp",
        "module": "container_manager_mcp",
    }
    assert jobs["publish-docker"]["needs"] == "verify-public-wheel"
    assert (
        jobs["publish-docker"]["uses"]
        == "Knuckles-Team/pipelines/.github/workflows/container_pipeline.yml@main"
    )


def test_version_publication_cannot_publish_the_previous_image_pin():
    root = Path(__file__).resolve().parents[1]
    package = yaml.safe_load((root / ".github/workflows/pipeline.yml").read_text())
    image = yaml.safe_load(
        (root / ".github/workflows/container-release.yml").read_text()
    )
    assert set(package["jobs"]) == {"publish-pypi"}
    triggers = image.get("on") or image[True]
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["push"]["paths"] == ["docker/Dockerfile"]


def test_package_publisher_skips_only_image_pin_and_bump_tracking_refresh():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/pipeline.yml").read_text())
    triggers = workflow.get("on") or workflow[True]
    ignored = triggers["push"]["paths-ignore"]
    assert set(ignored) == {"docker/Dockerfile", ".bumpversion.cfg"}
    for pin_path in ("docker/Dockerfile", ".bumpversion.cfg"):
        assert any(fnmatchcase(pin_path, pattern) for pattern in ignored)
    for package_path in (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "container_manager_mcp/mcp_server.py",
        "container_manager_mcp/ontology/container.ttl",
        ".github/workflows/pipeline.yml",
    ):
        assert not any(fnmatchcase(package_path, pattern) for pattern in ignored)

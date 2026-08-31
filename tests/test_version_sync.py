"""Regression checks for package version mirrors and bump configuration."""

import ast
import re
from configparser import ConfigParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEXT_VERSION = "3.1.1"
PYTHON_VERSION_FILES = {
    "container_manager_mcp/agent_server.py",
    "container_manager_mcp/container_manager.py",
    "container_manager_mcp/doctor.py",
    "container_manager_mcp/mcp_server.py",
}
EXPECTED_VERSION_FILES = {
    "README.md",
    "container_manager_mcp/agent_server.py",
    "container_manager_mcp/container_manager.py",
    "container_manager_mcp/doctor.py",
    "container_manager_mcp/mcp_server.py",
    "docker/Dockerfile",
    "pyproject.toml",
}
SECTION_PATH = re.compile(r"^bumpversion:file(?:\([^)]*\))?:(?P<path>.+)$")


def _configured_version_mirrors() -> tuple[str, list[tuple[str, str, str]]]:
    config = ConfigParser()
    config.read(ROOT / ".bumpversion.cfg")
    current_version = config["bumpversion"]["current_version"]
    entries = []
    for section in config.sections():
        match = SECTION_PATH.match(section)
        if match:
            entries.append(
                (
                    match.group("path"),
                    config[section]["search"],
                    config[section]["replace"],
                )
            )
    return current_version, entries


def _module_versions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    versions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            versions.append(node.value.value)
    return versions


def test_patch_bump_covers_every_version_mirror():
    """Every configured mirror has the current value and a valid patch replacement."""
    current_version, entries = _configured_version_mirrors()
    configured_files = {path for path, _, _ in entries}

    assert EXPECTED_VERSION_FILES <= configured_files
    assert "container_manager_mcp/doctor.py" in configured_files

    for path, search_template, replace_template in entries:
        source = (ROOT / path).read_text(encoding="utf-8")
        search = search_template.replace("{current_version}", current_version)
        replacement = replace_template.replace("{new_version}", NEXT_VERSION)

        assert source.count(search) == 1, f"{path} is not synchronized"
        bumped = source.replace(search, replacement)
        assert search not in bumped
        assert bumped.count(replacement) == 1

    for path in PYTHON_VERSION_FILES:
        assert _module_versions(ROOT / path) == [current_version]

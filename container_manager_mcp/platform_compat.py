"""Portable platform helpers: nothing here assumes a POSIX host."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DOCKER_POSIX_ENDPOINT = "unix:///var/run/docker.sock"
DOCKER_WINDOWS_ENDPOINT = "npipe:////./pipe/docker_engine"


def is_windows() -> bool:
    """True on native Windows (``os.name == "nt"``)."""
    return os.name == "nt" or sys.platform == "win32"


def default_docker_endpoint() -> str:
    """The Docker daemon endpoint used when ``DOCKER_HOST`` is unset."""
    return DOCKER_WINDOWS_ENDPOINT if is_windows() else DOCKER_POSIX_ENDPOINT


def docker_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for ``docker.from_env`` with an explicit per-OS default host."""
    env = dict(os.environ if environ is None else environ)
    env.setdefault("DOCKER_HOST", default_docker_endpoint())
    return env


def current_uid() -> int | None:
    """The effective user id, or None where the OS has no such concept."""
    getuid = getattr(os, "getuid", None)
    return getuid() if getuid is not None else None


def home_dir() -> Path:
    """The user's home directory (HOME on POSIX, USERPROFILE on Windows)."""
    return Path.home()


def default_kubeconfig() -> str:
    """``~/.kube/config`` resolved through pathlib."""
    return str(home_dir() / ".kube" / "config")


def restrict_to_owner(path: str | os.PathLike[str], mode: int) -> None:
    """Apply an owner-only POSIX mode; Windows ACLs ignore mode bits, so skip there."""
    if not is_windows():
        os.chmod(path, mode)


def write_private_text(path: Path, text: str) -> None:
    """Create/truncate ``path`` with owner-only permissions where supported."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    if hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)

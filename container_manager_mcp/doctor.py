#!/usr/bin/env python
"""Guided environment doctor for container-manager-mcp.

Diagnoses AND helps resolve the three connection surfaces a user must wire up
before container-manager-mcp is useful:

* **inventory** — the tunnel-manager SSH host inventory (Docker/Podman over SSH),
* **kubernetes** — kubeconfig / contexts (remote clusters),
* **docker / podman** — the local (or remote) container runtime daemons,

plus the **backends** (python client libraries + CLIs) and **config/env** that
select them. Every probe is real — it constructs the same managers the server
uses (``create_manager``), loads the same inventory (``HostManager``), and reaches
the same kubeconfig — and every non-OK check carries concrete remediation so a
user is walked through connecting to their environments.

Exposed as the ``container-manager-doctor`` CLI and the ``cm_doctor`` MCP tool.
"""

import argparse
import importlib.util
import json
import os
import socket
import sys

__version__ = "2.0.1"

# --- Guarded reuse of the server's own managers/inventory --------------------
# These are the real code paths the MCP server uses; importing them here means
# the doctor probes exactly what the server would. All are optional at runtime
# (kubernetes/podman libs may be absent), so imports are guarded and every probe
# degrades gracefully instead of raising.
try:
    from container_manager_mcp.container_manager import create_manager, is_app_installed
except Exception:  # pragma: no cover - only when the package itself is broken
    create_manager = None  # type: ignore[assignment]

    def is_app_installed(app_name: str = "docker") -> bool:  # type: ignore[misc]
        import shutil

        return shutil.which(app_name.lower()) is not None


try:
    from tunnel_manager.tunnel_manager import HostManager, default_inventory_path
except Exception:  # pragma: no cover - tunnel-manager is a hard dependency
    HostManager = None  # type: ignore[assignment]

    def default_inventory_path() -> str:  # type: ignore[misc]
        return os.path.join(
            os.path.expanduser("~/.config"), "agent-utilities", "inventory.yml"
        )


# ---------------------------------------------------------------------------
# Check primitives
# ---------------------------------------------------------------------------
def _check(
    name: str, category: str, status: str, detail: str, remediation: str = ""
) -> dict:
    """Build a single check record.

    ``status`` is one of ``ok`` | ``warn`` | ``fail``.
    """
    return {
        "name": name,
        "category": category,
        "status": status,
        "detail": detail,
        "remediation": remediation,
    }


def _module_available(name: str) -> bool:
    """True if ``name`` is importable, without importing it (cheap + safe)."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _parse_context_config(config_str: str) -> dict[str, str]:
    """Parse a ``"name=value;name2=value2"`` context string.

    Mirrors ``MultiContextManager._parse_context_config`` so the doctor reports
    exactly what the multi-context manager will see for ``K8S_CONTEXTS`` etc.
    """
    contexts: dict[str, str] = {}
    for item in (config_str or "").split(";"):
        if "=" in item:
            name, value = item.split("=", 1)
            contexts[name.strip()] = value.strip()
    return contexts


def _probe_tcp(hostname: str, port: int, timeout: float = 5.0) -> tuple[bool, str]:
    """Real TCP reachability probe for an SSH host (no credentials required)."""
    try:
        with socket.create_connection((hostname, int(port)), timeout=timeout):
            return True, f"TCP {hostname}:{port} reachable"
    except Exception as e:
        return False, f"cannot reach {hostname}:{port}: {e}"


_CLI_INSTALL = {
    "docker": "Install Docker Engine (https://docs.docker.com/engine/install/)",
    "podman": "Install Podman (https://podman.io/docs/installation)",
    "kubectl": "Install kubectl (https://kubernetes.io/docs/tasks/tools/)",
}


# ---------------------------------------------------------------------------
# Category probes — each returns a list[check] and NEVER raises.
# ---------------------------------------------------------------------------
def _check_backends() -> list[dict]:
    """Python client libraries importable + CLIs present on PATH."""
    checks: list[dict] = []
    for mod, extra in (
        ("docker", "docker"),
        ("podman", "podman"),
        ("kubernetes", "kubernetes"),
    ):
        if _module_available(mod):
            checks.append(
                _check(f"{mod} python library", "backends", "ok", f"'{mod}' importable")
            )
        else:
            checks.append(
                _check(
                    f"{mod} python library",
                    "backends",
                    "warn",
                    f"'{mod}' not importable",
                    f"pip install 'container-manager-mcp[{extra}]' to enable the {mod} backend",
                )
            )
    for cli in ("docker", "podman", "kubectl"):
        try:
            present = is_app_installed(cli)
        except Exception:
            present = False
        if present:
            checks.append(
                _check(f"{cli} CLI", "backends", "ok", f"'{cli}' found on PATH")
            )
        else:
            checks.append(
                _check(
                    f"{cli} CLI",
                    "backends",
                    "warn",
                    f"'{cli}' not found on PATH",
                    _CLI_INSTALL.get(cli, f"Install {cli}"),
                )
            )
    return checks


def _check_config() -> list[dict]:
    """CONTAINER_MANAGER_TYPE resolution, tool toggles, and K8S_CONTEXTS parse."""
    checks: list[dict] = []

    cmt = os.environ.get("CONTAINER_MANAGER_TYPE")
    if cmt:
        checks.append(
            _check("CONTAINER_MANAGER_TYPE", "config", "ok", f"set to '{cmt}'")
        )
    else:
        detected = (
            "podman"
            if is_app_installed("podman")
            else ("docker" if is_app_installed("docker") else None)
        )
        if detected:
            checks.append(
                _check(
                    "CONTAINER_MANAGER_TYPE",
                    "config",
                    "warn",
                    f"unset; will auto-detect '{detected}'",
                    "Set CONTAINER_MANAGER_TYPE=docker|podman|kubernetes|multi to make the backend explicit",
                )
            )
        else:
            checks.append(
                _check(
                    "CONTAINER_MANAGER_TYPE",
                    "config",
                    "fail",
                    "unset and neither the docker nor podman CLI is installed",
                    "Install Docker or Podman, or set CONTAINER_MANAGER_TYPE explicitly (e.g. kubernetes)",
                )
            )

    toggles = {k: v for k, v in os.environ.items() if k.endswith("TOOL")}
    disabled = sorted(
        k for k, v in toggles.items() if str(v).lower() not in ("true", "1", "yes")
    )
    detail = (
        f"{len(toggles)} *TOOL toggle(s) set"
        if toggles
        else "no *TOOL toggles set (all default on)"
    )
    if disabled:
        detail += f"; disabled: {', '.join(disabled)}"
    checks.append(_check("tool toggles", "config", "ok", detail))

    raw = os.environ.get("K8S_CONTEXTS", "")
    default_ctx = os.environ.get("DEFAULT_K8S_CONTEXT")
    if not raw:
        checks.append(
            _check(
                "K8S_CONTEXTS",
                "config",
                "ok",
                "not set (single-context / non-kubernetes mode)",
            )
        )
    else:
        parsed = _parse_context_config(raw)
        if not parsed:
            checks.append(
                _check(
                    "K8S_CONTEXTS",
                    "config",
                    "fail",
                    f"could not parse K8S_CONTEXTS={raw!r}",
                    'Use the "name=kubecontext;name2=kubecontext2" format',
                )
            )
        else:
            names = ", ".join(parsed)
            if default_ctx and default_ctx not in parsed:
                checks.append(
                    _check(
                        "K8S_CONTEXTS",
                        "config",
                        "warn",
                        f"parsed {len(parsed)} context(s): {names}; "
                        f"DEFAULT_K8S_CONTEXT='{default_ctx}' is not among them",
                        f"Set DEFAULT_K8S_CONTEXT to one of: {names}",
                    )
                )
            else:
                checks.append(
                    _check(
                        "K8S_CONTEXTS",
                        "config",
                        "ok",
                        f"parsed {len(parsed)} context(s): {names}",
                    )
                )
    return checks


def _check_inventory(
    inventory: str | None = None, host: str | None = None, guided: bool = False
) -> list[dict]:
    """Inventory file exists + parses via HostManager; optional per-host reachability."""
    checks: list[dict] = []
    path = inventory or default_inventory_path()

    if HostManager is None:
        checks.append(
            _check(
                "inventory library",
                "inventory",
                "fail",
                "tunnel_manager is not importable",
                "pip install tunnel-manager",
            )
        )
        return checks

    if not os.path.exists(path):
        checks.append(
            _check(
                "inventory file",
                "inventory",
                "fail",
                f"no inventory file at {path}",
                "Create ~/.config/agent-utilities/inventory.yml (tunnel-manager format): "
                "run `tunnel-manager inventory init` or the ssh-bootstrap skill, then add your hosts",
            )
        )
        return checks

    try:
        hm = HostManager(path)
        hosts = hm.list_hosts()
    except Exception as e:
        checks.append(
            _check(
                "inventory file",
                "inventory",
                "fail",
                f"failed to parse {path}: {e}",
                "Fix the YAML (Ansible-style tree or a flat alias map); see the tunnel-manager inventory schema",
            )
        )
        return checks

    if not hosts:
        checks.append(
            _check(
                "inventory file",
                "inventory",
                "warn",
                f"{path} parsed but defines 0 hosts",
                "Add hosts via `tunnel-manager inventory add <alias> ...` or the ssh-bootstrap skill",
            )
        )
        return checks

    checks.append(
        _check(
            "inventory file",
            "inventory",
            "ok",
            f"{len(hosts)} host(s): {', '.join(sorted(hosts))} ({path})",
        )
    )

    if host:
        targets = [host]
    elif guided:
        targets = sorted(hosts)
    else:
        targets = []

    for alias in targets:
        try:
            hc = hm.get_host(alias)
        except Exception:
            hc = None
        if hc is None:
            checks.append(
                _check(
                    f"host '{alias}'",
                    "inventory",
                    "fail",
                    f"alias '{alias}' not found in inventory",
                    f"Add '{alias}' to {path} or pass a known alias "
                    f"(available: {', '.join(sorted(hosts))})",
                )
            )
            continue
        hostname = getattr(hc, "hostname", None) or alias
        port = getattr(hc, "port", 22) or 22
        ok, detail = _probe_tcp(hostname, port)
        if ok:
            checks.append(_check(f"host '{alias}'", "inventory", "ok", detail))
        else:
            checks.append(
                _check(
                    f"host '{alias}'",
                    "inventory",
                    "fail",
                    detail,
                    f"Host offline or SSH port closed. Confirm the box is up and reachable, "
                    f"then run the ssh-bootstrap skill to establish key-based auth for '{alias}'",
                )
            )
    return checks


def _check_docker(host: str | None = None, focused: bool = False) -> list[dict]:
    """Reach the Docker daemon (local socket or remote host alias) via get_version."""
    bad = "fail" if focused else "warn"
    if create_manager is None:
        return [
            _check(
                "docker daemon",
                "docker",
                bad,
                "container_manager.create_manager is unavailable",
                "pip install 'container-manager-mcp[docker]'",
            )
        ]
    try:
        manager = create_manager("docker", host=host)
        version = manager.get_version()
        v = version.get("version") if isinstance(version, dict) else version
        where = f"remote host '{host}' (Docker over SSH)" if host else "local socket"
        return [
            _check("docker daemon", "docker", "ok", f"Docker {v} reachable via {where}")
        ]
    except Exception as e:
        if host:
            remediation = (
                f"Ensure the tunnel-manager alias '{host}' is reachable over SSH and its "
                "Docker daemon is running (run this doctor with --backend inventory --host "
                f"{host} to test SSH reachability)"
            )
            where = f"remote host '{host}' (via SSH)"
        else:
            remediation = (
                "Start dockerd (`systemctl start docker`), check /var/run/docker.sock "
                "permissions, or set CONTAINER_MANAGER_HOST to a tunnel-manager alias for a remote daemon"
            )
            where = "local /var/run/docker.sock"
        return [
            _check(
                "docker daemon",
                "docker",
                bad,
                f"cannot reach Docker at {where}: {e}",
                remediation,
            )
        ]


def _check_podman(focused: bool = False) -> list[dict]:
    """Reach the Podman service socket via get_version."""
    bad = "fail" if focused else "warn"
    if create_manager is None:
        return [
            _check(
                "podman service",
                "podman",
                bad,
                "container_manager.create_manager is unavailable",
                "pip install 'container-manager-mcp[podman]'",
            )
        ]
    try:
        manager = create_manager("podman")
        version = manager.get_version()
        v = version.get("version") if isinstance(version, dict) else version
        return [_check("podman service", "podman", "ok", f"Podman {v} reachable")]
    except Exception as e:
        return [
            _check(
                "podman service",
                "podman",
                bad,
                f"cannot reach Podman: {e}",
                "Enable the Podman API socket: `systemctl --user enable --now podman.socket` "
                "(or set CONTAINER_MANAGER_PODMAN_BASE_URL to an explicit socket URL)",
            )
        ]


def _check_kubernetes(context: str | None = None, focused: bool = False) -> list[dict]:
    """kubeconfig present + validate + list contexts + probe each target context."""
    checks: list[dict] = []
    bad = "fail" if focused else "warn"

    in_cluster = bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    kubeconfig = os.environ.get("KUBECONFIG") or os.path.expanduser("~/.kube/config")
    present = in_cluster or any(
        os.path.exists(p) for p in kubeconfig.split(os.pathsep) if p
    )
    if not present:
        checks.append(
            _check(
                "kubeconfig",
                "kubernetes",
                bad,
                f"no kubeconfig found (checked {kubeconfig}) and not running in-cluster",
                "Place a kubeconfig at ~/.kube/config or set KUBECONFIG; for multiple clusters set "
                "K8S_CONTEXTS/DEFAULT_K8S_CONTEXT; use the kubernetes-mesh-provisioner skill to stand up an RKE2 cluster",
            )
        )
        return checks
    checks.append(
        _check(
            "kubeconfig",
            "kubernetes",
            "ok",
            (
                "in-cluster service account"
                if in_cluster
                else f"kubeconfig at {kubeconfig}"
            ),
        )
    )

    if create_manager is None:
        checks.append(
            _check(
                "kubernetes client",
                "kubernetes",
                bad,
                "container_manager.create_manager is unavailable",
                "pip install 'container-manager-mcp[kubernetes]'",
            )
        )
        return checks

    parsed = _parse_context_config(os.environ.get("K8S_CONTEXTS", ""))
    if context:
        targets: list[tuple[str, str | None]] = [
            (context, parsed.get(context, context))
        ]
    elif parsed:
        targets = list(parsed.items())
    else:
        targets = [("(current-context)", None)]

    for label, ctx_value in targets:
        try:
            manager = create_manager("kubernetes", host=ctx_value)
        except Exception as e:
            checks.append(
                _check(
                    f"kubernetes context {label}",
                    "kubernetes",
                    "fail",
                    f"cannot construct a client for context '{label}': {e}",
                    "Install the kubernetes client and ensure the context exists in kubeconfig "
                    "(`kubectl config get-contexts`)",
                )
            )
            continue

        try:
            validation = manager.validate_kubeconfig()
            if (
                isinstance(validation, dict)
                and validation.get("status")
                and validation["status"] != "valid"
            ):
                checks.append(
                    _check(
                        f"kubeconfig valid ({label})",
                        "kubernetes",
                        "fail",
                        f"validate_kubeconfig reported: {validation.get('error', 'invalid')}",
                        "Fix the kubeconfig for this context (`kubectl config view`)",
                    )
                )
        except Exception:
            # Non-fatal: the reachability probe below is the authoritative signal.
            pass

        try:
            version = manager.get_version()
            v = version.get("version") if isinstance(version, dict) else version
            try:
                nodes = manager.list_nodes()
                node_detail = f", {len(nodes)} node(s)"
            except Exception as ne:
                node_detail = f" (node list failed: {ne})"
            checks.append(
                _check(
                    f"kubernetes context {label}",
                    "kubernetes",
                    "ok",
                    f"API reachable, server {v}{node_detail}",
                )
            )
        except Exception as e:
            checks.append(
                _check(
                    f"kubernetes context {label}",
                    "kubernetes",
                    "fail",
                    f"cluster unreachable for context '{label}': {e}",
                    f"Verify the cluster is up and context '{label}' points at a reachable API server "
                    f"(`kubectl --context {label} get nodes`); use the kubernetes-mesh-provisioner "
                    "skill to (re)provision RKE2",
                )
            )
    return checks


# ---------------------------------------------------------------------------
# Engine entry point
# ---------------------------------------------------------------------------
def _summarize(checks: list[dict]) -> dict:
    counts = {"ok": 0, "warn": 0, "fail": 0}
    for c in checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    status = "fail" if counts["fail"] else ("warn" if counts["warn"] else "ok")
    return {
        "total": len(checks),
        "ok": counts["ok"],
        "warn": counts["warn"],
        "fail": counts["fail"],
        "status": status,
    }


def run_doctor(
    backend: str = "all",
    host: str | None = None,
    context: str | None = None,
    inventory: str | None = None,
    guided: bool = False,
) -> dict:
    """Run the environment doctor and return a structured diagnostic report.

    ``backend`` selects the surface(s) to probe: ``all`` (everything),
    ``backends`` (libs + CLIs + config only), ``inventory``, ``docker``,
    ``podman``, or ``kubernetes``.

    Returns ``{"backend", "host", "context", "inventory", "checks": [...],
    "summary": {...}}`` where each check is
    ``{"name", "category", "status" (ok|warn|fail), "detail", "remediation"}``.
    Never raises — every probe degrades to a fail/warn check with the message.
    """
    backend = (backend or "all").lower()
    checks: list[dict] = []
    try:
        if backend in ("all", "backends"):
            checks += _check_backends()
            checks += _check_config()
        if backend in ("all", "inventory"):
            checks += _check_inventory(inventory, host, guided)
        if backend in ("all", "docker"):
            checks += _check_docker(host, focused=(backend == "docker"))
        if backend in ("all", "podman"):
            checks += _check_podman(focused=(backend == "podman"))
        if backend in ("all", "kubernetes"):
            checks += _check_kubernetes(context, focused=(backend == "kubernetes"))
    except Exception as e:  # pragma: no cover - defensive; probes are self-guarding
        checks.append(
            _check(
                "doctor",
                "doctor",
                "fail",
                f"unexpected internal error: {e}",
                "This is a bug in the doctor — please report it",
            )
        )

    return {
        "backend": backend,
        "host": host,
        "context": context,
        "inventory": inventory or default_inventory_path(),
        "checks": checks,
        "summary": _summarize(checks),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_COLORS = {"ok": "\033[32m", "warn": "\033[33m", "fail": "\033[31m"}
_SYMBOL = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}
_RESET = "\033[0m"


def _print_report(report: dict, guided: bool = False) -> None:
    use_color = sys.stdout.isatty()
    checks = report["checks"]
    print(f"container-manager-doctor — backend={report['backend']}")
    print("-" * 72)
    for c in checks:
        status = c["status"]
        label = _SYMBOL[status]
        if use_color:
            label = f"{_COLORS[status]}{label}{_RESET}"
        print(f"[{label}] {c['category']:<10} {c['name']}: {c['detail']}")
    summary = report["summary"]
    print("-" * 72)
    print(
        f"Summary: {summary['ok']} OK, {summary['warn']} WARN, {summary['fail']} FAIL "
        f"(status={summary['status']})"
    )

    non_ok = [c for c in checks if c["status"] != "ok" and c.get("remediation")]
    if non_ok:
        print("\nNext steps:")
        for c in non_ok:
            print(f"  - ({c['status'].upper()}) {c['name']}: {c['remediation']}")
            if guided:
                print(f"      detail: {c['detail']}")
    else:
        print("\nAll checks passed — the environment is fully wired.")


def doctor(argv: list[str] | None = None) -> int:
    """``container-manager-doctor`` CLI. Exit 0 if no failing check, else 1."""
    parser = argparse.ArgumentParser(
        prog="container-manager-doctor",
        description=(
            "Diagnose and help resolve the container-manager-mcp environment: the "
            "tunnel-manager SSH inventory, kubeconfig/contexts, and docker/podman runtimes."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["all", "docker", "podman", "kubernetes", "inventory"],
        default="all",
        help="Which surface to diagnose (default: all).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="tunnel-manager host alias to probe (Docker-over-SSH / SSH reachability).",
    )
    parser.add_argument("--context", default=None, help="kubeconfig context to probe.")
    parser.add_argument(
        "--inventory", default=None, help="Path to a tunnel-manager inventory file."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the raw run_doctor dict as JSON."
    )
    parser.add_argument(
        "--guided",
        "-g",
        action="store_true",
        help="Verbose remediation + probe every inventory host.",
    )
    args = parser.parse_args(argv)

    report = run_doctor(
        backend=args.backend,
        host=args.host,
        context=args.context,
        inventory=args.inventory,
        guided=args.guided,
    )
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_report(report, guided=args.guided)
    return 0 if report["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(doctor())

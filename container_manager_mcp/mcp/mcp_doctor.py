"""MCP tool for the container-manager environment doctor.

Wraps :func:`container_manager_mcp.doctor.run_doctor` as the ``cm_doctor`` tool so
an agent can diagnose AND get remediation for the inventory (tunnel-manager SSH
hosts), kubernetes (kubeconfig/contexts), and docker/podman environment
configuration through the same real probes the CLI uses.
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.doctor import run_doctor

_ACTION_BACKEND = {
    "run": None,  # honours the ``backend`` argument (default all)
    "check_backends": "backends",
    "check_inventory": "inventory",
    "check_docker": "docker",
    "check_podman": "podman",
    "check_kubernetes": "kubernetes",
}


def register_doctor_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Container Manager Environment Doctor",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        tags={"doctor"},
    )
    async def cm_doctor(
        action: Literal[
            "run",
            "check_backends",
            "check_inventory",
            "check_docker",
            "check_podman",
            "check_kubernetes",
        ] = Field(
            default="run",
            description=(
                "Diagnostic to run: 'run' (everything, honours 'backend'), or a focused "
                "'check_backends' (libs+CLIs+config) / 'check_inventory' / 'check_docker' / "
                "'check_podman' / 'check_kubernetes'."
            ),
        ),
        host: str | None = Field(
            default=None,
            description=(
                "tunnel-manager host alias to probe (SSH reachability + Docker-over-SSH). "
                "Call cm_list_hosts to see aliases."
            ),
        ),
        context: str | None = Field(
            default=None,
            description="kubeconfig context to probe for the kubernetes checks.",
        ),
        backend: str | None = Field(
            default=None,
            description=(
                "For action='run': restrict to one surface — all|docker|podman|kubernetes|inventory "
                "(default all)."
            ),
        ),
        ctx: Context | None = None,
    ) -> dict:
        """Diagnose + get remediation for the inventory / kubernetes / docker / podman environment.

        Runs real probes (constructs the same managers, loads the same inventory,
        reaches the same kubeconfig the server uses) and returns per-check
        ``{name, category, status: ok|warn|fail, detail, remediation}`` plus a summary.
        """
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_doctor: {action}")

        mapped = _ACTION_BACKEND.get(action)
        resolved_backend = mapped if mapped is not None else (backend or "all")
        try:
            return await run_blocking(
                run_doctor,
                backend=resolved_backend,
                host=host,
                context=context,
            )
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing cm_doctor {action}: {e}")
            return {"error": str(e), "action": action}

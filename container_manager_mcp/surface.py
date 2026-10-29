"""The declared tool surface: operation-tool sources and call-context wiring."""

from typing import Any

from agent_connector_sdk.connector_hooks import RegistrationContext
from agent_connector_sdk.mcp.tool_surface import registered_tools

from container_manager_mcp.call_context import (
    CallContextMiddleware,
    advertise_context_argument,
    manager_for,
)
from container_manager_mcp.container_manager import (
    DockerManager,
    PodmanManager,
    create_manager,
)
from container_manager_mcp.k8s.manager import KubernetesManager

# Tools target the per-call manager from create_manager(host=...), not a
# Depends(get_client) client; each register_*_tools is gated on its <TAG>TOOL
# setting. Every manager class is an operation-tool source for the verbose tier.
VERBOSE_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "client_cls": DockerManager,
        "get_client": lambda: manager_for("docker"),
        "prefix": "docker",
        # ``docker_service_logs`` wraps ``service_logs`` under another
        # argument shape; keep both tools addressable.
        "manifest": [
            {"method": "docker_service_logs", "name": "swarm_service_logs"},
            {"method": "docker_service_ps", "name": "swarm_service_ps"},
        ],
    },
    {
        "client_cls": PodmanManager,
        "get_client": lambda: create_manager("podman"),
        "prefix": "podman",
    },
    {
        "client_cls": KubernetesManager,
        "get_client": lambda: manager_for("kubernetes"),
        "prefix": "k8s",
    },
)


def wire_call_context(context: RegistrationContext) -> None:
    """Advertise the ``context`` argument and bind it per call."""
    advertise_context_argument(registered_tools(context.server))
    context.server.add_middleware(CallContextMiddleware())

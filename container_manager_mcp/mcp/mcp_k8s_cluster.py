"""MCP tools for Kubernetes cluster operations.

Themed dispatcher covering nodes (list/inspect/cordon/drain/taint/affinity/
conditions), kubeconfig contexts, certificate signing requests, API resources,
cluster info, and admission plugins.
"""

import logging
from typing import Literal

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log

_UNHANDLED = object()  # sentinel: this action doesn't belong to this dispatch group


_NODE_BASIC_ACTIONS_NO_NAME = frozenset({"list_nodes", "list_node_taints"})
_NODE_BASIC_ACTIONS_WITH_NAME = frozenset(
    {"inspect_node", "cordon_node", "uncordon_node", "get_node_conditions"}
)


async def _dispatch_node_basic_action(action, manager, node_name):
    """Dispatch a node action whose manager method name equals `action`.

    Every action here is either a no-argument node listing, or a lookup/
    mutation keyed by `node_name` -- so the manager method to call and the
    required-parameter guard are both derivable directly from `action`.
    """
    if action in _NODE_BASIC_ACTIONS_NO_NAME:
        return await run_blocking(getattr(manager, action))
    if action in _NODE_BASIC_ACTIONS_WITH_NAME:
        if not node_name:
            return f"Error: 'node_name' is required for {action}"
        return await run_blocking(getattr(manager, action), node_name)
    return _UNHANDLED


async def _dispatch_node_taint_drain_action(
    action, manager, grace_period_seconds, node_name, taint_key, taints
):
    if action == "drain_node":
        if not node_name:
            return "Error: 'node_name' is required for drain_node"
        return await run_blocking(
            manager.drain_node, node_name, grace_period_seconds or 120
        )
    elif action == "taint_node":
        if not node_name or not taints:
            return "Error: 'node_name' and 'taints' are required for taint_node"
        return await run_blocking(manager.taint_node, node_name, taints)
    elif action == "untaint_node":
        if not node_name or not taint_key:
            return "Error: 'node_name' and 'taint_key' are required for untaint_node"
        return await run_blocking(manager.untaint_node, node_name, taint_key)
    return _UNHANDLED


_NODE_AFFINITY_ACTIONS = {
    "set_node_affinity": (
        ("pod_name", "namespace", "affinity"),
        "'pod_name', 'namespace', and 'affinity' are required for set_node_affinity",
    ),
    "get_node_affinity": (
        ("pod_name", "namespace"),
        "'pod_name' and 'namespace' are required for get_node_affinity",
    ),
    "set_pod_anti_affinity": (
        ("pod_name", "namespace", "anti_affinity"),
        "'pod_name', 'namespace', and 'anti_affinity' are required for set_pod_anti_affinity",
    ),
}


async def _dispatch_node_affinity_action(
    action, manager, affinity, anti_affinity, namespace, pod_name
):
    """Dispatch a node/pod affinity action via `_NODE_AFFINITY_ACTIONS`.

    Each entry's manager method name equals `action`; only the required
    fields and their pinned error message differ per action.
    """
    spec = _NODE_AFFINITY_ACTIONS.get(action)
    if spec is None:
        return _UNHANDLED
    required, error_message = spec
    values = {
        "pod_name": pod_name,
        "namespace": namespace,
        "affinity": affinity,
        "anti_affinity": anti_affinity,
    }
    if not all(values[name] for name in required):
        return f"Error: {error_message}"
    return await run_blocking(
        getattr(manager, action), *(values[name] for name in required)
    )


async def _dispatch_context_action(action, manager, context_name, new_context_name):
    if action == "list_contexts":
        return await run_blocking(manager.list_contexts)
    elif action == "use_context":
        if not context_name:
            return "Error: 'context_name' is required for use_context"
        return await run_blocking(manager.use_context, context_name=context_name)
    elif action == "get_config":
        return await run_blocking(manager.get_config)
    elif action == "rename_context":
        if not context_name or not new_context_name:
            return "Error: 'context_name' and 'new_context_name' are required for rename_context"
        return await run_blocking(
            manager.rename_context,
            current_name=context_name,
            new_name=new_context_name,
        )
    elif action == "validate_kubeconfig":
        return await run_blocking(manager.validate_kubeconfig)
    return _UNHANDLED


async def _dispatch_csr_action(action, manager, csr_name, reason):
    if action == "list_csr":
        return await run_blocking(manager.list_certificate_signing_requests)
    elif action == "approve_csr":
        if not csr_name:
            return "Error: 'csr_name' is required for approve_csr"
        return await run_blocking(manager.approve_csr, csr_name)
    elif action == "deny_csr":
        if not csr_name:
            return "Error: 'csr_name' is required for deny_csr"
        if reason:
            return await run_blocking(manager.deny_csr, csr_name, reason)
        return await run_blocking(manager.deny_csr, csr_name)
    return _UNHANDLED


async def _dispatch_api_resource_action(action, manager, name):
    if action == "list_api_resources":
        return await run_blocking(manager.list_api_resources)
    elif action == "describe_api_resource":
        if not name:
            return "Error: 'name' is required for describe_api_resource"
        return await run_blocking(manager.describe_api_resource, name)
    return _UNHANDLED


async def _dispatch_cluster_info_action(action, manager, output_dir):
    if action == "cluster_info_dump":
        if not output_dir:
            return "Error: 'output_dir' is required for cluster_info_dump"
        return await run_blocking(manager.cluster_info_dump, output_dir)
    elif action == "get_cluster_info":
        return await run_blocking(manager.get_cluster_info)
    elif action == "get_api_server_info":
        return await run_blocking(manager.get_api_server_info)
    return _UNHANDLED


async def _dispatch_admission_plugin_action(
    action, manager, name, plugin_type, test_resource
):
    if action == "list_cluster_plugins":
        return await run_blocking(manager.list_cluster_plugins)
    elif action == "describe_cluster_plugin":
        if not name or not plugin_type:
            return "Error: 'name' and 'plugin_type' are required for describe_cluster_plugin"
        return await run_blocking(manager.describe_cluster_plugin, name, plugin_type)
    elif action == "test_cluster_plugin":
        if not name or not plugin_type or not test_resource:
            return "Error: 'name', 'plugin_type', and 'test_resource' are required for test_cluster_plugin"
        return await run_blocking(
            manager.test_cluster_plugin, name, plugin_type, test_resource
        )
    return _UNHANDLED


_ACTION_GROUPS: dict[str, str] = {
    "list_nodes": "node_basic",
    "inspect_node": "node_basic",
    "cordon_node": "node_basic",
    "uncordon_node": "node_basic",
    "get_node_conditions": "node_basic",
    "list_node_taints": "node_basic",
    "drain_node": "node_taint_drain",
    "taint_node": "node_taint_drain",
    "untaint_node": "node_taint_drain",
    "set_node_affinity": "node_affinity",
    "get_node_affinity": "node_affinity",
    "set_pod_anti_affinity": "node_affinity",
    "list_contexts": "context",
    "use_context": "context",
    "get_config": "context",
    "rename_context": "context",
    "validate_kubeconfig": "context",
    "list_csr": "csr",
    "approve_csr": "csr",
    "deny_csr": "csr",
    "list_api_resources": "api_resource",
    "describe_api_resource": "api_resource",
    "cluster_info_dump": "cluster_info",
    "get_cluster_info": "cluster_info",
    "get_api_server_info": "cluster_info",
    "list_cluster_plugins": "admission_plugin",
    "describe_cluster_plugin": "admission_plugin",
    "test_cluster_plugin": "admission_plugin",
}

_GROUP_FUNCS = {
    "node_basic": _dispatch_node_basic_action,
    "node_taint_drain": _dispatch_node_taint_drain_action,
    "node_affinity": _dispatch_node_affinity_action,
    "context": _dispatch_context_action,
    "csr": _dispatch_csr_action,
    "api_resource": _dispatch_api_resource_action,
    "cluster_info": _dispatch_cluster_info_action,
    "admission_plugin": _dispatch_admission_plugin_action,
}

_GROUP_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "node_basic": ("node_name",),
    "node_taint_drain": (
        "grace_period_seconds",
        "node_name",
        "taint_key",
        "taints",
    ),
    "node_affinity": (
        "affinity",
        "anti_affinity",
        "namespace",
        "pod_name",
    ),
    "context": (
        "context_name",
        "new_context_name",
    ),
    "csr": (
        "csr_name",
        "reason",
    ),
    "api_resource": ("name",),
    "cluster_info": ("output_dir",),
    "admission_plugin": (
        "name",
        "plugin_type",
        "test_resource",
    ),
}


def register_k8scluster_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Cluster Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "cluster"},
    )
    async def cm_k8s_cluster(
        action: Literal[
            # Nodes
            "list_nodes",
            "inspect_node",
            "cordon_node",
            "uncordon_node",
            "drain_node",
            "get_node_conditions",
            "taint_node",
            "untaint_node",
            "list_node_taints",
            "set_node_affinity",
            "get_node_affinity",
            "set_pod_anti_affinity",
            # Contexts
            "list_contexts",
            "use_context",
            "get_config",
            "rename_context",
            "validate_kubeconfig",
            "save_context",
            # Certificate signing requests
            "list_csr",
            "approve_csr",
            "deny_csr",
            # API resources
            "list_api_resources",
            "describe_api_resource",
            # Cluster info
            "cluster_info_dump",
            "get_cluster_info",
            "get_api_server_info",
            # Admission plugins
            "list_cluster_plugins",
            "describe_cluster_plugin",
            "test_cluster_plugin",
        ] = Field(
            description="Cluster action to perform (nodes, contexts, CSRs, API resources, cluster info, admission plugins)."
        ),
        node_name: str | None = Field(
            default=None, description="Node name for node operations"
        ),
        namespace: str | None = Field(
            default=None, description="Target namespace for affinity operations"
        ),
        pod_name: str | None = Field(
            default=None, description="Pod name for affinity operations"
        ),
        taints: list | None = Field(default=None, description="Taints for taint_node"),
        taint_key: str | None = Field(
            default=None, description="Taint key for untaint_node"
        ),
        affinity: dict | None = Field(
            default=None, description="Affinity configuration for set_node_affinity"
        ),
        anti_affinity: dict | None = Field(
            default=None,
            description="Anti-affinity configuration for set_pod_anti_affinity",
        ),
        grace_period_seconds: int | None = Field(
            default=None, description="Grace period for drain (default: 120)"
        ),
        context_name: str | None = Field(default=None, description="Context name"),
        new_context_name: str | None = Field(
            default=None, description="New context name for rename_context"
        ),
        csr_name: str | None = Field(
            default=None, description="CertificateSigningRequest name"
        ),
        reason: str | None = Field(default=None, description="Reason for deny_csr"),
        name: str | None = Field(
            default=None, description="Resource name (API resource, plugin)"
        ),
        output_dir: str | None = Field(
            default=None, description="Output directory for cluster_info_dump"
        ),
        plugin_type: str | None = Field(
            default=None, description="Plugin type (validating/mutating)"
        ),
        test_resource: dict | None = Field(
            default=None, description="Test resource for test_cluster_plugin"
        ),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        # --- save_context: register a kube environment into the kubeconfig cm reads ---
        server: str | None = Field(
            default=None,
            description="save_context (explicit): API server URL, e.g. https://10.0.0.10:6443",
        ),
        token: str | None = Field(
            default=None,
            description="save_context (explicit): bearer token for authentication",
        ),
        client_cert: str | None = Field(
            default=None,
            description="save_context (explicit): client cert as a file path or PEM text",
        ),
        client_key: str | None = Field(
            default=None,
            description="save_context (explicit): client key as a file path or PEM text",
        ),
        ca_cert: str | None = Field(
            default=None,
            description="save_context (explicit): cluster CA cert as a file path or PEM text",
        ),
        insecure_skip_tls_verify: bool = Field(
            default=False,
            description="save_context (explicit): skip TLS verification instead of pinning a CA",
        ),
        username: str | None = Field(
            default=None,
            description="save_context (oidc): username for an OIDC-backed cluster (requires oidc_issuer + oidc_client_id; the API server has no basic auth)",
        ),
        password: str | None = Field(
            default=None,
            description="save_context (oidc): password for the OIDC resource-owner password grant",
        ),
        oidc_issuer: str | None = Field(
            default=None,
            description="save_context (oidc): OIDC issuer URL, e.g. https://keycloak/realms/<realm>",
        ),
        oidc_client_id: str | None = Field(
            default=None,
            description="save_context (oidc): OIDC client id used for the login",
        ),
        oidc_client_secret: str | None = Field(
            default=None,
            description="save_context (oidc): OIDC client secret for confidential clients (optional)",
        ),
        source_file: str | None = Field(
            default=None,
            description="save_context (import): path to an existing kubeconfig file to merge",
        ),
        source_yaml: str | None = Field(
            default=None,
            description="save_context (import): raw kubeconfig YAML to merge",
        ),
        capture_current: bool = Field(
            default=False,
            description="save_context (capture): export the cluster cm is currently on and save it under 'context_name'",
        ),
        kubeconfig_path: str | None = Field(
            default=None,
            description="save_context: target kubeconfig to write (default: $KUBECONFIG first entry, else ~/.kube/config)",
        ),
        overwrite: bool = Field(
            default=False,
            description="save_context: replace an existing context on name collision (default: error)",
        ),
        use: bool = Field(
            default=False,
            description="save_context: set current-context to the saved context",
        ),
        validate: bool = Field(
            default=True,
            description="save_context: after saving, load the context and list_nodes to validate reachability",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """Manage Kubernetes cluster resources (nodes, contexts, CSRs, API resources, cluster info, admission plugins).

        The ``save_context`` action registers a NEW Kubernetes environment into
        the kubeconfig container-manager-mcp reads so it can be reused later
        (via ``use_context`` / ``CONTAINER_MANAGER_KUBECONTEXT`` / ``K8S_CONTEXTS``).
        Four input modes, non-destructive (a context-name collision errors unless
        ``overwrite``): **token** (``context_name`` + ``server`` + ``token`` +
        ``ca_cert`` or ``insecure_skip_tls_verify``), **mTLS** (``context_name`` +
        ``server`` + ``client_cert`` + ``client_key`` + ``ca_cert``), **oidc**
        (``context_name`` + ``server`` + ``username`` + ``password`` +
        ``oidc_issuer`` + ``oidc_client_id`` — drives an OIDC password-grant login
        and embeds the id-token; the API server has no basic auth, so this REQUIRES
        the OIDC params and fails clearly otherwise), and **import** (``source_file``/
        ``source_yaml`` to merge an existing kubeconfig, or ``capture_current=True``
        to export the cluster you are currently on). ``namespace`` is optional. With
        ``use`` it becomes current-context; with ``validate`` the saved context is
        loaded and node count returned.
        """
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_cluster: {action}")

        # save_context is handled WITHOUT constructing a manager: registering a
        # new environment must work even when no cluster is yet reachable (that
        # is precisely what this action bootstraps).
        if action == "save_context":
            from container_manager_mcp.k8s.kubeconfig_import import save_kube_context

            try:
                return await run_blocking(
                    save_kube_context,
                    name=context_name,
                    kubeconfig_path=kubeconfig_path,
                    source_file=source_file,
                    source_yaml=source_yaml,
                    server=server,
                    token=token,
                    client_cert=client_cert,
                    client_key=client_key,
                    ca_cert=ca_cert,
                    insecure_skip_tls_verify=insecure_skip_tls_verify,
                    namespace=namespace,
                    username=username,
                    password=password,
                    oidc_issuer=oidc_issuer,
                    oidc_client_id=oidc_client_id,
                    oidc_client_secret=oidc_client_secret,
                    capture_current=capture_current,
                    overwrite=overwrite,
                    use=use,
                    validate=validate,
                )
            except Exception as e:
                if ctx:
                    ctx_log(ctx, logging.ERROR, f"Error executing save_context: {e}")
                return f"Error executing save_context: {e}"

        manager = create_manager(manager_type or "kubernetes")

        try:
            group = _ACTION_GROUPS.get(action)
            if group is None:
                return f"Error: Unknown action '{action}'"
            all_values = {
                "affinity": affinity,
                "anti_affinity": anti_affinity,
                "context_name": context_name,
                "csr_name": csr_name,
                "grace_period_seconds": grace_period_seconds,
                "name": name,
                "namespace": namespace,
                "new_context_name": new_context_name,
                "node_name": node_name,
                "output_dir": output_dir,
                "plugin_type": plugin_type,
                "pod_name": pod_name,
                "reason": reason,
                "taint_key": taint_key,
                "taints": taints,
                "test_resource": test_resource,
            }
            group_kwargs = {n: all_values[n] for n in _GROUP_PARAM_NAMES[group]}
            return await _GROUP_FUNCS[group](action, manager, **group_kwargs)
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"

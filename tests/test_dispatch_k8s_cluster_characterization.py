"""Characterization tests for CXA-FL-CONTAINERMANAGERMCP-01's
register_k8scluster_tools.cm_k8s_cluster (CCN 67), before decomposition.

cm_k8s_cluster is a 29-action dispatcher, structurally in two parts:

1. `save_context` is handled specially, BEFORE `manager = create_manager(...)`
   is even called (registering a new kubeconfig context must work when no
   cluster is reachable yet -- that's the point of the action). It has its
   own try/except and calls `save_kube_context` directly, not a
   `manager.<method>`.
2. The other 28 actions (nodes, contexts, CSRs, API resources, cluster info,
   admission plugins) are a standard `if action == "x": ... elif ...`
   dispatcher against `manager`, wrapped in one shared try/except.

This file pins: for save_context, that it calls `save_kube_context` with
every kwarg forwarded and returns its result without ever calling
create_manager; for the other 28: which manager.<method> each calls and
with exactly which args (including deny_csr's optional trailing `reason`
arg, included only when truthy -- not an `is not None` guard), each
required-param guard's exact error string with the manager left uncalled,
the unknown-action fallback, and manager-exception formatting.

Harness pattern (`_capture_tool`) matches the repo's own precedent in
tests/test_multi_context_manager.py. Guard tests pass the guarded
parameter(s) explicitly as `None` (see other dispatch characterization
files in this lane for why: the bare function's real defaults are FastMCP
FieldInfo objects, not None).
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from container_manager_mcp.mcp import mcp_k8s_cluster


@pytest.fixture(autouse=True)
def _restore_create_manager():
    original = mcp_k8s_cluster.create_manager
    yield
    mcp_k8s_cluster.create_manager = original


def _capture_tool(register_fn):
    captured = {}

    def tool_decorator(*args, **kwargs):
        def wrapper(fn):
            captured["fn"] = fn
            return fn

        return wrapper

    fake_mcp = MagicMock()
    fake_mcp.tool = tool_decorator
    register_fn(fake_mcp)
    return captured["fn"]


def _make_tool():
    manager = MagicMock()
    mcp_k8s_cluster.create_manager = lambda manager_type=None: manager
    tool = _capture_tool(mcp_k8s_cluster.register_k8scluster_tools)
    return manager, tool


def test_cm_k8s_cluster_save_context_dispatches_to_save_kube_context_without_creating_a_manager():
    manager, tool = _make_tool()
    create_manager_calls = []
    original_create_manager = mcp_k8s_cluster.create_manager

    def _tracking_create_manager(manager_type=None):
        create_manager_calls.append(manager_type)
        return original_create_manager(manager_type)

    mcp_k8s_cluster.create_manager = _tracking_create_manager

    from container_manager_mcp.k8s import kubeconfig_import

    fake_save = MagicMock(return_value={"ok": True})
    original_save = kubeconfig_import.save_kube_context
    kubeconfig_import.save_kube_context = fake_save
    try:
        result = asyncio.run(
            tool(
                action="save_context",
                context_name="ctx1",
                kubeconfig_path="/tmp/kubeconfig",
                source_file=None,
                source_yaml=None,
                server="https://10.0.0.10:6443",
                token="tok",
                client_cert=None,
                client_key=None,
                ca_cert="ca-pem",
                insecure_skip_tls_verify=False,
                namespace=None,
                username=None,
                password=None,
                oidc_issuer=None,
                oidc_client_id=None,
                oidc_client_secret=None,
                capture_current=False,
                overwrite=False,
                use=False,
                validate=True,
                ctx=None,
            )
        )
    finally:
        kubeconfig_import.save_kube_context = original_save

    assert result == {"ok": True}
    fake_save.assert_called_once_with(
        name="ctx1",
        kubeconfig_path="/tmp/kubeconfig",
        source_file=None,
        source_yaml=None,
        server="https://10.0.0.10:6443",
        token="tok",
        client_cert=None,
        client_key=None,
        ca_cert="ca-pem",
        insecure_skip_tls_verify=False,
        namespace=None,
        username=None,
        password=None,
        oidc_issuer=None,
        oidc_client_id=None,
        oidc_client_secret=None,
        capture_current=False,
        overwrite=False,
        use=False,
        validate=True,
    )
    assert create_manager_calls == [], (
        "save_context must never construct a manager -- it must work when no "
        "cluster is reachable yet"
    )


def test_cm_k8s_cluster_save_context_exception_is_caught_and_formatted():
    from container_manager_mcp.k8s import kubeconfig_import

    fake_save = MagicMock(side_effect=RuntimeError("boom"))
    original_save = kubeconfig_import.save_kube_context
    kubeconfig_import.save_kube_context = fake_save
    manager, tool = _make_tool()
    try:
        result = asyncio.run(
            tool(
                action="save_context",
                context_name="ctx1",
                ctx=None,
            )
        )
    finally:
        kubeconfig_import.save_kube_context = original_save
    assert result == "Error executing save_context: boom"


def test_cm_k8s_cluster_list_nodes_dispatches_to_list_nodes():
    manager, tool = _make_tool()
    manager.list_nodes.return_value = "SENTINEL_RESULT_list_nodes"
    result = asyncio.run(
        tool(
            action="list_nodes",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_nodes"
    manager.list_nodes.assert_called_once_with()


def test_cm_k8s_cluster_inspect_node_dispatches_to_inspect_node():
    manager, tool = _make_tool()
    manager.inspect_node.return_value = "SENTINEL_RESULT_inspect_node"
    result = asyncio.run(
        tool(
            action="inspect_node",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_inspect_node"
    manager.inspect_node.assert_called_once_with("node_name_val")


def test_cm_k8s_cluster_inspect_node_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="inspect_node",
            node_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'node_name' is required for inspect_node"
    manager.inspect_node.assert_not_called()


def test_cm_k8s_cluster_cordon_node_dispatches_to_cordon_node():
    manager, tool = _make_tool()
    manager.cordon_node.return_value = "SENTINEL_RESULT_cordon_node"
    result = asyncio.run(
        tool(
            action="cordon_node",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_cordon_node"
    manager.cordon_node.assert_called_once_with("node_name_val")


def test_cm_k8s_cluster_cordon_node_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="cordon_node",
            node_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'node_name' is required for cordon_node"
    manager.cordon_node.assert_not_called()


def test_cm_k8s_cluster_uncordon_node_dispatches_to_uncordon_node():
    manager, tool = _make_tool()
    manager.uncordon_node.return_value = "SENTINEL_RESULT_uncordon_node"
    result = asyncio.run(
        tool(
            action="uncordon_node",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_uncordon_node"
    manager.uncordon_node.assert_called_once_with("node_name_val")


def test_cm_k8s_cluster_uncordon_node_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="uncordon_node",
            node_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'node_name' is required for uncordon_node"
    manager.uncordon_node.assert_not_called()


def test_cm_k8s_cluster_drain_node_dispatches_to_drain_node():
    manager, tool = _make_tool()
    manager.drain_node.return_value = "SENTINEL_RESULT_drain_node"
    result = asyncio.run(
        tool(
            action="drain_node",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_drain_node"
    manager.drain_node.assert_called_once_with("node_name_val", 7)


def test_cm_k8s_cluster_drain_node_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="drain_node",
            node_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'node_name' is required for drain_node"
    manager.drain_node.assert_not_called()


def test_cm_k8s_cluster_get_node_conditions_dispatches_to_get_node_conditions():
    manager, tool = _make_tool()
    manager.get_node_conditions.return_value = "SENTINEL_RESULT_get_node_conditions"
    result = asyncio.run(
        tool(
            action="get_node_conditions",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_get_node_conditions"
    manager.get_node_conditions.assert_called_once_with("node_name_val")


def test_cm_k8s_cluster_get_node_conditions_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="get_node_conditions",
            node_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'node_name' is required for get_node_conditions"
    manager.get_node_conditions.assert_not_called()


def test_cm_k8s_cluster_taint_node_dispatches_to_taint_node():
    manager, tool = _make_tool()
    manager.taint_node.return_value = "SENTINEL_RESULT_taint_node"
    result = asyncio.run(
        tool(
            action="taint_node",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_taint_node"
    manager.taint_node.assert_called_once_with("node_name_val", ["taints_item"])


def test_cm_k8s_cluster_taint_node_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="taint_node",
            node_name=None,
            taints=None,
            ctx=None,
        )
    )
    assert result == "Error: 'node_name' and 'taints' are required for taint_node"
    manager.taint_node.assert_not_called()


def test_cm_k8s_cluster_untaint_node_dispatches_to_untaint_node():
    manager, tool = _make_tool()
    manager.untaint_node.return_value = "SENTINEL_RESULT_untaint_node"
    result = asyncio.run(
        tool(
            action="untaint_node",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_untaint_node"
    manager.untaint_node.assert_called_once_with("node_name_val", "taint_key_val")


def test_cm_k8s_cluster_untaint_node_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="untaint_node",
            node_name=None,
            taint_key=None,
            ctx=None,
        )
    )
    assert result == "Error: 'node_name' and 'taint_key' are required for untaint_node"
    manager.untaint_node.assert_not_called()


def test_cm_k8s_cluster_list_node_taints_dispatches_to_list_node_taints():
    manager, tool = _make_tool()
    manager.list_node_taints.return_value = "SENTINEL_RESULT_list_node_taints"
    result = asyncio.run(
        tool(
            action="list_node_taints",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_node_taints"
    manager.list_node_taints.assert_called_once_with()


def test_cm_k8s_cluster_set_node_affinity_dispatches_to_set_node_affinity():
    manager, tool = _make_tool()
    manager.set_node_affinity.return_value = "SENTINEL_RESULT_set_node_affinity"
    result = asyncio.run(
        tool(
            action="set_node_affinity",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_set_node_affinity"
    manager.set_node_affinity.assert_called_once_with(
        "pod_name_val", "namespace_val", {"affinity_k": "affinity_v"}
    )


def test_cm_k8s_cluster_set_node_affinity_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="set_node_affinity",
            affinity=None,
            namespace=None,
            pod_name=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'pod_name', 'namespace', and 'affinity' are required for set_node_affinity"
    )
    manager.set_node_affinity.assert_not_called()


def test_cm_k8s_cluster_get_node_affinity_dispatches_to_get_node_affinity():
    manager, tool = _make_tool()
    manager.get_node_affinity.return_value = "SENTINEL_RESULT_get_node_affinity"
    result = asyncio.run(
        tool(
            action="get_node_affinity",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_get_node_affinity"
    manager.get_node_affinity.assert_called_once_with("pod_name_val", "namespace_val")


def test_cm_k8s_cluster_get_node_affinity_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="get_node_affinity",
            namespace=None,
            pod_name=None,
            ctx=None,
        )
    )
    assert (
        result == "Error: 'pod_name' and 'namespace' are required for get_node_affinity"
    )
    manager.get_node_affinity.assert_not_called()


def test_cm_k8s_cluster_set_pod_anti_affinity_dispatches_to_set_pod_anti_affinity():
    manager, tool = _make_tool()
    manager.set_pod_anti_affinity.return_value = "SENTINEL_RESULT_set_pod_anti_affinity"
    result = asyncio.run(
        tool(
            action="set_pod_anti_affinity",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_set_pod_anti_affinity"
    manager.set_pod_anti_affinity.assert_called_once_with(
        "pod_name_val", "namespace_val", {"anti_affinity_k": "anti_affinity_v"}
    )


def test_cm_k8s_cluster_set_pod_anti_affinity_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="set_pod_anti_affinity",
            anti_affinity=None,
            namespace=None,
            pod_name=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'pod_name', 'namespace', and 'anti_affinity' are required for set_pod_anti_affinity"
    )
    manager.set_pod_anti_affinity.assert_not_called()


def test_cm_k8s_cluster_list_contexts_dispatches_to_list_contexts():
    manager, tool = _make_tool()
    manager.list_contexts.return_value = "SENTINEL_RESULT_list_contexts"
    result = asyncio.run(
        tool(
            action="list_contexts",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_contexts"
    manager.list_contexts.assert_called_once_with()


def test_cm_k8s_cluster_use_context_dispatches_to_use_context():
    manager, tool = _make_tool()
    manager.use_context.return_value = "SENTINEL_RESULT_use_context"
    result = asyncio.run(
        tool(
            action="use_context",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_use_context"
    manager.use_context.assert_called_once_with(context_name="context_name_val")


def test_cm_k8s_cluster_use_context_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="use_context",
            context_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'context_name' is required for use_context"
    manager.use_context.assert_not_called()


def test_cm_k8s_cluster_get_config_dispatches_to_get_config():
    manager, tool = _make_tool()
    manager.get_config.return_value = "SENTINEL_RESULT_get_config"
    result = asyncio.run(
        tool(
            action="get_config",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_get_config"
    manager.get_config.assert_called_once_with()


def test_cm_k8s_cluster_rename_context_dispatches_to_rename_context():
    manager, tool = _make_tool()
    manager.rename_context.return_value = "SENTINEL_RESULT_rename_context"
    result = asyncio.run(
        tool(
            action="rename_context",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_rename_context"
    manager.rename_context.assert_called_once_with(
        current_name="context_name_val", new_name="new_context_name_val"
    )


def test_cm_k8s_cluster_rename_context_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="rename_context",
            context_name=None,
            new_context_name=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'context_name' and 'new_context_name' are required for rename_context"
    )
    manager.rename_context.assert_not_called()


def test_cm_k8s_cluster_validate_kubeconfig_dispatches_to_validate_kubeconfig():
    manager, tool = _make_tool()
    manager.validate_kubeconfig.return_value = "SENTINEL_RESULT_validate_kubeconfig"
    result = asyncio.run(
        tool(
            action="validate_kubeconfig",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_validate_kubeconfig"
    manager.validate_kubeconfig.assert_called_once_with()


def test_cm_k8s_cluster_list_csr_dispatches_to_list_certificate_signing_requests():
    manager, tool = _make_tool()
    manager.list_certificate_signing_requests.return_value = "SENTINEL_RESULT_list_csr"
    result = asyncio.run(
        tool(
            action="list_csr",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_csr"
    manager.list_certificate_signing_requests.assert_called_once_with()


def test_cm_k8s_cluster_approve_csr_dispatches_to_approve_csr():
    manager, tool = _make_tool()
    manager.approve_csr.return_value = "SENTINEL_RESULT_approve_csr"
    result = asyncio.run(
        tool(
            action="approve_csr",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_approve_csr"
    manager.approve_csr.assert_called_once_with("csr_name_val")


def test_cm_k8s_cluster_approve_csr_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="approve_csr",
            csr_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'csr_name' is required for approve_csr"
    manager.approve_csr.assert_not_called()


def test_cm_k8s_cluster_deny_csr_with_reason_dispatches_to_deny_csr_with_reason():
    """deny_csr has an optional trailing `reason` arg: `if reason:` includes
    it in the call, the fallthrough omits it -- not an is-not-None guard."""
    manager, tool = _make_tool()
    manager.deny_csr.return_value = "SENTINEL_RESULT_deny_csr"
    result = asyncio.run(
        tool(
            action="deny_csr",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_deny_csr"
    manager.deny_csr.assert_called_once_with("csr_name_val", "reason_val")


def test_cm_k8s_cluster_deny_csr_without_reason_omits_reason_arg():
    manager, tool = _make_tool()
    manager.deny_csr.return_value = "SENTINEL_RESULT_deny_csr"
    result = asyncio.run(
        tool(
            action="deny_csr",
            csr_name="csr_name_val",
            reason=None,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_deny_csr"
    manager.deny_csr.assert_called_once_with("csr_name_val")


def test_cm_k8s_cluster_deny_csr_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="deny_csr",
            csr_name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'csr_name' is required for deny_csr"
    manager.deny_csr.assert_not_called()


def test_cm_k8s_cluster_list_api_resources_dispatches_to_list_api_resources():
    manager, tool = _make_tool()
    manager.list_api_resources.return_value = "SENTINEL_RESULT_list_api_resources"
    result = asyncio.run(
        tool(
            action="list_api_resources",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_api_resources"
    manager.list_api_resources.assert_called_once_with()


def test_cm_k8s_cluster_describe_api_resource_dispatches_to_describe_api_resource():
    manager, tool = _make_tool()
    manager.describe_api_resource.return_value = "SENTINEL_RESULT_describe_api_resource"
    result = asyncio.run(
        tool(
            action="describe_api_resource",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_describe_api_resource"
    manager.describe_api_resource.assert_called_once_with("name_val")


def test_cm_k8s_cluster_describe_api_resource_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="describe_api_resource",
            name=None,
            ctx=None,
        )
    )
    assert result == "Error: 'name' is required for describe_api_resource"
    manager.describe_api_resource.assert_not_called()


def test_cm_k8s_cluster_cluster_info_dump_dispatches_to_cluster_info_dump():
    manager, tool = _make_tool()
    manager.cluster_info_dump.return_value = "SENTINEL_RESULT_cluster_info_dump"
    result = asyncio.run(
        tool(
            action="cluster_info_dump",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_cluster_info_dump"
    manager.cluster_info_dump.assert_called_once_with("output_dir_val")


def test_cm_k8s_cluster_cluster_info_dump_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="cluster_info_dump",
            output_dir=None,
            ctx=None,
        )
    )
    assert result == "Error: 'output_dir' is required for cluster_info_dump"
    manager.cluster_info_dump.assert_not_called()


def test_cm_k8s_cluster_get_cluster_info_dispatches_to_get_cluster_info():
    manager, tool = _make_tool()
    manager.get_cluster_info.return_value = "SENTINEL_RESULT_get_cluster_info"
    result = asyncio.run(
        tool(
            action="get_cluster_info",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_get_cluster_info"
    manager.get_cluster_info.assert_called_once_with()


def test_cm_k8s_cluster_get_api_server_info_dispatches_to_get_api_server_info():
    manager, tool = _make_tool()
    manager.get_api_server_info.return_value = "SENTINEL_RESULT_get_api_server_info"
    result = asyncio.run(
        tool(
            action="get_api_server_info",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_get_api_server_info"
    manager.get_api_server_info.assert_called_once_with()


def test_cm_k8s_cluster_list_cluster_plugins_dispatches_to_list_cluster_plugins():
    manager, tool = _make_tool()
    manager.list_cluster_plugins.return_value = "SENTINEL_RESULT_list_cluster_plugins"
    result = asyncio.run(
        tool(
            action="list_cluster_plugins",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_list_cluster_plugins"
    manager.list_cluster_plugins.assert_called_once_with()


def test_cm_k8s_cluster_describe_cluster_plugin_dispatches_to_describe_cluster_plugin():
    manager, tool = _make_tool()
    manager.describe_cluster_plugin.return_value = (
        "SENTINEL_RESULT_describe_cluster_plugin"
    )
    result = asyncio.run(
        tool(
            action="describe_cluster_plugin",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_describe_cluster_plugin"
    manager.describe_cluster_plugin.assert_called_once_with(
        "name_val", "plugin_type_val"
    )


def test_cm_k8s_cluster_describe_cluster_plugin_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="describe_cluster_plugin",
            name=None,
            plugin_type=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'name' and 'plugin_type' are required for describe_cluster_plugin"
    )
    manager.describe_cluster_plugin.assert_not_called()


def test_cm_k8s_cluster_test_cluster_plugin_dispatches_to_test_cluster_plugin():
    manager, tool = _make_tool()
    manager.test_cluster_plugin.return_value = "SENTINEL_RESULT_test_cluster_plugin"
    result = asyncio.run(
        tool(
            action="test_cluster_plugin",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "SENTINEL_RESULT_test_cluster_plugin"
    manager.test_cluster_plugin.assert_called_once_with(
        "name_val", "plugin_type_val", {"test_resource_k": "test_resource_v"}
    )


def test_cm_k8s_cluster_test_cluster_plugin_guard_0_returns_error_without_calling_manager():
    manager, tool = _make_tool()
    result = asyncio.run(
        tool(
            action="test_cluster_plugin",
            name=None,
            plugin_type=None,
            test_resource=None,
            ctx=None,
        )
    )
    assert (
        result
        == "Error: 'name', 'plugin_type', and 'test_resource' are required for test_cluster_plugin"
    )
    manager.test_cluster_plugin.assert_not_called()


def test_cm_k8s_cluster_unknown_action_returns_error():
    manager, tool = _make_tool()
    result = asyncio.run(tool(action="bogus_action_xyz", ctx=None))
    assert result == "Error: Unknown action 'bogus_action_xyz'"


def test_cm_k8s_cluster_manager_exception_is_caught_and_formatted():
    manager, tool = _make_tool()
    manager.list_nodes.side_effect = RuntimeError("boom")
    result = asyncio.run(
        tool(
            action="list_nodes",
            node_name="node_name_val",
            namespace="namespace_val",
            pod_name="pod_name_val",
            taints=["taints_item"],
            taint_key="taint_key_val",
            affinity={"affinity_k": "affinity_v"},
            anti_affinity={"anti_affinity_k": "anti_affinity_v"},
            grace_period_seconds=7,
            context_name="context_name_val",
            new_context_name="new_context_name_val",
            csr_name="csr_name_val",
            reason="reason_val",
            name="name_val",
            output_dir="output_dir_val",
            plugin_type="plugin_type_val",
            test_resource={"test_resource_k": "test_resource_v"},
            server="server_val",
            token="token_val",
            client_cert="client_cert_val",
            client_key="client_key_val",
            ca_cert="ca_cert_val",
            insecure_skip_tls_verify=True,
            username="username_val",
            password="password_val",
            oidc_issuer="oidc_issuer_val",
            oidc_client_id="oidc_client_id_val",
            oidc_client_secret="test_secret",
            source_file="source_file_val",
            source_yaml="source_yaml_val",
            capture_current=True,
            kubeconfig_path="kubeconfig_path_val",
            overwrite=True,
            use=True,
            validate=True,
            ctx=None,
        )
    )
    assert result == "Error executing list_nodes: RuntimeError"

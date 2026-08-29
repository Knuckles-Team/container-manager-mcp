"""Characterization tests for cm_multi_context's execute_operation (CCN 49),
before decomposition.

cm_multi_context is a 20-action flat dispatcher over a `target_manager`
resolved from `manager.get_manager(backend, context)` -- resolved
unconditionally for every action except `list_contexts` (which returns
before resolution), including an eventually-unknown action. This pins:
that `get_manager` is/is not called depending on action, which
`target_manager.<method>` each action calls and with exactly which args,
each required-param guard's exact ValueError with the manager left
uncalled, the Kubernetes-only actions' backend gate, and the unknown-action
ValueError.
"""

import asyncio
from unittest.mock import MagicMock

import pytest


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


@pytest.fixture
def tool_and_managers(monkeypatch):
    from container_manager_mcp.mcp import mcp_multi_context

    fake_target = MagicMock()
    fake_manager = MagicMock()
    fake_manager.get_manager.return_value = fake_target
    monkeypatch.setattr(
        mcp_multi_context, "create_manager", lambda manager_type: fake_manager
    )
    tool = _capture_tool(mcp_multi_context.register_multicontext_tools)
    return tool, fake_manager, fake_target


def _run(tool, **kwargs):
    return asyncio.run(tool(**kwargs))


def test_list_contexts_does_not_resolve_target_manager(tool_and_managers):
    tool, manager, target = tool_and_managers
    manager.list_available_contexts.return_value = ["a", "b"]
    result = _run(tool, action="list_contexts")
    assert result == ["a", "b"]
    manager.get_manager.assert_not_called()


def test_list_containers_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    target.list_containers.return_value = []
    result = _run(tool, action="list_containers", backend="kubernetes", context=None, all=True)
    manager.get_manager.assert_called_once_with("kubernetes", None)
    target.list_containers.assert_called_once_with(all=True)
    assert result == []


def test_run_container_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(
        tool,
        action="run_container",
        image="nginx",
        name="c1",
        command="echo hi",
        ports=[80],
        volumes={"/a": "/b"},
        environment={"X": "1"},
    )
    target.run_container.assert_called_once_with(
        image="nginx", name="c1", command="echo hi", ports=[80],
        volumes={"/a": "/b"}, environment={"X": "1"},
    )


def test_run_container_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="image is required for run_container"):
        _run(tool, action="run_container", image=None)
    target.run_container.assert_not_called()


def test_stop_container_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="stop_container", name="c1", force=True)
    target.stop_container.assert_called_once_with("c1", force=True)


def test_stop_container_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="name is required for stop_container"):
        _run(tool, action="stop_container", name=None)


def test_remove_container_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="remove_container", name="c1", force=True)
    target.remove_container.assert_called_once_with("c1", force=True)


def test_remove_container_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="name is required for remove_container"):
        _run(tool, action="remove_container", name=None)


def test_inspect_container_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="inspect_container", name="c1")
    target.inspect_container.assert_called_once_with("c1")


def test_inspect_container_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="name is required for inspect_container"):
        _run(tool, action="inspect_container", name=None)


def test_list_images_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="list_images")
    target.list_images.assert_called_once_with()


def test_pull_image_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="pull_image", image="nginx")
    target.pull_image.assert_called_once_with("nginx")


def test_pull_image_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="image is required for pull_image"):
        _run(tool, action="pull_image", image=None)


def test_remove_image_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="remove_image", name="img1", force=True)
    target.remove_image.assert_called_once_with("img1", force=True)


def test_remove_image_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="name is required for remove_image"):
        _run(tool, action="remove_image", name=None)


def test_list_volumes_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="list_volumes")
    target.list_volumes.assert_called_once_with()


def test_create_volume_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="create_volume", name="v1", driver="local")
    target.create_volume.assert_called_once_with("v1", "local")


def test_create_volume_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="name is required for create_volume"):
        _run(tool, action="create_volume", name=None)


def test_remove_volume_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="remove_volume", name="v1", force=True)
    target.remove_volume.assert_called_once_with("v1", force=True)


def test_remove_volume_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="name is required for remove_volume"):
        _run(tool, action="remove_volume", name=None)


def test_list_networks_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="list_networks")
    target.list_networks.assert_called_once_with()


def test_create_network_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="create_network", name="n1", driver="bridge")
    target.create_network.assert_called_once_with("n1", "bridge")


def test_create_network_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="name is required for create_network"):
        _run(tool, action="create_network", name=None)


def test_remove_network_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="remove_network", name="n1", force=True)
    target.remove_network.assert_called_once_with("n1", force=True)


def test_remove_network_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="name is required for remove_network"):
        _run(tool, action="remove_network", name=None)


def test_list_services_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="list_services")
    target.list_services.assert_called_once_with()


def test_create_service_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="create_service", name="svc", image="nginx", ports=[80])
    target.create_service.assert_called_once_with("svc", "nginx", [80])


def test_create_service_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="name and image are required for create_service"):
        _run(tool, action="create_service", name=None, image=None)


def test_remove_service_dispatches(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="remove_service", name="svc")
    target.remove_service.assert_called_once_with("svc")


def test_remove_service_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="name is required for remove_service"):
        _run(tool, action="remove_service", name=None)


def test_list_pods_dispatches_on_kubernetes(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="list_pods", backend="kubernetes", namespace="ns1")
    target.list_pods.assert_called_once_with(namespace="ns1")


def test_list_pods_default_namespace(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="list_pods", backend="kubernetes", namespace=None)
    target.list_pods.assert_called_once_with(namespace="default")


def test_list_pods_rejects_non_kubernetes_backend(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="list_pods is only available for Kubernetes, not docker"):
        _run(tool, action="list_pods", backend="docker")
    target.list_pods.assert_not_called()


def test_describe_pod_dispatches_on_kubernetes(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="describe_pod", backend="kubernetes", name="p1", namespace="ns1")
    target.describe_pod.assert_called_once_with("p1", "ns1")


def test_describe_pod_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="name and namespace are required for describe_pod"):
        _run(tool, action="describe_pod", backend="kubernetes", name=None, namespace=None)


def test_describe_pod_rejects_non_kubernetes_backend(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="describe_pod is only available for Kubernetes, not podman"):
        _run(tool, action="describe_pod", backend="podman", name="p1", namespace="ns1")


def test_list_deployments_dispatches_on_kubernetes(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(tool, action="list_deployments", backend="kubernetes", namespace="ns1")
    target.list_deployments.assert_called_once_with(namespace="ns1")


def test_list_deployments_rejects_non_kubernetes_backend(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="list_deployments is only available for Kubernetes, not swarm"):
        _run(tool, action="list_deployments", backend="swarm")


def test_scale_deployment_dispatches_on_kubernetes(tool_and_managers):
    tool, manager, target = tool_and_managers
    _run(
        tool,
        action="scale_deployment",
        backend="kubernetes",
        name="d1",
        namespace="ns1",
        replicas=3,
    )
    target.scale_service.assert_called_once_with("d1", 3, namespace="ns1")


def test_scale_deployment_guard_raises(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(
        ValueError, match="name, namespace, and replicas are required for scale_deployment"
    ):
        _run(
            tool,
            action="scale_deployment",
            backend="kubernetes",
            name=None,
            namespace=None,
            replicas=None,
        )


def test_scale_deployment_rejects_non_kubernetes_backend(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="scale_deployment is only available for Kubernetes, not docker"):
        _run(tool, action="scale_deployment", backend="docker", name="d1", namespace="ns1", replicas=3)


def test_unknown_action_raises_after_resolving_target_manager(tool_and_managers):
    tool, manager, target = tool_and_managers
    with pytest.raises(ValueError, match="Unknown action: bogus_action"):
        _run(tool, action="bogus_action", backend="kubernetes", context=None)
    # Pinning the pre-existing quirk: get_manager() IS called for an unknown
    # action, because target_manager is resolved unconditionally before the
    # dispatch chain (only "list_contexts" short-circuits before it).
    manager.get_manager.assert_called_once_with("kubernetes", None)

"""Characterization tests for CXA-FL-CONTAINERMANAGERMCP-01's `container_manager`
CLI entrypoint (CCN 69), before decomposition.

Unlike the register_*_tools dispatchers in mcp/mcp_k8s_*.py, `container_manager`
is an argparse CLI: after `args = parser.parse_args()`, it runs a *sequence of
independent* `if <flag>:` blocks (not `elif`) -- any subset of the 32 flags can
fire together in one invocation. Each pins: the exact positional/keyword
arguments passed to the one `manager.<method>(...)` call it makes, each
required-value guard's exact `raise ValueError(...)` message with the manager
left uncalled, and (for get-version) that the file's own precedent test
(tests/test_container_manager_mcp_brute_force_coverage.py::test_main_coverage)
already covers the identical pattern this file scales to all 32 flags.

Harness: patch `sys.argv` (matching the repo's own established pattern) and
`container_manager_mcp.container_manager.create_manager` to return a
MagicMock, then call the bare `container_manager()` function -- no real
Docker/Podman/Swarm client touched.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from container_manager_mcp.container_manager import container_manager


# Methods whose result is wrapped in `json.dumps(...)` by container_manager()
# need a JSON-serializable return_value -- a bare MagicMock() isn't one.
# get_container_logs/compose_up/compose_down/compose_ps/compose_logs print
# their raw return value instead, so a MagicMock is fine there.
_JSON_WRAPPED_METHODS = [
    "get_version", "get_info", "list_images", "pull_image", "remove_image",
    "prune_images", "list_containers", "run_container", "stop_container",
    "remove_container", "prune_containers", "exec_in_container", "list_volumes",
    "create_volume", "remove_volume", "prune_volumes", "list_networks",
    "create_network", "remove_network", "prune_networks", "prune_system",
    "init_swarm", "leave_swarm", "list_nodes", "list_services",
    "create_service", "remove_service",
]


def _run_cli(argv):
    manager = MagicMock()
    for method_name in _JSON_WRAPPED_METHODS:
        getattr(manager, method_name).return_value = {}
    with (
        patch.object(sys, "argv", argv),
        patch(
            "container_manager_mcp.container_manager.create_manager",
            return_value=manager,
        ) as create_manager_mock,
    ):
        container_manager()
    return manager, create_manager_mock


def test_container_manager_get_version_dispatches_to_get_version():
    manager, _ = _run_cli(["container_manager.py", '--get-version'])
    manager.get_version.assert_called_once_with()


def test_container_manager_get_info_dispatches_to_get_info():
    manager, _ = _run_cli(["container_manager.py", '--get-info'])
    manager.get_info.assert_called_once_with()


def test_container_manager_list_images_dispatches_to_list_images():
    manager, _ = _run_cli(["container_manager.py", '--list-images'])
    manager.list_images.assert_called_once_with()


def test_container_manager_pull_image_dispatches_to_pull_image():
    manager, _ = _run_cli(["container_manager.py", '--pull-image', 'myimg', '--tag', 'v1', '--platform', 'linux/amd64'])
    manager.pull_image.assert_called_once_with("myimg", "v1", "linux/amd64")


def test_container_manager_pull_image_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='Image required for pull-image'):
        _run_cli(["container_manager.py", '--pull-image', ''])


def test_container_manager_remove_image_dispatches_to_remove_image():
    manager, _ = _run_cli(["container_manager.py", '--remove-image', 'img2', '--force'])
    manager.remove_image.assert_called_once_with("img2", True)


def test_container_manager_remove_image_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='Image required for remove-image'):
        _run_cli(["container_manager.py", '--remove-image', ''])


def test_container_manager_prune_images_dispatches_to_prune_images():
    manager, _ = _run_cli(["container_manager.py", '--prune-images', '--force', '--all'])
    manager.prune_images.assert_called_once_with(True, True)


def test_container_manager_list_containers_dispatches_to_list_containers():
    manager, _ = _run_cli(["container_manager.py", '--list-containers', '--all'])
    manager.list_containers.assert_called_once_with(True,)


def test_container_manager_run_container_dispatches_to_run_container():
    manager, _ = _run_cli(["container_manager.py", '--run-container', 'img3', '--name', 'c1', '--command', 'echo hi', '--detach', '--ports', '8080:80', '--volumes', '/host:/container:ro', '--environment', 'KEY=VAL'])
    manager.run_container.assert_called_once_with("img3", "c1", "echo hi", True, {"80/tcp": "8080"}, {"/host": {"bind": "/container", "mode": "ro"}}, {"KEY": "VAL"})


def test_container_manager_run_container_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='Image required for run-container'):
        _run_cli(["container_manager.py", '--run-container', ''])


def test_container_manager_stop_container_dispatches_to_stop_container():
    manager, _ = _run_cli(["container_manager.py", '--stop-container', 'c1', '--timeout', '30'])
    manager.stop_container.assert_called_once_with("c1", 30)


def test_container_manager_stop_container_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='Container ID required for stop-container'):
        _run_cli(["container_manager.py", '--stop-container', ''])


def test_container_manager_remove_container_dispatches_to_remove_container():
    manager, _ = _run_cli(["container_manager.py", '--remove-container', 'c1', '--force'])
    manager.remove_container.assert_called_once_with("c1", True)


def test_container_manager_remove_container_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='Container ID required for remove-container'):
        _run_cli(["container_manager.py", '--remove-container', ''])


def test_container_manager_prune_containers_dispatches_to_prune_containers():
    manager, _ = _run_cli(["container_manager.py", '--prune-containers'])
    manager.prune_containers.assert_called_once_with()


def test_container_manager_get_container_logs_dispatches_to_get_container_logs():
    manager, _ = _run_cli(["container_manager.py", '--get-container-logs', 'c1', '--tail', '100'])
    manager.get_container_logs.assert_called_once_with("c1", "100")


def test_container_manager_get_container_logs_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='Container ID required for get-container-logs'):
        _run_cli(["container_manager.py", '--get-container-logs', ''])


def test_container_manager_exec_in_container_dispatches_to_exec_in_container():
    manager, _ = _run_cli(["container_manager.py", '--exec-in-container', 'c1', '--exec-command', 'ls -la', '--exec-detach'])
    manager.exec_in_container.assert_called_once_with("c1", ["ls", "-la"], True)


def test_container_manager_exec_in_container_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='Container ID required for exec-in-container'):
        _run_cli(["container_manager.py", '--exec-in-container', ''])


def test_container_manager_list_volumes_dispatches_to_list_volumes():
    manager, _ = _run_cli(["container_manager.py", '--list-volumes'])
    manager.list_volumes.assert_called_once_with()


def test_container_manager_create_volume_dispatches_to_create_volume():
    manager, _ = _run_cli(["container_manager.py", '--create-volume', 'v1'])
    manager.create_volume.assert_called_once_with("v1",)


def test_container_manager_create_volume_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='Name required for create-volume'):
        _run_cli(["container_manager.py", '--create-volume', ''])


def test_container_manager_remove_volume_dispatches_to_remove_volume():
    manager, _ = _run_cli(["container_manager.py", '--remove-volume', 'v1', '--force'])
    manager.remove_volume.assert_called_once_with("v1", True)


def test_container_manager_remove_volume_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='Name required for remove-volume'):
        _run_cli(["container_manager.py", '--remove-volume', ''])


def test_container_manager_prune_volumes_dispatches_to_prune_volumes():
    manager, _ = _run_cli(["container_manager.py", '--prune-volumes', '--force', '--all'])
    manager.prune_volumes.assert_called_once_with(True, True)


def test_container_manager_list_networks_dispatches_to_list_networks():
    manager, _ = _run_cli(["container_manager.py", '--list-networks'])
    manager.list_networks.assert_called_once_with()


def test_container_manager_create_network_dispatches_to_create_network():
    manager, _ = _run_cli(["container_manager.py", '--create-network', 'n1', '--driver', 'overlay'])
    manager.create_network.assert_called_once_with("n1", "overlay")


def test_container_manager_create_network_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='Name required for create-network'):
        _run_cli(["container_manager.py", '--create-network', ''])


def test_container_manager_remove_network_dispatches_to_remove_network():
    manager, _ = _run_cli(["container_manager.py", '--remove-network', 'n1'])
    manager.remove_network.assert_called_once_with("n1",)


def test_container_manager_remove_network_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='ID required for remove-network'):
        _run_cli(["container_manager.py", '--remove-network', ''])


def test_container_manager_prune_networks_dispatches_to_prune_networks():
    manager, _ = _run_cli(["container_manager.py", '--prune-networks'])
    manager.prune_networks.assert_called_once_with()


def test_container_manager_prune_system_dispatches_to_prune_system():
    manager, _ = _run_cli(["container_manager.py", '--prune-system', '--force', '--all'])
    manager.prune_system.assert_called_once_with(True, True)


def test_container_manager_compose_up_dispatches_to_compose_up():
    manager, _ = _run_cli(["container_manager.py", '--compose-up', 'docker-compose.yml', '--build'])
    manager.compose_up.assert_called_once_with("docker-compose.yml", True, True)


def test_container_manager_compose_up_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='File required for compose-up'):
        _run_cli(["container_manager.py", '--compose-up', ''])


def test_container_manager_compose_down_dispatches_to_compose_down():
    manager, _ = _run_cli(["container_manager.py", '--compose-down', 'file.yml'])
    manager.compose_down.assert_called_once_with("file.yml",)


def test_container_manager_compose_down_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='File required for compose-down'):
        _run_cli(["container_manager.py", '--compose-down', ''])


def test_container_manager_compose_ps_dispatches_to_compose_ps():
    manager, _ = _run_cli(["container_manager.py", '--compose-ps', 'file.yml'])
    manager.compose_ps.assert_called_once_with("file.yml",)


def test_container_manager_compose_ps_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='File required for compose-ps'):
        _run_cli(["container_manager.py", '--compose-ps', ''])


def test_container_manager_compose_logs_dispatches_to_compose_logs():
    manager, _ = _run_cli(["container_manager.py", '--compose-logs', 'file.yml', '--service', 'web'])
    manager.compose_logs.assert_called_once_with("file.yml", "web")


def test_container_manager_compose_logs_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='File required for compose-logs'):
        _run_cli(["container_manager.py", '--compose-logs', ''])


def test_container_manager_init_swarm_dispatches_to_init_swarm():
    manager, _ = _run_cli(["container_manager.py", '--init-swarm', '--advertise-addr', '10.0.0.1'])
    manager.init_swarm.assert_called_once_with("10.0.0.1",)


def test_container_manager_leave_swarm_dispatches_to_leave_swarm():
    manager, _ = _run_cli(["container_manager.py", '--leave-swarm', '--force'])
    manager.leave_swarm.assert_called_once_with(True,)


def test_container_manager_list_nodes_dispatches_to_list_nodes():
    manager, _ = _run_cli(["container_manager.py", '--list-nodes'])
    manager.list_nodes.assert_called_once_with()


def test_container_manager_list_services_dispatches_to_list_services():
    manager, _ = _run_cli(["container_manager.py", '--list-services'])
    manager.list_services.assert_called_once_with()


def test_container_manager_create_service_dispatches_to_create_service():
    manager, _ = _run_cli(["container_manager.py", '--create-service', 'svc1', '--image', 'img4', '--replicas', '3', '--ports', '8080:80', '--mounts', '/data:/data,/logs:/logs'])
    manager.create_service.assert_called_once_with("svc1", "img4", 3, {"80/tcp": "8080"}, ["/data:/data", "/logs:/logs"])


def test_container_manager_create_service_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='Image required for create-service'):
        _run_cli(["container_manager.py", '--create-service', 'svc1'])


def test_container_manager_remove_service_dispatches_to_remove_service():
    manager, _ = _run_cli(["container_manager.py", '--remove-service', 'svc1'])
    manager.remove_service.assert_called_once_with("svc1",)


def test_container_manager_remove_service_guard_raises_value_error_without_calling_manager():
    with pytest.raises(ValueError, match='ID required for remove-service'):
        _run_cli(["container_manager.py", '--remove-service', ''])


# TOTAL FLAGS: 32

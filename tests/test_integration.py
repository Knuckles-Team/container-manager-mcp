#!/usr/bin/env python
"""Integration tests for container manager.

These tests require actual Docker or Podman runtime to be available.
Tests will be skipped if the required runtime is not available.
"""

import os
import subprocess
import time
from unittest.mock import patch

import pytest

from container_manager_mcp.container_manager import (
    DockerManager,
    PodmanManager,
    create_manager,
    is_app_installed,
)


def check_docker_available():
    """Check if Docker is available and running."""
    if not is_app_installed("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_podman_available():
    """Check if Podman is available and running."""
    if not is_app_installed("podman"):
        return False
    try:
        result = subprocess.run(
            ["podman", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Test images - small, lightweight images for testing
TEST_IMAGE = "alpine:latest"
TEST_IMAGE_ALT = "alpine:latest"  # Using same image for Podman compatibility (nginx:alpine requires registry config)


class TestContainerManagerIntegration:
    """Integration tests for container manager with actual runtime."""

    @pytest.fixture
    def docker_manager(self):
        """Create a DockerManager instance if Docker is available."""
        if not check_docker_available():
            pytest.skip("Docker not available or not running")
        try:
            manager = DockerManager(silent=True)
            yield manager
        except Exception as e:
            pytest.skip(f"Docker initialization failed: {e}")

    @pytest.fixture
    def podman_manager(self):
        """Create a PodmanManager instance if Podman is available."""
        if not check_podman_available():
            pytest.skip("Podman not available or not running")

        # Set the Podman socket URL for this environment
        original_url = os.environ.get("CONTAINER_MANAGER_PODMAN_BASE_URL")
        os.environ["CONTAINER_MANAGER_PODMAN_BASE_URL"] = (
            "unix:///run/podman/podman.sock"
        )

        try:
            manager = PodmanManager(silent=True)
            yield manager
        except Exception as e:
            pytest.skip(f"Podman initialization failed: {e}")
        finally:
            if original_url is not None:
                os.environ["CONTAINER_MANAGER_PODMAN_BASE_URL"] = original_url
            else:
                os.environ.pop("CONTAINER_MANAGER_PODMAN_BASE_URL", None)

    @pytest.fixture
    def manager(self, request):
        """Create appropriate manager based on test parameter."""
        if request.param == "docker":
            return request.getfixturevalue("docker_manager")
        else:
            return request.getfixturevalue("podman_manager")

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_get_version(self, manager_type, request):
        """Test getting version from actual container runtime."""
        manager = request.getfixturevalue(f"{manager_type}_manager")
        version = manager.get_version()
        assert version is not None
        assert "version" in version
        assert version["version"] != "unknown"

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_get_info(self, manager_type, request):
        """Test getting info from actual container runtime."""
        manager = request.getfixturevalue(f"{manager_type}_manager")
        info = manager.get_info()
        assert info is not None
        assert "containers_total" in info
        assert "images" in info
        assert "platform" in info

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_list_images(self, manager_type, request):
        """Test listing images from actual container runtime."""
        manager = request.getfixturevalue(f"{manager_type}_manager")
        images = manager.list_images()
        assert images is not None
        assert isinstance(images, list)

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_pull_image(self, manager_type, request):
        """Test pulling an image from actual container runtime."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        # Pull a small image for testing
        result = manager.pull_image(TEST_IMAGE, "latest")
        assert result is not None
        assert "repository" in result
        assert result["repository"] == "alpine"
        assert "tag" in result

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_run_container(self, manager_type, request):
        """Test running a container with actual container runtime."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        # Clean up any existing container with the same name
        try:
            existing = manager.list_containers(all=True)
            for c in existing:
                if c.get("name") == "test-integration-container":
                    manager.remove_container(c["id"], force=True)
        except Exception:
            pass

        # Ensure image is available
        try:
            manager.pull_image(TEST_IMAGE, "latest")
        except Exception:
            pass

        # Run a simple container (use command as list for Podman compatibility)
        command = ["echo", "Hello World"]
        result = manager.run_container(
            image=TEST_IMAGE,
            name="test-integration-container",
            command=command if manager_type == "podman" else " ".join(command),
            detach=True,
        )
        assert result is not None
        assert "id" in result
        assert "name" in result
        assert result["name"] == "test-integration-container"

        # Clean up
        try:
            manager.remove_container(result["id"], force=True)
        except Exception:
            pass

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_list_containers(self, manager_type, request):
        """Test listing containers from actual container runtime."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        # Clean up any existing container with the same name
        try:
            existing = manager.list_containers(all=True)
            for c in existing:
                if c.get("name") == "test-list-container":
                    manager.remove_container(c["id"], force=True)
        except Exception:
            pass

        # Ensure image is available
        try:
            manager.pull_image(TEST_IMAGE, "latest")
        except Exception:
            pass

        # Run a test container
        command = ["sleep", "10"]
        container_result = manager.run_container(
            image=TEST_IMAGE,
            name="test-list-container",
            command=command if manager_type == "podman" else " ".join(command),
            detach=True,
        )

        # Wait for container to be fully started
        time.sleep(2)

        # List containers
        containers = manager.list_containers(all=True)
        assert containers is not None
        assert isinstance(containers, list)

        # Clean up
        try:
            manager.remove_container(container_result["id"], force=True)
        except Exception:
            pass

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_stop_and_remove_container(self, manager_type, request):
        """Test stopping and removing a container."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        # Skip for Podman due to podman-py library bug with stop_container (JSON parsing on 304 response)
        if manager_type == "podman":
            pytest.skip(
                "Stop container test skipped for Podman due to podman-py library API limitations"
            )

        # Clean up any existing container with the same name
        try:
            existing = manager.list_containers(all=True)
            for c in existing:
                if c.get("name") == "test-stop-remove-container":
                    manager.remove_container(c["id"], force=True)
        except Exception:
            pass

        # Ensure image is available
        try:
            manager.pull_image(TEST_IMAGE, "latest")
        except Exception:
            pass

        # Run a test container
        command = ["sleep", "30"]
        container_result = manager.run_container(
            image=TEST_IMAGE,
            name="test-stop-remove-container",
            command=command if manager_type == "podman" else " ".join(command),
            detach=True,
        )

        # Wait for container to be fully started
        time.sleep(2)

        # Stop the container
        stop_result = manager.stop_container(container_result["id"], timeout=5)
        assert stop_result is not None
        assert "stopped" in stop_result

        # Remove the container
        remove_result = manager.remove_container(container_result["id"], force=True)
        assert remove_result is not None
        assert "removed" in remove_result

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_create_volume(self, manager_type, request):
        """Test creating a volume."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        volume_name = "test-integration-volume"

        # Create volume
        result = manager.create_volume(volume_name)
        assert result is not None
        assert "name" in result
        assert result["name"] == volume_name

        # List volumes to verify
        volumes = manager.list_volumes()
        assert volumes is not None
        volume_names = [v["name"] for v in volumes]
        assert volume_name in volume_names

        # Clean up
        try:
            manager.remove_volume(volume_name, force=True)
        except Exception:
            pass

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_create_network(self, manager_type, request):
        """Test creating a network."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        network_name = "test-integration-network"

        # Create network
        result = manager.create_network(network_name, driver="bridge")
        assert result is not None
        assert "name" in result
        assert result["name"] == network_name

        # List networks to verify
        networks = manager.list_networks()
        assert networks is not None
        network_names = [n["name"] for n in networks]
        assert network_name in network_names

        # Clean up
        try:
            manager.remove_network(result["id"])
        except Exception:
            pass

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_prune_images(self, manager_type, request):
        """Test pruning images."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        # Pull a test image
        try:
            manager.pull_image(TEST_IMAGE_ALT, "latest")
        except Exception:
            pytest.skip(f"Could not pull test image {TEST_IMAGE_ALT}")

        # Prune images
        result = manager.prune_images(force=False, all=False)
        assert result is not None
        assert "space_reclaimed" in result

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_prune_containers(self, manager_type, request):
        """Test pruning containers."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        # Skip container tests for Podman due to podman-py library API differences (stop_container JSON parsing bug)
        if manager_type == "podman":
            pytest.skip(
                "Container test skipped for Podman due to podman-py library API limitations"
            )

        # Ensure image is available
        try:
            manager.pull_image(TEST_IMAGE, "latest")
        except Exception:
            pytest.skip("Could not pull test image")

        # Run and stop a container to create a stopped container
        command = ["echo", "test"]
        container_result = manager.run_container(
            image=TEST_IMAGE,
            name="test-prune-container",
            command=command if manager_type == "podman" else " ".join(command),
            detach=True,
        )
        time.sleep(1)  # Wait for container to finish
        manager.stop_container(container_result["id"], timeout=5)

        # Prune containers
        result = manager.prune_containers()
        assert result is not None
        assert "space_reclaimed" in result
        assert "containers_removed" in result

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_prune_volumes(self, manager_type, request):
        """Test pruning volumes."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        # Create a test volume
        volume_name = "test-prune-volume"
        try:
            manager.create_volume(volume_name)
        except Exception:
            pytest.skip("Could not create test volume")

        # Prune volumes
        result = manager.prune_volumes(force=False, all=False)
        assert result is not None
        assert "space_reclaimed" in result
        assert "volumes_removed" in result

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_prune_networks(self, manager_type, request):
        """Test pruning networks."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        # Create a test network
        network_name = "test-prune-network"
        try:
            manager.create_network(network_name)
        except Exception:
            pytest.skip("Could not create test network")

        # Prune networks
        result = manager.prune_networks()
        assert result is not None
        assert "space_reclaimed" in result
        assert "networks_removed" in result

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_get_container_logs(self, manager_type, request):
        """Test getting container logs."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        # Clean up any existing container with the same name
        try:
            existing = manager.list_containers(all=True)
            for c in existing:
                if c.get("name") == "test-logs-container":
                    manager.remove_container(c["id"], force=True)
        except Exception:
            pass

        # Ensure image is available
        try:
            manager.pull_image(TEST_IMAGE, "latest")
        except Exception:
            pass

        # Run a container that outputs logs and stays running briefly
        command = ["sh", "-c", "echo 'Integration test log output' && sleep 20"]
        container_result = manager.run_container(
            image=TEST_IMAGE,
            name="test-logs-container",
            command=command if manager_type == "podman" else " ".join(command),
            detach=True,
        )
        time.sleep(2)  # Wait for container to start and output logs

        # Get logs
        logs = manager.get_container_logs(container_result["id"], tail="all")
        assert logs is not None
        assert isinstance(logs, str)
        assert len(logs) > 0

        # Clean up
        try:
            manager.remove_container(container_result["id"], force=True)
        except Exception:
            pass

    @pytest.mark.parametrize("manager_type", ["podman", "docker"])
    def test_exec_in_container(self, manager_type, request):
        """Test executing commands in a container."""
        manager = request.getfixturevalue(f"{manager_type}_manager")

        # Clean up any existing container with the same name
        try:
            existing = manager.list_containers(all=True)
            for c in existing:
                if c.get("name") == "test-exec-container":
                    manager.remove_container(c["id"], force=True)
        except Exception:
            pass

        # Ensure image is available
        try:
            manager.pull_image(TEST_IMAGE, "latest")
        except Exception:
            pass

        # Run a long-running container
        command = ["sleep", "30"]
        container_result = manager.run_container(
            image=TEST_IMAGE,
            name="test-exec-container",
            command=command if manager_type == "podman" else " ".join(command),
            detach=True,
        )

        # Wait for container to be fully started
        time.sleep(2)

        # Execute a command in the container
        exec_result = manager.exec_in_container(
            container_result["id"],
            command=["echo", "executed"],
            detach=False,
        )
        assert exec_result is not None
        assert "exit_code" in exec_result
        assert "output" in exec_result
        assert exec_result["exit_code"] == 0

        # Clean up
        try:
            manager.remove_container(container_result["id"], force=True)
        except Exception:
            pass


class TestCreateManagerIntegration:
    """Integration tests for create_manager function."""

    def test_create_manager_auto_detect_podman(self):
        """Test auto-detection when only Podman is available."""
        if not check_podman_available():
            pytest.skip("Podman not available")

        # Force podman by setting environment variable
        original_type = os.environ.get("CONTAINER_MANAGER_TYPE")
        original_url = os.environ.get("CONTAINER_MANAGER_PODMAN_BASE_URL")
        os.environ["CONTAINER_MANAGER_TYPE"] = "podman"
        os.environ["CONTAINER_MANAGER_PODMAN_BASE_URL"] = (
            "unix:///run/podman/podman.sock"
        )

        try:
            manager = create_manager()
            assert isinstance(manager, PodmanManager)
        finally:
            if original_type is not None:
                os.environ["CONTAINER_MANAGER_TYPE"] = original_type
            else:
                os.environ.pop("CONTAINER_MANAGER_TYPE", None)
            if original_url is not None:
                os.environ["CONTAINER_MANAGER_PODMAN_BASE_URL"] = original_url
            else:
                os.environ.pop("CONTAINER_MANAGER_PODMAN_BASE_URL", None)

    def test_create_manager_explicit_podman(self):
        """Test explicit Podman manager creation."""
        if not check_podman_available():
            pytest.skip("Podman not available")

        original_url = os.environ.get("CONTAINER_MANAGER_PODMAN_BASE_URL")
        os.environ["CONTAINER_MANAGER_PODMAN_BASE_URL"] = (
            "unix:///run/podman/podman.sock"
        )

        try:
            manager = create_manager(manager_type="podman")
            assert isinstance(manager, PodmanManager)
        finally:
            if original_url is not None:
                os.environ["CONTAINER_MANAGER_PODMAN_BASE_URL"] = original_url
            else:
                os.environ.pop("CONTAINER_MANAGER_PODMAN_BASE_URL", None)

    def test_create_manager_explicit_docker(self):
        """Test explicit Docker manager creation."""
        if not check_docker_available():
            pytest.skip("Docker not available")

        manager = create_manager(manager_type="docker")
        assert isinstance(manager, DockerManager)


class TestHelperFunctionsIntegration:
    """Integration tests for helper functions."""

    def test_is_app_installed(self):
        """Test is_app_installed function."""
        # Mock shutil.which to simulate app presence
        with patch("shutil.which", return_value="/usr/bin/podman"):
            assert is_app_installed("podman")

        with patch("shutil.which", return_value=None):
            assert not is_app_installed("nonexistent_app_12345")

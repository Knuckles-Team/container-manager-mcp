#!/usr/bin/env python


import argparse
import base64
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from container_manager_mcp.models import (
    ContainerInfo,
    ImageInfo,
    NetworkInfo,
    VolumeInfo,
)

__version__ = "2.3.0"

try:
    from docker.errors import DockerException

    import docker  # type: ignore
except ImportError:
    docker = None  # type: ignore
    DockerException = Exception

try:
    from podman import PodmanClient
    from podman.errors import PodmanError
except ImportError:
    PodmanClient = None
    PodmanError = Exception


def _build_exec_result(
    container: Any, command: list[str], detach: bool, binary: bool
) -> dict:
    """Run ``command`` in ``container`` and shape the result dict.

    Shared by the Docker and Podman managers so the exec logic lives once.
    Binary mode (CONCEPT:CN-ECO.mcp.eco-2) demuxes stdout/stderr and base64-encodes the
    stdout bytes so non-text output (e.g. a screenshot PNG from ``maim`` /
    ``scrot -o /dev/stdout``) survives the JSON boundary uncorrupted — used by
    the computer-use actuator driving a gui-sandbox over the ssh:// docker socket.
    """
    if binary:
        exit_code, output = container.exec_run(command, detach=detach, demux=True)
        stdout_bytes, stderr_bytes = output if output else (None, None)
        return {
            "exit_code": exit_code,
            "output_b64": (
                base64.b64encode(stdout_bytes).decode("ascii") if stdout_bytes else None
            ),
            "stderr": stderr_bytes.decode("utf-8", "replace") if stderr_bytes else None,
            "command": command,
            "binary": True,
        }
    exit_code, output = container.exec_run(command, detach=detach)
    return {
        "exit_code": exit_code,
        "output": output.decode("utf-8") if output and not detach else None,
        "command": command,
    }


def _privacy_safe_shape(value: Any) -> str:
    """Describe a value without serializing operator data into logs."""

    if value is None:
        return "none"
    if isinstance(value, dict):
        return f"mapping(keys={len(value)})"
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"collection(items={len(value)})"
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        return f"{type(value).__name__}(length={len(value)})"
    return type(value).__name__


class ContainerManagerBase(ABC):
    def __init__(self, silent: bool = False, log_file: str | None = None):
        self.silent = silent
        self.setup_logging(log_file)

    def setup_logging(self, log_file: str | None = None):
        if not log_file:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_file = os.path.join(script_dir, "container_manager.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Container manager logging initialized")

    def log_action(
        self,
        action: str,
        params: dict | None = None,
        result: Any | None = None,
        error: Exception | None = None,
    ):
        safe_action = (
            action
            if action
            and len(action) <= 64
            and all(char.isalnum() or char in "_.-" for char in action)
            else "custom_action"
        )
        parameter_count = len(params) if isinstance(params, dict) else 0
        self.logger.info(
            "Container action started: action=%s parameter_count=%d",
            safe_action,
            parameter_count,
        )
        if result is not None:
            self.logger.info(
                "Container action completed: action=%s result_shape=%s",
                safe_action,
                _privacy_safe_shape(result),
            )
        if error:
            self.logger.error(
                "Container action failed: action=%s error_type=%s",
                safe_action,
                type(error).__name__,
            )

    def _extract_image_labels(self, attrs: dict) -> dict[str, str] | None:
        """Return an image's OCI labels from either list-summary or inspect ``attrs``.

        The Docker/Podman ``/images/json`` list summary carries ``Labels`` at the
        top level; the ``/images/{id}/json`` inspect shape nests them under
        ``Config.Labels``. Checked in that order so either source works.
        """
        labels = attrs.get("Labels")
        if not labels:
            labels = (attrs.get("Config") or {}).get("Labels")
        return labels or None

    def _format_size(self, size_bytes: int | float) -> str:
        """Helper to format bytes to human-readable (e.g., 1.23GB)."""
        size: float = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.2f}{unit}" if unit != "B" else f"{size}{unit}"
            size /= 1024.0
        return f"{size:.2f}PB"

    def _parse_timestamp(self, timestamp: Any) -> str:
        """Parse timestamp (integer, float, or string) to ISO 8601 string."""
        if not timestamp:
            return "unknown"

        if isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%S")
            except (ValueError, OSError):
                return "unknown"

        if isinstance(timestamp, str):
            formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%d",
            ]
            for fmt in formats:
                try:
                    parsed = datetime.strptime(timestamp, fmt)
                    return parsed.strftime("%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    continue
            return "unknown"

        return "unknown"

    @abstractmethod
    def get_version(self) -> dict:
        pass

    @abstractmethod
    def get_info(self) -> dict:
        pass

    @abstractmethod
    def list_images(self) -> list[ImageInfo]:
        pass

    @abstractmethod
    def pull_image(
        self, image: str, tag: str = "latest", platform: str | None = None
    ) -> dict:
        pass

    @abstractmethod
    def remove_image(self, image: str, force: bool = False) -> dict:
        pass

    @abstractmethod
    def prune_images(self, force: bool = False, all: bool = False) -> dict:
        pass

    @abstractmethod
    def list_containers(self, all: bool = False) -> list[ContainerInfo]:
        pass

    @abstractmethod
    def run_container(
        self,
        image: str,
        name: str | None = None,
        command: str | None = None,
        detach: bool = False,
        ports: dict[str, str] | None = None,
        volumes: dict[str, dict] | None = None,
        environment: dict[str, str] | list[str] | None = None,
        labels: dict[str, str] | None = None,
    ) -> dict:
        pass

    @abstractmethod
    def inspect_container(self, container_id: str) -> dict:
        pass

    @abstractmethod
    def stop_container(self, container_id: str, timeout: int = 10) -> dict:
        pass

    @abstractmethod
    def remove_container(self, container_id: str, force: bool = False) -> dict:
        pass

    @abstractmethod
    def prune_containers(self) -> dict:
        pass

    @abstractmethod
    def get_container_logs(self, container_id: str, tail: str = "50") -> str:
        pass

    @abstractmethod
    def exec_in_container(
        self,
        container_id: str,
        command: list[str],
        detach: bool = False,
        binary: bool = False,
    ) -> dict:
        pass

    @abstractmethod
    def list_volumes(self) -> list[VolumeInfo]:
        pass

    @abstractmethod
    def create_volume(self, name: str) -> VolumeInfo:
        pass

    @abstractmethod
    def remove_volume(self, name: str, force: bool = False) -> dict:
        pass

    @abstractmethod
    def prune_volumes(self, force: bool = False, all: bool = False) -> dict:
        pass

    @abstractmethod
    def list_networks(self) -> list[NetworkInfo]:
        pass

    @abstractmethod
    def create_network(self, name: str, driver: str = "bridge") -> NetworkInfo:
        pass

    @abstractmethod
    def remove_network(self, network_id: str) -> dict:
        pass

    @abstractmethod
    def prune_networks(self) -> dict:
        pass

    @abstractmethod
    def prune_system(self, force: bool = False, all: bool = False) -> dict:
        pass

    @abstractmethod
    def compose_up(
        self, compose_file: str, detach: bool = True, build: bool = False
    ) -> str:
        pass

    @abstractmethod
    def compose_down(self, compose_file: str) -> str:
        pass

    @abstractmethod
    def compose_ps(self, compose_file: str) -> str:
        pass

    @abstractmethod
    def compose_logs(self, compose_file: str, service: str | None = None) -> str:
        pass

    @abstractmethod
    def init_swarm(self, advertise_addr: str | None = None) -> dict:
        pass

    @abstractmethod
    def leave_swarm(self, force: bool = False) -> dict:
        pass

    @abstractmethod
    def list_nodes(self) -> list[dict]:
        pass

    @abstractmethod
    def list_services(self) -> list[dict]:
        pass

    @abstractmethod
    def create_service(
        self,
        name: str,
        image: str,
        replicas: int = 1,
        ports: dict[str, str] | None = None,
        mounts: list[str] | None = None,
    ) -> dict:
        pass

    @abstractmethod
    def remove_service(self, service_id: str) -> dict:
        pass

    @abstractmethod
    def inspect_node(self, node_id: str) -> dict:
        pass

    @abstractmethod
    def update_node(
        self,
        node_id: str,
        labels: dict[str, str] | None = None,
        role: str | None = None,
        availability: str | None = None,
        replace_labels: bool = False,
    ) -> dict:
        pass

    @abstractmethod
    def remove_node(self, node_id: str, force: bool = False) -> dict:
        pass

    @abstractmethod
    def inspect_service(self, service_id: str) -> dict:
        pass

    @abstractmethod
    def scale_service(self, service_id: str, replicas: int) -> dict:
        pass

    @abstractmethod
    def update_service(
        self,
        service_id: str,
        image: str | None = None,
        replicas: int | None = None,
        env: list[str] | None = None,
        constraints: list[str] | None = None,
        labels: dict[str, str] | None = None,
        force: bool = False,
    ) -> dict:
        pass

    @abstractmethod
    def service_ps(self, service_id: str) -> list[dict]:
        pass

    @abstractmethod
    def service_logs(self, service_id: str, tail: int = 100) -> dict:
        pass


def list_inventory_hosts() -> dict:
    """List the host aliases available for the ``host`` parameter.

    Each alias targets a remote machine's Docker daemon over SSH
    (``ssh://<user>@<hostname>:<port>``), resolved from the tunnel-manager
    inventory (``~/.config/agent-utilities/inventory.yaml``). Omit ``host`` to
    use the local Docker socket."""
    from tunnel_manager.tunnel_manager import HostManager

    hm = HostManager()
    hosts = {
        name: {
            "hostname": info.get("hostname"),
            "user": info.get("user", "genius"),
            "port": info.get("port", 22),
        }
        for name, info in (hm.hosts or {}).items()
    }
    return {
        "inventory_path": getattr(hm, "config_file", None),
        "count": len(hosts),
        "hosts": hosts,
    }


def resolve_host_from_inventory(host: str) -> dict:
    from tunnel_manager.tunnel_manager import HostManager

    hm = HostManager()
    if not hm.hosts:
        raise FileNotFoundError(
            f"No valid hosts configuration or inventory found (attempted: {hm.config_file})"
        )

    if host not in hm.hosts:
        raise ValueError(
            f"Host '{host}' not configured in inventory ({hm.config_file}). "
            f"Available hosts: {sorted(hm.hosts)} — or call cm_list_hosts."
        )

    hinfo = hm.hosts[host]
    return {
        "hostname": hinfo.get("hostname"),
        "user": hinfo.get("user", "genius"),
        "port": hinfo.get("port", 22),
        "identity_file": hinfo.get("key_path") or hinfo.get("identity_file"),
        "password": hinfo.get("password"),
    }


class DockerManager(ContainerManagerBase):
    def __init__(
        self, host: str | None = None, silent: bool = False, log_file: str | None = None
    ):
        super().__init__(silent, log_file)
        if docker is None:
            raise ImportError("Please install docker-py: pip install docker")
        try:
            if host:
                host_info = resolve_host_from_inventory(host)
                user = host_info.get("user")
                hostname = host_info.get("hostname")
                port = host_info.get("port", 22)
                if not hostname:
                    raise ValueError(
                        "No hostname was specified for the configured host"
                    )

                authority = f"{user}@" if user else ""
                base_url = f"ssh://{authority}{hostname}:{port}"
                self.logger.info("Connecting to configured remote Docker daemon")
                self.client = docker.DockerClient(base_url=base_url)  # type: ignore
            else:
                self.client = docker.from_env()  # type: ignore
        except Exception as e:
            self.logger.error("Operation failed: error_type=%s", type(e).__name__)
            raise RuntimeError("Configured Docker daemon is unavailable") from e

    def prune_system(self, force: bool = False, all: bool = False) -> dict:
        params = {"force": force, "all": all}
        try:
            filters = {"until": None} if all else {}
            result = self.client.system.prune(filters=filters, volumes=all)
            if result is None:
                result = {
                    "SpaceReclaimed": 0,
                    "ImagesDeleted": [],
                    "ContainersDeleted": [],
                    "VolumesDeleted": [],
                    "NetworksDeleted": [],
                }
            self.logger.debug(
                "Container prune result received: shape=%s",
                _privacy_safe_shape(result),
            )
            pruned = {
                "space_reclaimed": self._format_size(result.get("SpaceReclaimed", 0)),
                "images_removed": (
                    [img["Id"][7:19] for img in result.get("ImagesDeleted", [])]
                ),
                "containers_removed": (
                    [
                        (c.get("Id", "")[:12] if isinstance(c, dict) else str(c)[:12])
                        for c in (result.get("ContainersDeleted") or [])
                    ]
                ),
                "volumes_removed": (
                    [
                        (v.get("Name", "") if isinstance(v, dict) else str(v))
                        for v in (result.get("VolumesDeleted") or [])
                    ]
                ),
                "networks_removed": (
                    [
                        (n.get("Id", "")[:12] if isinstance(n, dict) else str(n)[:12])
                        for n in (result.get("NetworksDeleted") or [])
                    ]
                ),
            }
            self.log_action("prune_system", params, pruned)
            return pruned
        except Exception as e:
            self.log_action("prune_system", params, error=e)
            raise RuntimeError("Failed to prune system") from e

    def get_version(self) -> dict:
        params: dict[str, Any] = {}
        try:
            version = self.client.version()
            result = {
                "version": version.get("Version", "unknown"),
                "api_version": version.get("ApiVersion", "unknown"),
                "os": version.get("Os", "unknown"),
                "arch": version.get("Arch", "unknown"),
                "build_time": version.get("BuildTime", "unknown"),
            }
            self.log_action("get_version", params, result)
            return result
        except Exception as e:
            self.log_action("get_version", params, error=e)
            raise RuntimeError("Failed to get version") from e

    def get_info(self) -> dict:
        params: dict[str, Any] = {}
        try:
            info = self.client.info()
            result = {
                "containers_total": info.get("Containers", 0),
                "containers_running": info.get("ContainersRunning", 0),
                "images": info.get("Images", 0),
                "driver": info.get("Driver", "unknown"),
                "platform": f"{info.get('OperatingSystem', 'unknown')} {info.get('Architecture', 'unknown')}",
                "memory_total": self._format_size(info.get("MemTotal", 0)),
                "swap_total": self._format_size(info.get("SwapTotal", 0)),
            }
            self.log_action("get_info", params, result)
            return result
        except Exception as e:
            self.log_action("get_info", params, error=e)
            raise RuntimeError("Failed to get info") from e

    def list_images(self) -> list[ImageInfo]:
        params: dict[str, Any] = {}
        try:
            images = self.client.images.list()
            result = []
            for img in images:
                attrs = img.attrs
                repo_tags = attrs.get("RepoTags", [])
                repo_tag = repo_tags[0] if repo_tags else "<none>:<none>"
                repository, tag = (
                    repo_tag.rsplit(":", 1) if ":" in repo_tag else ("<none>", "<none>")
                )

                created = attrs.get("Created", None)
                created_str = self._parse_timestamp(created)

                size_bytes = attrs.get("Size", 0)
                size_str = self._format_size(size_bytes) if size_bytes else "0B"

                simplified: dict[str, Any] = {
                    "repository": repository,
                    "tag": tag,
                    "id": (
                        attrs.get("Id", "unknown")[7:19]
                        if attrs.get("Id")
                        else "unknown"
                    ),
                    "created": created_str,
                    "size": size_str,
                    "labels": self._extract_image_labels(attrs),
                }
                result.append(ImageInfo(**simplified))

            self.log_action("list_images", params, [i.model_dump() for i in result])
            return result
        except Exception as e:
            self.log_action("list_images", params, error=e)
            raise RuntimeError("Failed to list images") from e

    def pull_image(
        self, image: str, tag: str = "latest", platform: str | None = None
    ) -> dict:
        params = {"image": image, "tag": tag, "platform": platform}
        try:
            # Don't append tag if image already contains one
            image_ref = f"{image}:{tag}" if ":" not in image else image
            img = self.client.images.pull(image_ref, platform=platform)
            attrs = img.attrs
            repo_tags = attrs.get("RepoTags", [])
            repo_tag = repo_tags[0] if repo_tags else image_ref
            repository, tag = (
                repo_tag.rsplit(":", 1) if ":" in repo_tag else (image, tag)
            )
            created = attrs.get("Created", None)
            created_str = self._parse_timestamp(created)
            size_bytes = attrs.get("Size", 0)
            size_str = self._format_size(size_bytes) if size_bytes else "0B"
            result = {
                "repository": repository,
                "tag": tag,
                "id": (
                    attrs.get("Id", "unknown")[7:19] if attrs.get("Id") else "unknown"
                ),
                "created": created_str,
                "size": size_str,
            }
            self.log_action("pull_image", params, result)
            return result
        except Exception as e:
            self.log_action("pull_image", params, error=e)
            raise RuntimeError("Failed to pull image") from e

    def remove_image(self, image: str, force: bool = False) -> dict:
        params = {"image": image, "force": force}
        try:
            self.client.images.remove(image, force=force)
            result = {"removed": image}
            self.log_action("remove_image", params, result)
            return result
        except Exception as e:
            self.log_action("remove_image", params, error=e)
            raise RuntimeError("Failed to remove image") from e

    def prune_images(self, force: bool = False, all: bool = False) -> dict:
        params = {"force": force, "all": all}
        try:
            if all:
                images = self.client.images.list(all=True)
                removed = []
                for img in images:
                    try:
                        for tag in img.attrs.get("RepoTags", []):
                            self.client.images.remove(tag, force=force)
                            removed.append(img.attrs["Id"][7:19])
                    except Exception as e:
                        self.logger.info(
                            "Image removal failed: error_type=%s", type(e).__name__
                        )
                        continue
                result = {
                    "images_removed": removed,
                    "space_reclaimed": "N/A (all images)",
                }
            else:
                filters = {"dangling": True} if not all else {}
                result = self.client.images.prune(filters=filters)
                if result is None:
                    result = {"SpaceReclaimed": 0, "ImagesDeleted": []}  # type: ignore
                self.logger.debug(
                    "Image prune result received: shape=%s",
                    _privacy_safe_shape(result),
                )
                space_reclaimed = result.get("SpaceReclaimed", 0)
                if not isinstance(space_reclaimed, (int, float)):
                    space_reclaimed = 0
                pruned = {
                    "space_reclaimed": self._format_size(space_reclaimed),
                    "images_removed": (
                        [
                            (
                                (
                                    img.get("Deleted")
                                    or img.get("Untagged")
                                    or img.get("Id", "")
                                )[-12:]
                                if isinstance(img, dict)
                                else str(img)[-12:]
                            )
                            for img in (result.get("ImagesDeleted") or [])
                        ]
                    ),
                }
                result = pruned
            self.log_action("prune_images", params, result)
            return result
        except Exception as e:
            self.log_action("prune_images", params, error=e)
            raise RuntimeError("Failed to prune images") from e

    def list_containers(self, all: bool = False) -> list[ContainerInfo]:
        params = {"all": all}
        try:
            containers = self.client.containers.list(all=all, ignore_removed=True)
            result = []
            for c in containers:
                attrs = c.attrs
                ports = attrs.get("NetworkSettings", {}).get("Ports", {})
                port_mappings = []
                for container_port, host_ports in ports.items():
                    if host_ports:
                        for hp in host_ports:
                            port_mappings.append(
                                f"{hp.get('HostIp', '0.0.0.0')}:{hp.get('HostPort')}->{container_port}"  # nosec B104
                            )
                created = attrs.get("Created", None)
                created_str = self._parse_timestamp(created)
                simplified = {
                    "id": attrs.get("Id", "unknown")[:12],
                    "image": attrs.get("Config", {}).get("Image", "unknown"),
                    "name": attrs.get("Name", "unknown").lstrip("/"),
                    "status": attrs.get("State", {}).get("Status", "unknown"),
                    "ports": ", ".join(port_mappings) if port_mappings else "none",
                    "created": created_str,
                }
                result.append(ContainerInfo(**simplified))
            self.log_action("list_containers", params, [c.model_dump() for c in result])
            return result
        except Exception as e:
            self.log_action("list_containers", params, error=e)
            raise RuntimeError("Failed to list containers") from e

    def run_container(
        self,
        image: str,
        name: str | None = None,
        command: str | None = None,
        detach: bool = False,
        ports: dict[str, str] | None = None,
        volumes: dict[str, dict] | None = None,
        environment: dict[str, str] | list[str] | None = None,
        labels: dict[str, str] | None = None,
    ) -> dict:
        params = {
            "image": image,
            "name": name,
            "command": command,
            "detach": detach,
            "ports": ports,
            "volumes": volumes,
            "environment": environment,
            "labels": labels,
        }
        try:
            container = self.client.containers.run(  # type: ignore
                image,
                name=name,
                command=command,
                detach=detach,
                ports=ports,
                volumes=volumes,
                environment=environment,
                labels=labels,
            )
            if not detach:
                result = {"output": container.decode("utf-8") if container else ""}
                self.log_action("run_container", params, result)
                return result
            attrs = container.attrs
            port_mappings = []
            if ports:
                network_settings = attrs.get("NetworkSettings", {})
                container_ports = network_settings.get("Ports", {})
                if container_ports:
                    for container_port, host_ports in container_ports.items():
                        if host_ports:
                            for hp in host_ports:
                                port_mappings.append(
                                    f"{hp.get('HostIp', '0.0.0.0')}:{hp.get('HostPort')}->{container_port}"  # nosec B104
                                )
            created = attrs.get("Created", None)
            created_str = self._parse_timestamp(created)
            result = {
                "id": attrs.get("Id", "unknown")[:12],
                "image": attrs.get("Config", {}).get("Image", image),
                "name": attrs.get("Name", name or "unknown").lstrip("/"),
                "status": attrs.get("State", {}).get("Status", "unknown"),
                "ports": ", ".join(port_mappings) if port_mappings else "none",
                "created": created_str,
            }
            self.log_action("run_container", params, result)
            return result
        except Exception as e:
            self.log_action("run_container", params, error=e)
            raise RuntimeError("Failed to run container") from e

    def inspect_container(self, container_id: str) -> dict:
        params = {"container_id": container_id}
        try:
            container = self.client.containers.get(container_id)
            result = container.attrs
            self.log_action("inspect_container", params, {"id": container_id})
            return result
        except Exception as e:
            self.log_action("inspect_container", params, error=e)
            raise RuntimeError("Failed to inspect container") from e

    def stop_container(self, container_id: str, timeout: int = 10) -> dict:
        params = {"container_id": container_id, "timeout": timeout}
        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=timeout)
            result = {"stopped": container_id}
            self.log_action("stop_container", params, result)
            return result
        except Exception as e:
            self.log_action("stop_container", params, error=e)
            raise RuntimeError("Failed to stop container") from e

    def remove_container(self, container_id: str, force: bool = False) -> dict:
        params = {"container_id": container_id, "force": force}
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=force)
            result = {"removed": container_id}
            self.log_action("remove_container", params, result)
            return result
        except Exception as e:
            self.log_action("remove_container", params, error=e)
            raise RuntimeError("Failed to remove container") from e

    def prune_containers(self) -> dict:
        params: dict[str, Any] = {}
        try:
            result = self.client.containers.prune()
            self.logger.debug(
                "Container prune result received: shape=%s",
                _privacy_safe_shape(result),
            )
            if result is None:
                result = {"SpaceReclaimed": 0, "ContainersDeleted": []}
            pruned = {
                "space_reclaimed": self._format_size(result.get("SpaceReclaimed", 0)),
                "containers_removed": (
                    [
                        (c.get("Id", "")[:12] if isinstance(c, dict) else str(c)[:12])
                        for c in (result.get("ContainersDeleted") or [])
                    ]
                ),
            }
            self.log_action("prune_containers", params, pruned)
            return pruned
        except TypeError as e:
            self.logger.error("Container prune failed: error_type=TypeError")
            self.log_action("prune_containers", params, error=e)
            raise RuntimeError("Failed to prune containers") from e
        except Exception as e:
            self.logger.error(
                "Container prune failed: error_type=%s", type(e).__name__
            )
            self.log_action("prune_containers", params, error=e)
            raise RuntimeError("Failed to prune containers") from e

    def get_container_logs(self, container_id: str, tail: str = "50") -> str:
        params = {"container_id": container_id, "tail": tail}
        try:
            container = self.client.containers.get(container_id)
            logs = container.logs(tail=tail).decode("utf-8")
            self.log_action("get_container_logs", params, logs[:1000])
            return logs
        except Exception as e:
            self.log_action("get_container_logs", params, error=e)
            raise RuntimeError("Failed to get container logs") from e

    def exec_in_container(
        self,
        container_id: str,
        command: list[str],
        detach: bool = False,
        binary: bool = False,
    ) -> dict:
        params = {
            "container_id": container_id,
            "command": command,
            "detach": detach,
            "binary": binary,
        }
        try:
            container = self.client.containers.get(container_id)
            result = _build_exec_result(container, command, detach, binary)
            # Don't log a full base64 PNG — summarise binary results.
            log_result = (
                {"exit_code": result["exit_code"], "binary": True} if binary else result
            )
            self.log_action("exec_in_container", params, log_result)
            return result
        except Exception as e:
            self.log_action("exec_in_container", params, error=e)
            raise RuntimeError("Failed to exec in container") from e

    def list_volumes(self) -> list[VolumeInfo]:
        params: dict[str, Any] = {}
        try:
            volumes = self.client.volumes.list()
            result = [
                VolumeInfo(
                    **{
                        "name": v.attrs.get("Name", "unknown"),
                        "driver": v.attrs.get("Driver", "unknown"),
                        "mountpoint": v.attrs.get("Mountpoint", "unknown"),
                        "created": v.attrs.get("CreatedAt", "unknown"),
                    }
                )
                for v in volumes
            ]
            self.log_action("list_volumes", params, [v.model_dump() for v in result])
            return result
        except Exception as e:
            self.log_action("list_volumes", params, error=e)
            raise RuntimeError("Failed to list volumes") from e

    def create_volume(self, name: str) -> VolumeInfo:
        params = {"name": name}
        try:
            volume = self.client.volumes.create(name=name)
            attrs = volume.attrs
            result = VolumeInfo(
                **{
                    "name": attrs.get("Name", name),
                    "driver": attrs.get("Driver", "unknown"),
                    "mountpoint": attrs.get("Mountpoint", "unknown"),
                    "created": attrs.get("CreatedAt", "unknown"),
                }
            )
            self.log_action("create_volume", params, result.model_dump())
            return result
        except Exception as e:
            self.log_action("create_volume", params, error=e)
            raise RuntimeError("Failed to create volume") from e

    def remove_volume(self, name: str, force: bool = False) -> dict:
        params = {"name": name, "force": force}
        try:
            volume = self.client.volumes.get(name)
            volume.remove(force=force)
            result = {"removed": name}
            self.log_action("remove_volume", params, result)
            return result
        except Exception as e:
            self.log_action("remove_volume", params, error=e)
            raise RuntimeError("Failed to remove volume") from e

    def prune_volumes(self, force: bool = False, all: bool = False) -> dict:
        params = {"force": force, "all": all}
        try:
            if all:
                volumes = self.client.volumes.list(all=True)
                removed = []
                for v in volumes:
                    try:
                        v.remove(force=force)
                        removed.append(v.attrs["Name"])
                    except Exception as e:
                        self.logger.info(
                            "Volume removal failed: error_type=%s", type(e).__name__
                        )
                        continue
                result = {
                    "volumes_removed": removed,
                    "space_reclaimed": "N/A (all volumes)",
                }
            else:
                result = self.client.volumes.prune()
                if result is None:
                    result = {"SpaceReclaimed": 0, "VolumesDeleted": []}  # type: ignore
                self.logger.debug(
                    "Volume prune result received: shape=%s",
                    _privacy_safe_shape(result),
                )
                space_reclaimed = result.get("SpaceReclaimed", 0)
                if not isinstance(space_reclaimed, (int, float)):
                    space_reclaimed = 0
                pruned = {
                    "space_reclaimed": self._format_size(space_reclaimed),
                    "volumes_removed": (
                        [
                            (v.get("Name", "") if isinstance(v, dict) else str(v))
                            for v in (result.get("VolumesDeleted") or [])
                        ]
                    ),
                }
                result = pruned
            self.log_action("prune_volumes", params, result)
            return result
        except Exception as e:
            self.log_action("prune_volumes", params, error=e)
            raise RuntimeError("Failed to prune volumes") from e

    def list_networks(self) -> list[NetworkInfo]:
        params: dict[str, Any] = {}
        try:
            networks = self.client.networks.list()
            result = []
            for net in networks:
                attrs = net.attrs
                containers = len(attrs.get("Containers", {}))
                created = attrs.get("Created", None)
                created_str = self._parse_timestamp(created)
                simplified = {
                    "id": attrs.get("Id", "unknown")[:12],
                    "name": attrs.get("Name", "unknown"),
                    "driver": attrs.get("Driver", "unknown"),
                    "scope": attrs.get("Scope", "unknown"),
                    "containers": containers,
                    "created": created_str,
                }
                result.append(NetworkInfo(**simplified))
            self.log_action("list_networks", params, [n.model_dump() for n in result])
            return result
        except Exception as e:
            self.log_action("list_networks", params, error=e)
            raise RuntimeError("Failed to list networks") from e

    def create_network(self, name: str, driver: str = "bridge") -> NetworkInfo:
        params = {"name": name, "driver": driver}
        try:
            network = self.client.networks.create(name, driver=driver)
            attrs = network.attrs
            created = attrs.get("Created", None)
            created_str = self._parse_timestamp(created)
            result = NetworkInfo(
                **{
                    "id": attrs.get("Id", "unknown")[:12],
                    "name": attrs.get("Name", name),
                    "driver": attrs.get("Driver", driver),
                    "scope": attrs.get("Scope", "unknown"),
                    "created": created_str,
                }
            )
            self.log_action("create_network", params, result.model_dump())
            return result
        except Exception as e:
            self.log_action("create_network", params, error=e)
            raise RuntimeError("Failed to create network") from e

    def remove_network(self, network_id: str) -> dict:
        params = {"network_id": network_id}
        try:
            network = self.client.networks.get(network_id)
            network.remove()
            result = {"removed": network_id}
            self.log_action("remove_network", params, result)
            return result
        except Exception as e:
            self.log_action("remove_network", params, error=e)
            raise RuntimeError("Failed to remove network") from e

    def prune_networks(self) -> dict:
        params: dict[str, Any] = {}
        try:
            result = self.client.networks.prune()
            if result is None:
                result = {"SpaceReclaimed": 0, "NetworksDeleted": []}
            self.logger.debug(
                "Network prune result received: shape=%s",
                _privacy_safe_shape(result),
            )
            pruned = {
                "space_reclaimed": self._format_size(result.get("SpaceReclaimed", 0)),
                "networks_removed": (
                    [
                        (n.get("Id", "")[:12] if isinstance(n, dict) else str(n)[:12])
                        for n in (result.get("NetworksDeleted") or [])
                    ]
                ),
            }
            self.log_action("prune_networks", params, pruned)
            return pruned
        except Exception as e:
            self.log_action("prune_networks", params, error=e)
            raise RuntimeError("Failed to prune networks") from e

    def compose_up(
        self, compose_file: str, detach: bool = True, build: bool = False
    ) -> str:
        params = {"compose_file": compose_file, "detach": detach, "build": build}
        try:
            cmd = ["docker", "compose", "-f", compose_file, "up"]
            if build:
                cmd.append("--build")
            if detach:
                cmd.append("-d")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_up", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_up", params, error=e)
            raise RuntimeError("Failed to compose up") from e

    def compose_down(self, compose_file: str) -> str:
        params = {"compose_file": compose_file}
        try:
            cmd = ["docker", "compose", "-f", compose_file, "down"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_down", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_down", params, error=e)
            raise RuntimeError("Failed to compose down") from e

    def compose_ps(self, compose_file: str) -> str:
        params = {"compose_file": compose_file}
        try:
            cmd = ["docker", "compose", "-f", compose_file, "ps"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_ps", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_ps", params, error=e)
            raise RuntimeError("Failed to compose ps") from e

    def compose_logs(self, compose_file: str, service: str | None = None) -> str:
        params = {"compose_file": compose_file, "service": service}
        try:
            cmd = ["docker", "compose", "-f", compose_file, "logs"]
            if service:
                cmd.append(service)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_logs", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_logs", params, error=e)
            raise RuntimeError("Failed to compose logs") from e

    def init_swarm(self, advertise_addr: str | None = None) -> dict:
        params = {"advertise_addr": advertise_addr}
        try:
            swarm_id = self.client.swarm.init(advertise_addr=advertise_addr)
            result = {"swarm_id": swarm_id}
            self.log_action("init_swarm", params, result)
            return result
        except Exception as e:
            self.log_action("init_swarm", params, error=e)
            raise RuntimeError("Failed to init swarm") from e

    def leave_swarm(self, force: bool = False) -> dict:
        params = {"force": force}
        try:
            self.client.swarm.leave(force=force)
            result = {"left": True}
            self.log_action("leave_swarm", params, result)
            return result
        except Exception as e:
            self.log_action("leave_swarm", params, error=e)
            raise RuntimeError("Failed to leave swarm") from e

    def list_nodes(self) -> list[dict]:
        params: dict[str, Any] = {}
        try:
            nodes = self.client.nodes.list()
            result = []
            for node in nodes:
                attrs = node.attrs
                spec = attrs.get("Spec", {})
                status = attrs.get("Status", {})
                created = attrs.get("CreatedAt", "unknown")
                updated = attrs.get("UpdatedAt", "unknown")
                simplified = {
                    "id": attrs.get("ID", "unknown")[7:19],
                    "hostname": spec.get("Name", "unknown"),
                    "role": spec.get("Role", "unknown"),
                    "status": status.get("State", "unknown"),
                    "availability": spec.get("Availability", "unknown"),
                    "created": created,
                    "updated": updated,
                }
                result.append(simplified)
            self.log_action("list_nodes", params, result)
            return result
        except Exception as e:
            self.log_action("list_nodes", params, error=e)
            raise RuntimeError("Failed to list nodes") from e

    def list_services(self) -> list[dict]:
        params: dict[str, Any] = {}
        try:
            services = self.client.services.list()
            result = []
            for service in services:
                attrs = service.attrs
                spec = attrs.get("Spec", {})
                endpoint = attrs.get("Endpoint", {})
                ports = endpoint.get("Ports", [])
                port_mappings = [
                    f"{p.get('PublishedPort')}->{p.get('TargetPort')}/{p.get('Protocol')}"
                    for p in ports
                    if p.get("PublishedPort")
                ]
                created = attrs.get("CreatedAt", "unknown")
                updated = attrs.get("UpdatedAt", "unknown")
                simplified = {
                    "id": attrs.get("ID", "unknown")[7:19],
                    "name": spec.get("Name", "unknown"),
                    "image": spec.get("TaskTemplate", {})
                    .get("ContainerSpec", {})
                    .get("Image", "unknown"),
                    "replicas": spec.get("Mode", {})
                    .get("Replicated", {})
                    .get("Replicas", 0),
                    "ports": ", ".join(port_mappings) if port_mappings else "none",
                    "created": created,
                    "updated": updated,
                }
                result.append(simplified)
            self.log_action("list_services", params, result)
            return result
        except Exception as e:
            self.log_action("list_services", params, error=e)
            raise RuntimeError("Failed to list services") from e

    def create_service(
        self,
        name: str,
        image: str,
        replicas: int = 1,
        ports: dict[str, str] | None = None,
        mounts: list[str] | None = None,
    ) -> dict:
        params = {
            "name": name,
            "image": image,
            "replicas": replicas,
            "ports": ports,
            "mounts": mounts,
        }
        try:
            mode = {"mode": "replicated", "replicas": replicas}
            endpoint_spec = None
            if ports:
                port_list = [
                    {
                        "Protocol": "tcp",
                        "PublishedPort": int(host_port),
                        "TargetPort": int(container_port.split("/")[0]),
                    }
                    for container_port, host_port in ports.items()
                ]
                if docker is not None:
                    endpoint_spec = docker.types.EndpointSpec(ports=port_list)  # type: ignore
            service = self.client.services.create(
                image,
                name=name,
                mode=mode,
                mounts=mounts,
                endpoint_spec=endpoint_spec,
            )
            attrs = service.attrs
            spec = attrs.get("Spec", {})
            endpoint = attrs.get("Endpoint", {})
            ports_list = endpoint.get("Ports", []) or []
            port_mappings = [
                f"{p.get('PublishedPort')}->{p.get('TargetPort')}/{p.get('Protocol')}"
                for p in ports_list
                if isinstance(p, dict) and p.get("PublishedPort")
            ]
            created = attrs.get("CreatedAt", "unknown")
            result = {
                "id": attrs.get("ID", "unknown")[7:19],
                "name": spec.get("Name", name),
                "image": spec.get("TaskTemplate", {})
                .get("ContainerSpec", {})
                .get("Image", image),
                "replicas": spec.get("Mode", {})
                .get("Replicated", {})
                .get("Replicas", replicas),
                "ports": ", ".join(port_mappings) if port_mappings else "none",
                "created": created,
            }
            self.log_action("create_service", params, result)
            return result
        except Exception as e:
            self.log_action("create_service", params, error=e)
            raise RuntimeError("Failed to create service") from e

    def remove_service(self, service_id: str) -> dict:
        params = {"service_id": service_id}
        try:
            service = self.client.services.get(service_id)
            service.remove()
            result = {"removed": service_id}
            self.log_action("remove_service", params, result)
            return result
        except Exception as e:
            self.log_action("remove_service", params, error=e)
            raise RuntimeError("Failed to remove service") from e

    # ------------------------------------------------------------------
    # Swarm node operations
    # ------------------------------------------------------------------
    def _resolve_node(self, node_id: str):
        """Resolve a swarm node by full/short ID or hostname."""
        try:
            return self.client.nodes.get(node_id)
        except Exception:
            for node in self.client.nodes.list():
                attrs = node.attrs
                hostname = attrs.get("Description", {}).get("Hostname")
                if (
                    attrs.get("ID", "").startswith(node_id)
                    or hostname == node_id
                    or attrs.get("Spec", {}).get("Name") == node_id
                ):
                    return node
            raise RuntimeError(f"Node '{node_id}' not found") from None

    @staticmethod
    def _node_summary(node) -> dict:
        attrs = node.attrs
        spec = attrs.get("Spec", {})
        desc = attrs.get("Description", {})
        status = attrs.get("Status", {})
        return {
            "id": attrs.get("ID", "unknown"),
            "hostname": desc.get("Hostname", "unknown"),
            "role": spec.get("Role", "unknown"),
            "availability": spec.get("Availability", "unknown"),
            "state": status.get("State", "unknown"),
            "addr": status.get("Addr", "unknown"),
            "labels": spec.get("Labels", {}) or {},
            "engine_version": desc.get("Engine", {}).get("EngineVersion", "unknown"),
            "platform": desc.get("Platform", {}),
            "manager": attrs.get("ManagerStatus"),
        }

    def inspect_node(self, node_id: str) -> dict:
        params = {"node_id": node_id}
        try:
            node = self._resolve_node(node_id)
            result = self._node_summary(node)
            self.log_action("inspect_node", params, result)
            return result
        except Exception as e:
            self.log_action("inspect_node", params, error=e)
            raise RuntimeError("Failed to inspect node") from e

    def update_node(
        self,
        node_id: str,
        labels: dict[str, str] | None = None,
        role: str | None = None,
        availability: str | None = None,
        replace_labels: bool = False,
    ) -> dict:
        """Update a node's labels (merge by default), role, or availability.

        ``role`` is one of ``manager``/``worker`` (promote/demote);
        ``availability`` is ``active``/``pause``/``drain``.
        """
        params = {
            "node_id": node_id,
            "labels": labels,
            "role": role,
            "availability": availability,
            "replace_labels": replace_labels,
        }
        try:
            node = self._resolve_node(node_id)
            spec = dict(node.attrs.get("Spec", {}) or {})
            current_labels = dict(spec.get("Labels") or {})
            if labels is not None:
                current_labels = (
                    dict(labels) if replace_labels else {**current_labels, **labels}
                )
            new_spec: dict[str, Any] = {
                "Availability": availability or spec.get("Availability", "active"),
                "Role": role or spec.get("Role", "worker"),
                "Labels": current_labels,
            }
            if spec.get("Name"):
                new_spec["Name"] = spec["Name"]
            node.update(new_spec)
            node.reload()
            result = self._node_summary(node)
            self.log_action("update_node", params, result)
            return result
        except Exception as e:
            self.log_action("update_node", params, error=e)
            raise RuntimeError("Failed to update node") from e

    def remove_node(self, node_id: str, force: bool = False) -> dict:
        params = {"node_id": node_id, "force": force}
        try:
            node = self._resolve_node(node_id)
            self.client.api.remove_node(node.id, force=force)
            result = {"removed": node.id}
            self.log_action("remove_node", params, result)
            return result
        except Exception as e:
            self.log_action("remove_node", params, error=e)
            raise RuntimeError("Failed to remove node") from e

    # ------------------------------------------------------------------
    # Swarm service operations
    # ------------------------------------------------------------------
    def inspect_service(self, service_id: str) -> dict:
        params = {"service_id": service_id}
        try:
            service = self.client.services.get(service_id)
            result = service.attrs
            self.log_action("inspect_service", params, {"id": service.id})
            return result
        except Exception as e:
            self.log_action("inspect_service", params, error=e)
            raise RuntimeError("Failed to inspect service") from e

    def scale_service(self, service_id: str, replicas: int) -> dict:
        params = {"service_id": service_id, "replicas": replicas}
        try:
            service = self.client.services.get(service_id)
            service.scale(replicas)
            result = {"service": service.id, "replicas": replicas, "scaled": True}
            self.log_action("scale_service", params, result)
            return result
        except Exception as e:
            self.log_action("scale_service", params, error=e)
            raise RuntimeError("Failed to scale service") from e

    def update_service(
        self,
        service_id: str,
        image: str | None = None,
        replicas: int | None = None,
        env: list[str] | None = None,
        constraints: list[str] | None = None,
        labels: dict[str, str] | None = None,
        force: bool = False,
    ) -> dict:
        """Update a service in place, preserving unspecified spec fields.

        Performs a read-modify-write of the full service spec so that env,
        mounts, networks and other settings are not reset to defaults (the
        footgun of the high-level ``Service.update``).
        """
        params = {
            "service_id": service_id,
            "image": image,
            "replicas": replicas,
            "env": env,
            "constraints": constraints,
            "labels": labels,
            "force": force,
        }
        try:
            service = self.client.services.get(service_id)
            attrs = service.attrs
            spec = dict(attrs.get("Spec", {}) or {})
            version = attrs.get("Version", {}).get("Index")
            task_template = dict(spec.get("TaskTemplate", {}) or {})
            container_spec = dict(task_template.get("ContainerSpec", {}) or {})
            if image:
                container_spec["Image"] = image
            if env is not None:
                container_spec["Env"] = env
            task_template["ContainerSpec"] = container_spec
            if constraints is not None:
                placement = dict(task_template.get("Placement", {}) or {})
                placement["Constraints"] = constraints
                task_template["Placement"] = placement
            if force:
                task_template["ForceUpdate"] = (
                    int(task_template.get("ForceUpdate", 0) or 0) + 1
                )
            mode = spec.get("Mode")
            if replicas is not None and isinstance(mode, dict) and "Replicated" in mode:
                mode = {"Replicated": {"Replicas": replicas}}
            new_labels = spec.get("Labels")
            if labels is not None:
                new_labels = {**(spec.get("Labels") or {}), **labels}
            self.client.api.update_service(
                service.id,
                version,
                task_template=task_template,
                name=spec.get("Name"),
                labels=new_labels,
                mode=mode,
                endpoint_spec=spec.get("EndpointSpec"),
            )
            result = {
                "service": service.id,
                "updated": True,
                "image": container_spec.get("Image"),
            }
            self.log_action("update_service", params, result)
            return result
        except Exception as e:
            self.log_action("update_service", params, error=e)
            raise RuntimeError("Failed to update service") from e

    def service_ps(self, service_id: str) -> list[dict]:
        params = {"service_id": service_id}
        try:
            service = self.client.services.get(service_id)
            result = []
            for t in service.tasks():
                status = t.get("Status", {})
                result.append(
                    {
                        "id": t.get("ID", "")[:12],
                        "node": t.get("NodeID", "")[:12],
                        "desired_state": t.get("DesiredState", "unknown"),
                        "state": status.get("State", "unknown"),
                        "error": status.get("Err", ""),
                        "timestamp": status.get("Timestamp", "unknown"),
                    }
                )
            self.log_action("service_ps", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("service_ps", params, error=e)
            raise RuntimeError("Failed to list service tasks") from e

    def service_logs(self, service_id: str, tail: int = 100) -> dict:
        params = {"service_id": service_id, "tail": tail}
        try:
            service = self.client.services.get(service_id)
            raw = service.logs(stdout=True, stderr=True, tail=tail, timestamps=True)
            if isinstance(raw, bytes):
                text = raw.decode(errors="replace")
            else:
                text = b"".join(raw).decode(errors="replace")
            result = {"service": service.id, "logs": text}
            self.log_action("service_logs", params, {"service": service.id})
            return result
        except Exception as e:
            self.log_action("service_logs", params, error=e)
            raise RuntimeError("Failed to get service logs") from e

    # ------------------------------------------------------------------
    # Swarm / service / stack / config / secret / node operations
    # (the function-based cm_docker_swarm surface — real SDK/CLI calls)
    # ------------------------------------------------------------------
    def _docker_cli(self, args: list[str], action: str) -> str:
        """Run ``docker <args>`` and return stdout, raising on failure.

        Used only for surfaces the docker SDK does not expose (``docker
        stack`` has no SDK equivalent).
        """
        cmd = ["docker", *args]
        result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603 B607
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout

    def docker_swarm_init(
        self, advertise_addr: str, listen_addr: str | None = None
    ) -> dict:
        """Initialize a swarm (real ``swarm.init`` via :meth:`init_swarm`)."""
        params = {"advertise_addr": advertise_addr, "listen_addr": listen_addr}
        try:
            result = self.init_swarm(advertise_addr)
            self.log_action("docker_swarm_init", params, result)
            return result
        except Exception as e:
            self.log_action("docker_swarm_init", params, error=e)
            raise RuntimeError("Failed to initialize swarm") from e

    def docker_swarm_join(
        self, remote_addr: str, token: str, worker: bool = True
    ) -> dict:
        """Join an existing swarm as a worker/manager."""
        params = {"remote_addr": remote_addr, "worker": worker}
        try:
            joined = self.client.swarm.join(
                remote_addrs=[remote_addr], join_token=token
            )
            result = {"remote_addr": remote_addr, "worker": worker, "joined": joined}
            self.log_action("docker_swarm_join", params, result)
            return result
        except Exception as e:
            self.log_action("docker_swarm_join", params, error=e)
            raise RuntimeError("Failed to join swarm") from e

    def docker_swarm_leave(self, force: bool = False) -> dict:
        """Leave the swarm (real ``swarm.leave`` via :meth:`leave_swarm`)."""
        params = {"force": force}
        try:
            result = self.leave_swarm(force=force)
            self.log_action("docker_swarm_leave", params, result)
            return result
        except Exception as e:
            self.log_action("docker_swarm_leave", params, error=e)
            raise RuntimeError("Failed to leave swarm") from e

    @staticmethod
    def _ports_list_to_map(ports: list | None) -> dict[str, str] | None:
        """Translate ``["8080:80", ...]`` into ``{"80/tcp": "8080"}``."""
        if not ports:
            return None
        mapping: dict[str, str] = {}
        for spec in ports:
            text = str(spec)
            if ":" in text:
                host_port, container_port = text.split(":", 1)
            else:
                host_port = container_port = text
            key = container_port if "/" in container_port else f"{container_port}/tcp"
            mapping[key] = host_port
        return mapping

    def docker_service_create(
        self,
        service_name: str,
        image: str,
        replicas: int = 1,
        ports: list | None = None,
    ) -> dict:
        """Create a service (real ``services.create`` via :meth:`create_service`)."""
        params = {
            "service_name": service_name,
            "image": image,
            "replicas": replicas,
            "ports": ports,
        }
        try:
            result = self.create_service(
                service_name,
                image,
                replicas=replicas,
                ports=self._ports_list_to_map(ports),
            )
            self.log_action("docker_service_create", params, result)
            return result
        except Exception as e:
            self.log_action("docker_service_create", params, error=e)
            raise RuntimeError("Failed to create service") from e

    def docker_service_list(self) -> list[dict]:
        """List services (real ``services.list`` via :meth:`list_services`)."""
        return self.list_services()

    def docker_service_update(
        self, service_name: str, image: str | None = None, replicas: int | None = None
    ) -> dict:
        """Update a service in place (real update via :meth:`update_service`)."""
        return self.update_service(service_name, image=image, replicas=replicas)

    def docker_service_rm(self, service_name: str) -> dict:
        """Remove a service (real ``service.remove`` via :meth:`remove_service`)."""
        return self.remove_service(service_name)

    def docker_service_logs(self, service_name: str, tail_lines: int = 100) -> dict:
        """Fetch service logs (real ``service.logs`` via :meth:`service_logs`)."""
        return self.service_logs(service_name, tail=tail_lines)

    def docker_service_ps(self) -> list[dict]:
        """List every task across all services (real ``service.tasks()``)."""
        params: dict[str, Any] = {}
        try:
            result = []
            for service in self.client.services.list():
                name = service.attrs.get("Spec", {}).get("Name", service.id)
                for t in service.tasks():
                    status = t.get("Status", {})
                    result.append(
                        {
                            "service_name": name,
                            "task_id": t.get("ID", "")[:12],
                            "node": t.get("NodeID", "")[:12],
                            "desired_state": t.get("DesiredState", "unknown"),
                            "current_state": status.get("State", "unknown"),
                            "error": status.get("Err", ""),
                        }
                    )
            self.log_action("docker_service_ps", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("docker_service_ps", params, error=e)
            raise RuntimeError("Failed to list service tasks") from e

    def docker_stack_deploy(self, stack_name: str, compose_file: str) -> dict:
        """Deploy a stack via ``docker stack deploy`` (no SDK equivalent)."""
        params = {"stack_name": stack_name, "compose_file": compose_file}
        try:
            output = self._docker_cli(
                ["stack", "deploy", "-c", compose_file, stack_name],
                "docker_stack_deploy",
            )
            result = {
                "stack_name": stack_name,
                "compose_file": compose_file,
                "output": output.strip(),
                "status": "deployed",
            }
            self.log_action("docker_stack_deploy", params, result)
            return result
        except Exception as e:
            self.log_action("docker_stack_deploy", params, error=e)
            raise RuntimeError("Failed to deploy stack") from e

    def docker_stack_services(self, stack_name: str) -> list[dict]:
        """List a stack's services via ``docker stack services`` (no SDK)."""
        params = {"stack_name": stack_name}
        try:
            output = self._docker_cli(
                [
                    "stack",
                    "services",
                    "--format",
                    "{{.Name}}\t{{.Replicas}}\t{{.Image}}",
                    stack_name,
                ],
                "docker_stack_services",
            )
            result = []
            for line in output.splitlines():
                if not line.strip():
                    continue
                parts = line.split("\t")
                result.append(
                    {
                        "name": parts[0] if len(parts) > 0 else "",
                        "replicas": parts[1] if len(parts) > 1 else "",
                        "image": parts[2] if len(parts) > 2 else "",
                        "stack_name": stack_name,
                    }
                )
            self.log_action("docker_stack_services", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("docker_stack_services", params, error=e)
            raise RuntimeError("Failed to list stack services") from e

    def docker_stack_rm(self, stack_name: str) -> dict:
        """Remove a stack via ``docker stack rm`` (no SDK equivalent)."""
        params = {"stack_name": stack_name}
        try:
            output = self._docker_cli(["stack", "rm", stack_name], "docker_stack_rm")
            result = {
                "stack_name": stack_name,
                "output": output.strip(),
                "status": "removed",
            }
            self.log_action("docker_stack_rm", params, result)
            return result
        except Exception as e:
            self.log_action("docker_stack_rm", params, error=e)
            raise RuntimeError("Failed to remove stack") from e

    def docker_config_create(self, config_name: str, data: str) -> dict:
        """Create a swarm config (real ``configs.create``)."""
        params = {"config_name": config_name}
        try:
            config = self.client.configs.create(
                name=config_name, data=data.encode("utf-8")
            )
            result = {
                "config_name": config_name,
                "id": getattr(config, "id", None),
                "status": "created",
            }
            self.log_action("docker_config_create", params, result)
            return result
        except Exception as e:
            self.log_action("docker_config_create", params, error=e)
            raise RuntimeError("Failed to create config") from e

    def docker_config_list(self) -> list[dict]:
        """List swarm configs (real ``configs.list``)."""
        params: dict[str, Any] = {}
        try:
            result = []
            for c in self.client.configs.list():
                spec = c.attrs.get("Spec", {})
                result.append(
                    {
                        "id": c.id,
                        "name": spec.get("Name", "unknown"),
                        "created": c.attrs.get("CreatedAt", "unknown"),
                    }
                )
            self.log_action("docker_config_list", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("docker_config_list", params, error=e)
            raise RuntimeError("Failed to list configs") from e

    def docker_secret_create(self, secret_name: str, data: str) -> dict:
        """Create a swarm secret (real ``secrets.create``)."""
        params = {"secret_name": secret_name}
        try:
            secret = self.client.secrets.create(
                name=secret_name, data=data.encode("utf-8")
            )
            result = {
                "secret_name": secret_name,
                "id": getattr(secret, "id", None),
                "status": "created",
            }
            self.log_action("docker_secret_create", params, result)
            return result
        except Exception as e:
            self.log_action("docker_secret_create", params, error=e)
            raise RuntimeError("Failed to create secret") from e

    def docker_secret_list(self) -> list[dict]:
        """List swarm secrets (real ``secrets.list``)."""
        params: dict[str, Any] = {}
        try:
            result = []
            for s in self.client.secrets.list():
                spec = s.attrs.get("Spec", {})
                result.append(
                    {
                        "id": s.id,
                        "name": spec.get("Name", "unknown"),
                        "created": s.attrs.get("CreatedAt", "unknown"),
                    }
                )
            self.log_action("docker_secret_list", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("docker_secret_list", params, error=e)
            raise RuntimeError("Failed to list secrets") from e

    def docker_node_ls(self) -> list[dict]:
        """List swarm nodes (real ``nodes.list`` via :meth:`list_nodes`)."""
        return self.list_nodes()

    def docker_node_update(self, node_id: str, availability: str) -> dict:
        """Update a node's availability (real update via :meth:`update_node`)."""
        return self.update_node(node_id, availability=availability)

    def docker_node_inspect(self, node_id: str) -> dict:
        """Inspect a swarm node (real ``nodes.get`` via :meth:`inspect_node`)."""
        return self.inspect_node(node_id)


class PodmanManager(ContainerManagerBase):
    def __init__(self, silent: bool = False, log_file: str | None = None):
        super().__init__(silent, log_file)
        if PodmanClient is None:
            raise ImportError("Please install podman-py: pip install podman")
        base_url = self._autodetect_podman_url()
        if base_url is None:
            self.logger.error(
                "No valid Podman socket found after trying all known locations"
            )
            raise RuntimeError("Failed to connect to Podman: No valid socket found")
        try:
            self.client = PodmanClient(base_url=base_url)
            self.logger.info("Connected to configured Podman daemon")
        except PodmanError as e:
            self.logger.error(
                "Failed to connect to Podman daemon: error_type=%s",
                type(e).__name__,
            )
            raise RuntimeError("Configured Podman daemon is unavailable") from e

    def _is_wsl(self) -> bool:
        """Check if running inside WSL2."""
        try:
            with open("/proc/version") as f:
                return "WSL" in f.read()
        except FileNotFoundError:
            return "WSL_DISTRO_NAME" in os.environ

    def _is_podman_machine_running(self) -> bool:
        """Check if Podman machine is running (for Windows/WSL2)."""
        try:
            result = subprocess.run(  # nosec B607 B603
                ["podman", "machine", "list", "--format", "{{.Running}}"],
                capture_output=True,
                text=True,
                check=False,
            )
            return "true" in result.stdout.lower()
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _try_connect(self, base_url: str) -> "PodmanClient | None":
        """Attempt to connect to Podman with the given base_url."""
        try:
            client = PodmanClient(base_url=base_url)
            client.version()
            return client
        except (PodmanError, Exception) as e:
            self.logger.debug("Operation failed: error_type=%s", type(e).__name__)
            return None

    def _get_podman_cli_sockets(self) -> list[str]:
        """Get socket URLs from 'podman system connection list'."""
        try:
            result = subprocess.run(  # nosec B607 B603
                ["podman", "system", "connection", "list", "--format", "json"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                try:
                    connections = json.loads(result.stdout)
                    urls: list[str] = []

                    for conn in connections:
                        uri = conn.get("URI")
                        if not uri:
                            continue
                        if conn.get("Default"):
                            urls.insert(0, uri)
                        else:
                            urls.append(uri)
                    return urls
                except json.JSONDecodeError:
                    return []
        except Exception as e:
            self.logger.debug("Operation failed: error_type=%s", type(e).__name__)
        return []

    def _autodetect_podman_url(self) -> str | None:
        """Autodetect the appropriate Podman socket URL based on platform."""

        base_url = os.environ.get("CONTAINER_MANAGER_PODMAN_BASE_URL")
        if base_url:
            self.logger.info(
                f"Using CONTAINER_MANAGER_PODMAN_BASE_URL from environment: {base_url}"
            )
            return base_url

        socket_candidates = []

        cli_sockets = self._get_podman_cli_sockets()
        if cli_sockets:
            socket_candidates.extend(cli_sockets)

        system = platform.system()
        is_wsl = self._is_wsl()

        if system == "Windows" and not is_wsl:
            socket_candidates.extend(
                [
                    "npipe:////./pipe/podman-machine-default",
                    "npipe:////./pipe/podman-machine-default-root",
                    "npipe:////./pipe/docker_engine",
                    "tcp://127.0.0.1:8080",
                ]
            )

            socket_candidates.extend(
                [
                    "unix:///mnt/wsl/podman-sockets/podman-machine-default/podman-user.sock",
                    "unix:///mnt/wsl/podman-sockets/podman-machine-default/podman-root.sock",
                ]
            )
        elif system == "Linux" or is_wsl:
            uid = os.getuid()

            socket_candidates.extend(
                [
                    f"unix:///run/user/{uid}/podman/podman.sock",
                    "unix:///run/podman/podman.sock",
                    "unix:///var/run/podman/podman.sock",
                ]
            )

            socket_candidates.extend(
                [
                    "/mnt/wsl/podman-sockets/podman-machine-default/podman-user.sock",
                    "/mnt/wsl/podman-sockets/podman-machine-default/podman-root.sock",
                ]
            )

        for url in socket_candidates:
            if url.startswith("/") and not url.startswith("unix://"):
                url = f"unix://{url}"

            client = self._try_connect(url)
            if client:
                self.logger.info("Autodetected a Podman socket")
                return url

        return None

    def prune_images(self, force: bool = False, all: bool = False) -> dict:
        params = {"force": force, "all": all}
        try:
            if all:
                images = self.client.images.list(all=True)
                removed = []
                for img in images:
                    try:
                        for tag in img.attrs.get("Names", []):
                            self.client.images.remove(tag, force=force)
                            removed.append(img.attrs["Id"][7:19])
                    except Exception as e:
                        self.logger.info(
                            "Image removal failed: error_type=%s", type(e).__name__
                        )
                        continue
                result = {
                    "images_removed": removed,
                    "space_reclaimed": "N/A (all images)",
                }
            else:
                filters = {"dangling": True} if not all else {}
                result = self.client.images.prune(filters=filters)
                if result is None:
                    result = {"SpaceReclaimed": 0, "ImagesRemoved": []}  # type: ignore
                self.logger.debug(
                    "Image prune result received: shape=%s",
                    _privacy_safe_shape(result),
                )
                space_reclaimed = result.get("SpaceReclaimed", 0)
                if not isinstance(space_reclaimed, (int, float)):
                    space_reclaimed = 0
                pruned = {
                    "space_reclaimed": self._format_size(space_reclaimed),
                    "images_removed": (
                        [img["Id"][7:19] for img in result.get("ImagesRemoved", [])]
                        or [img["Id"][7:19] for img in result.get("ImagesDeleted", [])]
                    ),
                }
                result = pruned
            self.log_action("prune_images", params, result)
            return result
        except Exception as e:
            self.log_action("prune_images", params, error=e)
            raise RuntimeError("Failed to prune images") from e

    def prune_containers(self) -> dict:
        params: dict[str, Any] = {}
        try:
            result = self.client.containers.prune()
            self.logger.debug(
                "Container prune result received: shape=%s",
                _privacy_safe_shape(result),
            )
            if result is None:
                result = {"SpaceReclaimed": 0, "ContainersDeleted": []}  # type: ignore
            pruned = {
                "space_reclaimed": self._format_size(result.get("SpaceReclaimed", 0)),
                "containers_removed": (
                    [
                        (c.get("Id", "")[:12] if isinstance(c, dict) else str(c)[:12])
                        for c in (result.get("ContainersDeleted") or [])
                    ]
                    or [c["Id"][7:19] for c in result.get("ContainersRemoved", [])]
                ),
            }
            self.log_action("prune_containers", params, pruned)
            return pruned
        except PodmanError as e:
            self.logger.error(
                "Container prune failed: error_type=%s", type(e).__name__
            )
            self.log_action("prune_containers", params, error=e)
            raise RuntimeError("Failed to prune containers") from e
        except Exception as e:
            self.logger.error(
                "Container prune failed: error_type=%s", type(e).__name__
            )
            self.log_action("prune_containers", params, error=e)
            raise RuntimeError("Failed to prune containers") from e

    def prune_volumes(self, force: bool = False, all: bool = False) -> dict:
        params = {"force": force, "all": all}
        try:
            if all:
                volumes = self.client.volumes.list(all=True)
                removed = []
                for v in volumes:
                    try:
                        v.remove(force=force)
                        removed.append(v.attrs["Name"])
                    except Exception as e:
                        self.logger.info(
                            "Volume removal failed: error_type=%s", type(e).__name__
                        )
                        continue
                result = {
                    "volumes_removed": removed,
                    "space_reclaimed": "N/A (all volumes)",
                }
            else:
                result = self.client.volumes.prune()
                if result is None:
                    result = {"SpaceReclaimed": 0, "VolumesRemoved": []}  # type: ignore
                self.logger.debug(
                    "Volume prune result received: shape=%s",
                    _privacy_safe_shape(result),
                )
                space_reclaimed = result.get("SpaceReclaimed", 0)
                if not isinstance(space_reclaimed, (int, float)):
                    space_reclaimed = 0

                # Handle different API response formats
                volumes_removed_data = result.get("VolumesRemoved", []) or result.get(
                    "VolumesDeleted", []
                )
                volumes_removed = []
                for v in volumes_removed_data:
                    if isinstance(v, dict):
                        volumes_removed.append(v.get("Name", "unknown"))
                    elif isinstance(v, str):
                        volumes_removed.append(v)

                pruned = {
                    "space_reclaimed": self._format_size(space_reclaimed),
                    "volumes_removed": volumes_removed,
                }
                result = pruned
            self.log_action("prune_volumes", params, result)
            return result
        except Exception as e:
            self.log_action("prune_volumes", params, error=e)
            raise RuntimeError("Failed to prune volumes") from e

    def prune_networks(self) -> dict:
        params: dict[str, Any] = {}
        try:
            result = self.client.networks.prune()
            if result is None:
                result = {"SpaceReclaimed": 0, "NetworksRemoved": []}
            self.logger.debug(
                "Network prune result received: shape=%s",
                _privacy_safe_shape(result),
            )
            pruned = {
                "space_reclaimed": self._format_size(result.get("SpaceReclaimed", 0)),
                "networks_removed": (
                    [
                        n["Id"][7:19] if isinstance(n, dict) and "Id" in n else n
                        for n in result.get("NetworksRemoved", [])
                    ]
                    or [
                        n["Id"][7:19] if isinstance(n, dict) and "Id" in n else n
                        for n in result.get("NetworksDeleted", [])
                    ]
                ),
            }
            self.log_action("prune_networks", params, pruned)
            return pruned
        except Exception as e:
            self.log_action("prune_networks", params, error=e)
            raise RuntimeError("Failed to prune networks") from e

    def prune_system(self, force: bool = False, all: bool = False) -> dict:
        params = {"force": force, "all": all}
        try:
            cmd = (
                ["podman", "system", "prune", "--force"]
                if force
                else ["podman", "system", "prune"]
            )
            if all:
                cmd.append("--all")
                if all:
                    cmd.append("--volumes")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.logger.debug(
                "System prune command completed: output_shape=%s",
                _privacy_safe_shape(result.stdout),
            )
            pruned = {
                "output": result.stdout.strip(),
                "space_reclaimed": "Check output",
                "images_removed": [],
                "containers_removed": [],
                "volumes_removed": [],
                "networks_removed": [],
            }
            self.log_action("prune_system", params, pruned)
            return pruned
        except Exception as e:
            self.log_action("prune_system", params, error=e)
            raise RuntimeError("Failed to prune system") from e

    def get_version(self) -> dict:
        params: dict[str, Any] = {}
        try:
            version = self.client.version()
            result = {
                "version": version.get("Version", "unknown"),
                "api_version": version.get("APIVersion", "unknown"),
                "os": version.get("Os", "unknown"),
                "arch": version.get("Arch", "unknown"),
                "build_time": version.get("BuildTime", "unknown"),
            }
            self.log_action("get_version", params, result)
            return result
        except Exception as e:
            self.log_action("get_version", params, error=e)
            raise RuntimeError("Failed to get version") from e

    def get_info(self) -> dict:
        params: dict[str, Any] = {}
        try:
            info = self.client.info()
            host = info.get("host", {})
            result = {
                "containers_total": info.get("store", {}).get("containers", 0),
                "containers_running": host.get("runningContainers", 0),
                "images": info.get("store", {}).get("images", 0),
                "driver": host.get("graphDriverName", "unknown"),
                "platform": f"{host.get('os', 'unknown')} {host.get('arch', 'unknown')}",
                "memory_total": self._format_size(host.get("memTotal", 0)),
                "swap_total": self._format_size(host.get("swapTotal", 0)),
            }
            self.log_action("get_info", params, result)
            return result
        except Exception as e:
            self.log_action("get_info", params, error=e)
            raise RuntimeError("Failed to get info") from e

    def list_images(self) -> list[ImageInfo]:
        params: dict[str, Any] = {}
        try:
            images = self.client.images.list()
            result = []
            for img in images:
                attrs = img.attrs
                repo_tags = attrs.get("Names", [])
                repo_tag = repo_tags[0] if repo_tags else "<none>:<none>"
                repository, tag = (
                    repo_tag.rsplit(":", 1) if ":" in repo_tag else ("<none>", "<none>")
                )
                created = attrs.get("Created", None)
                created_str = self._parse_timestamp(created)
                size_bytes = attrs.get("Size", 0)
                size_str = self._format_size(size_bytes) if size_bytes else "0B"
                simplified: dict[str, Any] = {
                    "repository": repository,
                    "tag": tag,
                    "id": (
                        attrs.get("Id", "unknown")[7:19]
                        if attrs.get("Id")
                        else "unknown"
                    ),
                    "created": created_str,
                    "size": size_str,
                    "labels": self._extract_image_labels(attrs),
                }
                result.append(ImageInfo(**simplified))
            self.log_action("list_images", params, [i.model_dump() for i in result])
            return result
        except Exception as e:
            self.log_action("list_images", params, error=e)
            raise RuntimeError("Failed to list images") from e

    def pull_image(
        self, image: str, tag: str = "latest", platform: str | None = None
    ) -> dict:
        params = {"image": image, "tag": tag, "platform": platform}
        try:
            # Don't append tag if image already contains one
            image_ref = f"{image}:{tag}" if ":" not in image else image
            img = self.client.images.pull(image_ref, platform=platform)
            attrs = img[0].attrs if isinstance(img, list) else img.attrs
            repo_tags = attrs.get("Names", [])
            repo_tag = repo_tags[0] if repo_tags else image_ref
            repository, tag = (
                repo_tag.rsplit(":", 1) if ":" in repo_tag else (image, tag)
            )
            created = attrs.get("Created", None)
            created_str = self._parse_timestamp(created)
            size_bytes = attrs.get("Size", 0)
            size_str = self._format_size(size_bytes) if size_bytes else "0B"
            result = {
                "repository": repository,
                "tag": tag,
                "id": (
                    attrs.get("Id", "unknown")[7:19] if attrs.get("Id") else "unknown"
                ),
                "created": created_str,
                "size": size_str,
            }
            self.log_action("pull_image", params, result)
            return result
        except Exception as e:
            self.log_action("pull_image", params, error=e)
            raise RuntimeError("Failed to pull image") from e

    def remove_image(self, image: str, force: bool = False) -> dict:
        params = {"image": image, "force": force}
        try:
            self.client.images.remove(image, force=force)
            result = {"removed": image}
            self.log_action("remove_image", params, result)
            return result
        except Exception as e:
            self.log_action("remove_image", params, error=e)
            raise RuntimeError("Failed to remove image") from e

    def list_containers(self, all: bool = False) -> list[ContainerInfo]:
        params = {"all": all}
        try:
            containers = self.client.containers.list(all=all)
            result = []
            for c in containers:
                attrs = c.attrs
                ports = attrs.get("Ports", []) or []
                port_mappings = [
                    f"{p.get('host_ip', '0.0.0.0')}:{p.get('host_port')}->{p.get('container_port')}/{p.get('protocol', 'tcp')}"  # nosec
                    for p in ports
                    if p.get("host_port")
                ]
                created = attrs.get("Created", None)
                created_str = self._parse_timestamp(created)
                simplified = {
                    "id": attrs.get("Id", "unknown")[:12],
                    "image": attrs.get("Image", "unknown"),
                    "name": attrs.get("Names", ["unknown"])[0].lstrip("/"),
                    "status": attrs.get("State", "unknown"),
                    "ports": ", ".join(port_mappings) if port_mappings else "none",
                    "created": created_str,
                }
                result.append(ContainerInfo(**simplified))
            self.log_action("list_containers", params, [c.model_dump() for c in result])
            return result
        except Exception as e:
            self.log_action("list_containers", params, error=e)
            raise RuntimeError("Failed to list containers") from e

    def run_container(
        self,
        image: str,
        name: str | None = None,
        command: str | None = None,
        detach: bool = False,
        ports: dict[str, str] | None = None,
        volumes: dict[str, dict] | None = None,
        environment: dict[str, str] | list[str] | None = None,
        labels: dict[str, str] | None = None,
    ) -> dict:
        params = {
            "image": image,
            "name": name,
            "command": command,
            "detach": detach,
            "ports": ports,
            "volumes": volumes,
            "environment": environment,
            "labels": labels,
        }
        try:
            # Build kwargs, filtering out None values for podman-py compatibility
            run_kwargs: dict[str, Any] = {}
            run_kwargs["detach"] = detach
            if name is not None:
                run_kwargs["name"] = name
            if command is not None:
                run_kwargs["command"] = command
            if ports is not None:
                run_kwargs["ports"] = ports
            if volumes is not None:
                run_kwargs["volumes"] = volumes
            if environment is not None:
                run_kwargs["environment"] = environment
            if labels is not None:
                run_kwargs["labels"] = labels

            container = self.client.containers.run(image, **run_kwargs)
            if not detach:
                result = {"output": container.decode("utf-8") if container else ""}
                self.log_action("run_container", params, result)
                return result
            attrs = container.attrs
            port_mappings = []
            if ports:
                container_ports = attrs.get("Ports", [])
                if container_ports:
                    port_mappings = [
                        f"{p.get('host_ip', '0.0.0.0')}:{p.get('host_port')}->{p.get('container_port')}/{p.get('protocol', 'tcp')}"  # nosec
                        for p in container_ports
                        if p.get("host_port")
                    ]
            created = attrs.get("Created", None)
            created_str = self._parse_timestamp(created)
            result = {
                "id": attrs.get("Id", "unknown"),
                "image": attrs.get("Image", image),
                "name": attrs.get("Names", [name or "unknown"])[0].lstrip("/"),
                "status": attrs.get("State", "unknown"),
                "ports": ", ".join(port_mappings) if port_mappings else "none",
                "created": created_str,
            }
            self.log_action("run_container", params, result)
            return result
        except Exception as e:
            self.log_action("run_container", params, error=e)
            raise RuntimeError("Failed to run container") from e

    def inspect_container(self, container_id: str) -> dict:
        params = {"container_id": container_id}
        try:
            container = self.client.containers.get(container_id)
            result = container.attrs
            self.log_action("inspect_container", params, {"id": container_id})
            return result
        except Exception as e:
            self.log_action("inspect_container", params, error=e)
            raise RuntimeError("Failed to inspect container") from e

    def stop_container(self, container_id: str, timeout: int = 10) -> dict:
        params = {"container_id": container_id, "timeout": timeout}
        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=timeout)
            result = {"stopped": container_id}
            self.log_action("stop_container", params, result)
            return result
        except Exception as e:
            self.log_action("stop_container", params, error=e)
            raise RuntimeError("Failed to stop container") from e

    def remove_container(self, container_id: str, force: bool = False) -> dict:
        params = {"container_id": container_id, "force": force}
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=force)
            result = {"removed": container_id}
            self.log_action("remove_container", params, result)
            return result
        except Exception as e:
            self.log_action("remove_container", params, error=e)
            raise RuntimeError("Failed to remove container") from e

    def get_container_logs(self, container_id: str, tail: str = "50") -> str:
        params = {"container_id": container_id, "tail": tail}
        try:
            container = self.client.containers.get(container_id)
            logs_output = container.logs(tail=tail)
            # Handle podman-py returning a generator vs docker-py returning bytes
            if hasattr(logs_output, "__iter__") and not isinstance(
                logs_output, (bytes, str)
            ):
                logs = "".join(
                    chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
                    for chunk in logs_output
                )
            else:
                logs = (
                    logs_output.decode("utf-8")
                    if isinstance(logs_output, bytes)
                    else str(logs_output)
                )
            self.log_action("get_container_logs", params, logs[:1000])
            return logs
        except Exception as e:
            self.log_action("get_container_logs", params, error=e)
            raise RuntimeError("Failed to get container logs") from e

    def exec_in_container(
        self,
        container_id: str,
        command: list[str],
        detach: bool = False,
        binary: bool = False,
    ) -> dict:
        params = {
            "container_id": container_id,
            "command": command,
            "detach": detach,
            "binary": binary,
        }
        try:
            container = self.client.containers.get(container_id)
            result = _build_exec_result(container, command, detach, binary)
            # Don't log a full base64 PNG — summarise binary results.
            log_result = (
                {"exit_code": result["exit_code"], "binary": True} if binary else result
            )
            self.log_action("exec_in_container", params, log_result)
            return result
        except Exception as e:
            self.log_action("exec_in_container", params, error=e)
            raise RuntimeError("Failed to exec in container") from e

    def list_volumes(self) -> list[VolumeInfo]:
        params: dict[str, Any] = {}
        try:
            volumes = self.client.volumes.list()
            result = [
                VolumeInfo(
                    **{
                        "name": v.attrs.get("Name", "unknown"),
                        "driver": v.attrs.get("Driver", "unknown"),
                        "mountpoint": v.attrs.get("Mountpoint", "unknown"),
                        "created": v.attrs.get("CreatedAt", "unknown"),
                    }
                )
                for v in volumes
            ]
            self.log_action("list_volumes", params, [v.model_dump() for v in result])
            return result
        except Exception as e:
            self.log_action("list_volumes", params, error=e)
            raise RuntimeError("Failed to list volumes") from e

    def create_volume(self, name: str) -> VolumeInfo:
        params = {"name": name}
        try:
            volume = self.client.volumes.create(name=name)
            attrs = volume.attrs
            result = VolumeInfo(
                **{
                    "name": attrs.get("Name", name),
                    "driver": attrs.get("Driver", "unknown"),
                    "mountpoint": attrs.get("Mountpoint", "unknown"),
                    "created": attrs.get("CreatedAt", "unknown"),
                }
            )
            self.log_action("create_volume", params, result.model_dump())
            return result
        except Exception as e:
            self.log_action("create_volume", params, error=e)
            raise RuntimeError("Failed to create volume") from e

    def remove_volume(self, name: str, force: bool = False) -> dict:
        params = {"name": name, "force": force}
        try:
            volume = self.client.volumes.get(name)
            volume.remove(force=force)
            result = {"removed": name}
            self.log_action("remove_volume", params, result)
            return result
        except Exception as e:
            self.log_action("remove_volume", params, error=e)
            raise RuntimeError("Failed to remove volume") from e

    def list_networks(self) -> list[NetworkInfo]:
        params: dict[str, Any] = {}
        try:
            networks = self.client.networks.list()
            result = []
            for net in networks:
                attrs = net.attrs
                containers = len(attrs.get("Containers", {}))
                created = attrs.get("Created", None)
                created_str = self._parse_timestamp(created)
                # Try different possible name field locations for Podman compatibility
                name = attrs.get("Name") or attrs.get("name") or net.name or "unknown"
                simplified = {
                    "id": attrs.get("Id", "unknown")[:12],
                    "name": name,
                    "driver": attrs.get("Driver", "unknown"),
                    "scope": attrs.get("Scope", "unknown"),
                    "containers": containers,
                    "created": created_str,
                }
                result.append(NetworkInfo(**simplified))
            self.log_action("list_networks", params, [n.model_dump() for n in result])
            return result
        except Exception as e:
            self.log_action("list_networks", params, error=e)
            raise RuntimeError("Failed to list networks") from e

    def create_network(self, name: str, driver: str = "bridge") -> NetworkInfo:
        params = {"name": name, "driver": driver}
        try:
            network = self.client.networks.create(name, driver=driver)
            attrs = network.attrs
            created = attrs.get("Created", None)
            created_str = self._parse_timestamp(created)
            result = NetworkInfo(
                **{
                    "id": attrs.get("Id", "unknown")[:12],
                    "name": attrs.get("Name", name),
                    "driver": attrs.get("Driver", driver),
                    "scope": attrs.get("Scope", "unknown"),
                    "created": created_str,
                }
            )
            self.log_action("create_network", params, result.model_dump())
            return result
        except Exception as e:
            self.log_action("create_network", params, error=e)
            raise RuntimeError("Failed to create network") from e

    def remove_network(self, network_id: str) -> dict:
        params = {"network_id": network_id}
        try:
            network = self.client.networks.get(network_id)
            network.remove()
            result = {"removed": network_id}
            self.log_action("remove_network", params, result)
            return result
        except Exception as e:
            self.log_action("remove_network", params, error=e)
            raise RuntimeError("Failed to remove network") from e

    def compose_up(
        self, compose_file: str, detach: bool = True, build: bool = False
    ) -> str:
        params = {"compose_file": compose_file, "detach": detach, "build": build}
        try:
            cmd = ["podman-compose", "-f", compose_file, "up"]
            if build:
                cmd.append("--build")
            if detach:
                cmd.append("-d")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_up", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_up", params, error=e)
            raise RuntimeError("Failed to compose up") from e

    def compose_down(self, compose_file: str) -> str:
        params = {"compose_file": compose_file}
        try:
            cmd = ["podman-compose", "-f", compose_file, "down"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_down", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_down", params, error=e)
            raise RuntimeError("Failed to compose down") from e

    def compose_ps(self, compose_file: str) -> str:
        params = {"compose_file": compose_file}
        try:
            cmd = ["podman-compose", "-f", compose_file, "ps"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_ps", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_ps", params, error=e)
            raise RuntimeError("Failed to compose ps") from e

    def compose_logs(self, compose_file: str, service: str | None = None) -> str:
        params = {"compose_file": compose_file, "service": service}
        try:
            cmd = ["podman-compose", "-f", compose_file, "logs"]
            if service:
                cmd.append(service)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_logs", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_logs", params, error=e)
            raise RuntimeError("Failed to compose logs") from e

    def init_swarm(self, advertise_addr: str | None = None) -> dict:
        raise RuntimeError("Swarm not supported in Podman")

    def leave_swarm(self, force: bool = False) -> dict:
        raise RuntimeError("Swarm not supported in Podman")

    def list_nodes(self) -> list[dict]:
        raise RuntimeError("Swarm not supported in Podman")

    def list_services(self) -> list[dict]:
        raise RuntimeError("Swarm not supported in Podman")

    def create_service(
        self,
        name: str,
        image: str,
        replicas: int = 1,
        ports: dict[str, str] | None = None,
        mounts: list[str] | None = None,
    ) -> dict:
        raise RuntimeError("Swarm not supported in Podman")

    def remove_service(self, service_id: str) -> dict:
        raise RuntimeError("Swarm not supported in Podman")

    def inspect_node(self, node_id: str) -> dict:
        raise RuntimeError("Swarm not supported in Podman")

    def update_node(
        self,
        node_id: str,
        labels: dict[str, str] | None = None,
        role: str | None = None,
        availability: str | None = None,
        replace_labels: bool = False,
    ) -> dict:
        raise RuntimeError("Swarm not supported in Podman")

    def remove_node(self, node_id: str, force: bool = False) -> dict:
        raise RuntimeError("Swarm not supported in Podman")

    def inspect_service(self, service_id: str) -> dict:
        raise RuntimeError("Swarm not supported in Podman")

    def scale_service(self, service_id: str, replicas: int) -> dict:
        raise RuntimeError("Swarm not supported in Podman")

    def update_service(
        self,
        service_id: str,
        image: str | None = None,
        replicas: int | None = None,
        env: list[str] | None = None,
        constraints: list[str] | None = None,
        labels: dict[str, str] | None = None,
        force: bool = False,
    ) -> dict:
        raise RuntimeError("Swarm not supported in Podman")

    def service_ps(self, service_id: str) -> list[dict]:
        raise RuntimeError("Swarm not supported in Podman")

    def service_logs(self, service_id: str, tail: int = 100) -> dict:
        raise RuntimeError("Swarm not supported in Podman")

    # ------------------------------------------------------------------
    # Pod / network / volume / kube-interop / checkpoint operations
    # (the function-based cm_podman surface — real SDK/CLI calls)
    # ------------------------------------------------------------------
    def _podman_cli(self, args: list[str], action: str) -> str:
        """Run ``podman <args>`` and return stdout, raising on failure.

        Used for surfaces podman-py does not expose (pod stats/top/logs,
        healthcheck, generate/play kube, checkpoint/restore, system prune).
        """
        cmd = ["podman", *args]
        result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603 B607
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout

    def podman_pod_create(
        self, pod_name: str, image: str, command: str | None = None
    ) -> dict:
        """Create a pod (real ``pods.create``)."""
        params = {"pod_name": pod_name, "image": image, "command": command}
        try:
            pod = self.client.pods.create(name=pod_name)
            attrs = getattr(pod, "attrs", {}) or {}
            result = {
                "pod_name": pod_name,
                "id": attrs.get("Id") or attrs.get("ID") or getattr(pod, "id", None),
                "image": image,
                "command": command,
                "status": "created",
            }
            self.log_action("podman_pod_create", params, result)
            return result
        except Exception as e:
            self.log_action("podman_pod_create", params, error=e)
            raise RuntimeError("Failed to create pod") from e

    def podman_pod_list(self) -> list[dict]:
        """List pods (real ``pods.list``)."""
        params: dict[str, Any] = {}
        try:
            result = []
            for pod in self.client.pods.list():
                attrs = getattr(pod, "attrs", {}) or {}
                result.append(
                    {
                        "name": attrs.get("Name") or getattr(pod, "name", "unknown"),
                        "id": attrs.get("Id")
                        or attrs.get("ID")
                        or getattr(pod, "id", "unknown"),
                        "status": attrs.get("Status") or attrs.get("State", "unknown"),
                        "infrastructure": "podman",
                    }
                )
            self.log_action("podman_pod_list", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("podman_pod_list", params, error=e)
            raise RuntimeError("Failed to list pods") from e

    def podman_pod_inspect(self, pod_name: str) -> dict:
        """Inspect a pod (real ``pods.get(...).attrs``)."""
        params = {"pod_name": pod_name}
        try:
            pod = self.client.pods.get(pod_name)
            result = getattr(pod, "attrs", {}) or {}
            self.log_action("podman_pod_inspect", params, {"pod_name": pod_name})
            return result
        except Exception as e:
            self.log_action("podman_pod_inspect", params, error=e)
            raise RuntimeError("Failed to inspect pod") from e

    def podman_pod_stats(self, pod_name: str) -> dict:
        """Get pod stats via ``podman pod stats`` (not in podman-py)."""
        params = {"pod_name": pod_name}
        try:
            output = self._podman_cli(
                ["pod", "stats", "--no-stream", "--format", "json", pod_name],
                "podman_pod_stats",
            )
            try:
                stats = json.loads(output) if output.strip() else []
            except json.JSONDecodeError:
                stats = output.strip()
            result = {"pod_name": pod_name, "stats": stats}
            self.log_action("podman_pod_stats", params, result)
            return result
        except Exception as e:
            self.log_action("podman_pod_stats", params, error=e)
            raise RuntimeError("Failed to get pod stats") from e

    def podman_pod_top(self, pod_name: str) -> dict:
        """Get pod processes via ``podman pod top`` (not in podman-py)."""
        params = {"pod_name": pod_name}
        try:
            output = self._podman_cli(["pod", "top", pod_name], "podman_pod_top")
            result = {"pod_name": pod_name, "processes": output.strip()}
            self.log_action("podman_pod_top", params, result)
            return result
        except Exception as e:
            self.log_action("podman_pod_top", params, error=e)
            raise RuntimeError("Failed to get pod top") from e

    def podman_pod_logs(self, pod_name: str, tail_lines: int = 100) -> dict:
        """Get pod logs via ``podman pod logs`` (not in podman-py)."""
        params = {"pod_name": pod_name, "tail_lines": tail_lines}
        try:
            output = self._podman_cli(
                ["pod", "logs", "--tail", str(tail_lines), pod_name],
                "podman_pod_logs",
            )
            result = {"pod_name": pod_name, "tail_lines": tail_lines, "logs": output}
            self.log_action("podman_pod_logs", params, {"pod_name": pod_name})
            return result
        except Exception as e:
            self.log_action("podman_pod_logs", params, error=e)
            raise RuntimeError("Failed to get pod logs") from e

    def podman_pod_stop(self, pod_name: str) -> dict:
        """Stop a pod (real ``pods.get(...).stop()``)."""
        params = {"pod_name": pod_name}
        try:
            self.client.pods.get(pod_name).stop()
            result = {"pod_name": pod_name, "status": "stopped"}
            self.log_action("podman_pod_stop", params, result)
            return result
        except Exception as e:
            self.log_action("podman_pod_stop", params, error=e)
            raise RuntimeError("Failed to stop pod") from e

    def podman_pod_rm(self, pod_name: str) -> dict:
        """Remove a pod (real ``pods.get(...).remove()``)."""
        params = {"pod_name": pod_name}
        try:
            self.client.pods.get(pod_name).remove()
            result = {"pod_name": pod_name, "status": "removed"}
            self.log_action("podman_pod_rm", params, result)
            return result
        except Exception as e:
            self.log_action("podman_pod_rm", params, error=e)
            raise RuntimeError("Failed to remove pod") from e

    def podman_network_create(
        self, network_name: str, driver: str = "bridge", subnet: str | None = None
    ) -> dict:
        """Create a network (real ``networks.create``)."""
        params = {"network_name": network_name, "driver": driver, "subnet": subnet}
        try:
            kwargs: dict[str, Any] = {"driver": driver}
            if subnet:
                kwargs["subnet"] = subnet
            network = self.client.networks.create(network_name, **kwargs)
            attrs = getattr(network, "attrs", {}) or {}
            result = {
                "network_name": network_name,
                "id": attrs.get("Id") or getattr(network, "id", None),
                "driver": driver,
                "subnet": subnet,
                "status": "created",
            }
            self.log_action("podman_network_create", params, result)
            return result
        except Exception as e:
            self.log_action("podman_network_create", params, error=e)
            raise RuntimeError("Failed to create network") from e

    def podman_network_list(self) -> list[dict]:
        """List networks (real ``networks.list``)."""
        params: dict[str, Any] = {}
        try:
            result = []
            for net in self.client.networks.list():
                attrs = getattr(net, "attrs", {}) or {}
                result.append(
                    {
                        "name": attrs.get("Name") or getattr(net, "name", "unknown"),
                        "id": attrs.get("Id") or getattr(net, "id", "unknown"),
                        "driver": attrs.get("Driver", "unknown"),
                    }
                )
            self.log_action("podman_network_list", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("podman_network_list", params, error=e)
            raise RuntimeError("Failed to list networks") from e

    def podman_network_inspect(self, network_name: str) -> dict:
        """Inspect a network (real ``networks.get(...).attrs``)."""
        params = {"network_name": network_name}
        try:
            net = self.client.networks.get(network_name)
            result = getattr(net, "attrs", {}) or {}
            self.log_action(
                "podman_network_inspect", params, {"network_name": network_name}
            )
            return result
        except Exception as e:
            self.log_action("podman_network_inspect", params, error=e)
            raise RuntimeError("Failed to inspect network") from e

    def podman_volume_create(self, volume_name: str, driver: str = "local") -> dict:
        """Create a volume (real ``volumes.create``)."""
        params = {"volume_name": volume_name, "driver": driver}
        try:
            volume = self.client.volumes.create(name=volume_name, driver=driver)
            attrs = getattr(volume, "attrs", {}) or {}
            result = {
                "volume_name": volume_name,
                "driver": attrs.get("Driver", driver),
                "mountpoint": attrs.get("Mountpoint"),
                "status": "created",
            }
            self.log_action("podman_volume_create", params, result)
            return result
        except Exception as e:
            self.log_action("podman_volume_create", params, error=e)
            raise RuntimeError("Failed to create volume") from e

    def podman_volume_list(self) -> list[dict]:
        """List volumes (real ``volumes.list``)."""
        params: dict[str, Any] = {}
        try:
            result = []
            for v in self.client.volumes.list():
                attrs = getattr(v, "attrs", {}) or {}
                result.append(
                    {
                        "name": attrs.get("Name") or getattr(v, "name", "unknown"),
                        "driver": attrs.get("Driver", "unknown"),
                        "mountpoint": attrs.get("Mountpoint", "unknown"),
                    }
                )
            self.log_action("podman_volume_list", params, {"count": len(result)})
            return result
        except Exception as e:
            self.log_action("podman_volume_list", params, error=e)
            raise RuntimeError("Failed to list volumes") from e

    def podman_volume_inspect(self, volume_name: str) -> dict:
        """Inspect a volume (real ``volumes.get(...).attrs``)."""
        params = {"volume_name": volume_name}
        try:
            volume = self.client.volumes.get(volume_name)
            result = getattr(volume, "attrs", {}) or {}
            self.log_action(
                "podman_volume_inspect", params, {"volume_name": volume_name}
            )
            return result
        except Exception as e:
            self.log_action("podman_volume_inspect", params, error=e)
            raise RuntimeError("Failed to inspect volume") from e

    def podman_system_prune(self) -> dict:
        """Prune unused resources via ``podman system prune -f``."""
        params: dict[str, Any] = {}
        try:
            output = self._podman_cli(["system", "prune", "-f"], "podman_system_prune")
            result = {"status": "pruned", "output": output.strip()}
            self.log_action("podman_system_prune", params, result)
            return result
        except Exception as e:
            self.log_action("podman_system_prune", params, error=e)
            raise RuntimeError("Failed to prune system") from e

    def podman_health_check(self, container_id: str, config: dict) -> dict:
        """Run a container's healthcheck via ``podman healthcheck run``."""
        params = {"container_id": container_id, "config": config}
        try:
            output = self._podman_cli(
                ["healthcheck", "run", container_id], "podman_health_check"
            )
            result = {
                "container_id": container_id,
                "status": "healthy",
                "output": output.strip(),
            }
            self.log_action("podman_health_check", params, result)
            return result
        except Exception as e:
            self.log_action("podman_health_check", params, error=e)
            raise RuntimeError("Failed to run health check") from e

    def podman_generate_kube_yaml(
        self, pod_name: str, namespace: str = "default"
    ) -> dict:
        """Generate Kubernetes YAML via ``podman generate kube``."""
        params = {"pod_name": pod_name, "namespace": namespace}
        try:
            output = self._podman_cli(
                ["generate", "kube", pod_name], "podman_generate_kube_yaml"
            )
            result = {
                "pod_name": pod_name,
                "namespace": namespace,
                "yaml": output,
                "status": "generated",
            }
            self.log_action("podman_generate_kube_yaml", params, {"pod_name": pod_name})
            return result
        except Exception as e:
            self.log_action("podman_generate_kube_yaml", params, error=e)
            raise RuntimeError("Failed to generate kube YAML") from e

    def podman_play_kube_yaml(self, yaml_path: str) -> dict:
        """Apply a Kubernetes YAML via ``podman play kube``."""
        params = {"yaml_path": yaml_path}
        try:
            if not os.path.exists(yaml_path):
                raise FileNotFoundError("Configured YAML file was not found")
            output = self._podman_cli(
                ["play", "kube", yaml_path], "podman_play_kube_yaml"
            )
            result = {
                "yaml_path": yaml_path,
                "output": output.strip(),
                "status": "played",
            }
            self.log_action("podman_play_kube_yaml", params, result)
            return result
        except Exception as e:
            self.log_action("podman_play_kube_yaml", params, error=e)
            raise RuntimeError("Failed to play kube YAML") from e

    def podman_checkpoint(self, container_id: str, checkpoint_dir: str) -> dict:
        """Checkpoint a container via ``podman container checkpoint``."""
        params = {"container_id": container_id, "checkpoint_dir": checkpoint_dir}
        try:
            args = ["container", "checkpoint"]
            if checkpoint_dir:
                args += ["--export", checkpoint_dir]
            args.append(container_id)
            output = self._podman_cli(args, "podman_checkpoint")
            result = {
                "container_id": container_id,
                "checkpoint_dir": checkpoint_dir,
                "output": output.strip(),
                "status": "checkpointed",
            }
            self.log_action("podman_checkpoint", params, result)
            return result
        except Exception as e:
            self.log_action("podman_checkpoint", params, error=e)
            raise RuntimeError("Failed to create checkpoint") from e

    def podman_restore(self, container_id: str, checkpoint_dir: str) -> dict:
        """Restore a container via ``podman container restore``."""
        params = {"container_id": container_id, "checkpoint_dir": checkpoint_dir}
        try:
            args = ["container", "restore"]
            if checkpoint_dir:
                args += ["--import", checkpoint_dir]
            args.append(container_id)
            output = self._podman_cli(args, "podman_restore")
            result = {
                "container_id": container_id,
                "checkpoint_dir": checkpoint_dir,
                "output": output.strip(),
                "status": "restored",
            }
            self.log_action("podman_restore", params, result)
            return result
        except Exception as e:
            self.log_action("podman_restore", params, error=e)
            raise RuntimeError("Failed to restore checkpoint") from e


def is_app_installed(app_name: str = "docker") -> bool:
    return shutil.which(app_name.lower()) is not None


def create_manager(
    manager_type: str | None = None,
    silent: bool = False,
    log_file: str | None = None,
    host: str | None = None,
    multi_context: bool = False,
) -> ContainerManagerBase:
    """Create a container manager instance.

    Args:
        manager_type: Type of manager ('docker', 'podman', 'kubernetes', 'swarm', or 'multi')
        silent: Suppress logging output
        log_file: Path to log file
        host: Remote host for Docker
        multi_context: Enable multi-context mode (returns MultiContextManager)

    Returns:
        ContainerManagerBase instance. In multi-context mode, a specific
        ``manager_type`` (e.g. "kubernetes"/"docker"/"podman"/"swarm") is
        resolved to that backend's default-context concrete manager (a
        ``KubernetesManager``/``DockerManager``/``PodmanManager``) rather than
        the ``MultiContextManager`` pool itself, so callers see the full verb
        surface they expect. Only ``manager_type=None`` or ``"multi"`` returns
        the raw ``MultiContextManager`` (pool-level operations).
    """
    if host is None:
        host = os.environ.get("CONTAINER_MANAGER_HOST", None)

    # Multi-context mode
    if (
        multi_context
        or manager_type == "multi"
        or os.environ.get("MULTI_CONTEXT_MODE", "false").lower() in ("true", "1", "yes")
    ):
        from container_manager_mcp.multi_context_manager import MultiContextManager

        multi_manager = MultiContextManager(silent=silent, log_file=log_file)

        if manager_type is None or manager_type == "multi":
            return multi_manager

        # A specific backend was requested while multi-context mode is
        # active (e.g. the themed cm_k8s_*/cm_docker_swarm/cm_podman tools
        # call create_manager("kubernetes"/"docker"/"podman")). Resolve to
        # that backend's default-context manager so callers get the real
        # ~40-verb interface instead of the pooling MultiContextManager,
        # which only exposes pool-management plus a handful of generic
        # delegated methods and would otherwise raise
        # AttributeError for any backend-specific verb (e.g. get_cluster_info,
        # list_nodes). Mirrors the get_manager(backend, context) resolution
        # already used by the cm_multi_context tool.
        backend = manager_type.lower()
        if backend in ("rke2", "k3s"):
            backend = "kubernetes"
        return multi_manager.get_manager(backend)

    if manager_type is None:
        manager_type = os.environ.get("CONTAINER_MANAGER_TYPE", None)
    if manager_type is None:
        if is_app_installed("podman"):
            manager_type = "podman"
        if is_app_installed("docker"):
            manager_type = "docker"
    if manager_type is None:
        raise ValueError(
            "No supported container manager detected. Set CONTAINER_MANAGER_TYPE or install Docker/Podman."
        )
    if manager_type.lower() in ["docker", "swarm"]:
        return DockerManager(host=host, silent=silent, log_file=log_file)
    elif manager_type.lower() == "podman":
        if host:
            raise NotImplementedError(
                "Multi-host support is not implemented for Podman"
            )
        return PodmanManager(silent=silent, log_file=log_file)
    elif manager_type.lower() in ["kubernetes", "k8s", "rke2", "k3s"]:
        from container_manager_mcp.k8s_manager import KubernetesManager

        return KubernetesManager(context=host, silent=silent, log_file=log_file)
    else:
        raise ValueError(f"Unsupported container manager type: {manager_type}")


def container_manager():
    print(f"container_manager v{__version__}", file=sys.stderr)
    parser = argparse.ArgumentParser(
        description="Container Manager: A tool to manage containers with Docker, Podman, and Docker Swarm!",
    )
    parser.add_argument("-s", "--silent", action="store_true", help="Suppress output")
    parser.add_argument(
        "-m",
        "--manager",
        type=str,
        default=None,
        help="Container manager type: docker, podman, swarm (default: auto-detect)",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file")
    parser.add_argument("--get-version", action="store_true", help="Get version info")
    parser.add_argument("--get-info", action="store_true", help="Get system info")
    parser.add_argument("--list-images", action="store_true", help="List images")
    parser.add_argument("--pull-image", type=str, default=None, help="Image to pull")
    parser.add_argument("--tag", type=str, default="latest", help="Image tag")
    parser.add_argument("--platform", type=str, default=None, help="Platform")
    parser.add_argument(
        "--remove-image", type=str, default=None, help="Image to remove"
    )
    parser.add_argument("--prune-images", action="store_true", help="Prune images")
    parser.add_argument(
        "--list-containers", action="store_true", help="List containers"
    )
    parser.add_argument("--all", action="store_true", help="Show all containers")
    parser.add_argument("--run-container", type=str, default=None, help="Image to run")
    parser.add_argument("--name", type=str, default=None, help="Container name")
    parser.add_argument("--command", type=str, default=None, help="Command to run")
    parser.add_argument("--detach", action="store_true", help="Detach mode")
    parser.add_argument("--ports", type=str, default=None, help="Port mappings")
    parser.add_argument("--volumes", type=str, default=None, help="Volume mappings")
    parser.add_argument(
        "--environment", type=str, default=None, help="Environment vars"
    )
    parser.add_argument(
        "--stop-container", type=str, default=None, help="Container to stop"
    )
    parser.add_argument("--timeout", type=int, default=10, help="Timeout in seconds")
    parser.add_argument(
        "--remove-container", type=str, default=None, help="Container to remove"
    )
    parser.add_argument(
        "--prune-containers", action="store_true", help="Prune containers"
    )
    parser.add_argument(
        "--get-container-logs", type=str, default=None, help="Container logs"
    )
    parser.add_argument("--tail", type=str, default="50", help="Tail lines")
    parser.add_argument(
        "--exec-in-container", type=str, default=None, help="Container to exec"
    )
    parser.add_argument("--exec-command", type=str, default=None, help="Exec command")
    parser.add_argument("--exec-detach", action="store_true", help="Detach exec")
    parser.add_argument("--list-volumes", action="store_true", help="List volumes")
    parser.add_argument(
        "--create-volume", type=str, default=None, help="Volume to create"
    )
    parser.add_argument(
        "--remove-volume", type=str, default=None, help="Volume to remove"
    )
    parser.add_argument("--prune-volumes", action="store_true", help="Prune volumes")
    parser.add_argument("--list-networks", action="store_true", help="List networks")
    parser.add_argument(
        "--create-network", type=str, default=None, help="Network to create"
    )
    parser.add_argument("--driver", type=str, default="bridge", help="Network driver")
    parser.add_argument(
        "--remove-network", type=str, default=None, help="Network to remove"
    )
    parser.add_argument("--prune-networks", action="store_true", help="Prune networks")
    parser.add_argument("--prune-system", action="store_true", help="Prune system")
    parser.add_argument("--compose-up", type=str, default=None, help="Compose file up")
    parser.add_argument("--build", action="store_true", help="Build images")
    parser.add_argument(
        "--compose-detach", action="store_true", default=True, help="Detach compose"
    )
    parser.add_argument(
        "--compose-down", type=str, default=None, help="Compose file down"
    )
    parser.add_argument("--compose-ps", type=str, default=None, help="Compose ps")
    parser.add_argument("--compose-logs", type=str, default=None, help="Compose logs")
    parser.add_argument("--service", type=str, default=None, help="Specific service")
    parser.add_argument("--init-swarm", action="store_true", help="Init swarm")
    parser.add_argument(
        "--advertise-addr", type=str, default=None, help="Advertise address"
    )
    parser.add_argument("--leave-swarm", action="store_true", help="Leave swarm")
    parser.add_argument("--list-nodes", action="store_true", help="List swarm nodes")
    parser.add_argument(
        "--list-services", action="store_true", help="List swarm services"
    )
    parser.add_argument(
        "--create-service", type=str, default=None, help="Service to create"
    )
    parser.add_argument("--image", type=str, default=None, help="Service image")
    parser.add_argument("--replicas", type=int, default=1, help="Replicas")
    parser.add_argument("--mounts", type=str, default=None, help="Mounts")
    parser.add_argument(
        "--remove-service", type=str, default=None, help="Service to remove"
    )
    parser.add_argument("--force", action="store_true", help="Force removal")

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, "help") and args.help:
        parser.print_help()
        sys.exit(0)

    get_version = args.get_version
    get_info = args.get_info
    list_images = args.list_images
    pull_image = args.pull_image is not None
    pull_image_str = args.pull_image
    tag = args.tag
    platform = args.platform
    remove_image = args.remove_image is not None
    remove_image_str = args.remove_image
    prune_images = args.prune_images
    prune_images_all = args.all if prune_images else False
    force = args.force
    list_containers = args.list_containers
    all_containers = args.all if list_containers else False
    run_container = args.run_container is not None
    run_image = args.run_container
    name = args.name
    command = args.command
    detach = args.detach
    ports_str = args.ports
    volumes_str = args.volumes
    environment_str = args.environment
    stop_container = args.stop_container is not None
    stop_container_id = args.stop_container
    timeout = args.timeout
    remove_container = args.remove_container is not None
    remove_container_id = args.remove_container
    prune_containers = args.prune_containers
    get_container_logs = args.get_container_logs is not None
    container_logs_id = args.get_container_logs
    tail = args.tail
    exec_in_container = args.exec_in_container is not None
    exec_container_id = args.exec_in_container
    exec_command = args.exec_command
    exec_detach = args.exec_detach
    list_volumes = args.list_volumes
    create_volume = args.create_volume is not None
    create_volume_name = args.create_volume
    remove_volume = args.remove_volume is not None
    remove_volume_name = args.remove_volume
    prune_volumes = args.prune_volumes
    prune_volumes_all = args.all if prune_volumes else False
    list_networks = args.list_networks
    create_network = args.create_network is not None
    create_network_name = args.create_network
    driver = args.driver
    remove_network = args.remove_network is not None
    remove_network_id = args.remove_network
    prune_networks = args.prune_networks
    prune_system = args.prune_system
    prune_system_all = args.all if prune_system else False
    compose_up = args.compose_up is not None
    compose_up_file = args.compose_up
    compose_build = args.build
    compose_detach = args.compose_detach
    compose_down = args.compose_down is not None
    compose_down_file = args.compose_down
    compose_ps = args.compose_ps is not None
    compose_ps_file = args.compose_ps
    compose_logs = args.compose_logs is not None
    compose_logs_file = args.compose_logs
    compose_service = args.service
    init_swarm = args.init_swarm
    advertise_addr = args.advertise_addr
    leave_swarm = args.leave_swarm
    list_nodes = args.list_nodes
    list_services = args.list_services
    create_service = args.create_service is not None
    create_service_name = args.create_service
    service_image = args.image
    replicas = args.replicas
    mounts_str = args.mounts
    remove_service = args.remove_service is not None
    remove_service_id = args.remove_service
    manager_type = args.manager
    silent = args.silent
    log_file = args.log_file

    manager = create_manager(manager_type, silent, log_file)

    if get_version:
        print(json.dumps(manager.get_version(), indent=2), file=sys.stderr)

    if get_info:
        print(json.dumps(manager.get_info(), indent=2), file=sys.stderr)

    if list_images:
        print(json.dumps(manager.list_images(), indent=2), file=sys.stderr)

    if pull_image:
        if not pull_image_str:
            raise ValueError("Image required for pull-image")
        print(
            json.dumps(manager.pull_image(pull_image_str, tag, platform), indent=2),
            file=sys.stderr,
        )

    if remove_image:
        if not remove_image_str:
            raise ValueError("Image required for remove-image")
        print(
            json.dumps(manager.remove_image(remove_image_str, force), indent=2),
            file=sys.stderr,
        )

    if prune_images:
        print(
            json.dumps(manager.prune_images(force, prune_images_all), indent=2),
            file=sys.stderr,
        )

    if list_containers:
        print(
            json.dumps(manager.list_containers(all_containers), indent=2),
            file=sys.stderr,
        )

    if run_container:
        if not run_image:
            raise ValueError("Image required for run-container")
        ports = None
        if ports_str:
            ports = {}
            for p in ports_str.split(","):
                host, cont = p.split(":")
                ports[cont + "/tcp"] = host
        volumes = None
        if volumes_str:
            volumes = {}
            for v in volumes_str.split(","):
                parts = v.split(":")
                host = parts[0]
                cont = parts[1]
                mode = parts[2] if len(parts) > 2 else "rw"
                volumes[host] = {"bind": cont, "mode": mode}
        env = None
        if environment_str:
            env = dict(e.split("=") for e in environment_str.split(","))
        print(
            json.dumps(
                manager.run_container(
                    run_image, name, command, detach, ports, volumes, env
                ),
                indent=2,
            ),
            file=sys.stderr,
        )

    if stop_container:
        if not stop_container_id:
            raise ValueError("Container ID required for stop-container")
        print(
            json.dumps(manager.stop_container(stop_container_id, timeout), indent=2),
            file=sys.stderr,
        )

    if remove_container:
        if not remove_container_id:
            raise ValueError("Container ID required for remove-container")
        print(
            json.dumps(manager.remove_container(remove_container_id, force), indent=2),
            file=sys.stderr,
        )

    if prune_containers:
        print(json.dumps(manager.prune_containers(), indent=2), file=sys.stderr)

    if get_container_logs:
        if not container_logs_id:
            raise ValueError("Container ID required for get-container-logs")
        print(manager.get_container_logs(container_logs_id, tail), file=sys.stderr)

    if exec_in_container:
        if not exec_container_id:
            raise ValueError("Container ID required for exec-in-container")
        cmd_list = exec_command.split() if exec_command else []
        print(
            json.dumps(
                manager.exec_in_container(exec_container_id, cmd_list, exec_detach),
                indent=2,
            ),
            file=sys.stderr,
        )

    if list_volumes:
        print(json.dumps(manager.list_volumes(), indent=2), file=sys.stderr)

    if create_volume:
        if not create_volume_name:
            raise ValueError("Name required for create-volume")
        print(
            json.dumps(manager.create_volume(create_volume_name), indent=2),
            file=sys.stderr,
        )

    if remove_volume:
        if not remove_volume_name:
            raise ValueError("Name required for remove-volume")
        print(
            json.dumps(manager.remove_volume(remove_volume_name, force), indent=2),
            file=sys.stderr,
        )

    if prune_volumes:
        print(
            json.dumps(manager.prune_volumes(force, prune_volumes_all), indent=2),
            file=sys.stderr,
        )

    if list_networks:
        print(json.dumps(manager.list_networks(), indent=2), file=sys.stderr)

    if create_network:
        if not create_network_name:
            raise ValueError("Name required for create-network")
        print(
            json.dumps(manager.create_network(create_network_name, driver), indent=2),
            file=sys.stderr,
        )

    if remove_network:
        if not remove_network_id:
            raise ValueError("ID required for remove-network")
        print(
            json.dumps(manager.remove_network(remove_network_id), indent=2),
            file=sys.stderr,
        )

    if prune_networks:
        print(json.dumps(manager.prune_networks(), indent=2), file=sys.stderr)

    if prune_system:
        print(
            json.dumps(manager.prune_system(force, prune_system_all), indent=2),
            file=sys.stderr,
        )

    if compose_up:
        if not compose_up_file:
            raise ValueError("File required for compose-up")
        print(
            manager.compose_up(compose_up_file, compose_detach, compose_build),
            file=sys.stderr,
        )

    if compose_down:
        if not compose_down_file:
            raise ValueError("File required for compose-down")
        print(manager.compose_down(compose_down_file), file=sys.stderr)

    if compose_ps:
        if not compose_ps_file:
            raise ValueError("File required for compose-ps")
        print(manager.compose_ps(compose_ps_file), file=sys.stderr)

    if compose_logs:
        if not compose_logs_file:
            raise ValueError("File required for compose-logs")
        print(manager.compose_logs(compose_logs_file, compose_service), file=sys.stderr)

    if init_swarm:
        print(json.dumps(manager.init_swarm(advertise_addr), indent=2), file=sys.stderr)

    if leave_swarm:
        print(json.dumps(manager.leave_swarm(force), indent=2), file=sys.stderr)

    if list_nodes:
        print(json.dumps(manager.list_nodes(), indent=2), file=sys.stderr)

    if list_services:
        print(json.dumps(manager.list_services(), indent=2), file=sys.stderr)

    if create_service:
        if not create_service_name:
            raise ValueError("Name required for create-service")
        if not service_image:
            raise ValueError("Image required for create-service")
        service_ports: dict[str, str] | None = None
        if ports_str:
            service_ports = {}
            for p in ports_str.split(","):
                host, cont = p.split(":")
                service_ports[cont + "/tcp"] = host
        mounts = None
        if mounts_str:
            mounts = mounts_str.split(",")
        print(
            json.dumps(
                manager.create_service(
                    create_service_name, service_image, replicas, service_ports, mounts
                ),
                indent=2,
            ),
            file=sys.stderr,
        )

    if remove_service:
        if not remove_service_id:
            raise ValueError("ID required for remove-service")
        print(
            json.dumps(manager.remove_service(remove_service_id), indent=2),
            file=sys.stderr,
        )

    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    container_manager()

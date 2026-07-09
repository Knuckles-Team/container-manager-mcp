"""UnsupportedMixin for KubernetesManager (split from k8s_manager.py)."""

from container_manager_mcp.models import (
    ContainerInfo,
    ImageInfo,
    NetworkInfo,
    VolumeInfo,
)


class UnsupportedMixin:
    def list_images(self) -> list[ImageInfo]:
        raise self._unsupported("list_images")

    def pull_image(self, image: str, tag: str = "latest", platform: str | None = None):
        raise self._unsupported("pull_image")

    def remove_image(self, image: str, force: bool = False) -> dict:
        raise self._unsupported("remove_image")

    def prune_images(self, force: bool = False, all: bool = False) -> dict:
        raise self._unsupported("prune_images")

    def list_containers(self, all: bool = False) -> list[ContainerInfo]:
        raise self._unsupported("list_containers")

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
        raise self._unsupported("run_container")

    def inspect_container(self, container_id: str) -> dict:
        raise self._unsupported("inspect_container")

    def stop_container(self, container_id: str, timeout: int = 10) -> dict:
        raise self._unsupported("stop_container")

    def remove_container(self, container_id: str, force: bool = False) -> dict:
        raise self._unsupported("remove_container")

    def prune_containers(self) -> dict:
        raise self._unsupported("prune_containers")

    def get_container_logs(self, container_id: str, tail: str = "50") -> str:
        raise self._unsupported("get_container_logs")

    def exec_in_container(
        self, container_id: str, command: list[str], detach: bool = False
    ) -> dict:
        raise self._unsupported("exec_in_container")

    def list_volumes(self) -> list[VolumeInfo]:
        raise self._unsupported("list_volumes")

    def create_volume(self, name: str) -> VolumeInfo:
        raise self._unsupported("create_volume")

    def remove_volume(self, name: str, force: bool = False) -> dict:
        raise self._unsupported("remove_volume")

    def prune_volumes(self, force: bool = False, all: bool = False) -> dict:
        raise self._unsupported("prune_volumes")

    def list_networks(self) -> list[NetworkInfo]:
        raise self._unsupported("list_networks")

    def create_network(self, name: str, driver: str = "bridge") -> NetworkInfo:
        raise self._unsupported("create_network")

    def remove_network(self, network_id: str) -> dict:
        raise self._unsupported("remove_network")

    def prune_networks(self) -> dict:
        raise self._unsupported("prune_networks")

    def prune_system(self, force: bool = False, all: bool = False) -> dict:
        raise self._unsupported("prune_system")

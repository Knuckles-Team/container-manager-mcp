"""Composed :class:`KubernetesManager` (mixin composition)."""

from container_manager_mcp.container_manager import ContainerManagerBase
from container_manager_mcp.k8s.base import _K8sBase
from container_manager_mcp.k8s.workloads import WorkloadsMixin
from container_manager_mcp.k8s.config import ConfigMixin
from container_manager_mcp.k8s.networking import NetworkingMixin
from container_manager_mcp.k8s.storage import StorageMixin
from container_manager_mcp.k8s.rbac import RbacMixin
from container_manager_mcp.k8s.cluster_nodes import ClusterNodesMixin
from container_manager_mcp.k8s.governance import GovernanceMixin
from container_manager_mcp.k8s.observability import ObservabilityMixin
from container_manager_mcp.k8s.compose import ComposeMixin
from container_manager_mcp.k8s.interop import InteropMixin
from container_manager_mcp.k8s.unsupported import UnsupportedMixin


class KubernetesManager(
    WorkloadsMixin,
    ConfigMixin,
    NetworkingMixin,
    StorageMixin,
    RbacMixin,
    ClusterNodesMixin,
    GovernanceMixin,
    ObservabilityMixin,
    ComposeMixin,
    InteropMixin,
    UnsupportedMixin,
    _K8sBase,
    ContainerManagerBase,
):
    """ContainerManagerBase implementation backed by a Kubernetes cluster."""

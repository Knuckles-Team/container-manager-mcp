"""Multi-context container manager supporting K8S, Docker, Podman, and Swarm simultaneously."""

import logging
import os
from typing import Any

from container_manager_mcp.container_manager import (
    ContainerManagerBase,
    DockerManager,
    PodmanManager,
    create_manager,
)


class MultiContextManager:
    """Manages multiple container backends (K8S, Docker, Podman, Swarm) with multiple contexts each."""
    
    def __init__(self, silent: bool = False, log_file: str | None = None):
        super().__init__(silent, log_file)
        self.setup_logging(log_file)
        
        # Manager pools for each backend
        self.k8s_managers: dict[str, Any] = {}  # context_name -> KubernetesManager
        self.docker_managers: dict[str, Any] = {}  # context_name -> DockerManager
        self.podman_manager: Any = None  # Single Podman manager (local only)
        self.swarm_managers: dict[str, Any] = {}  # context_name -> DockerManager (Swarm mode)
        
        # Default contexts
        self.default_k8s_context: str | None = None
        self.default_docker_context: str | None = None
        self.default_swarm_context: str | None = None
        
        self._initialize_managers()
    
    def setup_logging(self, log_file: str | None = None):
        if not log_file:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_file = os.path.join(script_dir, "multi_context_manager.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Multi-Context Manager logging initialized to {log_file}")
    
    def _initialize_managers(self):
        """Initialize all configured managers from environment variables."""
        self.logger.info("Initializing multi-context managers...")
        
        # Initialize Kubernetes contexts
        k8s_contexts = self._parse_context_config("K8S_CONTEXTS")
        if k8s_contexts:
            for context_name, context_value in k8s_contexts.items():
                try:
                    self._add_k8s_context(context_name, context_value)
                except Exception as e:
                    self.logger.error(f"Failed to initialize K8S context '{context_name}': {e}")
        
        # Set default K8S context
        self.default_k8s_context = os.environ.get("DEFAULT_K8S_CONTEXT")
        if self.default_k8s_context and self.default_k8s_context in self.k8s_managers:
            self.logger.info(f"Default K8S context set to: {self.default_k8s_context}")
        elif self.k8s_managers:
            self.default_k8s_context = list(self.k8s_managers.keys())[0]
            self.logger.info(f"Default K8S context auto-selected: {self.default_k8s_context}")
        
        # Initialize Docker contexts
        docker_contexts = self._parse_context_config("DOCKER_CONTEXTS")
        if docker_contexts:
            for context_name, host in docker_contexts.items():
                try:
                    self._add_docker_context(context_name, host)
                except Exception as e:
                    self.logger.error(f"Failed to initialize Docker context '{context_name}': {e}")
        
        # Set default Docker context
        self.default_docker_context = os.environ.get("DEFAULT_DOCKER_CONTEXT")
        if self.default_docker_context and self.default_docker_context in self.docker_managers:
            self.logger.info(f"Default Docker context set to: {self.default_docker_context}")
        elif self.docker_managers:
            self.default_docker_context = list(self.docker_managers.keys())[0]
            self.logger.info(f"Default Docker context auto-selected: {self.default_docker_context}")
        
        # Initialize Swarm contexts
        swarm_contexts = self._parse_context_config("SWARM_CONTEXTS")
        if swarm_contexts:
            for context_name, host in swarm_contexts.items():
                try:
                    self._add_swarm_context(context_name, host)
                except Exception as e:
                    self.logger.error(f"Failed to initialize Swarm context '{context_name}': {e}")
        
        # Set default Swarm context
        self.default_swarm_context = os.environ.get("DEFAULT_SWARM_CONTEXT")
        if self.default_swarm_context and self.default_swarm_context in self.swarm_managers:
            self.logger.info(f"Default Swarm context set to: {self.default_swarm_context}")
        elif self.swarm_managers:
            self.default_swarm_context = list(self.swarm_managers.keys())[0]
            self.logger.info(f"Default Swarm context auto-selected: {self.default_swarm_context}")
        
        # Initialize Podman (local only)
        podman_enabled = os.environ.get("PODMAN_ENABLED", "true").lower() in ("true", "1", "yes")
        if podman_enabled:
            try:
                self._add_podman_manager()
            except Exception as e:
                self.logger.error(f"Failed to initialize Podman manager: {e}")
        
        self.logger.info(
            f"Multi-Context Manager initialized: "
            f"K8S={len(self.k8s_managers)} contexts, "
            f"Docker={len(self.docker_managers)} contexts, "
            f"Swarm={len(self.swarm_managers)} contexts, "
            f"Podman={1 if self.podman_manager else 0}"
        )
    
    def _parse_context_config(self, env_var: str) -> dict[str, str]:
        """Parse context configuration from environment variable.
        
        Format: "context1=value1;context2=value2;context3=value3"
        """
        config_str = os.environ.get(env_var, "")
        if not config_str:
            return {}
        
        contexts = {}
        for item in config_str.split(";"):
            if "=" in item:
                name, value = item.split("=", 1)
                contexts[name.strip()] = value.strip()
        
        return contexts
    
    def _add_k8s_context(self, context_name: str, context_value: str):
        """Add a Kubernetes context to the manager pool."""
        from container_manager_mcp.k8s_manager import KubernetesManager
        
        self.logger.info(f"Adding K8S context: {context_name} -> {context_value}")
        manager = KubernetesManager(
            context=context_value,
            silent=self.silent,
            log_file=self.log_file
        )
        self.k8s_managers[context_name] = manager
    
    def _add_docker_context(self, context_name: str, host: str):
        """Add a Docker context to the manager pool."""
        self.logger.info(f"Adding Docker context: {context_name} -> {host}")
        manager = DockerManager(
            host=host if host else None,
            silent=self.silent,
            log_file=self.log_file
        )
        self.docker_managers[context_name] = manager
    
    def _add_swarm_context(self, context_name: str, host: str):
        """Add a Swarm context to the manager pool."""
        self.logger.info(f"Adding Swarm context: {context_name} -> {host}")
        manager = DockerManager(
            host=host if host else None,
            silent=self.silent,
            log_file=self.log_file
        )
        self.swarm_managers[context_name] = manager
    
    def _add_podman_manager(self):
        """Add Podman manager (local only)."""
        self.logger.info("Adding Podman manager (local)")
        self.podman_manager = PodmanManager(
            silent=self.silent,
            log_file=self.log_file
        )
    
    def get_k8s_manager(self, context_name: str | None = None) -> Any:
        """Get a Kubernetes manager by context name."""
        if context_name is None:
            context_name = self.default_k8s_context
        
        if context_name not in self.k8s_managers:
            available = ", ".join(self.k8s_managers.keys())
            raise ValueError(
                f"K8S context '{context_name}' not found. "
                f"Available contexts: {available}"
            )
        
        return self.k8s_managers[context_name]
    
    def get_docker_manager(self, context_name: str | None = None) -> Any:
        """Get a Docker manager by context name."""
        if context_name is None:
            context_name = self.default_docker_context
        
        if context_name not in self.docker_managers:
            available = ", ".join(self.docker_managers.keys())
            raise ValueError(
                f"Docker context '{context_name}' not found. "
                f"Available contexts: {available}"
            )
        
        return self.docker_managers[context_name]
    
    def get_swarm_manager(self, context_name: str | None = None) -> Any:
        """Get a Swarm manager by context name."""
        if context_name is None:
            context_name = self.default_swarm_context
        
        if context_name not in self.swarm_managers:
            available = ", ".join(self.swarm_managers.keys())
            raise ValueError(
                f"Swarm context '{context_name}' not found. "
                f"Available contexts: {available}"
            )
        
        return self.swarm_managers[context_name]
    
    def get_podman_manager(self) -> Any:
        """Get the Podman manager."""
        if self.podman_manager is None:
            raise ValueError("Podman manager is not enabled or failed to initialize")
        return self.podman_manager
    
    def get_manager(
        self,
        backend: str = "kubernetes",
        context: str | None = None
    ) -> ContainerManagerBase:
        """Get a manager by backend type and context name.
        
        Args:
            backend: One of 'kubernetes', 'docker', 'podman', 'swarm'
            context: Context name (uses default if None)
        
        Returns:
            ContainerManagerBase instance
        """
        backend = backend.lower()
        
        if backend in ["kubernetes", "k8s"]:
            return self.get_k8s_manager(context)
        elif backend == "docker":
            return self.get_docker_manager(context)
        elif backend == "podman":
            return self.get_podman_manager()
        elif backend == "swarm":
            return self.get_swarm_manager(context)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def list_available_contexts(self) -> dict[str, dict]:
        """List all available contexts across all backends."""
        return {
            "kubernetes": {
                "contexts": list(self.k8s_managers.keys()),
                "default": self.default_k8s_context
            },
            "docker": {
                "contexts": list(self.docker_managers.keys()),
                "default": self.default_docker_context
            },
            "swarm": {
                "contexts": list(self.swarm_managers.keys()),
                "default": self.default_swarm_context
            },
            "podman": {
                "enabled": self.podman_manager is not None
            }
        }
    
    # ------------------------------------------------------------------
    # Delegated operations - route to appropriate backend
    # ------------------------------------------------------------------
    
    def list_containers(self, backend: str = "kubernetes", context: str | None = None, **kwargs):
        """List containers from specified backend and context."""
        manager = self.get_manager(backend, context)
        return manager.list_containers(**kwargs)
    
    def list_images(self, backend: str = "kubernetes", context: str | None = None, **kwargs):
        """List images from specified backend and context."""
        manager = self.get_manager(backend, context)
        return manager.list_images(**kwargs)
    
    def list_volumes(self, backend: str = "kubernetes", context: str | None = None, **kwargs):
        """List volumes from specified backend and context."""
        manager = self.get_manager(backend, context)
        return manager.list_volumes(**kwargs)
    
    def list_networks(self, backend: str = "kubernetes", context: str | None = None, **kwargs):
        """List networks from specified backend and context."""
        manager = self.get_manager(backend, context)
        return manager.list_networks(**kwargs)

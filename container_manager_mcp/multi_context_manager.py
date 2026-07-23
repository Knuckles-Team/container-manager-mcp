"""Multi-context container manager supporting K8S, Docker, Podman, and Swarm simultaneously."""

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from container_manager_mcp.container_manager import (
    ContainerManagerBase,
    DockerManager,
    PodmanManager,
)

# Default number of worker threads used to fan operations out across contexts.
DEFAULT_MAX_WORKERS = 8

# How long (in seconds) a positive/negative health check result is trusted before
# a fresh probe is made.
HEALTH_CHECK_TTL_SECONDS = float(os.environ.get("HEALTH_CHECK_TTL_SECONDS", "30"))


class MultiContextManager:
    """Manages multiple container backends (K8S, Docker, Podman, Swarm) with multiple contexts each.

    This is intentionally NOT a ContainerManagerBase subclass: it pools other
    concrete managers rather than implementing the ~40 abstract container verbs
    itself.
    """

    def __init__(self, silent: bool = False, log_file: str | None = None):
        # Set these before anything else touches self.silent / self.log_file
        # (e.g. _initialize_managers -> _add_*_context reads them).
        self.silent = silent
        self.log_file = log_file
        self.setup_logging(log_file)

        # Manager pools for each backend
        self.k8s_managers: dict[str, Any] = {}  # context_name -> KubernetesManager
        self.docker_managers: dict[str, Any] = {}  # context_name -> DockerManager
        self.podman_manager: Any = None  # Single Podman manager (local only)
        self.swarm_managers: dict[str, Any] = (
            {}
        )  # context_name -> DockerManager (Swarm mode)

        # Default contexts
        self.default_k8s_context: str | None = None
        self.default_docker_context: str | None = None
        self.default_swarm_context: str | None = None

        # Original config values per (backend, context_name), captured at add-time
        # so a lazy reconnect never needs to re-parse environment variables.
        self._context_config: dict[str, dict[str, str]] = {
            "kubernetes": {},
            "docker": {},
            "swarm": {},
        }

        # Health-check cache: (backend, context_name) -> (is_healthy, checked_at)
        self._health_cache: dict[tuple[str, str], tuple[bool, float]] = {}
        # Per-context locks guarding lazy reconnects so concurrent callers don't
        # race to reconnect the same context.
        self._context_locks: dict[tuple[str, str], threading.Lock] = {}
        self._context_locks_guard = threading.Lock()

        max_workers = int(
            os.environ.get("MULTI_CONTEXT_MAX_WORKERS", str(DEFAULT_MAX_WORKERS))
        )
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="multi-context-manager"
        )

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
        self.logger.info("Multi-context manager logging initialized")

    def _initialize_managers(self):
        """Initialize all configured managers from environment variables."""
        self.logger.info("Initializing multi-context managers...")

        # Initialize Kubernetes contexts
        k8s_contexts = self._parse_context_config(os.environ.get("K8S_CONTEXTS", ""))
        if k8s_contexts:
            for context_name, context_value in k8s_contexts.items():
                try:
                    self._add_k8s_context(context_name, context_value)
                except Exception as e:
                    self.logger.error(
                        "Failed to initialize Kubernetes context: error_type=%s",
                        type(e).__name__,
                    )

        # In-cluster fallback: when running inside a Kubernetes pod with no
        # explicit K8S_CONTEXTS, register the pod's own service-account context
        # so the k8s tools target the cluster we're deployed into. The
        # KubernetesManager already loads in-cluster config when
        # KUBERNETES_SERVICE_HOST is set (k8s/base.py), so an empty context
        # value resolves to load_incluster_config().
        if not self.k8s_managers and os.environ.get("KUBERNETES_SERVICE_HOST"):
            try:
                self._add_k8s_context("in-cluster", "")
                self.logger.info(
                    "Registered in-cluster K8S context from the pod service account"
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize in-cluster K8S context: {e}")

        # Set default K8S context
        self.default_k8s_context = os.environ.get("DEFAULT_K8S_CONTEXT")
        if self.default_k8s_context and self.default_k8s_context in self.k8s_managers:
            self.logger.info("Default Kubernetes context configured")
        elif self.k8s_managers:
            self.default_k8s_context = list(self.k8s_managers.keys())[0]
            self.logger.info(
                f"Default K8S context auto-selected: {self.default_k8s_context}"
            )

        # Initialize Docker contexts
        docker_contexts = self._parse_context_config(
            os.environ.get("DOCKER_CONTEXTS", "")
        )
        if docker_contexts:
            for context_name, host in docker_contexts.items():
                try:
                    self._add_docker_context(context_name, host)
                except Exception as e:
                    self.logger.error(
                        "Failed to initialize Docker context: error_type=%s",
                        type(e).__name__,
                    )

        # Set default Docker context
        self.default_docker_context = os.environ.get("DEFAULT_DOCKER_CONTEXT")
        if (
            self.default_docker_context
            and self.default_docker_context in self.docker_managers
        ):
            self.logger.info(
                f"Default Docker context set to: {self.default_docker_context}"
            )
        elif self.docker_managers:
            self.default_docker_context = list(self.docker_managers.keys())[0]
            self.logger.info(
                f"Default Docker context auto-selected: {self.default_docker_context}"
            )

        # Initialize Swarm contexts
        swarm_contexts = self._parse_context_config(
            os.environ.get("SWARM_CONTEXTS", "")
        )
        if swarm_contexts:
            for context_name, host in swarm_contexts.items():
                try:
                    self._add_swarm_context(context_name, host)
                except Exception as e:
                    self.logger.error(
                        "Failed to initialize Swarm context: error_type=%s",
                        type(e).__name__,
                    )

        # Set default Swarm context
        self.default_swarm_context = os.environ.get("DEFAULT_SWARM_CONTEXT")
        if (
            self.default_swarm_context
            and self.default_swarm_context in self.swarm_managers
        ):
            self.logger.info(
                f"Default Swarm context set to: {self.default_swarm_context}"
            )
        elif self.swarm_managers:
            self.default_swarm_context = list(self.swarm_managers.keys())[0]
            self.logger.info(
                f"Default Swarm context auto-selected: {self.default_swarm_context}"
            )

        # Initialize Podman (local only)
        podman_enabled = os.environ.get("PODMAN_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        if podman_enabled:
            try:
                self._add_podman_manager()
            except Exception as e:
                self.logger.error("Operation failed: error_type=%s", type(e).__name__)

        self.logger.info(
            f"Multi-Context Manager initialized: "
            f"K8S={len(self.k8s_managers)} contexts, "
            f"Docker={len(self.docker_managers)} contexts, "
            f"Swarm={len(self.swarm_managers)} contexts, "
            f"Podman={1 if self.podman_manager else 0}"
        )

    def _parse_context_config(self, config_str: str) -> dict[str, str]:
        """Parse a context configuration string.

        Format: "context1=value1;context2=value2;context3=value3"
        """
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

        self.logger.info("Adding configured Kubernetes context")
        manager = KubernetesManager(
            context=context_value, silent=self.silent, log_file=self.log_file
        )
        self.k8s_managers[context_name] = manager
        self._context_config["kubernetes"][context_name] = context_value
        self._health_cache.pop(("kubernetes", context_name), None)

    def _add_docker_context(self, context_name: str, host: str):
        """Add a Docker context to the manager pool."""
        self.logger.info("Adding configured Docker context")
        manager = DockerManager(
            host=host if host else None, silent=self.silent, log_file=self.log_file
        )
        self.docker_managers[context_name] = manager
        self._context_config["docker"][context_name] = host
        self._health_cache.pop(("docker", context_name), None)

    def _add_swarm_context(self, context_name: str, host: str):
        """Add a Swarm context to the manager pool."""
        self.logger.info("Adding configured Swarm context")
        manager = DockerManager(
            host=host if host else None, silent=self.silent, log_file=self.log_file
        )
        self.swarm_managers[context_name] = manager
        self._context_config["swarm"][context_name] = host
        self._health_cache.pop(("swarm", context_name), None)

    def _add_podman_manager(self):
        """Add Podman manager (local only)."""
        self.logger.info("Adding Podman manager (local)")
        self.podman_manager = PodmanManager(silent=self.silent, log_file=self.log_file)
        self._health_cache.pop(("podman", "local"), None)

    # ------------------------------------------------------------------
    # Health checks + lazy reconnect
    # ------------------------------------------------------------------

    def _get_context_lock(self, backend: str, context_name: str) -> threading.Lock:
        key = (backend, context_name)
        with self._context_locks_guard:
            lock = self._context_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._context_locks[key] = lock
            return lock

    def _is_healthy(
        self, manager: Any, backend: str | None = None, context_name: str | None = None
    ) -> bool:
        """Cheaply probe whether a pooled manager is still usable.

        Results are cached for HEALTH_CHECK_TTL_SECONDS keyed by (backend,
        context_name) when those are provided, to avoid hammering the
        underlying daemon/cluster on every call.
        """
        cache_key = (backend, context_name) if backend and context_name else None
        if cache_key is not None:
            cached = self._health_cache.get(cache_key)
            if cached is not None:
                is_healthy, checked_at = cached
                if (time.monotonic() - checked_at) < HEALTH_CHECK_TTL_SECONDS:
                    return is_healthy

        healthy = False
        try:
            if manager is not None:
                manager.get_version()
                healthy = True
        except Exception as e:
            self.logger.warning(
                "Context health check failed: backend_type=%s error_type=%s",
                backend,
                type(e).__name__,
            )
            healthy = False

        if cache_key is not None:
            self._health_cache[cache_key] = (healthy, time.monotonic())

        return healthy

    def _reconnect_context(self, backend: str, context_name: str) -> None:
        """Lazily re-create a pooled manager using its originally-captured config."""
        backend_config = self._context_config.get(backend, {})
        if context_name not in backend_config:
            raise ValueError(
                f"No stored configuration for {backend} context '{context_name}'; cannot reconnect"
            )
        original_value = backend_config[context_name]

        self.logger.info("Reconnecting configured context: backend_type=%s", backend)
        if backend == "kubernetes":
            self._add_k8s_context(context_name, original_value)
        elif backend == "docker":
            self._add_docker_context(context_name, original_value)
        elif backend == "swarm":
            self._add_swarm_context(context_name, original_value)
        else:
            raise ValueError(f"Unsupported backend for reconnect: {backend}")

    def _entitled(self, namespace: str, names: list[str]) -> list[str]:
        """Filter a context pool to what the calling identity may reach.

        Routes the pool through agent-utilities' shared identity-scoped resolver
        (CONCEPT:AU-OS.identity.identity-scoped-resource-autoload): a caller's
        Okta/Keycloak groups decide which contexts auto-load for them.
        Unauthenticated/local callers (no ambient actor bound at all) see the
        full pool, so behaviour is unchanged until a real identity scopes it
        down. Degrades to the full pool if agent-utilities predates the
        resolver, or has no ambient identity bound (``IdentityRequiredError``)
        — an authenticated caller who is simply under-entitled still gets
        filtered/denied by ``identity_scoped_resources`` itself, not here.
        """
        try:
            from agent_utilities.security.brain_context import IdentityRequiredError
            from agent_utilities.security.entitlements import (
                identity_scoped_resources,
            )
        except Exception:
            return names
        try:
            return list(identity_scoped_resources(namespace, names))
        except IdentityRequiredError:
            return names

    def _resolve_context(
        self, namespace: str, pool: dict[str, Any], default: str | None,
        context_name: str | None,
    ) -> str:
        """Resolve + authorize a context against the caller's entitlements."""
        entitled = self._entitled(namespace, list(pool.keys()))
        if context_name is None:
            if default in entitled:
                return default
            if entitled:
                return entitled[0]
            raise ValueError(
                f"No {namespace} contexts are available to your identity. "
                "Your Okta/Keycloak groups grant none of the configured contexts."
            )
        if context_name not in pool:
            raise ValueError(
                f"{namespace} context '{context_name}' not found. "
                f"Available contexts: {', '.join(pool.keys())}"
            )
        if context_name not in entitled:
            raise PermissionError(
                f"Your identity is not entitled to the {namespace} context "
                f"'{context_name}'. Entitled: {', '.join(entitled) or '(none)'}"
            )
        return context_name

    def get_k8s_manager(self, context_name: str | None = None) -> Any:
        """Get a Kubernetes manager by context name, lazily reconnecting if unhealthy.

        The context is resolved against the caller's identity entitlements: an
        omitted ``context_name`` auto-selects the caller's entitled default, and
        a named context they are not entitled to is denied.
        """
        context_name = self._resolve_context(
            "k8s", self.k8s_managers, self.default_k8s_context, context_name
        )

        manager = self.k8s_managers[context_name]
        if not self._is_healthy(manager, "kubernetes", context_name):
            lock = self._get_context_lock("kubernetes", context_name)
            with lock:
                manager = self.k8s_managers.get(context_name)
                if not self._is_healthy(manager, "kubernetes", context_name):
                    self._reconnect_context("kubernetes", context_name)
                    manager = self.k8s_managers[context_name]

        return manager

    def get_docker_manager(self, context_name: str | None = None) -> Any:
        """Get a Docker manager by context name, lazily reconnecting if unhealthy.

        Context resolved against the caller's identity entitlements (see
        :meth:`get_k8s_manager`)."""
        context_name = self._resolve_context(
            "docker", self.docker_managers, self.default_docker_context, context_name
        )

        manager = self.docker_managers[context_name]
        if not self._is_healthy(manager, "docker", context_name):
            lock = self._get_context_lock("docker", context_name)
            with lock:
                manager = self.docker_managers.get(context_name)
                if not self._is_healthy(manager, "docker", context_name):
                    self._reconnect_context("docker", context_name)
                    manager = self.docker_managers[context_name]

        return manager

    def get_swarm_manager(self, context_name: str | None = None) -> Any:
        """Get a Swarm manager by context name, lazily reconnecting if unhealthy.

        Context resolved against the caller's identity entitlements (see
        :meth:`get_k8s_manager`)."""
        context_name = self._resolve_context(
            "swarm", self.swarm_managers, self.default_swarm_context, context_name
        )

        manager = self.swarm_managers[context_name]
        if not self._is_healthy(manager, "swarm", context_name):
            lock = self._get_context_lock("swarm", context_name)
            with lock:
                manager = self.swarm_managers.get(context_name)
                if not self._is_healthy(manager, "swarm", context_name):
                    self._reconnect_context("swarm", context_name)
                    manager = self.swarm_managers[context_name]

        return manager

    def get_podman_manager(self) -> Any:
        """Get the Podman manager."""
        if self.podman_manager is None:
            raise ValueError("Podman manager is not enabled or failed to initialize")

        if not self._is_healthy(self.podman_manager, "podman", "local"):
            lock = self._get_context_lock("podman", "local")
            with lock:
                if not self._is_healthy(self.podman_manager, "podman", "local"):
                    self.logger.info("Reconnecting Podman manager...")
                    self._add_podman_manager()

        return self.podman_manager

    def get_manager(
        self, backend: str = "kubernetes", context: str | None = None
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
        """List the contexts the CALLER may reach, per backend.

        Auto-scoped to the caller's Okta/Keycloak identity: each backend's pool
        is filtered to the contexts their groups entitle
        (CONCEPT:AU-OS.identity.identity-scoped-resource-autoload), so an
        operator connects to exactly the environments they have access to with
        no manual context selection. Unauthenticated/local (SYSTEM_ACTOR) sees
        all — unchanged from today."""
        k8s = self._entitled("k8s", list(self.k8s_managers.keys()))
        docker = self._entitled("docker", list(self.docker_managers.keys()))
        swarm = self._entitled("swarm", list(self.swarm_managers.keys()))
        return {
            "kubernetes": {
                "contexts": k8s,
                "default": self.default_k8s_context if self.default_k8s_context in k8s
                else (k8s[0] if k8s else None),
            },
            "docker": {
                "contexts": docker,
                "default": self.default_docker_context
                if self.default_docker_context in docker
                else (docker[0] if docker else None),
            },
            "swarm": {
                "contexts": swarm,
                "default": self.default_swarm_context
                if self.default_swarm_context in swarm
                else (swarm[0] if swarm else None),
            },
            "podman": {"enabled": self.podman_manager is not None},
        }

    # ------------------------------------------------------------------
    # Parallel fan-out
    # ------------------------------------------------------------------

    def _managers_for_backend(self, backend: str) -> dict[str, Any]:
        """Return the {context_name: manager} pool for a backend."""
        backend = backend.lower()
        if backend in ["kubernetes", "k8s"]:
            return dict(self.k8s_managers)
        elif backend == "docker":
            return dict(self.docker_managers)
        elif backend == "swarm":
            return dict(self.swarm_managers)
        elif backend == "podman":
            return (
                {"local": self.podman_manager}
                if self.podman_manager is not None
                else {}
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def fan_out(
        self,
        backend: str,
        op: str,
        contexts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run method `op` against every (or the listed) context of `backend` concurrently.

        Uses a thread pool since the underlying docker/podman/kubernetes SDK
        calls are synchronous and I/O-bound.

        Returns:
            {context_name: result} on success, or {context_name: {"error": str}}
            for any context whose call raised.
        """
        pool = self._managers_for_backend(backend)
        target_contexts = contexts if contexts is not None else list(pool.keys())

        futures: dict[str, Future] = {}
        for context_name in target_contexts:
            manager = pool.get(context_name)
            if manager is None:
                continue
            method = getattr(manager, op, None)
            if method is None:
                futures[context_name] = None  # marker handled below
                continue
            futures[context_name] = self._executor.submit(method, **kwargs)

        results: dict[str, Any] = {}
        for context_name in target_contexts:
            if context_name not in pool:
                results[context_name] = {
                    "error": f"Context '{context_name}' not found for backend '{backend}'"
                }
                continue

            future = futures.get(context_name)
            if future is None:
                results[context_name] = {
                    "error": f"Operation '{op}' is not supported by backend '{backend}'"
                }
                continue

            try:
                results[context_name] = future.result()
            except Exception as e:
                self.logger.error("Operation failed: error_type=%s", type(e).__name__)
                results[context_name] = {"error": "Operation failed"}

        return results

    def fan_out_all(self, op: str, **kwargs: Any) -> dict[str, dict[str, Any]]:
        """Fan a read-only operation across every context of every backend.

        Returns:
            {backend: {context_name: result_or_error}}
        """
        backends = ["kubernetes", "docker", "swarm", "podman"]
        results: dict[str, dict[str, Any]] = {}
        for backend in backends:
            pool = self._managers_for_backend(backend)
            if not pool:
                results[backend] = {}
                continue
            results[backend] = self.fan_out(backend, op, **kwargs)
        return results

    def shutdown(self) -> None:
        """Shut down the thread pool used for fan-out operations."""
        self._executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Delegated operations - route to appropriate backend
    # ------------------------------------------------------------------

    def list_containers(
        self, backend: str = "kubernetes", context: str | None = None, **kwargs
    ):
        """List containers from specified backend and context."""
        manager = self.get_manager(backend, context)
        return manager.list_containers(**kwargs)

    def list_images(
        self, backend: str = "kubernetes", context: str | None = None, **kwargs
    ):
        """List images from specified backend and context."""
        manager = self.get_manager(backend, context)
        return manager.list_images(**kwargs)

    def list_volumes(
        self, backend: str = "kubernetes", context: str | None = None, **kwargs
    ):
        """List volumes from specified backend and context."""
        manager = self.get_manager(backend, context)
        return manager.list_volumes(**kwargs)

    def list_networks(
        self, backend: str = "kubernetes", context: str | None = None, **kwargs
    ):
        """List networks from specified backend and context."""
        manager = self.get_manager(backend, context)
        return manager.list_networks(**kwargs)

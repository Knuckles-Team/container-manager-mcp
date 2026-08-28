"""MCP tools for Kubernetes workload operations.

Themed dispatcher covering pods, rollouts, deployment/update strategies,
StatefulSets, DaemonSets, ReplicaSets, Jobs, and CronJobs.
"""

import logging
from typing import Literal

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager
from container_manager_mcp.mcp_server import ctx_log

_UNHANDLED = object()  # sentinel: this action doesn't belong to this dispatch group


async def _describe_pod_action(manager, pod_name, namespace):
    if not pod_name:
        return "Error: 'pod_name' is required for describe_pod"
    return await run_blocking(
        manager.describe_pod, pod_name=pod_name, namespace=namespace
    )


async def _exec_pod_action(manager, pod_name, namespace, command, exec_command, exec_container):
    if not pod_name:
        return "Error: 'pod_name' is required for exec_pod"
    cmd = command if command else (exec_command.split() if exec_command else None)
    return await run_blocking(
        manager.exec_pod,
        pod_name=pod_name,
        namespace=namespace,
        command=cmd,
        container=exec_container,
    )


async def _port_forward_pod_action(manager, pod_name, namespace, local_port, remote_port):
    if not pod_name or not local_port or not remote_port:
        return "Error: 'pod_name', 'local_port', and 'remote_port' are required for port_forward_pod"
    return await run_blocking(
        manager.port_forward_pod,
        pod_name=pod_name,
        namespace=namespace,
        local_port=local_port,
        remote_port=remote_port,
    )


async def _attach_pod_action(manager, pod_name, namespace, attach_container):
    if not pod_name:
        return "Error: 'pod_name' is required for attach_pod"
    return await run_blocking(
        manager.attach_pod,
        pod_name=pod_name,
        namespace=namespace,
        container=attach_container,
    )


async def _copy_pod_action(manager, action, pod_name, ns, source, destination):
    if not pod_name or not source or not destination:
        return f"Error: 'pod_name', 'source', and 'destination' are required for {action}"
    return await run_blocking(getattr(manager, action), pod_name, ns, source, destination)


async def _dispatch_pod_action(
    action,
    manager,
    ns,
    attach_container,
    command,
    destination,
    exec_command,
    exec_container,
    label_selector,
    local_port,
    namespace,
    pod_name,
    remote_port,
    source,
):
    if action == "list_pods":
        return await run_blocking(
            manager.list_pods,
            namespace=namespace,
            label_selector=label_selector,
        )
    if action == "describe_pod":
        return await _describe_pod_action(manager, pod_name, namespace)
    if action == "exec_pod":
        return await _exec_pod_action(
            manager, pod_name, namespace, command, exec_command, exec_container
        )
    if action == "port_forward_pod":
        return await _port_forward_pod_action(manager, pod_name, namespace, local_port, remote_port)
    if action == "attach_pod":
        return await _attach_pod_action(manager, pod_name, namespace, attach_container)
    if action in ("copy_to_pod", "copy_from_pod"):
        return await _copy_pod_action(manager, action, pod_name, ns, source, destination)
    return _UNHANDLED


_ROLLOUT_ACTIONS = {
    "rollout_status": False,
    "rollout_history": False,
    "rollout_restart": False,
    "rollout_undo": True,
    "rollout_pause": False,
    "rollout_resume": False,
}


async def _dispatch_rollout_action(
    action, manager, ns, namespace, resource_name, resource_type, rollout_revision
):
    """Dispatch a rollout action via `_ROLLOUT_ACTIONS`; manager method name == action."""
    takes_revision = _ROLLOUT_ACTIONS.get(action)
    if takes_revision is None:
        return _UNHANDLED
    if not resource_type or not resource_name:
        return f"Error: 'resource_type' and 'resource_name' are required for {action}"
    kwargs = {
        "resource_type": resource_type,
        "name": resource_name,
        "namespace": namespace,
    }
    if takes_revision:
        kwargs["revision"] = rollout_revision
    return await run_blocking(getattr(manager, action), **kwargs)


_STRATEGY_SET_ACTIONS = frozenset(
    {
        "set_deployment_strategy",
        "set_daemonset_update_strategy",
        "set_statefulset_update_strategy",
    }
)
_STRATEGY_GET_ACTIONS = frozenset(
    {
        "get_deployment_strategy",
        "get_daemonset_update_strategy",
        "get_statefulset_update_strategy",
    }
)


async def _dispatch_strategy_action(action, manager, ns, name, spec):
    """Dispatch a strategy get/set action; manager method name == action."""
    if action in _STRATEGY_SET_ACTIONS:
        if not name or not spec:
            return f"Error: 'name' and 'spec' are required for {action}"
        return await run_blocking(getattr(manager, action), name, ns, spec)
    if action in _STRATEGY_GET_ACTIONS:
        if not name:
            return f"Error: 'name' is required for {action}"
        return await run_blocking(getattr(manager, action), name, ns)
    return _UNHANDLED


async def _dispatch_statefulset_action(
    action, manager, ns, name, namespace, replicas, spec
):
    if action == "list_statefulsets":
        return await run_blocking(manager.list_statefulsets, namespace=namespace)
    elif action == "create_stateful_set":
        if not name or not spec:
            return "Error: 'name' and 'spec' are required for create_stateful_set"
        return await run_blocking(manager.create_stateful_set, name, ns, spec)
    elif action == "scale_statefulset":
        if not name:
            return "Error: 'name' is required for scale_statefulset"
        return await run_blocking(
            manager.scale_statefulset,
            name=name,
            namespace=namespace,
            replicas=1 if replicas is None else replicas,
        )
    return _UNHANDLED


async def _dispatch_daemonset_action(action, manager, ns, name, namespace, spec):
    if action == "list_daemonsets":
        return await run_blocking(manager.list_daemonsets, namespace=namespace)
    elif action == "create_daemon_set":
        if not name or not spec:
            return "Error: 'name' and 'spec' are required for create_daemon_set"
        return await run_blocking(manager.create_daemon_set, name, ns, spec)
    return _UNHANDLED


async def _dispatch_replicaset_action(action, manager, ns, name, namespace, replicas):
    if action == "list_replicasets":
        return await run_blocking(manager.list_replica_sets, namespace=namespace)
    elif action == "describe_replicaset":
        if not name:
            return "Error: 'name' is required for describe_replicaset"
        return await run_blocking(manager.describe_replica_set, name, ns)
    elif action == "scale_replicaset":
        if not name or replicas is None:
            return "Error: 'name' and 'replicas' are required for scale_replicaset"
        return await run_blocking(manager.scale_replica_set, name, ns, replicas)
    return _UNHANDLED


async def _dispatch_job_action(action, manager, ns, name, namespace, spec):
    if action == "list_jobs":
        return await run_blocking(manager.list_jobs, namespace=namespace)
    elif action == "describe_job":
        if not name:
            return "Error: 'name' is required for describe_job"
        return await run_blocking(manager.describe_job, name, ns)
    elif action == "create_job":
        if not name or not spec:
            return "Error: 'name' and 'spec' are required for create_job"
        return await run_blocking(manager.create_job, name, ns, spec)
    elif action == "delete_job":
        if not name:
            return "Error: 'name' is required for delete_job"
        return await run_blocking(manager.delete_job, name, ns)
    return _UNHANDLED


async def _dispatch_cronjob_action(action, manager, ns, name, namespace, spec):
    if action == "list_cron_jobs":
        return await run_blocking(manager.list_cron_jobs, namespace=namespace)
    elif action == "describe_cron_job":
        if not name:
            return "Error: 'name' is required for describe_cron_job"
        return await run_blocking(manager.describe_cron_job, name, ns)
    elif action == "create_cron_job":
        if not name or not spec:
            return "Error: 'name' and 'spec' are required for create_cron_job"
        return await run_blocking(manager.create_cron_job, name, ns, spec)
    elif action == "delete_cron_job":
        if not name:
            return "Error: 'name' is required for delete_cron_job"
        return await run_blocking(manager.delete_cron_job, name, ns)
    return _UNHANDLED


_ACTION_GROUPS: dict[str, str] = {
    "list_pods": "pod",
    "describe_pod": "pod",
    "exec_pod": "pod",
    "port_forward_pod": "pod",
    "attach_pod": "pod",
    "copy_to_pod": "pod",
    "copy_from_pod": "pod",
    "rollout_status": "rollout",
    "rollout_history": "rollout",
    "rollout_restart": "rollout",
    "rollout_undo": "rollout",
    "rollout_pause": "rollout",
    "rollout_resume": "rollout",
    "set_deployment_strategy": "strategy",
    "get_deployment_strategy": "strategy",
    "set_daemonset_update_strategy": "strategy",
    "get_daemonset_update_strategy": "strategy",
    "set_statefulset_update_strategy": "strategy",
    "get_statefulset_update_strategy": "strategy",
    "list_statefulsets": "statefulset",
    "create_stateful_set": "statefulset",
    "scale_statefulset": "statefulset",
    "list_daemonsets": "daemonset",
    "create_daemon_set": "daemonset",
    "list_replicasets": "replicaset",
    "describe_replicaset": "replicaset",
    "scale_replicaset": "replicaset",
    "list_jobs": "job",
    "describe_job": "job",
    "create_job": "job",
    "delete_job": "job",
    "list_cron_jobs": "cronjob",
    "describe_cron_job": "cronjob",
    "create_cron_job": "cronjob",
    "delete_cron_job": "cronjob",
}


_GROUP_FUNCS = {
    "pod": _dispatch_pod_action,
    "rollout": _dispatch_rollout_action,
    "strategy": _dispatch_strategy_action,
    "statefulset": _dispatch_statefulset_action,
    "daemonset": _dispatch_daemonset_action,
    "replicaset": _dispatch_replicaset_action,
    "job": _dispatch_job_action,
    "cronjob": _dispatch_cronjob_action,
}


_GROUP_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "pod": (
        "attach_container",
        "command",
        "destination",
        "exec_command",
        "exec_container",
        "label_selector",
        "local_port",
        "namespace",
        "pod_name",
        "remote_port",
        "source",
    ),
    "rollout": (
        "namespace",
        "resource_name",
        "resource_type",
        "rollout_revision",
    ),
    "strategy": (
        "name",
        "spec",
    ),
    "statefulset": (
        "name",
        "namespace",
        "replicas",
        "spec",
    ),
    "daemonset": (
        "name",
        "namespace",
        "spec",
    ),
    "replicaset": (
        "name",
        "namespace",
        "replicas",
    ),
    "job": (
        "name",
        "namespace",
        "spec",
    ),
    "cronjob": (
        "name",
        "namespace",
        "spec",
    ),
}


def register_k8sworkloads_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Kubernetes Workload Operations",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"kubernetes", "workloads"},
    )
    async def cm_k8s_workloads(
        action: Literal[
            # Pods
            "list_pods",
            "describe_pod",
            "exec_pod",
            "port_forward_pod",
            "attach_pod",
            "copy_to_pod",
            "copy_from_pod",
            # Rollouts
            "rollout_status",
            "rollout_history",
            "rollout_restart",
            "rollout_undo",
            "rollout_pause",
            "rollout_resume",
            # Deployment / update strategies
            "set_deployment_strategy",
            "get_deployment_strategy",
            "set_daemonset_update_strategy",
            "get_daemonset_update_strategy",
            "set_statefulset_update_strategy",
            "get_statefulset_update_strategy",
            # StatefulSets
            "list_statefulsets",
            "create_stateful_set",
            "scale_statefulset",
            # DaemonSets
            "list_daemonsets",
            "create_daemon_set",
            # ReplicaSets
            "list_replicasets",
            "describe_replicaset",
            "scale_replicaset",
            # Jobs
            "list_jobs",
            "describe_job",
            "create_job",
            "delete_job",
            # CronJobs
            "list_cron_jobs",
            "describe_cron_job",
            "create_cron_job",
            "delete_cron_job",
        ] = Field(
            description="Workload action to perform (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs)."
        ),
        pod_name: str | None = Field(
            default=None, description="Pod name for pod operations"
        ),
        namespace: str | None = Field(
            default=None, description="Target namespace (default: from config)"
        ),
        label_selector: str | None = Field(
            default=None, description="Label selector for filtering pods"
        ),
        exec_command: str | None = Field(
            default=None,
            description="Command to execute in pod (space-separated string)",
        ),
        command: list | None = Field(
            default=None, description="Command to execute in pod (list form)"
        ),
        exec_container: str | None = Field(
            default=None, description="Container name for exec"
        ),
        local_port: int | None = Field(
            default=None, description="Local port for port-forward"
        ),
        remote_port: int | None = Field(
            default=None, description="Remote port for port-forward"
        ),
        attach_container: str | None = Field(
            default=None, description="Container name for attach"
        ),
        source: str | None = Field(
            default=None, description="Source path for copy operations"
        ),
        destination: str | None = Field(
            default=None, description="Destination path for copy operations"
        ),
        resource_type: str | None = Field(
            default=None, description="Resource type for rollout operations"
        ),
        resource_name: str | None = Field(
            default=None, description="Resource name for rollout operations"
        ),
        rollout_revision: int | None = Field(
            default=None, description="Revision number for rollout undo"
        ),
        name: str | None = Field(
            default=None,
            description="Resource name for workload objects (jobs, cronjobs, statefulsets, etc.)",
        ),
        spec: dict | None = Field(
            default=None, description="Resource specification for create/set operations"
        ),
        replicas: int | None = Field(
            default=None, description="Number of replicas for scaling operations"
        ),
        manager_type: str | None = Field(
            default=None,
            description="Container manager: kubernetes (default: auto-detect)",
        ),
        ctx: Context | None = None,
    ) -> dict | list | str:
        """Manage Kubernetes workloads (pods, rollouts, strategies, statefulsets, daemonsets, replicasets, jobs, cronjobs)."""
        manager = create_manager(manager_type or "kubernetes")
        if ctx:
            ctx_log(ctx, logging.INFO, f"Executing cm_k8s_workloads: {action}")

        try:
            ns = namespace or getattr(manager, "namespace", namespace)

            group = _ACTION_GROUPS.get(action)
            if group is None:
                return f"Error: Unknown action '{action}'"
            all_values = {
                "attach_container": attach_container,
                "command": command,
                "destination": destination,
                "exec_command": exec_command,
                "exec_container": exec_container,
                "label_selector": label_selector,
                "local_port": local_port,
                "name": name,
                "namespace": namespace,
                "pod_name": pod_name,
                "remote_port": remote_port,
                "replicas": replicas,
                "resource_name": resource_name,
                "resource_type": resource_type,
                "rollout_revision": rollout_revision,
                "source": source,
                "spec": spec,
            }
            group_kwargs = {n: all_values[n] for n in _GROUP_PARAM_NAMES[group]}
            return await _GROUP_FUNCS[group](action, manager, ns, **group_kwargs)
        except Exception as e:
            if ctx:
                ctx_log(
                    ctx, logging.ERROR, f"Error executing {action}: {type(e).__name__}"
                )
            return f"Error executing {action}: {type(e).__name__}"

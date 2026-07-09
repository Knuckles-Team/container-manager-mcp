"""MCP tools for Kubernetes workload operations.

Themed dispatcher covering pods, rollouts, deployment/update strategies,
StatefulSets, DaemonSets, ReplicaSets, Jobs, and CronJobs.
"""

import logging
from typing import Literal

from agent_utilities.mcp_utilities import ctx_log, run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from container_manager_mcp.container_manager import create_manager


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
        pod_name: str | None = Field(default=None, description="Pod name for pod operations"),
        namespace: str | None = Field(default=None, description="Target namespace (default: from config)"),
        label_selector: str | None = Field(default=None, description="Label selector for filtering pods"),
        exec_command: str | None = Field(default=None, description="Command to execute in pod (space-separated string)"),
        command: list | None = Field(default=None, description="Command to execute in pod (list form)"),
        exec_container: str | None = Field(default=None, description="Container name for exec"),
        local_port: int | None = Field(default=None, description="Local port for port-forward"),
        remote_port: int | None = Field(default=None, description="Remote port for port-forward"),
        attach_container: str | None = Field(default=None, description="Container name for attach"),
        source: str | None = Field(default=None, description="Source path for copy operations"),
        destination: str | None = Field(default=None, description="Destination path for copy operations"),
        resource_type: str | None = Field(default=None, description="Resource type for rollout operations"),
        resource_name: str | None = Field(default=None, description="Resource name for rollout operations"),
        rollout_revision: int | None = Field(default=None, description="Revision number for rollout undo"),
        name: str | None = Field(default=None, description="Resource name for workload objects (jobs, cronjobs, statefulsets, etc.)"),
        spec: dict | None = Field(default=None, description="Resource specification for create/set operations"),
        replicas: int | None = Field(default=None, description="Number of replicas for scaling operations"),
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

            # Pods
            if action == "list_pods":
                return await run_blocking(
                    manager.list_pods, namespace=namespace, label_selector=label_selector
                )
            elif action == "describe_pod":
                if not pod_name:
                    return "Error: 'pod_name' is required for describe_pod"
                return await run_blocking(
                    manager.describe_pod, pod_name=pod_name, namespace=namespace
                )
            elif action == "exec_pod":
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
            elif action == "port_forward_pod":
                if not pod_name or not local_port or not remote_port:
                    return "Error: 'pod_name', 'local_port', and 'remote_port' are required for port_forward_pod"
                return await run_blocking(
                    manager.port_forward_pod,
                    pod_name=pod_name,
                    namespace=namespace,
                    local_port=local_port,
                    remote_port=remote_port,
                )
            elif action == "attach_pod":
                if not pod_name:
                    return "Error: 'pod_name' is required for attach_pod"
                return await run_blocking(
                    manager.attach_pod, pod_name=pod_name, namespace=namespace, container=attach_container
                )
            elif action == "copy_to_pod":
                if not pod_name or not source or not destination:
                    return "Error: 'pod_name', 'source', and 'destination' are required for copy_to_pod"
                return await run_blocking(
                    manager.copy_to_pod, pod_name, ns, source, destination
                )
            elif action == "copy_from_pod":
                if not pod_name or not source or not destination:
                    return "Error: 'pod_name', 'source', and 'destination' are required for copy_from_pod"
                return await run_blocking(
                    manager.copy_from_pod, pod_name, ns, source, destination
                )

            # Rollouts
            elif action == "rollout_status":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_status"
                return await run_blocking(
                    manager.rollout_status, resource_type=resource_type, name=resource_name, namespace=namespace
                )
            elif action == "rollout_history":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_history"
                return await run_blocking(
                    manager.rollout_history, resource_type=resource_type, name=resource_name, namespace=namespace
                )
            elif action == "rollout_restart":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_restart"
                return await run_blocking(
                    manager.rollout_restart, resource_type=resource_type, name=resource_name, namespace=namespace
                )
            elif action == "rollout_undo":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_undo"
                return await run_blocking(
                    manager.rollout_undo, resource_type=resource_type, name=resource_name, namespace=namespace, revision=rollout_revision
                )
            elif action == "rollout_pause":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_pause"
                return await run_blocking(
                    manager.rollout_pause, resource_type=resource_type, name=resource_name, namespace=namespace
                )
            elif action == "rollout_resume":
                if not resource_type or not resource_name:
                    return "Error: 'resource_type' and 'resource_name' are required for rollout_resume"
                return await run_blocking(
                    manager.rollout_resume, resource_type=resource_type, name=resource_name, namespace=namespace
                )

            # Deployment / update strategies
            elif action == "set_deployment_strategy":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for set_deployment_strategy"
                return await run_blocking(manager.set_deployment_strategy, name, ns, spec)
            elif action == "get_deployment_strategy":
                if not name:
                    return "Error: 'name' is required for get_deployment_strategy"
                return await run_blocking(manager.get_deployment_strategy, name, ns)
            elif action == "set_daemonset_update_strategy":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for set_daemonset_update_strategy"
                return await run_blocking(manager.set_daemonset_update_strategy, name, ns, spec)
            elif action == "get_daemonset_update_strategy":
                if not name:
                    return "Error: 'name' is required for get_daemonset_update_strategy"
                return await run_blocking(manager.get_daemonset_update_strategy, name, ns)
            elif action == "set_statefulset_update_strategy":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for set_statefulset_update_strategy"
                return await run_blocking(manager.set_statefulset_update_strategy, name, ns, spec)
            elif action == "get_statefulset_update_strategy":
                if not name:
                    return "Error: 'name' is required for get_statefulset_update_strategy"
                return await run_blocking(manager.get_statefulset_update_strategy, name, ns)

            # StatefulSets
            elif action == "list_statefulsets":
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

            # DaemonSets
            elif action == "list_daemonsets":
                return await run_blocking(manager.list_daemonsets, namespace=namespace)
            elif action == "create_daemon_set":
                if not name or not spec:
                    return "Error: 'name' and 'spec' are required for create_daemon_set"
                return await run_blocking(manager.create_daemon_set, name, ns, spec)

            # ReplicaSets
            elif action == "list_replicasets":
                return await run_blocking(manager.list_replica_sets, namespace=namespace)
            elif action == "describe_replicaset":
                if not name:
                    return "Error: 'name' is required for describe_replicaset"
                return await run_blocking(manager.describe_replica_set, name, ns)
            elif action == "scale_replicaset":
                if not name or replicas is None:
                    return "Error: 'name' and 'replicas' are required for scale_replicaset"
                return await run_blocking(manager.scale_replica_set, name, ns, replicas)

            # Jobs
            elif action == "list_jobs":
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

            # CronJobs
            elif action == "list_cron_jobs":
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

            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            if ctx:
                ctx_log(ctx, logging.ERROR, f"Error executing {action}: {e}")
            return f"Error executing {action}: {e}"

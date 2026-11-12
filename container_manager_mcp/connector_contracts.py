"""Surface contracts: tool use, links, data locations and business roles."""

from scion_sdk.surface_contracts import (
    BusinessRole,
    DataLocation,
    Refusal,
    SurfaceLink,
    ToolStep,
    ToolUseContract,
)

CONTRACTS = (
    ToolUseContract(
        task="inspect-cm-diagnostics",
        description="Read cm backend diagnostics.",
        steps=(
            ToolStep(
                tool="cm_doctor_backend",
                params=("host", "context", "backend"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_doctor_check_backends",
                params=("host", "context", "backend"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_doctor_check_docker",
                params=("host", "context", "backend"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_doctor_check_inventory",
                params=("host", "context", "backend"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_doctor_check_kubernetes",
                params=("host", "context", "backend"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_doctor_check_podman",
                params=("host", "context", "backend"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_doctor_run",
                params=("host", "context", "backend"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-cm-inventory",
        description="Read cm the host inventory.",
        steps=(
            ToolStep(
                tool="cm_list_hosts",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-cm-inventory",
        description="Change cm the host inventory: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="cm_list_hosts",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_ingest_inventory",
                params=("modality", "host", "all_containers", "manager_type"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-docker-compose-and-stacks",
        description="Read docker compose projects and stacks.",
        steps=(
            ToolStep(
                tool="docker_compose_logs",
                params=("compose_file", "service", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_compose_ps",
                params=("compose_file", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-docker-compose-and-stacks",
        description="Change docker compose projects and stacks: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="docker_compose_logs",
                params=("compose_file", "service", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_compose_ps",
                params=("compose_file", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_compose_down",
                params=("compose_file", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_compose_up",
                params=("compose_file", "detach", "build", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_stack_deploy",
                params=("stack_name", "compose_file", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_stack_rm",
                params=("stack_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_stack_services",
                params=("stack_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-docker-containers",
        description="Read docker containers.",
        steps=(
            ToolStep(
                tool="docker_get_container_logs",
                params=("container_id", "tail", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_get_info",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_get_version",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_inspect_container",
                params=("container_id", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_list_containers",
                params=("all", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-docker-containers",
        description="Change docker containers: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="docker_get_container_logs",
                params=("container_id", "tail", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_get_info",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_exec_in_container",
                params=("container_id", "command", "detach", "binary", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_remove_container",
                params=("container_id", "force", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_run_container",
                params=(
                    "image",
                    "name",
                    "command",
                    "detach",
                    "ports",
                    "volumes",
                    "environment",
                    "labels",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_stop_container",
                params=("container_id", "timeout", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-docker-images-networks-volumes",
        description="Read docker images, networks and volumes.",
        steps=(
            ToolStep(
                tool="docker_list_images",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_list_networks",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_list_volumes",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-docker-images-networks-volumes",
        description="Change docker images, networks and volumes: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="docker_list_images",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_list_networks",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_create_network",
                params=("name", "driver", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_create_volume",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_prune_containers",
                params=("confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_prune_images",
                params=("force", "all", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_prune_networks",
                params=("confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_prune_system",
                params=("force", "all", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_prune_volumes",
                params=("force", "all", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_pull_image",
                params=("image", "tag", "platform", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_remove_image",
                params=("image", "force", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_remove_network",
                params=("network_id", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_remove_volume",
                params=("name", "force", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-docker-services-and-secrets",
        description="Read docker services, secrets and configs.",
        steps=(
            ToolStep(
                tool="docker_config_list",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_inspect_service",
                params=("service_id", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_list_services",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_secret_list",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_service_list",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_service_logs",
                params=("service_id", "tail", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_service_ps",
                params=("service_id", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-docker-services-and-secrets",
        description="Change docker services, secrets and configs: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="docker_config_list",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_inspect_service",
                params=("service_id", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_config_create",
                params=("config_name", "data", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_create_service",
                params=("name", "image", "replicas", "ports", "mounts", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_remove_service",
                params=("service_id", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_scale_service",
                params=("service_id", "replicas", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_secret_create",
                params=("secret_name", "data", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_service_create",
                params=("service_name", "image", "replicas", "ports", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_service_rm",
                params=("service_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_service_update",
                params=("service_name", "image", "replicas", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_update_service",
                params=(
                    "service_id",
                    "image",
                    "replicas",
                    "env",
                    "constraints",
                    "labels",
                    "force",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-docker-swarm-and-nodes",
        description="Read docker swarm membership and nodes.",
        steps=(
            ToolStep(
                tool="docker_inspect_node",
                params=("node_id", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_list_nodes",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_node_inspect",
                params=("node_id", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_node_ls",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_swarm_service_logs",
                params=("service_name", "tail_lines", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_swarm_service_ps",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-docker-swarm-and-nodes",
        description="Change docker swarm membership and nodes: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="docker_inspect_node",
                params=("node_id", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_list_nodes",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_swarm_operations_host",
                params=(
                    "service_id",
                    "node_id",
                    "labels",
                    "replace_labels",
                    "role",
                    "availability",
                    "env",
                    "constraints",
                    "tail",
                    "advertise_addr",
                    "image",
                    "name",
                    "ports",
                    "mounts",
                    "replicas",
                    "force",
                    "host",
                    "manager_type",
                    "confirm",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_init_swarm",
                params=("advertise_addr", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_leave_swarm",
                params=("force", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_node_update",
                params=("node_id", "availability", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_remove_node",
                params=("node_id", "force", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_swarm_init",
                params=("advertise_addr", "listen_addr", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_swarm_join",
                params=("remote_addr", "token", "worker", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="docker_swarm_leave",
                params=("force", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="docker_update_node",
                params=(
                    "node_id",
                    "labels",
                    "role",
                    "availability",
                    "replace_labels",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-k8s-cluster",
        description="Read k8s cluster and node state.",
        steps=(
            ToolStep(
                tool="cm_k8s_cluster_list_csr",
                params=(
                    "node_name",
                    "namespace",
                    "pod_name",
                    "taints",
                    "taint_key",
                    "affinity",
                    "anti_affinity",
                    "grace_period_seconds",
                    "context_name",
                    "new_context_name",
                    "csr_name",
                    "reason",
                    "name",
                    "output_dir",
                    "plugin_type",
                    "test_resource",
                    "manager_type",
                    "server",
                    "token",
                    "client_cert",
                    "client_key",
                    "ca_cert",
                    "insecure_skip_tls_verify",
                    "username",
                    "password",
                    "oidc_issuer",
                    "oidc_client_id",
                    "oidc_client_secret",
                    "source_file",
                    "source_yaml",
                    "capture_current",
                    "kubeconfig_path",
                    "overwrite",
                    "use",
                    "validate",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_cluster_info_dump",
                params=("output_dir", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_cluster_topology",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_cluster_plugin",
                params=("name", "plugin_type", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_cluster_info",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_cluster_resource_summary",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_node_affinity",
                params=("pod_name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_node_conditions",
                params=("node_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_cluster_plugins",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_node_conditions",
                params=("node_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_node_taints",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_nodes",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_test_cluster_plugin",
                params=("name", "plugin_type", "test_resource", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_watch_resource",
                params=("resource_type", "name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-k8s-cluster",
        description="Change k8s cluster and node state: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="cm_k8s_cluster_list_csr",
                params=(
                    "node_name",
                    "namespace",
                    "pod_name",
                    "taints",
                    "taint_key",
                    "affinity",
                    "anti_affinity",
                    "grace_period_seconds",
                    "context_name",
                    "new_context_name",
                    "csr_name",
                    "reason",
                    "name",
                    "output_dir",
                    "plugin_type",
                    "test_resource",
                    "manager_type",
                    "server",
                    "token",
                    "client_cert",
                    "client_key",
                    "ca_cert",
                    "insecure_skip_tls_verify",
                    "username",
                    "password",
                    "oidc_issuer",
                    "oidc_client_id",
                    "oidc_client_secret",
                    "source_file",
                    "source_yaml",
                    "capture_current",
                    "kubeconfig_path",
                    "overwrite",
                    "use",
                    "validate",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_cluster_info_dump",
                params=("output_dir", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_k8s_cluster_save_context",
                params=(
                    "namespace",
                    "context_name",
                    "server",
                    "token",
                    "client_cert",
                    "client_key",
                    "ca_cert",
                    "insecure_skip_tls_verify",
                    "username",
                    "password",
                    "oidc_issuer",
                    "oidc_client_id",
                    "oidc_client_secret",
                    "source_file",
                    "source_yaml",
                    "capture_current",
                    "kubeconfig_path",
                    "overwrite",
                    "use",
                    "validate",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_cordon_node",
                params=("node_name", "unschedulable", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_drain_node",
                params=(
                    "node_name",
                    "force",
                    "grace_period_seconds",
                    "confirm",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_set_node_affinity",
                params=("pod_name", "namespace", "affinity", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_taint_node",
                params=("node_name", "taints", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_uncordon_node",
                params=("node_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_untaint_node",
                params=("node_name", "taint_key", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_update_node",
                params=(
                    "node_id",
                    "labels",
                    "role",
                    "availability",
                    "replace_labels",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-k8s-config",
        description="Read k8s configuration: namespaces, config maps, secrets, manifests and contexts.",
        steps=(
            ToolStep(
                tool="k8s_describe_api_resource",
                params=("resource_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_crd",
                params=("crd_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_diff_manifest",
                params=("manifest_yaml", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_export_manifest",
                params=("api_version", "kind", "name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_api_server_info",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_config",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_resource",
                params=(
                    "api_version",
                    "kind",
                    "name",
                    "namespace",
                    "reveal",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_secret",
                params=("name", "namespace", "reveal", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_secret_state_hash",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_api_resources",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_configmaps",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_contexts",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_crds",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_custom_resources",
                params=("group", "version", "plural", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_external_secrets",
                params=("namespace", "all_namespaces", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_field_selector",
                params=("resource_type", "field_selector", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_namespaces",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_resources",
                params=(
                    "api_version",
                    "kind",
                    "namespace",
                    "label_selector",
                    "all_namespaces",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_secrets",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_track_resource_version",
                params=("resource_type", "name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_validate_kubeconfig",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_wait_for_resource_version",
                params=(
                    "resource_type",
                    "name",
                    "namespace",
                    "target_version",
                    "timeout",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-k8s-config",
        description="Change k8s configuration: namespaces, config maps, secrets, manifests and contexts: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="k8s_describe_api_resource",
                params=("resource_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_crd",
                params=("crd_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_annotate_resource",
                params=("resource_type", "name", "namespace", "annotations", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_api_resources",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_apply_manifest",
                params=(
                    "manifest_yaml",
                    "namespace",
                    "dry_run",
                    "force_conflicts",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_compare_configmap_state",
                params=("name", "namespace", "expected_data", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_configmap",
                params=("name", "namespace", "data", "from_file", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_namespace",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_secret",
                params=("name", "namespace", "secret_type", "data", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_namespace",
                params=("name", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_resource",
                params=(
                    "api_version",
                    "kind",
                    "name",
                    "namespace",
                    "propagation_policy",
                    "confirm",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_force_sync_external_secret",
                params=(
                    "name",
                    "namespace",
                    "recreate_target",
                    "timeout_seconds",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_label_resource",
                params=("resource_type", "name", "namespace", "labels", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_patch_resource",
                params=(
                    "resource_type",
                    "name",
                    "namespace",
                    "patch_body",
                    "patch_type",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_rename_context",
                params=("current_name", "new_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_sync_configmap_from_file",
                params=("name", "namespace", "file_path", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_use_context",
                params=("context_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-k8s-container-runtime-compat",
        description="Read k8s container-runtime compatible operations on the cluster.",
        steps=(
            ToolStep(
                tool="k8s_compose_logs",
                params=("compose_file", "service", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_compose_ps",
                params=("compose_file", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_container_logs",
                params=("container_id", "tail", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_inspect_container",
                params=("container_id", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_inspect_node",
                params=("node_id", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_inspect_service",
                params=("service_id", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_containers",
                params=("all", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_images",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_networks",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_volumes",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_service_logs",
                params=("service_id", "tail", "follow", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_service_ps",
                params=("service_id", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-k8s-container-runtime-compat",
        description="Change k8s container-runtime compatible operations on the cluster: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="k8s_compose_logs",
                params=("compose_file", "service", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_compose_ps",
                params=("compose_file", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_compose_down",
                params=("compose_file", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_compose_up",
                params=("compose_file", "detach", "build", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_network",
                params=("name", "driver", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_network_policy_with_cidr",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_networkpolicy",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_volume",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_volume_snapshot",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_exec_in_container",
                params=("container_id", "command", "detach", "binary", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_prune_containers",
                params=("confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_prune_images",
                params=("force", "all", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_prune_networks",
                params=("confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_prune_system",
                params=("force", "all", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_prune_volumes",
                params=("force", "all", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_pull_image",
                params=("image", "tag", "platform", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_remove_container",
                params=("container_id", "force", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_remove_image",
                params=("image", "force", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_remove_network",
                params=("network_id", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_remove_node",
                params=("node_id", "force", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_remove_service",
                params=("service_id", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_remove_volume",
                params=("name", "force", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_run_container",
                params=(
                    "image",
                    "name",
                    "command",
                    "detach",
                    "ports",
                    "volumes",
                    "environment",
                    "labels",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_scale_service",
                params=("service_id", "replicas", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_stop_container",
                params=("container_id", "timeout", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-k8s-networking",
        description="Read k8s networking: services, ingress, DNS and network policy.",
        steps=(
            ToolStep(
                tool="cm_k8s_networking_get_k8s_service",
                params=(
                    "namespace",
                    "ingress_name",
                    "ingress_spec",
                    "netpol_name",
                    "netpol_spec",
                    "name",
                    "spec",
                    "rules",
                    "pod_name",
                    "hostname",
                    "service_name",
                    "target",
                    "service_spec",
                    "service_ports",
                    "service_selector",
                    "service_type",
                    "manager_type",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_k8s_networking_list_k8s_services",
                params=(
                    "namespace",
                    "ingress_name",
                    "ingress_spec",
                    "netpol_name",
                    "netpol_spec",
                    "name",
                    "spec",
                    "rules",
                    "pod_name",
                    "hostname",
                    "service_name",
                    "target",
                    "service_spec",
                    "service_ports",
                    "service_selector",
                    "service_type",
                    "manager_type",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_check_dns_resolution",
                params=("namespace", "pod_name", "hostname", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_ingress_class",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_native_service",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_dns_endpoints",
                params=("namespace", "service_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_endpoints",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_endpointslices",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_ingress",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_ingress_classes",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_ingresses",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_native_services",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_networkpolicies",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_services",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_test_dns_connectivity",
                params=("namespace", "target", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_test_network_policy_connectivity",
                params=("namespace", "policy_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-k8s-networking",
        description="Change k8s networking: services, ingress, DNS and network policy: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="cm_k8s_networking_get_k8s_service",
                params=(
                    "namespace",
                    "ingress_name",
                    "ingress_spec",
                    "netpol_name",
                    "netpol_spec",
                    "name",
                    "spec",
                    "rules",
                    "pod_name",
                    "hostname",
                    "service_name",
                    "target",
                    "service_spec",
                    "service_ports",
                    "service_selector",
                    "service_type",
                    "manager_type",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_k8s_networking_list_k8s_services",
                params=(
                    "namespace",
                    "ingress_name",
                    "ingress_spec",
                    "netpol_name",
                    "netpol_spec",
                    "name",
                    "spec",
                    "rules",
                    "pod_name",
                    "hostname",
                    "service_name",
                    "target",
                    "service_spec",
                    "service_ports",
                    "service_selector",
                    "service_type",
                    "manager_type",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_k8s_networking_create_k8s_service",
                params=(
                    "namespace",
                    "ingress_name",
                    "ingress_spec",
                    "netpol_name",
                    "netpol_spec",
                    "name",
                    "spec",
                    "rules",
                    "pod_name",
                    "hostname",
                    "service_name",
                    "target",
                    "service_spec",
                    "service_ports",
                    "service_selector",
                    "service_type",
                    "manager_type",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_k8s_networking_delete_k8s_service",
                params=(
                    "namespace",
                    "ingress_name",
                    "ingress_spec",
                    "netpol_name",
                    "netpol_spec",
                    "name",
                    "spec",
                    "rules",
                    "pod_name",
                    "hostname",
                    "service_name",
                    "target",
                    "service_spec",
                    "service_ports",
                    "service_selector",
                    "service_type",
                    "manager_type",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_ingress",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_ingress_class",
                params=("name", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_native_service",
                params=(
                    "name",
                    "namespace",
                    "spec",
                    "ports",
                    "selector",
                    "type",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_service",
                params=("name", "image", "replicas", "ports", "mounts", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_ingress",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_native_service",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_networkpolicy",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_port_forward_pod",
                params=(
                    "pod_name",
                    "namespace",
                    "local_port",
                    "remote_port",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_set_default_ingress_class",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_update_network_policy_rules",
                params=("name", "namespace", "rules", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_update_service",
                params=(
                    "service_id",
                    "image",
                    "replicas",
                    "env",
                    "constraints",
                    "labels",
                    "force",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="trace_port_namespace",
                params=("port", "host", "manager_type"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-k8s-observability",
        description="Read k8s logs, metrics, events and debugging.",
        steps=(
            ToolStep(
                tool="k8s_get_node_metrics",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_resource_events",
                params=("resource_type", "name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_top_nodes",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_events",
                params=(
                    "namespace",
                    "field_selector",
                    "all_namespaces",
                    "limit",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_top_nodes",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-k8s-observability",
        description="Change k8s logs, metrics, events and debugging: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="k8s_get_node_metrics",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_resource_events",
                params=("resource_type", "name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_debug_node",
                params=("node_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_debug_service",
                params=("service_name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-k8s-rbac",
        description="Read k8s access control: roles, bindings, service accounts, tokens and certificate requests.",
        steps=(
            ToolStep(
                tool="k8s_auth_can_i",
                params=("verb", "resource", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_pod_security_policy",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_evaluate_pod_security",
                params=("namespace", "pod_spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_certificate_signing_requests",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_cluster_rolebindings",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_cluster_roles",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_pod_security_policies",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_rolebindings",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_roles",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_service_account_mapped_secrets",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_service_account_tokens",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_serviceaccounts",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_local_subject_access_review",
                params=("namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_subject_access_review",
                params=("spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-k8s-rbac",
        description="Change k8s access control: roles, bindings, service accounts, tokens and certificate requests: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="k8s_auth_can_i",
                params=("verb", "resource", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_pod_security_policy",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_approve_csr",
                params=("csr_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_aggregated_cluster_role",
                params=("name", "aggregation_rule", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_cluster_rolebinding",
                params=("name", "role_ref", "subjects", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_pod_security_policy",
                params=("name", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_role",
                params=("name", "namespace", "rules", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_rolebinding",
                params=("name", "namespace", "role_ref", "subjects", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_service_account_token",
                params=("name", "namespace", "token_spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_serviceaccount",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_cluster_rolebinding",
                params=("name", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_pod_security_policy",
                params=("name", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_role",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_rolebinding",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_service_account_token",
                params=("name", "namespace", "token_name", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_serviceaccount",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_deny_csr",
                params=("csr_name", "reason", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_map_secret_to_service_account",
                params=("secret_name", "sa_name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_unmap_secret_from_service_account",
                params=("secret_name", "sa_name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_update_aggregated_cluster_role",
                params=("name", "aggregation_rule", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-k8s-storage",
        description="Read k8s storage: volumes, claims, classes and snapshots.",
        steps=(
            ToolStep(
                tool="k8s_describe_csi_driver",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_csi_driver_capacity",
                params=("driver_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_storage_class_provisioner",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_csi_drivers",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_persistent_volume_claims",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_persistent_volumes",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_storage_classes",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_volume_snapshots",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-k8s-storage",
        description="Change k8s storage: volumes, claims, classes and snapshots: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="k8s_describe_csi_driver",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_csi_driver_capacity",
                params=("driver_name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_persistent_volume",
                params=("name", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_persistent_volume_claim",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_storage_class",
                params=("name", "provisioner", "parameters", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_persistent_volume_claim",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_expand_persistent_volume",
                params=("name", "namespace", "size", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_expand_pvc",
                params=("name", "namespace", "size", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_set_default_storage_class",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-k8s-workloads",
        description="Read k8s workloads: pods, controllers, jobs, rollouts and autoscaling.",
        steps=(
            ToolStep(
                tool="cm_k8s_workloads_describe_replicaset",
                params=(
                    "pod_name",
                    "namespace",
                    "label_selector",
                    "exec_command",
                    "command",
                    "exec_container",
                    "local_port",
                    "remote_port",
                    "attach_container",
                    "source",
                    "destination",
                    "resource_type",
                    "resource_name",
                    "rollout_revision",
                    "name",
                    "spec",
                    "replicas",
                    "manager_type",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_multi_context_list_deployments",
                params=(
                    "backend",
                    "context",
                    "name",
                    "namespace",
                    "image",
                    "command",
                    "all",
                    "force",
                    "spec",
                    "replicas",
                    "driver",
                    "ports",
                    "volumes",
                    "environment",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_cron_job",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_deployment",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_horizontal_pod_autoscaler",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_job",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_limit_range",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_pod",
                params=("pod_name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_pod_disruption_budget",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_priority_class",
                params=("name", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_replica_set",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_describe_resource_quota",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_autoscaler_history",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_autoscaler_metrics",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_daemonset_update_strategy",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_deployment_strategy",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_pod_metrics",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_pod_resource_usage",
                params=("pod_name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_statefulset_update_strategy",
                params=("name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_get_top_pods",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_cron_jobs",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_daemon_sets",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_daemonsets",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_horizontal_pod_autoscalers",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_jobs",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_limit_ranges",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_pod_disruption_budgets",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_pods",
                params=("namespace", "label_selector", "all_namespaces", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_priority_classes",
                params=("context",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_replica_sets",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_replicasets",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_resource_quotas",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_stateful_sets",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_list_statefulsets",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_read_pod_logs",
                params=(
                    "pod_name",
                    "namespace",
                    "container",
                    "tail_lines",
                    "previous",
                    "since_seconds",
                    "context",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_rollout_history",
                params=("resource_type", "name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_rollout_status",
                params=("resource_type", "name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_stream_pod_logs",
                params=("pod_name", "namespace", "tail_lines", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_top_pods",
                params=("namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-k8s-workloads",
        description="Change k8s workloads: pods, controllers, jobs, rollouts and autoscaling: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="cm_k8s_workloads_describe_replicaset",
                params=(
                    "pod_name",
                    "namespace",
                    "label_selector",
                    "exec_command",
                    "command",
                    "exec_container",
                    "local_port",
                    "remote_port",
                    "attach_container",
                    "source",
                    "destination",
                    "resource_type",
                    "resource_name",
                    "rollout_revision",
                    "name",
                    "spec",
                    "replicas",
                    "manager_type",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_multi_context_list_deployments",
                params=(
                    "backend",
                    "context",
                    "name",
                    "namespace",
                    "image",
                    "command",
                    "all",
                    "force",
                    "spec",
                    "replicas",
                    "driver",
                    "ports",
                    "volumes",
                    "environment",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="cm_k8s_workloads_scale_replicaset",
                params=(
                    "pod_name",
                    "namespace",
                    "label_selector",
                    "exec_command",
                    "command",
                    "exec_container",
                    "local_port",
                    "remote_port",
                    "attach_container",
                    "source",
                    "destination",
                    "resource_type",
                    "resource_name",
                    "rollout_revision",
                    "name",
                    "spec",
                    "replicas",
                    "manager_type",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_attach_pod",
                params=("pod_name", "namespace", "container", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_copy_from_pod",
                params=("pod_name", "namespace", "source", "destination", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_copy_to_pod",
                params=("pod_name", "namespace", "source", "destination", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_cp_pod",
                params=("pod_name", "namespace", "source", "destination", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_cron_job",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_daemon_set",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_horizontal_pod_autoscaler",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_job",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_limit_range",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_pod_disruption_budget",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_priority_class",
                params=("name", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_resource_quota",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_create_stateful_set",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_debug_deployment",
                params=("deployment_name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_debug_pod",
                params=("pod_name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_cron_job",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_horizontal_pod_autoscaler",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_job",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_limit_range",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_pod_disruption_budget",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_priority_class",
                params=("name", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_delete_resource_quota",
                params=("name", "namespace", "confirm", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_exec_pod",
                params=("pod_name", "namespace", "command", "container", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_rollout_pause",
                params=("resource_type", "name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_rollout_restart",
                params=("resource_type", "name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_rollout_resume",
                params=("resource_type", "name", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_rollout_undo",
                params=("resource_type", "name", "namespace", "revision", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_scale_deployment",
                params=("name", "replicas", "namespace", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_scale_deployment_autoscaler",
                params=("name", "namespace", "min_replicas", "max_replicas", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_scale_replica_set",
                params=("name", "namespace", "replicas", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_scale_statefulset",
                params=("name", "namespace", "replicas", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_set_autoscaler_metrics",
                params=("name", "namespace", "metrics", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_set_daemonset_update_strategy",
                params=("name", "namespace", "strategy", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_set_deployment_strategy",
                params=("name", "namespace", "strategy", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_set_pod_anti_affinity",
                params=("pod_name", "namespace", "anti_affinity", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_set_statefulset_update_strategy",
                params=("name", "namespace", "strategy", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_update_horizontal_pod_autoscaler",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="k8s_update_resource_quota",
                params=("name", "namespace", "spec", "context"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-podman-compose-and-stacks",
        description="Read podman compose projects and stacks.",
        steps=(
            ToolStep(
                tool="podman_compose_logs",
                params=("compose_file", "service"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_compose_ps",
                params=("compose_file",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-podman-compose-and-stacks",
        description="Change podman compose projects and stacks: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="podman_compose_logs",
                params=("compose_file", "service"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_compose_ps",
                params=("compose_file",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_compose_down",
                params=("compose_file",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_compose_up",
                params=("compose_file", "detach", "build"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-podman-containers",
        description="Read podman containers.",
        steps=(
            ToolStep(
                tool="podman_get_container_logs",
                params=("container_id", "tail"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_get_info",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_get_version",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_health_check",
                params=("container_id", "config"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_inspect_container",
                params=("container_id",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_list_containers",
                params=("all",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-podman-containers",
        description="Change podman containers: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="podman_get_container_logs",
                params=("container_id", "tail"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_get_info",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_exec_in_container",
                params=("container_id", "command", "detach", "binary"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_remove_container",
                params=("container_id", "force", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_run_container",
                params=(
                    "image",
                    "name",
                    "command",
                    "detach",
                    "ports",
                    "volumes",
                    "environment",
                    "labels",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_stop_container",
                params=("container_id", "timeout"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-podman-images-networks-volumes",
        description="Read podman images, networks and volumes.",
        steps=(
            ToolStep(
                tool="podman_list_images",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_list_networks",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_list_volumes",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_network_inspect",
                params=("network_name",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_network_list",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_volume_inspect",
                params=("volume_name",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_volume_list",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-podman-images-networks-volumes",
        description="Change podman images, networks and volumes: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="podman_list_images",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_list_networks",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_create_network",
                params=("name", "driver"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_create_volume",
                params=("name",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_network_create",
                params=("network_name", "driver", "subnet"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_prune_containers",
                params=("confirm",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_prune_images",
                params=("force", "all", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_prune_networks",
                params=("confirm",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_prune_system",
                params=("force", "all", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_prune_volumes",
                params=("force", "all", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_pull_image",
                params=("image", "tag", "platform"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_remove_image",
                params=("image", "force", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_remove_network",
                params=("network_id", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_remove_volume",
                params=("name", "force", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_system_prune",
                params=("confirm",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_volume_create",
                params=("volume_name", "driver"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-podman-pods",
        description="Read podman pods, kube YAML and checkpoints.",
        steps=(
            ToolStep(
                tool="podman_pod_inspect",
                params=("pod_name",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_pod_list",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_pod_logs",
                params=("pod_name", "tail_lines"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_pod_stats",
                params=("pod_name",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_pod_top",
                params=("pod_name",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-podman-pods",
        description="Change podman pods, kube YAML and checkpoints: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="podman_pod_inspect",
                params=("pod_name",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_pod_list",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_checkpoint",
                params=("container_id", "checkpoint_dir"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_generate_kube_yaml",
                params=("pod_name", "namespace"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_play_kube_yaml",
                params=("yaml_path",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_pod_create",
                params=("pod_name", "image", "command"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_pod_rm",
                params=("pod_name",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_pod_stop",
                params=("pod_name",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_restore",
                params=("container_id", "checkpoint_dir"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-podman-services-and-secrets",
        description="Read podman services, secrets and configs.",
        steps=(
            ToolStep(
                tool="podman_inspect_service",
                params=("service_id",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_list_services",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_service_logs",
                params=("service_id", "tail"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_service_ps",
                params=("service_id",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-podman-services-and-secrets",
        description="Change podman services, secrets and configs: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="podman_inspect_service",
                params=("service_id",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_list_services",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_create_service",
                params=("name", "image", "replicas", "ports", "mounts"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_remove_service",
                params=("service_id", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_scale_service",
                params=("service_id", "replicas"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_update_service",
                params=(
                    "service_id",
                    "image",
                    "replicas",
                    "env",
                    "constraints",
                    "labels",
                    "force",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-podman-swarm-and-nodes",
        description="Read podman swarm membership and nodes.",
        steps=(
            ToolStep(
                tool="podman_inspect_node",
                params=("node_id",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_list_nodes",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="change-podman-swarm-and-nodes",
        description="Change podman swarm membership and nodes: read the current state, apply the change, read it back.",
        steps=(
            ToolStep(
                tool="podman_inspect_node",
                params=("node_id",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_list_nodes",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="podman_init_swarm",
                params=("advertise_addr", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_leave_swarm",
                params=("force", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_remove_node",
                params=("node_id", "force", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="podman_update_node",
                params=("node_id", "labels", "role", "availability", "replace_labels"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
)

LINKS = (
    SurfaceLink(
        task="inspect-cm-diagnostics",
        tools=(
            "cm_doctor_backend",
            "cm_doctor_check_backends",
            "cm_doctor_check_docker",
            "cm_doctor_check_inventory",
            "cm_doctor_check_kubernetes",
            "cm_doctor_check_podman",
            "cm_doctor_run",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=(
            "container-manager-config-walkthrough",
            "container-manager-multi-context",
            "container-manager-kg-ingestion",
        ),
    ),
    SurfaceLink(
        task="inspect-cm-inventory",
        tools=("cm_list_hosts",),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=(
            "container-manager-config-walkthrough",
            "container-manager-multi-context",
            "container-manager-kg-ingestion",
        ),
    ),
    SurfaceLink(
        task="change-cm-inventory",
        tools=("cm_list_hosts", "cm_ingest_inventory"),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=(
            "container-manager-config-walkthrough",
            "container-manager-multi-context",
            "container-manager-kg-ingestion",
        ),
    ),
    SurfaceLink(
        task="inspect-docker-compose-and-stacks",
        tools=("docker_compose_logs", "docker_compose_ps"),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=(
            "container-manager-lifecycle",
            "container-manager-swarm",
            "container-manager-mcp-operations",
            "swarm-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-docker-compose-and-stacks",
        tools=(
            "docker_compose_logs",
            "docker_compose_ps",
            "docker_compose_down",
            "docker_compose_up",
            "docker_stack_deploy",
            "docker_stack_rm",
            "docker_stack_services",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=(
            "container-manager-lifecycle",
            "container-manager-swarm",
            "container-manager-mcp-operations",
            "swarm-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="inspect-docker-containers",
        tools=(
            "docker_get_container_logs",
            "docker_get_info",
            "docker_get_version",
            "docker_inspect_container",
            "docker_list_containers",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=(
            "container-manager-lifecycle",
            "container-manager-swarm",
            "container-manager-mcp-operations",
            "swarm-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-docker-containers",
        tools=(
            "docker_get_container_logs",
            "docker_get_info",
            "docker_exec_in_container",
            "docker_remove_container",
            "docker_run_container",
            "docker_stop_container",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=(
            "container-manager-lifecycle",
            "container-manager-swarm",
            "container-manager-mcp-operations",
            "swarm-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="inspect-docker-images-networks-volumes",
        tools=("docker_list_images", "docker_list_networks", "docker_list_volumes"),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=(
            "container-manager-lifecycle",
            "container-manager-swarm",
            "container-manager-mcp-operations",
            "swarm-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-docker-images-networks-volumes",
        tools=(
            "docker_list_images",
            "docker_list_networks",
            "docker_create_network",
            "docker_create_volume",
            "docker_prune_containers",
            "docker_prune_images",
            "docker_prune_networks",
            "docker_prune_system",
            "docker_prune_volumes",
            "docker_pull_image",
            "docker_remove_image",
            "docker_remove_network",
            "docker_remove_volume",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=(
            "container-manager-lifecycle",
            "container-manager-swarm",
            "container-manager-mcp-operations",
            "swarm-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="inspect-docker-services-and-secrets",
        tools=(
            "docker_config_list",
            "docker_inspect_service",
            "docker_list_services",
            "docker_secret_list",
            "docker_service_list",
            "docker_service_logs",
            "docker_service_ps",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=(
            "container-manager-lifecycle",
            "container-manager-swarm",
            "container-manager-mcp-operations",
            "swarm-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-docker-services-and-secrets",
        tools=(
            "docker_config_list",
            "docker_inspect_service",
            "docker_config_create",
            "docker_create_service",
            "docker_remove_service",
            "docker_scale_service",
            "docker_secret_create",
            "docker_service_create",
            "docker_service_rm",
            "docker_service_update",
            "docker_update_service",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=(
            "container-manager-lifecycle",
            "container-manager-swarm",
            "container-manager-mcp-operations",
            "swarm-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="inspect-docker-swarm-and-nodes",
        tools=(
            "docker_inspect_node",
            "docker_list_nodes",
            "docker_node_inspect",
            "docker_node_ls",
            "docker_swarm_service_logs",
            "docker_swarm_service_ps",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=(
            "container-manager-lifecycle",
            "container-manager-swarm",
            "container-manager-mcp-operations",
            "swarm-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-docker-swarm-and-nodes",
        tools=(
            "docker_inspect_node",
            "docker_list_nodes",
            "cm_swarm_operations_host",
            "docker_init_swarm",
            "docker_leave_swarm",
            "docker_node_update",
            "docker_remove_node",
            "docker_swarm_init",
            "docker_swarm_join",
            "docker_swarm_leave",
            "docker_update_node",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=(
            "container-manager-lifecycle",
            "container-manager-swarm",
            "container-manager-mcp-operations",
            "swarm-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="inspect-k8s-cluster",
        tools=(
            "cm_k8s_cluster_list_csr",
            "k8s_cluster_info_dump",
            "k8s_cluster_topology",
            "k8s_describe_cluster_plugin",
            "k8s_get_cluster_info",
            "k8s_get_cluster_resource_summary",
            "k8s_get_node_affinity",
            "k8s_get_node_conditions",
            "k8s_list_cluster_plugins",
            "k8s_list_node_conditions",
            "k8s_list_node_taints",
            "k8s_list_nodes",
            "k8s_test_cluster_plugin",
            "k8s_watch_resource",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "container-manager-mcp-platform-observer",
            "kubernetes-operator",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-k8s-cluster",
        tools=(
            "cm_k8s_cluster_list_csr",
            "k8s_cluster_info_dump",
            "cm_k8s_cluster_save_context",
            "k8s_cordon_node",
            "k8s_drain_node",
            "k8s_set_node_affinity",
            "k8s_taint_node",
            "k8s_uncordon_node",
            "k8s_untaint_node",
            "k8s_update_node",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "kubernetes-operator",
            "platform-reliability-engineer",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
            "container-manager-externalsecret-sync",
            "container-manager-gitops-persist",
        ),
    ),
    SurfaceLink(
        task="inspect-k8s-config",
        tools=(
            "k8s_describe_api_resource",
            "k8s_describe_crd",
            "k8s_diff_manifest",
            "k8s_export_manifest",
            "k8s_get_api_server_info",
            "k8s_get_config",
            "k8s_get_resource",
            "k8s_get_secret",
            "k8s_get_secret_state_hash",
            "k8s_list_api_resources",
            "k8s_list_configmaps",
            "k8s_list_contexts",
            "k8s_list_crds",
            "k8s_list_custom_resources",
            "k8s_list_external_secrets",
            "k8s_list_field_selector",
            "k8s_list_namespaces",
            "k8s_list_resources",
            "k8s_list_secrets",
            "k8s_track_resource_version",
            "k8s_validate_kubeconfig",
            "k8s_wait_for_resource_version",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "container-manager-mcp-platform-observer",
            "kubernetes-operator",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-k8s-config",
        tools=(
            "k8s_describe_api_resource",
            "k8s_describe_crd",
            "k8s_annotate_resource",
            "k8s_api_resources",
            "k8s_apply_manifest",
            "k8s_compare_configmap_state",
            "k8s_create_configmap",
            "k8s_create_namespace",
            "k8s_create_secret",
            "k8s_delete_namespace",
            "k8s_delete_resource",
            "k8s_force_sync_external_secret",
            "k8s_label_resource",
            "k8s_patch_resource",
            "k8s_rename_context",
            "k8s_sync_configmap_from_file",
            "k8s_use_context",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "kubernetes-operator",
            "platform-reliability-engineer",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
            "container-manager-externalsecret-sync",
            "container-manager-gitops-persist",
        ),
    ),
    SurfaceLink(
        task="inspect-k8s-container-runtime-compat",
        tools=(
            "k8s_compose_logs",
            "k8s_compose_ps",
            "k8s_get_container_logs",
            "k8s_inspect_container",
            "k8s_inspect_node",
            "k8s_inspect_service",
            "k8s_list_containers",
            "k8s_list_images",
            "k8s_list_networks",
            "k8s_list_volumes",
            "k8s_service_logs",
            "k8s_service_ps",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "container-manager-mcp-platform-observer",
            "kubernetes-operator",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-k8s-container-runtime-compat",
        tools=(
            "k8s_compose_logs",
            "k8s_compose_ps",
            "k8s_compose_down",
            "k8s_compose_up",
            "k8s_create_network",
            "k8s_create_network_policy_with_cidr",
            "k8s_create_networkpolicy",
            "k8s_create_volume",
            "k8s_create_volume_snapshot",
            "k8s_exec_in_container",
            "k8s_prune_containers",
            "k8s_prune_images",
            "k8s_prune_networks",
            "k8s_prune_system",
            "k8s_prune_volumes",
            "k8s_pull_image",
            "k8s_remove_container",
            "k8s_remove_image",
            "k8s_remove_network",
            "k8s_remove_node",
            "k8s_remove_service",
            "k8s_remove_volume",
            "k8s_run_container",
            "k8s_scale_service",
            "k8s_stop_container",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "kubernetes-operator",
            "platform-reliability-engineer",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
            "container-manager-externalsecret-sync",
            "container-manager-gitops-persist",
        ),
    ),
    SurfaceLink(
        task="inspect-k8s-networking",
        tools=(
            "cm_k8s_networking_get_k8s_service",
            "cm_k8s_networking_list_k8s_services",
            "k8s_check_dns_resolution",
            "k8s_describe_ingress_class",
            "k8s_get_native_service",
            "k8s_list_dns_endpoints",
            "k8s_list_endpoints",
            "k8s_list_endpointslices",
            "k8s_list_ingress",
            "k8s_list_ingress_classes",
            "k8s_list_ingresses",
            "k8s_list_native_services",
            "k8s_list_networkpolicies",
            "k8s_list_services",
            "k8s_test_dns_connectivity",
            "k8s_test_network_policy_connectivity",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "container-manager-mcp-platform-observer",
            "kubernetes-operator",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-k8s-networking",
        tools=(
            "cm_k8s_networking_get_k8s_service",
            "cm_k8s_networking_list_k8s_services",
            "cm_k8s_networking_create_k8s_service",
            "cm_k8s_networking_delete_k8s_service",
            "k8s_create_ingress",
            "k8s_create_ingress_class",
            "k8s_create_native_service",
            "k8s_create_service",
            "k8s_delete_ingress",
            "k8s_delete_native_service",
            "k8s_delete_networkpolicy",
            "k8s_port_forward_pod",
            "k8s_set_default_ingress_class",
            "k8s_update_network_policy_rules",
            "k8s_update_service",
            "trace_port_namespace",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "kubernetes-operator",
            "platform-reliability-engineer",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
            "container-manager-externalsecret-sync",
            "container-manager-gitops-persist",
        ),
    ),
    SurfaceLink(
        task="inspect-k8s-observability",
        tools=(
            "k8s_get_node_metrics",
            "k8s_get_resource_events",
            "k8s_get_top_nodes",
            "k8s_list_events",
            "k8s_top_nodes",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "container-manager-mcp-platform-observer",
            "kubernetes-operator",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-k8s-observability",
        tools=(
            "k8s_get_node_metrics",
            "k8s_get_resource_events",
            "k8s_debug_node",
            "k8s_debug_service",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "kubernetes-operator",
            "platform-reliability-engineer",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
            "container-manager-externalsecret-sync",
            "container-manager-gitops-persist",
        ),
    ),
    SurfaceLink(
        task="inspect-k8s-rbac",
        tools=(
            "k8s_auth_can_i",
            "k8s_describe_pod_security_policy",
            "k8s_evaluate_pod_security",
            "k8s_list_certificate_signing_requests",
            "k8s_list_cluster_rolebindings",
            "k8s_list_cluster_roles",
            "k8s_list_pod_security_policies",
            "k8s_list_rolebindings",
            "k8s_list_roles",
            "k8s_list_service_account_mapped_secrets",
            "k8s_list_service_account_tokens",
            "k8s_list_serviceaccounts",
            "k8s_local_subject_access_review",
            "k8s_subject_access_review",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "container-manager-mcp-platform-observer",
            "kubernetes-operator",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-k8s-rbac",
        tools=(
            "k8s_auth_can_i",
            "k8s_describe_pod_security_policy",
            "k8s_approve_csr",
            "k8s_create_aggregated_cluster_role",
            "k8s_create_cluster_rolebinding",
            "k8s_create_pod_security_policy",
            "k8s_create_role",
            "k8s_create_rolebinding",
            "k8s_create_service_account_token",
            "k8s_create_serviceaccount",
            "k8s_delete_cluster_rolebinding",
            "k8s_delete_pod_security_policy",
            "k8s_delete_role",
            "k8s_delete_rolebinding",
            "k8s_delete_service_account_token",
            "k8s_delete_serviceaccount",
            "k8s_deny_csr",
            "k8s_map_secret_to_service_account",
            "k8s_unmap_secret_from_service_account",
            "k8s_update_aggregated_cluster_role",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "kubernetes-operator",
            "platform-reliability-engineer",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
            "container-manager-externalsecret-sync",
            "container-manager-gitops-persist",
        ),
    ),
    SurfaceLink(
        task="inspect-k8s-storage",
        tools=(
            "k8s_describe_csi_driver",
            "k8s_get_csi_driver_capacity",
            "k8s_get_storage_class_provisioner",
            "k8s_list_csi_drivers",
            "k8s_list_persistent_volume_claims",
            "k8s_list_persistent_volumes",
            "k8s_list_storage_classes",
            "k8s_list_volume_snapshots",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "container-manager-mcp-platform-observer",
            "kubernetes-operator",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-k8s-storage",
        tools=(
            "k8s_describe_csi_driver",
            "k8s_get_csi_driver_capacity",
            "k8s_create_persistent_volume",
            "k8s_create_persistent_volume_claim",
            "k8s_create_storage_class",
            "k8s_delete_persistent_volume_claim",
            "k8s_expand_persistent_volume",
            "k8s_expand_pvc",
            "k8s_set_default_storage_class",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "kubernetes-operator",
            "platform-reliability-engineer",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
            "container-manager-externalsecret-sync",
            "container-manager-gitops-persist",
        ),
    ),
    SurfaceLink(
        task="inspect-k8s-workloads",
        tools=(
            "cm_k8s_workloads_describe_replicaset",
            "cm_multi_context_list_deployments",
            "k8s_describe_cron_job",
            "k8s_describe_deployment",
            "k8s_describe_horizontal_pod_autoscaler",
            "k8s_describe_job",
            "k8s_describe_limit_range",
            "k8s_describe_pod",
            "k8s_describe_pod_disruption_budget",
            "k8s_describe_priority_class",
            "k8s_describe_replica_set",
            "k8s_describe_resource_quota",
            "k8s_get_autoscaler_history",
            "k8s_get_autoscaler_metrics",
            "k8s_get_daemonset_update_strategy",
            "k8s_get_deployment_strategy",
            "k8s_get_pod_metrics",
            "k8s_get_pod_resource_usage",
            "k8s_get_statefulset_update_strategy",
            "k8s_get_top_pods",
            "k8s_list_cron_jobs",
            "k8s_list_daemon_sets",
            "k8s_list_daemonsets",
            "k8s_list_horizontal_pod_autoscalers",
            "k8s_list_jobs",
            "k8s_list_limit_ranges",
            "k8s_list_pod_disruption_budgets",
            "k8s_list_pods",
            "k8s_list_priority_classes",
            "k8s_list_replica_sets",
            "k8s_list_replicasets",
            "k8s_list_resource_quotas",
            "k8s_list_stateful_sets",
            "k8s_list_statefulsets",
            "k8s_read_pod_logs",
            "k8s_rollout_history",
            "k8s_rollout_status",
            "k8s_stream_pod_logs",
            "k8s_top_pods",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "container-manager-mcp-platform-observer",
            "kubernetes-operator",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
        ),
    ),
    SurfaceLink(
        task="change-k8s-workloads",
        tools=(
            "cm_k8s_workloads_describe_replicaset",
            "cm_multi_context_list_deployments",
            "cm_k8s_workloads_scale_replicaset",
            "k8s_attach_pod",
            "k8s_copy_from_pod",
            "k8s_copy_to_pod",
            "k8s_cp_pod",
            "k8s_create_cron_job",
            "k8s_create_daemon_set",
            "k8s_create_horizontal_pod_autoscaler",
            "k8s_create_job",
            "k8s_create_limit_range",
            "k8s_create_pod_disruption_budget",
            "k8s_create_priority_class",
            "k8s_create_resource_quota",
            "k8s_create_stateful_set",
            "k8s_debug_deployment",
            "k8s_debug_pod",
            "k8s_delete_cron_job",
            "k8s_delete_horizontal_pod_autoscaler",
            "k8s_delete_job",
            "k8s_delete_limit_range",
            "k8s_delete_pod_disruption_budget",
            "k8s_delete_priority_class",
            "k8s_delete_resource_quota",
            "k8s_exec_pod",
            "k8s_rollout_pause",
            "k8s_rollout_restart",
            "k8s_rollout_resume",
            "k8s_rollout_undo",
            "k8s_scale_deployment",
            "k8s_scale_deployment_autoscaler",
            "k8s_scale_replica_set",
            "k8s_scale_statefulset",
            "k8s_set_autoscaler_metrics",
            "k8s_set_daemonset_update_strategy",
            "k8s_set_deployment_strategy",
            "k8s_set_pod_anti_affinity",
            "k8s_set_statefulset_update_strategy",
            "k8s_update_horizontal_pod_autoscaler",
            "k8s_update_resource_quota",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
            "kubernetes-operator",
            "platform-reliability-engineer",
        ),
        skills=(
            "container-manager-kubernetes-operations",
            "container-manager-k8s-change-validation",
            "kubernetes-mesh-provisioner",
            "container-manager-externalsecret-sync",
            "container-manager-gitops-persist",
        ),
    ),
    SurfaceLink(
        task="inspect-podman-compose-and-stacks",
        tools=("podman_compose_logs", "podman_compose_ps"),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
    SurfaceLink(
        task="change-podman-compose-and-stacks",
        tools=(
            "podman_compose_logs",
            "podman_compose_ps",
            "podman_compose_down",
            "podman_compose_up",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
    SurfaceLink(
        task="inspect-podman-containers",
        tools=(
            "podman_get_container_logs",
            "podman_get_info",
            "podman_get_version",
            "podman_health_check",
            "podman_inspect_container",
            "podman_list_containers",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
    SurfaceLink(
        task="change-podman-containers",
        tools=(
            "podman_get_container_logs",
            "podman_get_info",
            "podman_exec_in_container",
            "podman_remove_container",
            "podman_run_container",
            "podman_stop_container",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
    SurfaceLink(
        task="inspect-podman-images-networks-volumes",
        tools=(
            "podman_list_images",
            "podman_list_networks",
            "podman_list_volumes",
            "podman_network_inspect",
            "podman_network_list",
            "podman_volume_inspect",
            "podman_volume_list",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
    SurfaceLink(
        task="change-podman-images-networks-volumes",
        tools=(
            "podman_list_images",
            "podman_list_networks",
            "podman_create_network",
            "podman_create_volume",
            "podman_network_create",
            "podman_prune_containers",
            "podman_prune_images",
            "podman_prune_networks",
            "podman_prune_system",
            "podman_prune_volumes",
            "podman_pull_image",
            "podman_remove_image",
            "podman_remove_network",
            "podman_remove_volume",
            "podman_system_prune",
            "podman_volume_create",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
    SurfaceLink(
        task="inspect-podman-pods",
        tools=(
            "podman_pod_inspect",
            "podman_pod_list",
            "podman_pod_logs",
            "podman_pod_stats",
            "podman_pod_top",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
    SurfaceLink(
        task="change-podman-pods",
        tools=(
            "podman_pod_inspect",
            "podman_pod_list",
            "podman_checkpoint",
            "podman_generate_kube_yaml",
            "podman_play_kube_yaml",
            "podman_pod_create",
            "podman_pod_rm",
            "podman_pod_stop",
            "podman_restore",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
    SurfaceLink(
        task="inspect-podman-services-and-secrets",
        tools=(
            "podman_inspect_service",
            "podman_list_services",
            "podman_service_logs",
            "podman_service_ps",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
    SurfaceLink(
        task="change-podman-services-and-secrets",
        tools=(
            "podman_inspect_service",
            "podman_list_services",
            "podman_create_service",
            "podman_remove_service",
            "podman_scale_service",
            "podman_update_service",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
    SurfaceLink(
        task="inspect-podman-swarm-and-nodes",
        tools=("podman_inspect_node", "podman_list_nodes"),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-observer",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
    SurfaceLink(
        task="change-podman-swarm-and-nodes",
        tools=(
            "podman_inspect_node",
            "podman_list_nodes",
            "podman_init_swarm",
            "podman_leave_swarm",
            "podman_remove_node",
            "podman_update_node",
        ),
        prompts=(
            "container-manager-mcp-operator",
            "container-manager-mcp-platform-reliability",
        ),
        skills=("container-manager-podman-operations", "podman-mesh-provisioner"),
    ),
)

LOCATIONS = (
    DataLocation(
        name="cluster-api",
        holds=("Pod", "Deployment", "Service", "Namespace"),
        instance="KUBECONFIG",
        read_tool="k8s_list_pods",
        federation_key="uid",
    ),
    DataLocation(
        name="node-local-volumes",
        holds=("ContainerVolume", "Container"),
        instance="CONTAINER_MANAGER_HOST",
        read_tool="docker_list_volumes",
        federation_key="name",
    ),
    DataLocation(
        name="secret-store",
        holds=("ExternalSecret",),
        instance="KUBECONFIG",
        read_tool="k8s_list_external_secrets",
        federation_key="name",
    ),
    DataLocation(
        name="services-manifests",
        holds=("Deployment", "Service"),
        instance="SERVICES_MANIFEST_REPOSITORY",
        read_tool="k8s_export_manifest",
        federation_key="name",
    ),
)

ROLES = (
    BusinessRole(
        name="platform-reliability",
        description="Run workloads reliably: change them safely, roll back failures and keep manifests in step with the cluster.",
        tasks=(
            "change-cm-inventory",
            "change-docker-compose-and-stacks",
            "change-docker-containers",
            "change-docker-images-networks-volumes",
            "change-docker-services-and-secrets",
            "change-docker-swarm-and-nodes",
            "change-k8s-cluster",
            "change-k8s-config",
            "change-k8s-container-runtime-compat",
            "change-k8s-networking",
            "change-k8s-observability",
            "change-k8s-rbac",
            "change-k8s-storage",
            "change-k8s-workloads",
            "change-podman-compose-and-stacks",
            "change-podman-containers",
            "change-podman-images-networks-volumes",
            "change-podman-pods",
            "change-podman-services-and-secrets",
            "change-podman-swarm-and-nodes",
            "inspect-k8s-cluster",
            "inspect-k8s-config",
            "inspect-k8s-container-runtime-compat",
            "inspect-k8s-networking",
            "inspect-k8s-observability",
            "inspect-k8s-rbac",
            "inspect-k8s-storage",
            "inspect-k8s-workloads",
        ),
    ),
    BusinessRole(
        name="platform observer",
        description="Report container and cluster health, capacity, logs and diagnostics without changing anything.",
        tasks=(
            "inspect-cm-diagnostics",
            "inspect-cm-inventory",
            "inspect-docker-compose-and-stacks",
            "inspect-docker-containers",
            "inspect-docker-images-networks-volumes",
            "inspect-docker-services-and-secrets",
            "inspect-docker-swarm-and-nodes",
            "inspect-k8s-cluster",
            "inspect-k8s-config",
            "inspect-k8s-container-runtime-compat",
            "inspect-k8s-networking",
            "inspect-k8s-observability",
            "inspect-k8s-rbac",
            "inspect-k8s-storage",
            "inspect-k8s-workloads",
            "inspect-podman-compose-and-stacks",
            "inspect-podman-containers",
            "inspect-podman-images-networks-volumes",
            "inspect-podman-pods",
            "inspect-podman-services-and-secrets",
            "inspect-podman-swarm-and-nodes",
        ),
    ),
)

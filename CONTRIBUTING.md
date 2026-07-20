# Contributing to Container Manager MCP

Thanks for helping improve `container-manager-mcp`. This project exposes Docker,
Podman, Docker Swarm, Kubernetes, multi-context, MCP, and A2A agent workflows, so
small, well-validated changes are easiest to review.

Before changing code, read `AGENTS.md`. It is the canonical agent and contributor
guidance for this repository; `CLAUDE.md` imports it to avoid drift.

## Reporting Bugs

Open an issue with enough detail to reproduce the failure:

- Package version and install path, such as PyPI, `uvx`, local checkout, or
  container image.
- Runtime and platform: Python version, operating system, Docker or Podman
  version, Kubernetes version, and whether the MCP server runs over `stdio`,
  `streamable-http`, or `sse`.
- The exact command or MCP tool call, including `cm_*` tool name, `action`, and
  non-secret arguments.
- Relevant `container-manager-doctor` output, MCP client output, container
  runtime logs, kubeconfig context, and `inventory.yml` host alias.
- Expected behavior, actual behavior, and whether the failure is local-only,
  remote inventory-based, multi-context, or Kubernetes-specific.

Do not include private keys, OIDC tokens, kubeconfigs, SSH tunnel details,
production hostnames, secrets, or raw customer data. Redact sensitive values
before attaching logs, `inventory.yml`, MCP traces, or Kubernetes manifests.

## Suggesting Features

For feature requests, describe:

- The backend or surface involved: Docker, Podman, Swarm, Kubernetes, inventory,
  multi-context, MCP tools, A2A agent, Knowledge Graph ingestion, or skills.
- The proposed `cm_*` tool and action shape, including required and optional
  inputs.
- How the feature should behave across local, inventory-routed, and multi-context
  execution.
- Any security or policy implications, especially for Eunomia, OIDC delegation,
  SSH, Kubernetes RBAC, and destructive operations.
- Documentation that would need to change.

## Development Setup

Use Python 3.11 or newer. `uv` is the preferred local workflow because the lockfile
is committed.

```bash
uv sync --all-extras --group dev
uv run container-manager-mcp --help
uv run container-manager-agent --help
uv run container-manager-doctor --help
```

If you use `pip`, install the project extras you need:

```bash
python -m pip install -e ".[all,test]"
```

For runtime checks, start with the doctor:

```bash
uv run container-manager-doctor
uv run container-manager-doctor --backend inventory --host <alias>
uv run container-manager-doctor --backend kubernetes --context <context>
```

Remote Docker and Podman hosts are resolved from the shared
`~/.config/agent-utilities/inventory.yml` file. Kubernetes targets use kubeconfig
contexts or `K8S_CONTEXTS`; they do not use the inventory host alias mechanism.

## Testing and Quality

Run the project quality gate before committing:

```bash
pre-commit run --all-files
```

For focused test runs:

```bash
uv run pytest
uv run pytest --timeout=60 -k "test_name_pattern"
```

When adding or changing behavior:

- Add or update pytest coverage in `tests/`.
- Cover both successful and failure paths for tool actions.
- Include MCP protocol validation when tool schemas, tool registration, or action
  routing changes.
- Include Docker, Podman, Swarm, Kubernetes, or inventory doctor validation when
  changing backend-specific behavior.
- Avoid tests that require live production infrastructure. Use mocks, fixtures,
  or explicit manual verification notes when a live runtime is unavoidable.

## Code Style

- Follow the patterns in `AGENTS.md`.
- Keep tools action-routed and idempotent where possible.
- Use `agent-utilities` helpers for shared MCP and agent patterns.
- Use Pydantic models for structured inputs and outputs.
- Keep public tool docstrings clear; they become LLM-facing tool descriptions.
- Check optional integrations with `try/except ImportError`.
- Prefer optional extras in `pyproject.toml` before adding new base dependencies.
- Do not write scratch scripts, logs, generated reports, databases, or debug files
  into the repository root.

## Documentation

Update docs in the same pull request when behavior changes:

- Root overview and generated MCP tool tables: `README.md`
- Installation and local setup: `docs/installation.md`, `docs/setup.md`
- Runtime usage and tool examples: `docs/usage.md`
- Deployment and MCP client configuration: `docs/deployment.md`
- Multi-host inventory behavior: `docs/multi_host.md`
- Kubernetes coverage and kubeconfig behavior: `docs/kubernetes.md`
- Agent skills: `container_manager_mcp/skills/*/SKILL.md`

The README has generated sections for MCP tools, environment variables, and MCP
config examples. Do not hand-edit generated blocks unless you are also running
the corresponding sync hook.

## Pull Request Process

1. Confirm no open pull request already covers the issue.
2. Keep the PR focused on one feature, bug fix, or documentation change.
3. Link the issue and explain the affected backend or MCP surface.
4. List the validation commands run, including doctor or manual runtime checks
   where applicable.
5. Note any checks skipped because Docker, Podman, Kubernetes, inventory hosts, or
   external services were unavailable.
6. Keep secrets, `.env` files, kubeconfigs, generated databases, and runtime logs
   out of the diff.

## Security

Report security-sensitive issues privately when they involve:

- Credential, token, kubeconfig, or SSH key exposure.
- Eunomia policy bypass or OIDC delegation problems.
- Kubernetes RBAC, service account token, or pod security regressions.
- Unsafe destructive operations across Docker, Podman, Swarm, or Kubernetes.
- MCP tool output leaking secrets or private infrastructure data.

Use placeholders in public issues and pull requests, such as `<host-alias>`,
`<kube-context>`, and `<token-redacted>`. Do not attach production
`inventory.yml`, kubeconfig files, SSH tunnel details, or raw MCP traces.

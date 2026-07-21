# Two right-sized images from ONE Dockerfile, selected by build --target. Both are
# slim multi-stage (build in a builder, ship only /usr/local — no dev-tools/source):
#
#   --target agent  (DEFAULT)  installs container-manager-mcp[agent] = the FULL runtime
#                              (agent-utilities[agent-runtime]: pydantic-ai + skills;
#                              the full epistemic-graph engine is a base dependency).
#                              The provider's [agent] also includes [mcp], so this image
#                              can run EITHER the agent OR the mcp server. Push as the
#                              default tags: latest / main / <version>.
#
#   --target mcp               installs container-manager-mcp[mcp]   = MCP-server tooling ONLY
#                              (agent-utilities[mcp]: FastMCP/FastAPI plus the mandatory
#                              full epistemic-graph base runtime). It omits the optional
#                              provider agent/frontends. Push as the :mcp / :mcp-<version> tag.
#
#   docker build --target agent -t container-manager-mcp:local .
#   docker build --target mcp   -t container-manager-mcp:mcp    .
FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3 AS builder-base
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=3600
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Slim MCP-only dependency closure.
FROM builder-base AS builder-mcp
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages container-manager-mcp[mcp]==2.3.0

# Full agent runtime dependency closure.
FROM builder-base AS builder-agent
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages container-manager-mcp[agent]==2.3.0

# ── Kubernetes auth CLIs ──────────────────────────────────────────────────────
# cm shells out to `kubectl` (copy_to_pod/copy_from_pod use `kubectl cp`) and,
# for managed-cloud kubeconfigs, to exec-credential plugins. Bake the LEAN single
# binaries that cover every conformant cluster: in-cluster SA + cert/token need
# only kubectl; EKS needs aws-iam-authenticator; AKS/Azure-AD needs kubelogin;
# generic OIDC (incl. our own Keycloak) needs `kubectl oidc-login`; GKE needs
# gke-gcloud-auth-plugin. Full gcloud/az/aws SDKs are intentionally NOT baked
# (too heavy for a slim image); mount them or extend this stage for interactive
# cloud login. Latest release tags are resolved at build time so pins never go stale.
FROM debian:12-slim@sha256:7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818 AS k8s-tools
ARG TARGETARCH=amd64
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates unzip apt-transport-https gnupg \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; mkdir -p /out; arch="${TARGETARCH}"; \
    latest() { curl -fsSL "https://api.github.com/repos/$1/releases/latest" | grep -oP '"tag_name":\s*"v?\K[^"]+' | head -1; }; \
    kver="$(curl -fsSL https://dl.k8s.io/release/stable.txt)"; \
    curl -fsSL -o /out/kubectl "https://dl.k8s.io/release/${kver}/bin/linux/${arch}/kubectl"; \
    aia="$(latest kubernetes-sigs/aws-iam-authenticator)"; \
    curl -fsSL -o /out/aws-iam-authenticator "https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v${aia}/aws-iam-authenticator_${aia}_linux_${arch}"; \
    akl="$(latest Azure/kubelogin)"; \
    curl -fsSL -o /tmp/azkl.zip "https://github.com/Azure/kubelogin/releases/download/v${akl}/kubelogin-linux-${arch}.zip"; \
    unzip -oq /tmp/azkl.zip -d /tmp/azkl; cp "/tmp/azkl/bin/linux_${arch}/kubelogin" /out/kubelogin; \
    okl="$(latest int128/kubelogin)"; \
    curl -fsSL -o /tmp/oidc.zip "https://github.com/int128/kubelogin/releases/download/v${okl}/kubelogin_linux_${arch}.zip"; \
    unzip -oq /tmp/oidc.zip -d /tmp/oidc; cp /tmp/oidc/kubelogin /out/kubectl-oidc_login; \
    chmod +x /out/*
# GKE: gke-gcloud-auth-plugin — standalone Go binary from Google's apt package
# (the full google-cloud-cli it pulls stays in THIS discarded stage; only the
# ~single plugin binary is copied into the runtime image).
RUN set -eux; \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends google-cloud-cli-gke-gcloud-auth-plugin; \
    cp "$(readlink -f "$(command -v gke-gcloud-auth-plugin)")" /out/gke-gcloud-auth-plugin; \
    rm -rf /var/lib/apt/lists/*

FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3 AS runtime-base
ARG HOST=127.0.0.1
ARG PORT=8000
ARG TRANSPORT="stdio"
ARG AUTH_TYPE="none"
ENV HOST=${HOST} \
    PORT=${PORT} \
    TRANSPORT=${TRANSPORT} \
    AUTH_TYPE=${AUTH_TYPE} \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:${PATH}" \
    UV_HTTP_TIMEOUT=3600 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
# kubectl + exec-credential plugins (EKS/AKS/GKE/OIDC) so cm drives ANY cluster.
COPY --from=k8s-tools /out/ /usr/local/bin/

# Slim image: MCP server only.
# Fixed identity for least-privilege runtime and stable volume ownership.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --no-create-home \
        --home-dir /tmp --shell /usr/sbin/nologin app
ENV HOME=/tmp \
    XDG_CONFIG_HOME=/tmp/.config \
    XDG_CACHE_HOME=/tmp/.cache

FROM runtime-base AS mcp
USER 10001:10001
COPY --from=builder-mcp /usr/local /usr/local
CMD ["container-manager-mcp"]

# Full image (DEFAULT target): agent runtime; can also run the mcp server.
FROM runtime-base AS agent
USER 10001:10001
COPY --from=builder-agent /usr/local /usr/local
CMD ["container-manager-agent"]

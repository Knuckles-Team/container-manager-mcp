#!/usr/bin/env python
"""Save/register a Kubernetes environment (kubeconfig context) into the file cm reads.

This is the core capability behind the ``save-context`` CLI
(``container-manager-save-context``) and the ``cm_k8s_cluster`` ``save_context``
MCP action. It lets a user *dynamically* add a kube context to the kubeconfig
container-manager-mcp loads (``$KUBECONFIG`` first entry, else ``~/.kube/config``)
and reuse it later via ``CONTAINER_MANAGER_KUBECONTEXT`` / ``K8S_CONTEXTS`` /
``use_context``.

Four input modes, all funnelling through one non-destructive merge
(``kubectl config view --merge`` semantics — existing entries are never
clobbered, a context-name collision errors unless ``overwrite=True``):

* **token** — ``name`` + ``server`` + ``token`` (bearer token: ServiceAccount /
  OIDC id-token) + ``ca_cert`` / ``insecure_skip_tls_verify``. The primary path.
* **mTLS** — ``name`` + ``server`` + ``client_cert`` + ``client_key`` +
  ``ca_cert`` (how an RKE2 admin kubeconfig authenticates). Certs accept file
  paths OR inline PEM/base64 data.
* **oidc** — ``name`` + ``server`` + ``username`` + ``password`` +
  ``oidc_issuer`` + ``oidc_client_id``: drives an OIDC resource-owner password
  grant against the issuer to obtain an id-token that is embedded as the user's
  bearer token. NOTE: the Kubernetes API server does **not** accept plain
  username/password (basic auth was removed) — username/password is only usable
  against an **OIDC-backed** cluster, so this mode *requires* ``oidc_issuer`` +
  ``oidc_client_id`` and fails clearly otherwise.
* **import** — merge an existing kubeconfig file (``source_file``) or raw
  kubeconfig YAML (``source_yaml``); or ``capture_current=True`` generates a
  *portable* kubeconfig for the cluster cm is currently pointed at (in-cluster
  service-account token + CA, or the current kubeconfig context with its certs
  embedded) and saves it under ``name`` — so you can export the cluster you are on.

The ``token`` and ``mTLS`` modes are both driven by the *explicit* code path
(they differ only in which auth fields you supply).

Secrets (bearer tokens, key data) are kept out of logs and out of the returned
summary. Certificate/key material passed as a file path or PEM text is embedded
as base64 ``*-data`` so the resulting context is self-contained/portable.
"""

import argparse
import base64
import json
import os
import sys
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML ships with the kubernetes client
    yaml = None  # type: ignore[assignment]

# In-cluster service-account mount (kubelet-projected). Used by capture mode when
# running inside a pod.
_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"


def default_kubeconfig_path() -> str:
    """Resolve the kubeconfig file cm reads: ``$KUBECONFIG`` first entry, else ``~/.kube/config``.

    Mirrors the kubernetes client's own precedence so the doctor, the manager,
    and this writer all agree on which file is authoritative.
    """
    env = os.environ.get("KUBECONFIG")
    if env:
        for part in env.split(os.pathsep):
            if part.strip():
                return os.path.expanduser(part.strip())
    return os.path.expanduser("~/.kube/config")


def _require_yaml():
    if yaml is None:  # pragma: no cover - defensive
        raise RuntimeError(
            "PyYAML is required to read/write kubeconfig files: "
            "pip install 'container-manager-mcp[kubernetes]'"
        )


def _empty_kubeconfig() -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Config",
        "preferences": {},
        "clusters": [],
        "users": [],
        "contexts": [],
        "current-context": "",
    }


def _load_kubeconfig_file(path: str) -> dict:
    """Load a kubeconfig file into a dict, returning an empty skeleton if absent."""
    _require_yaml()
    if not os.path.exists(path):
        return _empty_kubeconfig()
    with open(path) as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"kubeconfig at {path} is not a valid YAML mapping")
    for key in ("clusters", "users", "contexts"):
        loaded.setdefault(key, [])
    loaded.setdefault("apiVersion", "v1")
    loaded.setdefault("kind", "Config")
    return loaded


def _looks_like_pem(value: str) -> bool:
    return "-----BEGIN" in value


def _to_cert_data(value: str) -> str:
    """Normalise a cert/key given as a file path or inline PEM/base64 into base64 data.

    * an existing file path -> its bytes, base64-encoded;
    * inline PEM text (``-----BEGIN ...``) -> base64-encoded;
    * anything else -> assumed already base64 and returned verbatim.
    """
    if os.path.isfile(value):
        with open(value, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    if _looks_like_pem(value):
        return base64.b64encode(value.encode("utf-8")).decode("ascii")
    return value


def _build_explicit_kubeconfig(
    name: str,
    server: str,
    token: str | None = None,
    client_cert: str | None = None,
    client_key: str | None = None,
    ca_cert: str | None = None,
    insecure_skip_tls_verify: bool = False,
    namespace: str | None = None,
) -> dict:
    """Assemble a one-context kubeconfig dict from explicit connection params."""
    if not server:
        raise ValueError("explicit mode requires 'server' (the API server URL)")
    if not token and not (client_cert and client_key):
        raise ValueError(
            "explicit mode requires either 'token' or both "
            "'client_cert' and 'client_key'"
        )

    cluster: dict[str, Any] = {"server": server}
    if insecure_skip_tls_verify:
        cluster["insecure-skip-tls-verify"] = True
    elif ca_cert:
        cluster["certificate-authority-data"] = _to_cert_data(ca_cert)

    user: dict[str, Any] = {}
    if token:
        user["token"] = token
    if client_cert and client_key:
        user["client-certificate-data"] = _to_cert_data(client_cert)
        user["client-key-data"] = _to_cert_data(client_key)

    context: dict[str, Any] = {"cluster": name, "user": name}
    if namespace:
        context["namespace"] = namespace

    return {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [{"name": name, "cluster": cluster}],
        "users": [{"name": name, "user": user}],
        "contexts": [{"name": name, "context": context}],
        "current-context": name,
    }


def _oidc_verify_arg(ca_cert: str | None, insecure: bool):
    """Map ca_cert/insecure into the ``verify=`` argument for the OIDC HTTP calls."""
    if insecure:
        return False
    if ca_cert and os.path.isfile(ca_cert):
        return ca_cert
    return True


def _oidc_password_grant(
    issuer: str,
    client_id: str,
    username: str,
    password: str,
    client_secret: str | None = None,
    scope: str = "openid",
    ca_cert: str | None = None,
    insecure: bool = False,
) -> str:
    """Obtain an OIDC id-token via the resource-owner password grant.

    Discovers the token endpoint from ``{issuer}/.well-known/openid-configuration``
    and posts ``grant_type=password``. Returns the ``id_token`` (which the
    Kubernetes OIDC authenticator consumes). Raises a clear ``RuntimeError`` when
    the issuer is unreachable, the grant is rejected, or no id-token is returned.
    """
    try:
        import requests
    except ImportError as e:  # pragma: no cover - requests ships with docker
        raise RuntimeError(
            "the OIDC username/password mode requires the 'requests' library"
        ) from e

    verify = _oidc_verify_arg(ca_cert, insecure)
    disc_url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    try:
        disc = requests.get(disc_url, verify=verify, timeout=30)
        disc.raise_for_status()
        token_endpoint = disc.json().get("token_endpoint")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"OIDC discovery failed at {disc_url}: {e}. Confirm 'oidc_issuer' is the "
            "issuer URL of an OIDC provider the cluster trusts."
        ) from e
    if not token_endpoint:
        raise RuntimeError(
            f"OIDC discovery at {disc_url} did not advertise a token_endpoint"
        )

    data = {
        "grant_type": "password",
        "client_id": client_id,
        "username": username,
        "password": password,
        "scope": scope,
    }
    if client_secret:
        data["client_secret"] = client_secret
    try:
        resp = requests.post(token_endpoint, data=data, verify=verify, timeout=30)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"OIDC token request to {token_endpoint} failed: {e}") from e
    if resp.status_code >= 400:
        detail = ""
        try:
            body = resp.json()
            detail = (
                f": {body.get('error')} {body.get('error_description', '')}".strip()
            )
        except Exception:  # noqa: BLE001
            detail = f": {resp.text[:200]}"
        raise RuntimeError(
            f"OIDC password grant rejected (HTTP {resp.status_code}){detail}. "
            "Check the username/password, client-id, and that the provider allows "
            "the resource-owner password grant."
        )
    payload = resp.json()
    id_token = payload.get("id_token")
    if not id_token:
        raise RuntimeError(
            "OIDC provider returned no id_token (only an access_token?). The "
            "Kubernetes OIDC authenticator requires an id_token — request the "
            "'openid' scope and a client configured to issue id_tokens."
        )
    return id_token


def _read_sa_file(fname: str) -> str | None:
    path = os.path.join(_SA_DIR, fname)
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _capture_incluster_kubeconfig(name: str, namespace: str | None) -> dict:
    """Build a portable kubeconfig from the pod's mounted service account."""
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    if not host:
        raise RuntimeError(
            "capture_current in-cluster mode requires KUBERNETES_SERVICE_HOST"
        )
    server = f"https://{host}:{port}"
    token = _read_sa_file("token")
    if not token:
        raise RuntimeError(f"could not read service-account token under {_SA_DIR}")
    ns = namespace or _read_sa_file("namespace") or "default"
    ca_path = os.path.join(_SA_DIR, "ca.crt")
    return _build_explicit_kubeconfig(
        name=name,
        server=server,
        token=token,
        ca_cert=ca_path if os.path.exists(ca_path) else None,
        insecure_skip_tls_verify=not os.path.exists(ca_path),
        namespace=ns,
    )


def _embed_file_refs(entry: dict, mapping: dict[str, str]) -> dict:
    """Replace file-path cert refs with embedded ``*-data`` for portability."""
    out = dict(entry)
    for path_key, data_key in mapping.items():
        if out.get(path_key) and not out.get(data_key):
            try:
                out[data_key] = _to_cert_data(out.pop(path_key))
            except OSError:
                # Leave the path reference in place if unreadable.
                pass
    return out


def _capture_from_current_kubeconfig(
    name: str,
    kubeconfig_path: str,
    source_context: str | None,
    namespace: str | None,
) -> dict:
    """Extract the current (or named) context from kubeconfig, certs embedded, renamed to ``name``."""
    src = _load_kubeconfig_file(kubeconfig_path)
    if not os.path.exists(kubeconfig_path):
        raise RuntimeError(
            f"no kubeconfig at {kubeconfig_path} to capture the current cluster from"
        )
    ctx_name = source_context or src.get("current-context")
    if not ctx_name:
        raise RuntimeError(
            "no current-context set in kubeconfig; pass source_context to pick one"
        )
    ctx = next((c for c in src.get("contexts", []) if c.get("name") == ctx_name), None)
    if ctx is None:
        raise RuntimeError(f"context '{ctx_name}' not found in {kubeconfig_path}")
    ctx_body = dict(ctx.get("context", {}) or {})
    cluster_ref = ctx_body.get("cluster")
    user_ref = ctx_body.get("user")

    cluster = next(
        (c for c in src.get("clusters", []) if c.get("name") == cluster_ref), None
    )
    user = next((u for u in src.get("users", []) if u.get("name") == user_ref), None)
    if cluster is None:
        raise RuntimeError(f"cluster '{cluster_ref}' referenced by context not found")

    cluster_body = _embed_file_refs(
        dict(cluster.get("cluster", {}) or {}),
        {"certificate-authority": "certificate-authority-data"},
    )
    user_body = _embed_file_refs(
        dict((user or {}).get("user", {}) or {}),
        {
            "client-certificate": "client-certificate-data",
            "client-key": "client-key-data",
        },
    )
    new_ctx_body: dict[str, Any] = {"cluster": name, "user": name}
    ns = namespace or ctx_body.get("namespace")
    if ns:
        new_ctx_body["namespace"] = ns

    return {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [{"name": name, "cluster": cluster_body}],
        "users": [{"name": name, "user": user_body}],
        "contexts": [{"name": name, "context": new_ctx_body}],
        "current-context": name,
    }


def _index_by_name(entries: list[dict]) -> dict[str, int]:
    return {
        e.get("name"): i
        for i, e in enumerate(entries)
        if isinstance(e, dict) and e.get("name")
    }


def merge_kubeconfig(target: dict, incoming: dict, overwrite: bool = False) -> dict:
    """Non-destructively merge ``incoming``'s clusters/users/contexts into ``target``.

    Existing entries are preserved. A **context**-name collision raises unless
    ``overwrite=True`` (in which case the context and its referenced cluster/user
    are replaced). Returns a summary dict of what changed.
    """
    summary: dict[str, Any] = {
        "clusters_added": [],
        "clusters_replaced": [],
        "users_added": [],
        "users_replaced": [],
        "contexts_added": [],
        "contexts_replaced": [],
    }

    # Context collisions gate the whole merge.
    tgt_ctx_idx = _index_by_name(target.setdefault("contexts", []))
    for ctx in incoming.get("contexts", []):
        cname = ctx.get("name")
        if cname in tgt_ctx_idx and not overwrite:
            raise ValueError(
                f"context '{cname}' already exists in the target kubeconfig; "
                "pass overwrite=True to replace it"
            )

    def _merge_section(section: str, add_key: str, repl_key: str) -> None:
        tgt_list = target.setdefault(section, [])
        idx = _index_by_name(tgt_list)
        for entry in incoming.get(section, []):
            ename = entry.get("name")
            if not ename:
                continue
            if ename in idx:
                if overwrite:
                    tgt_list[idx[ename]] = entry
                    summary[repl_key].append(ename)
                # else: keep existing (non-destructive)
            else:
                tgt_list.append(entry)
                idx[ename] = len(tgt_list) - 1
                summary[add_key].append(ename)

    _merge_section("clusters", "clusters_added", "clusters_replaced")
    _merge_section("users", "users_added", "users_replaced")
    _merge_section("contexts", "contexts_added", "contexts_replaced")
    return summary


def _write_kubeconfig(path: str, config: dict) -> None:
    _require_yaml()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh, default_flow_style=False, sort_keys=False)
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - best effort on exotic filesystems
        pass


def _validate_context(context_name: str) -> dict:
    """Load ``context_name`` and call list_nodes; return node count or a clear error."""
    try:
        from container_manager_mcp.container_manager import create_manager

        manager = create_manager("kubernetes", host=context_name)
        nodes = manager.list_nodes()
        return {"status": "ok", "context": context_name, "node_count": len(nodes)}
    except Exception as e:  # noqa: BLE001 - report, never fail the save
        return {"status": "unreachable", "context": context_name, "error": str(e)}


def save_kube_context(
    name: str | None = None,
    kubeconfig_path: str | None = None,
    source_file: str | None = None,
    source_yaml: str | None = None,
    server: str | None = None,
    token: str | None = None,
    client_cert: str | None = None,
    client_key: str | None = None,
    ca_cert: str | None = None,
    insecure_skip_tls_verify: bool = False,
    namespace: str | None = None,
    username: str | None = None,
    password: str | None = None,
    oidc_issuer: str | None = None,
    oidc_client_id: str | None = None,
    oidc_client_secret: str | None = None,
    oidc_scope: str = "openid",
    capture_current: bool = False,
    source_context: str | None = None,
    overwrite: bool = False,
    use: bool = False,
    validate: bool = True,
) -> dict:
    """Save/register a kube context into the kubeconfig cm reads, then optionally validate it.

    Modes (resolved in this order): **import** (``source_file`` or
    ``source_yaml``), **capture** (``capture_current=True``), **oidc**
    (``username``/``password`` — requires ``oidc_issuer`` + ``oidc_client_id``),
    else **explicit** (``server`` + ``token`` OR ``client_cert``/``client_key``).
    All require ``name`` except pure import. The merge is non-destructive; a
    context-name collision raises unless ``overwrite=True``. ``use=True`` sets
    ``current-context``. When ``validate`` is set, the saved context is loaded
    and ``list_nodes`` is called; the node count (or a clear error) is returned.

    NOTE: the Kubernetes API server does not accept plain username/password
    (basic auth was removed), so the username/password path goes through OIDC and
    requires ``oidc_issuer`` + ``oidc_client_id``; it fails clearly otherwise.

    Returns ``{path, mode, contexts, merged, current_context, validation}``.
    Secrets are never included in the returned summary.
    """
    _require_yaml()
    path = (
        os.path.expanduser(kubeconfig_path)
        if kubeconfig_path
        else default_kubeconfig_path()
    )

    # --- resolve the incoming kubeconfig dict from the selected mode ---------
    if source_file or source_yaml:
        mode = "import"
        if source_file:
            incoming = _load_kubeconfig_file(os.path.expanduser(source_file))
            if not os.path.exists(os.path.expanduser(source_file)):
                raise FileNotFoundError(f"source kubeconfig not found: {source_file}")
        else:
            loaded = yaml.safe_load(source_yaml) or {}
            if not isinstance(loaded, dict):
                raise ValueError("source_yaml is not a valid kubeconfig YAML mapping")
            for key in ("clusters", "users", "contexts"):
                loaded.setdefault(key, [])
            incoming = loaded
        if not incoming.get("contexts"):
            raise ValueError("the source kubeconfig defines no contexts to import")
    elif capture_current:
        mode = "capture"
        if not name:
            raise ValueError("capture_current mode requires 'name' to save under")
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            incoming = _capture_incluster_kubeconfig(name, namespace)
        else:
            incoming = _capture_from_current_kubeconfig(
                name, path, source_context, namespace
            )
    elif username or password:
        mode = "oidc"
        if not name:
            raise ValueError("oidc mode requires 'name'")
        if not server:
            raise ValueError("oidc mode requires 'server' (the API server URL)")
        if not username or not password:
            raise ValueError("oidc mode requires both 'username' and 'password'")
        if not oidc_issuer or not oidc_client_id:
            raise ValueError(
                "username/password can only authenticate to an OIDC-backed cluster: "
                "the Kubernetes API server does not accept basic auth. Provide "
                "'oidc_issuer' and 'oidc_client_id' to drive the OIDC login, or use "
                "the token / client-cert mode instead."
            )
        id_token = _oidc_password_grant(
            issuer=oidc_issuer,
            client_id=oidc_client_id,
            username=username,
            password=password,
            client_secret=oidc_client_secret,
            scope=oidc_scope,
            ca_cert=ca_cert,
            insecure=insecure_skip_tls_verify,
        )
        incoming = _build_explicit_kubeconfig(
            name=name,
            server=server,
            token=id_token,
            ca_cert=ca_cert,
            insecure_skip_tls_verify=insecure_skip_tls_verify,
            namespace=namespace,
        )
    else:
        mode = "explicit"
        if not name:
            raise ValueError("explicit mode requires 'name'")
        incoming = _build_explicit_kubeconfig(
            name=name,
            server=server or "",
            token=token,
            client_cert=client_cert,
            client_key=client_key,
            ca_cert=ca_cert,
            insecure_skip_tls_verify=insecure_skip_tls_verify,
            namespace=namespace,
        )

    incoming_contexts = [
        c.get("name") for c in incoming.get("contexts", []) if c.get("name")
    ]

    # --- merge into the target file -----------------------------------------
    target = _load_kubeconfig_file(path)
    summary = merge_kubeconfig(target, incoming, overwrite=overwrite)

    # --- optionally set current-context -------------------------------------
    chosen = (
        name
        if name in incoming_contexts
        else (incoming_contexts[0] if incoming_contexts else None)
    )
    if use and chosen:
        target["current-context"] = chosen

    _write_kubeconfig(path, target)

    result: dict[str, Any] = {
        "path": path,
        "mode": mode,
        "contexts": incoming_contexts,
        "merged": summary,
        "current_context": target.get("current-context", ""),
        "used": bool(use and chosen),
    }

    # --- validate-after-save ------------------------------------------------
    if validate and chosen:
        result["validation"] = _validate_context(chosen)

    return result


# ---------------------------------------------------------------------------
# CLI — ``container-manager-save-context``
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="container-manager-save-context",
        description=(
            "Save/register a Kubernetes environment (kubeconfig context) into the "
            "kubeconfig container-manager-mcp reads ($KUBECONFIG first entry, else "
            "~/.kube/config), so it can be reused later. Supports importing an "
            "existing kubeconfig, explicit connection params, and capturing the "
            "cluster you are currently on."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Context name to save under (required for explicit/capture modes).",
    )
    parser.add_argument(
        "--kubeconfig",
        default=None,
        help="Target kubeconfig file to write (default: $KUBECONFIG first entry, else ~/.kube/config).",
    )
    # import mode
    parser.add_argument(
        "--from-file",
        default=None,
        help="Merge an existing kubeconfig file (import mode).",
    )
    parser.add_argument(
        "--from-yaml",
        default=None,
        help="Merge raw kubeconfig YAML (import mode); pass '-' to read from stdin.",
    )
    # explicit mode
    parser.add_argument(
        "--server", default=None, help="API server URL (explicit mode)."
    )
    parser.add_argument("--token", default=None, help="Bearer token (explicit mode).")
    parser.add_argument(
        "--client-cert", default=None, help="Client cert path or PEM (explicit mode)."
    )
    parser.add_argument(
        "--client-key", default=None, help="Client key path or PEM (explicit mode)."
    )
    parser.add_argument(
        "--ca-cert", default=None, help="Cluster CA cert path or PEM (explicit mode)."
    )
    parser.add_argument(
        "--insecure-skip-tls-verify",
        action="store_true",
        help="Skip TLS verification instead of pinning a CA (explicit mode).",
    )
    parser.add_argument(
        "--namespace", default=None, help="Default namespace for the context."
    )
    # oidc mode (username/password -> OIDC login)
    parser.add_argument(
        "--username",
        default=None,
        help=(
            "OIDC username (oidc mode). NOTE: the Kubernetes API server does NOT "
            "accept basic auth — username/password only works against an "
            "OIDC-backed cluster and requires --oidc-issuer + --oidc-client-id."
        ),
    )
    parser.add_argument("--password", default=None, help="OIDC password (oidc mode).")
    parser.add_argument(
        "--oidc-issuer",
        default=None,
        help="OIDC issuer URL, e.g. https://keycloak/realms/<realm> (oidc mode).",
    )
    parser.add_argument(
        "--oidc-client-id", default=None, help="OIDC client id (oidc mode)."
    )
    parser.add_argument(
        "--oidc-client-secret",
        default=None,
        help="OIDC client secret for confidential clients (oidc mode, optional).",
    )
    parser.add_argument(
        "--oidc-scope",
        default="openid",
        help="OIDC scope to request (oidc mode, default: openid).",
    )
    # capture mode
    parser.add_argument(
        "--capture-current",
        action="store_true",
        help="Capture the cluster cm is currently on (in-cluster SA or current kubeconfig context).",
    )
    parser.add_argument(
        "--source-context",
        default=None,
        help="For --capture-current: which existing context to capture (default: current-context).",
    )
    # behaviour
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing context/cluster/user on name collision (default: error).",
    )
    parser.add_argument(
        "--use", action="store_true", help="Set current-context to the saved context."
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the post-save validation (loading the context + list_nodes).",
    )
    parser.add_argument("--json", action="store_true", help="Emit the result as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """``container-manager-save-context`` CLI. Exit 0 on success, 1 on error."""
    print("container-manager-save-context", file=sys.stderr)
    parser = _build_parser()
    args = parser.parse_args(argv)

    from_yaml = args.from_yaml
    if from_yaml == "-":
        from_yaml = sys.stdin.read()

    try:
        result = save_kube_context(
            name=args.name,
            kubeconfig_path=args.kubeconfig,
            source_file=args.from_file,
            source_yaml=from_yaml,
            server=args.server,
            token=args.token,
            client_cert=args.client_cert,
            client_key=args.client_key,
            ca_cert=args.ca_cert,
            insecure_skip_tls_verify=args.insecure_skip_tls_verify,
            namespace=args.namespace,
            username=args.username,
            password=args.password,
            oidc_issuer=args.oidc_issuer,
            oidc_client_id=args.oidc_client_id,
            oidc_client_secret=args.oidc_client_secret,
            oidc_scope=args.oidc_scope,
            capture_current=args.capture_current,
            source_context=args.source_context,
            overwrite=args.overwrite,
            use=args.use,
            validate=not args.no_validate,
        )
    except Exception as e:  # noqa: BLE001 - surface a clean CLI error
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(
            f"Saved context(s) {result['contexts']} ({result['mode']} mode) "
            f"into {result['path']}"
        )
        merged = result["merged"]
        added = merged["contexts_added"] + merged["contexts_replaced"]
        print(f"  contexts written: {added or 'none (already present)'}")
        print(f"  current-context: {result['current_context']}")
        val = result.get("validation")
        if val:
            if val["status"] == "ok":
                print(f"  validation: OK — {val['node_count']} node(s) reachable")
            else:
                print(f"  validation: {val['status']} — {val.get('error', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

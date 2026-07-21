"""Tests for the save/register kube-context capability (``kubeconfig_import``).

Covers the pure-YAML merge surface (merge-into-empty, non-destructive merge,
name-collision, overwrite), the three input modes (import / explicit / capture),
and the validate-after-save path. No live cluster is required — the kubernetes
client is only touched by the validation path, which is mocked.
"""

import base64
import os
from unittest.mock import MagicMock, patch

import pytest
import yaml

from container_manager_mcp.k8s import kubeconfig_import as kci
from container_manager_mcp.k8s.kubeconfig_import import (
    default_kubeconfig_path,
    merge_kubeconfig,
    save_kube_context,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _ctx_names(cfg: dict) -> list[str]:
    return [c["name"] for c in cfg.get("contexts", [])]


def _one_context_config(name: str, server: str = "https://h:6443") -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": name,
                "cluster": {"server": server, "insecure-skip-tls-verify": True},
            }
        ],
        "users": [{"name": name, "user": {"token": "tok"}}],
        "contexts": [{"name": name, "context": {"cluster": name, "user": name}}],
        "current-context": name,
    }


# ---------------------------------------------------------------------------
# default path resolution
# ---------------------------------------------------------------------------
def test_default_kubeconfig_path_prefers_kubeconfig_env(monkeypatch, tmp_path):
    first = tmp_path / "a"
    monkeypatch.setenv("KUBECONFIG", f"{first}{os.pathsep}{tmp_path / 'b'}")
    assert default_kubeconfig_path() == str(first)


def test_default_kubeconfig_path_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("KUBECONFIG", raising=False)
    assert default_kubeconfig_path() == os.path.expanduser("~/.kube/config")


# ---------------------------------------------------------------------------
# merge semantics
# ---------------------------------------------------------------------------
def test_merge_into_empty():
    target = {"clusters": [], "users": [], "contexts": []}
    summary = merge_kubeconfig(target, _one_context_config("dev"))
    assert summary["contexts_added"] == ["dev"]
    assert _ctx_names(target) == ["dev"]


def test_merge_non_destructive_keeps_existing():
    target = _one_context_config("dev")
    summary = merge_kubeconfig(
        target, _one_context_config("prod", server="https://p:6443")
    )
    assert summary["contexts_added"] == ["prod"]
    # existing 'dev' untouched, 'prod' appended
    assert _ctx_names(target) == ["dev", "prod"]


def test_merge_name_collision_raises():
    target = _one_context_config("dev")
    with pytest.raises(ValueError, match="already exists"):
        merge_kubeconfig(
            target, _one_context_config("dev", server="https://other:6443")
        )


def test_merge_overwrite_replaces():
    target = _one_context_config("dev")
    summary = merge_kubeconfig(
        target, _one_context_config("dev", server="https://new:6443"), overwrite=True
    )
    assert summary["contexts_replaced"] == ["dev"]
    cluster = next(c for c in target["clusters"] if c["name"] == "dev")["cluster"]
    assert cluster["server"] == "https://new:6443"


# ---------------------------------------------------------------------------
# explicit params mode
# ---------------------------------------------------------------------------
def test_save_explicit_params_into_empty(tmp_path):
    kc = tmp_path / "config"
    result = save_kube_context(
        name="dev",
        kubeconfig_path=str(kc),
        server="https://1.2.3.4:6443",
        token="s3cr3t",
        insecure_skip_tls_verify=True,
        namespace="apps",
        validate=False,
    )
    assert result["mode"] == "explicit"
    assert result["contexts"] == ["dev"]
    cfg = _read(str(kc))
    assert _ctx_names(cfg) == ["dev"]
    ctx = cfg["contexts"][0]["context"]
    assert ctx["namespace"] == "apps"
    # secret is written to the file but NOT leaked into the returned summary
    assert "s3cr3t" not in str(result)
    # file is written 0600
    assert oct(os.stat(kc).st_mode & 0o777) == "0o600"


def test_save_explicit_requires_auth(tmp_path):
    with pytest.raises(ValueError, match="token"):
        save_kube_context(
            name="dev",
            kubeconfig_path=str(tmp_path / "c"),
            server="https://h:6443",
            validate=False,
        )


def test_save_explicit_embeds_cert_files(tmp_path):
    ca = tmp_path / "ca.crt"
    ca.write_text("-----BEGIN CERTIFICATE-----\nQ0E=\n-----END CERTIFICATE-----")
    result = save_kube_context(
        name="dev",
        kubeconfig_path=str(tmp_path / "config"),
        server="https://h:6443",
        token="t",
        ca_cert=str(ca),
        validate=False,
    )
    cfg = _read(result["path"])
    cluster = cfg["clusters"][0]["cluster"]
    assert "certificate-authority-data" in cluster
    decoded = base64.b64decode(cluster["certificate-authority-data"]).decode()
    assert "BEGIN CERTIFICATE" in decoded


# ---------------------------------------------------------------------------
# non-destructive save + collision at the save_kube_context level
# ---------------------------------------------------------------------------
def test_save_is_non_destructive_and_collision_errors(tmp_path):
    kc = str(tmp_path / "config")
    save_kube_context(
        name="dev",
        kubeconfig_path=kc,
        server="https://a:6443",
        token="t",
        insecure_skip_tls_verify=True,
        validate=False,
    )
    save_kube_context(
        name="prod",
        kubeconfig_path=kc,
        server="https://b:6443",
        token="t",
        insecure_skip_tls_verify=True,
        validate=False,
    )
    assert _ctx_names(_read(kc)) == ["dev", "prod"]
    with pytest.raises(ValueError, match="already exists"):
        save_kube_context(
            name="dev",
            kubeconfig_path=kc,
            server="https://c:6443",
            token="t",
            insecure_skip_tls_verify=True,
            validate=False,
        )


# ---------------------------------------------------------------------------
# import mode (file + raw yaml)
# ---------------------------------------------------------------------------
def test_import_from_file(tmp_path):
    src = tmp_path / "src"
    src.write_text(yaml.safe_dump(_one_context_config("imported")))
    result = save_kube_context(
        kubeconfig_path=str(tmp_path / "target"),
        source_file=str(src),
        validate=False,
    )
    assert result["mode"] == "import"
    assert result["contexts"] == ["imported"]


def test_import_from_raw_yaml(tmp_path):
    raw = yaml.safe_dump(_one_context_config("raw"))
    result = save_kube_context(
        kubeconfig_path=str(tmp_path / "target"),
        source_yaml=raw,
        validate=False,
    )
    assert result["contexts"] == ["raw"]


def test_import_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        save_kube_context(
            kubeconfig_path=str(tmp_path / "target"),
            source_file=str(tmp_path / "does-not-exist"),
            validate=False,
        )


# ---------------------------------------------------------------------------
# capture mode
# ---------------------------------------------------------------------------
def test_capture_from_current_kubeconfig_embeds_certs(tmp_path):
    ca = tmp_path / "ca.crt"
    ca.write_text("-----BEGIN CERTIFICATE-----\nQ0E=\n-----END CERTIFICATE-----")
    src = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": "c1",
                "cluster": {
                    "server": "https://s:6443",
                    "certificate-authority": str(ca),
                },
            }
        ],
        "users": [{"name": "u1", "user": {"token": "xyz"}}],
        "contexts": [
            {
                "name": "ctxA",
                "context": {"cluster": "c1", "user": "u1", "namespace": "ns1"},
            }
        ],
        "current-context": "ctxA",
    }
    kc = tmp_path / "config"
    kc.write_text(yaml.safe_dump(src))
    result = save_kube_context(
        name="exported",
        kubeconfig_path=str(kc),
        capture_current=True,
        source_context="ctxA",
        validate=False,
    )
    assert result["mode"] == "capture"
    cfg = _read(str(kc))
    cluster = next(c for c in cfg["clusters"] if c["name"] == "exported")["cluster"]
    assert "certificate-authority-data" in cluster
    assert "certificate-authority" not in cluster
    ctx = next(c for c in cfg["contexts"] if c["name"] == "exported")["context"]
    assert ctx["namespace"] == "ns1"


def test_capture_incluster(monkeypatch, tmp_path):
    sa_dir = tmp_path / "sa"
    sa_dir.mkdir()
    (sa_dir / "token").write_text("in-cluster-token")
    (sa_dir / "namespace").write_text("kube-system")
    (sa_dir / "ca.crt").write_text(
        "-----BEGIN CERTIFICATE-----\nQ0E=\n-----END CERTIFICATE-----"
    )
    monkeypatch.setattr(kci, "_SA_DIR", str(sa_dir))
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.43.0.1")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "443")
    result = save_kube_context(
        name="incluster",
        kubeconfig_path=str(tmp_path / "config"),
        capture_current=True,
        validate=False,
    )
    cfg = _read(result["path"])
    cluster = next(c for c in cfg["clusters"] if c["name"] == "incluster")["cluster"]
    assert cluster["server"] == "https://10.43.0.1:443"
    assert "incluster-token" not in str(result)  # token not leaked in summary


# ---------------------------------------------------------------------------
# oidc mode (username/password -> OIDC login)
# ---------------------------------------------------------------------------
def test_oidc_requires_issuer_and_client_id(tmp_path):
    with pytest.raises(ValueError, match="OIDC-backed"):
        save_kube_context(
            name="dev",
            kubeconfig_path=str(tmp_path / "config"),
            server="https://a:6443",
            username="alice",
            password="pw",
            validate=False,
        )


def test_oidc_requires_both_username_and_password(tmp_path):
    with pytest.raises(ValueError, match="both 'username' and 'password'"):
        save_kube_context(
            name="dev",
            kubeconfig_path=str(tmp_path / "config"),
            server="https://a:6443",
            username="alice",
            oidc_issuer="https://issuer",
            oidc_client_id="k8s",
            validate=False,
        )


def test_oidc_embeds_id_token(tmp_path):
    with patch.object(
        kci, "_oidc_password_grant", return_value="the-id-token"
    ) as grant:
        result = save_kube_context(
            name="oidc-dev",
            kubeconfig_path=str(tmp_path / "config"),
            server="https://a:6443",
            username="alice",
            password="pw",
            oidc_issuer="https://keycloak/realms/homelab",
            oidc_client_id="kubernetes",
            insecure_skip_tls_verify=True,
            validate=False,
        )
    assert result["mode"] == "oidc"
    grant.assert_called_once()
    cfg = _read(result["path"])
    user = next(u for u in cfg["users"] if u["name"] == "oidc-dev")["user"]
    assert user["token"] == "the-id-token"
    # password never leaks into the returned summary
    assert "pw" not in str(result["merged"])


def test_oidc_password_grant_discovers_and_posts():
    """The grant helper discovers the token endpoint then posts the password grant."""
    disc = MagicMock()
    disc.json.return_value = {"token_endpoint": "https://issuer/token"}
    disc.raise_for_status.return_value = None
    token_resp = MagicMock(status_code=200)
    token_resp.json.return_value = {"id_token": "abc.def.ghi"}
    fake_requests = MagicMock()
    fake_requests.get.return_value = disc
    fake_requests.post.return_value = token_resp
    with patch.dict("sys.modules", {"requests": fake_requests}):
        tok = kci._oidc_password_grant(
            issuer="https://issuer",
            client_id="k8s",
            username="alice",
            password="pw",
            insecure=True,
        )
    assert tok == "abc.def.ghi"
    assert fake_requests.get.call_args[0][0] == (
        "https://issuer/.well-known/openid-configuration"
    )
    posted = fake_requests.post.call_args
    assert posted[0][0] == "https://issuer/token"
    assert posted[1]["data"]["grant_type"] == "password"
    assert posted[1]["verify"] is False


def test_oidc_password_grant_rejected_raises_clear():
    disc = MagicMock()
    disc.json.return_value = {"token_endpoint": "https://issuer/token"}
    disc.raise_for_status.return_value = None
    token_resp = MagicMock(status_code=401)
    token_resp.json.return_value = {
        "error": "invalid_grant",
        "error_description": "bad creds",
    }
    fake_requests = MagicMock()
    fake_requests.get.return_value = disc
    fake_requests.post.return_value = token_resp
    with patch.dict("sys.modules", {"requests": fake_requests}):
        with pytest.raises(RuntimeError, match="password grant rejected"):
            kci._oidc_password_grant(
                issuer="https://issuer",
                client_id="k8s",
                username="alice",
                password="pw",
            )


# ---------------------------------------------------------------------------
# use + validate-after-save
# ---------------------------------------------------------------------------
def test_use_sets_current_context(tmp_path):
    kc = str(tmp_path / "config")
    save_kube_context(
        name="dev",
        kubeconfig_path=kc,
        server="https://a:6443",
        token="t",
        insecure_skip_tls_verify=True,
        validate=False,
    )
    result = save_kube_context(
        name="prod",
        kubeconfig_path=kc,
        server="https://b:6443",
        token="t",
        insecure_skip_tls_verify=True,
        use=True,
        validate=False,
    )
    assert result["used"] is True
    assert _read(kc)["current-context"] == "prod"


def test_validate_after_save_reports_node_count(tmp_path):
    fake_manager = MagicMock()
    fake_manager.list_nodes.return_value = [{"hostname": "n1"}, {"hostname": "n2"}]
    with patch(
        "container_manager_mcp.container_manager.create_manager",
        return_value=fake_manager,
    ) as cm:
        result = save_kube_context(
            name="dev",
            kubeconfig_path=str(tmp_path / "config"),
            server="https://a:6443",
            token="t",
            insecure_skip_tls_verify=True,
            validate=True,
        )
    cm.assert_called_once_with("kubernetes", host="dev")
    assert result["validation"]["status"] == "ok"
    assert result["validation"]["node_count"] == 2


def test_validate_after_save_reports_unreachable(tmp_path):
    with patch(
        "container_manager_mcp.container_manager.create_manager",
        side_effect=RuntimeError("no cluster"),
    ):
        result = save_kube_context(
            name="dev",
            kubeconfig_path=str(tmp_path / "config"),
            server="https://a:6443",
            token="t",
            insecure_skip_tls_verify=True,
            validate=True,
        )
    assert result["validation"]["status"] == "unreachable"
    assert "no cluster" in result["validation"]["error"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_explicit(tmp_path, capsys):
    kc = tmp_path / "config"
    rc = kci.main(
        [
            "--name",
            "clidev",
            "--kubeconfig",
            str(kc),
            "--server",
            "https://1.2.3.4:6443",
            "--token",
            "secret",
            "--insecure-skip-tls-verify",
            "--use",
            "--no-validate",
        ]
    )
    assert rc == 0
    assert _ctx_names(_read(str(kc))) == ["clidev"]
    out = capsys.readouterr().out
    assert "clidev" in out
    assert "secret" not in out  # token never printed


def test_cli_collision_returns_error(tmp_path):
    kc = str(tmp_path / "config")
    common = [
        "--kubeconfig",
        kc,
        "--server",
        "https://h:6443",
        "--token",
        "t",
        "--insecure-skip-tls-verify",
        "--no-validate",
    ]
    assert kci.main(["--name", "dev", *common]) == 0
    assert kci.main(["--name", "dev", *common]) == 1  # collision -> exit 1
    assert kci.main(["--name", "dev", "--overwrite", *common]) == 0  # overwrite ok

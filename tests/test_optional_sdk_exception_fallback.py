#!/usr/bin/env python
"""Regression tests for the optional-import exception fallback bug (GH issue #3).

``container_manager.py`` (docker, podman) and ``k8s_manager.py`` (kubernetes)
each guard an optional SDK import with ``try/except ImportError`` and bind a
placeholder for the SDK's exception class so the rest of the module can still
reference ``except DockerException:`` / ``except PodmanError:`` / ``except
_km.ApiException:`` even when the corresponding SDK isn't installed. The
placeholder used to be the bare ``Exception`` class::

    except ImportError:
        docker = None
        DockerException = Exception   # <- the bug

which made every one of those ``except`` clauses catch *all* errors -- not
just the SDK's own -- whenever the SDK was absent, silently masking unrelated
bugs. Reported by Elshayib.

The fix binds the placeholder to a dedicated, narrow ``_MissingRuntimeError``
class instead (defined once per module), which no unrelated error can ever be
an instance of.

Two kinds of coverage:

1. ``test_fallback_binds_to_dedicated_placeholder_not_bare_exception`` --
   forces the SDK import to *genuinely* fail (a fresh subprocess with the
   package poisoned in ``sys.modules``, so the real ``except ImportError:``
   branch executes) and asserts the guarded name resolves to the dedicated
   placeholder, never to ``Exception``. Run in a subprocess rather than an
   in-process ``importlib.reload`` because reloading ``container_manager.py``
   / ``k8s_manager.py`` in-process would rebind ``ContainerManagerBase`` and
   the guarded symbols for the rest of the test session -- other
   already-imported modules (e.g. the ``k8s`` mixins, which hold a reference
   to ``ContainerManagerBase`` from their own import time) would end up with
   a stale class object, corrupting isinstance/MRO checks in unrelated tests.
2. The ``Test*UnrelatedErrorPropagates`` classes below exercise real
   application code paths (``PodmanManager.__init__``,
   ``KubernetesManager.list_nodes``) whose ``except`` clause names the
   guarded symbol with no broader ``except Exception`` fallback beside it.
   They monkeypatch the already-imported module's guarded name directly
   (the same technique the rest of this test suite already uses to simulate
   "SDK present" - see ``tests/test_k8s_ops.py``), which is safe for the
   whole test session because ``unittest.mock.patch`` restores the original
   value on context exit. This proves an unrelated error is NOT swallowed
   when the SDK-absent placeholder is active, while a real SDK exception is
   still caught/wrapped exactly as before when the SDK is present.
"""

import importlib
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

import container_manager_mcp.k8s_manager as km
from container_manager_mcp.container_manager import PodmanManager
from container_manager_mcp.k8s_manager import KubernetesManager

# NOT `import container_manager_mcp.container_manager as cm`: the package's
# __init__.py re-exports every public name from container_manager.py (incl.
# the `container_manager` CLI function) onto the *package*, which shadows the
# `container_manager` submodule attribute that `import ... as` would resolve
# through. importlib.import_module reads sys.modules directly and sidesteps
# that (see the module docstring above for the same reasoning applied to the
# subprocess probe).
cm = importlib.import_module("container_manager_mcp.container_manager")

# ---------------------------------------------------------------------------
# 1. The fallback binding itself, with the SDK import genuinely forced absent.
# ---------------------------------------------------------------------------

# Poisons `sys.modules[sdk_module_name]` with None (the standard way to force
# `import <name>` to raise ImportError deterministically), then imports the
# target module fresh via importlib -- NOT `import container_manager_mcp.foo
# as m`, because container_manager_mcp/__init__.py re-exports every public
# name from container_manager.py (including the `container_manager` CLI
# function) onto the *package*, which shadows the `container_manager`
# submodule attribute; importlib.import_module reads sys.modules directly and
# sidesteps that.
_SUBPROCESS_SCRIPT = """
import sys, importlib
sdk_module_name, target_module, attr = sys.argv[1:4]
sys.modules[sdk_module_name] = None
m = importlib.import_module(target_module)
cls = getattr(m, attr)
assert cls is not Exception, (
    f"{attr} fell back to bare Exception when {sdk_module_name} is absent -- "
    f"this makes 'except {attr}:' catch every error, masking unrelated bugs "
    "(GH issue #3)"
)
assert cls is m._MissingRuntimeError, (
    f"expected the dedicated _MissingRuntimeError placeholder, got {cls!r}"
)
assert issubclass(cls, Exception)
assert not isinstance(ValueError("unrelated bug"), cls), (
    "an unrelated ValueError must not be an instance of the SDK-absent placeholder"
)
print("OK")
"""


@pytest.mark.parametrize(
    ("sdk_module_name", "target_module", "attr"),
    [
        ("docker", "container_manager_mcp.container_manager", "DockerException"),
        ("podman", "container_manager_mcp.container_manager", "PodmanError"),
        ("kubernetes", "container_manager_mcp.k8s_manager", "ApiException"),
    ],
    ids=["docker-DockerException", "podman-PodmanError", "kubernetes-ApiException"],
)
def test_fallback_binds_to_dedicated_placeholder_not_bare_exception(
    sdk_module_name, target_module, attr
):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _SUBPROCESS_SCRIPT,
            sdk_module_name,
            target_module,
            attr,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        f"SDK-absent fallback check failed for {target_module}.{attr}:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# 2a. Podman: PodmanManager.__init__'s `except PodmanError as e:` has no
#     broader `except Exception` fallback beside it, so it cleanly
#     demonstrates swallowed-vs-propagated.
# ---------------------------------------------------------------------------


class TestPodmanErrorFallbackDoesNotMaskUnrelatedErrors:
    @patch.object(
        PodmanManager,
        "_autodetect_podman_url",
        return_value="unix:///tmp/podman-test.sock",
    )
    @patch("container_manager_mcp.container_manager.PodmanClient")
    def test_unrelated_error_propagates_when_sdk_absent(
        self, mock_client_cls, _mock_autodetect
    ):
        """SDK-absent state: PodmanError bound to the narrow placeholder (what
        the fixed ImportError fallback now produces). An unrelated bug
        (TypeError) raised while connecting must propagate untouched -- NOT
        get caught by `except PodmanError` and rewrapped into a misleading
        'Failed to connect to Podman' RuntimeError.
        """
        mock_client_cls.side_effect = TypeError(
            "totally unrelated bug, not a Podman error"
        )
        with patch(
            "container_manager_mcp.container_manager.PodmanError",
            cm._MissingRuntimeError,
        ):
            with pytest.raises(TypeError, match="totally unrelated bug"):
                PodmanManager(silent=True)

    @patch.object(
        PodmanManager,
        "_autodetect_podman_url",
        return_value="unix:///tmp/podman-test.sock",
    )
    @patch("container_manager_mcp.container_manager.PodmanClient")
    def test_real_podman_error_still_caught_when_sdk_present(
        self, mock_client_cls, _mock_autodetect
    ):
        """SDK-present state (unaffected by the fix): a genuine PodmanError
        raised while connecting is still caught and wrapped into a
        RuntimeError, exactly as before the fix.
        """

        class _FakePodmanError(Exception):
            pass

        mock_client_cls.side_effect = _FakePodmanError("daemon not reachable")
        with patch(
            "container_manager_mcp.container_manager.PodmanError", _FakePodmanError
        ):
            # The genuine-PodmanError path deliberately raises a NON-LEAKING
            # message ("Configured Podman daemon is unavailable") and chains the
            # real error with `from e`, rather than interpolating the daemon's
            # own text into the message. "Failed to connect to Podman" belongs to
            # the separate no-socket-found path. Assert the real contract, and
            # assert the cause is preserved so the detail is still recoverable.
            with pytest.raises(
                RuntimeError, match="Configured Podman daemon is unavailable"
            ) as excinfo:
                PodmanManager(silent=True)
            assert isinstance(excinfo.value.__cause__, _FakePodmanError)


# ---------------------------------------------------------------------------
# 2b. Kubernetes: KubernetesManager.list_nodes' `except _km.ApiException as e:`
#     also has no broader fallback beside it -- a real `list_*` method.
# ---------------------------------------------------------------------------


def _make_k8s_manager() -> KubernetesManager:
    """Construct a KubernetesManager with the k8s client mocked (not absent),
    independent of what ApiException is bound to.
    """
    manager = KubernetesManager(namespace="default")
    manager.core = MagicMock()
    return manager


class TestApiExceptionFallbackDoesNotMaskUnrelatedErrors:
    def test_unrelated_error_in_list_nodes_propagates_when_sdk_absent(self):
        """SDK-absent state: ApiException bound to the narrow placeholder.
        An unrelated bug (KeyError) raised by the (mocked) API call inside
        `list_nodes` -- a real `list_*` method -- must propagate untouched,
        not be caught by `except _km.ApiException`.
        """
        with (
            patch.object(km, "k8s_client", MagicMock()),
            patch.object(km, "k8s_config", MagicMock()),
            patch.object(km, "ApiException", km._MissingRuntimeError),
        ):
            manager = _make_k8s_manager()
            manager.core.list_node.side_effect = KeyError(
                "unrelated bug, not a Kubernetes API error"
            )
            with pytest.raises(KeyError, match="unrelated bug"):
                manager.list_nodes()

    def test_real_api_exception_still_caught_when_sdk_present(self):
        """SDK-present state (unaffected by the fix): a genuine ApiException
        raised by the API call is still caught and wrapped into a
        RuntimeError, exactly as before the fix.
        """

        class _FakeApiException(Exception):
            pass

        with (
            patch.object(km, "k8s_client", MagicMock()),
            patch.object(km, "k8s_config", MagicMock()),
            patch.object(km, "ApiException", _FakeApiException),
        ):
            manager = _make_k8s_manager()
            manager.core.list_node.side_effect = _FakeApiException(
                "cluster unreachable"
            )
            with pytest.raises(RuntimeError, match="Failed to list nodes"):
                manager.list_nodes()

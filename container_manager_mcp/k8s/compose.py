"""ComposeMixin for KubernetesManager (split from k8s_manager.py)."""

import os
import subprocess
import tempfile


class ComposeMixin:
    def compose_up(
        self, compose_file: str, detach: bool = True, build: bool = False
    ) -> str:
        params = {"compose_file": compose_file, "detach": detach, "build": build}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "manifests.yaml")
                convert = subprocess.run(
                    ["kompose", "convert", "-f", compose_file, "-o", out],
                    capture_output=True,
                    text=True,
                )
                if convert.returncode != 0:
                    raise RuntimeError(convert.stderr)
                apply = subprocess.run(
                    ["kubectl", "apply", "-n", self.namespace, "-f", out],
                    capture_output=True,
                    text=True,
                )
                if apply.returncode != 0:
                    raise RuntimeError(apply.stderr)
                self.log_action("compose_up", params, apply.stdout)
                return apply.stdout
        except Exception as e:
            self.log_action("compose_up", params, error=e)
            raise RuntimeError("Failed to apply manifests") from e

    def compose_down(self, compose_file: str) -> str:
        params = {"compose_file": compose_file}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "manifests.yaml")
                convert = subprocess.run(
                    ["kompose", "convert", "-f", compose_file, "-o", out],
                    capture_output=True,
                    text=True,
                )
                if convert.returncode != 0:
                    raise RuntimeError(convert.stderr)
                delete = subprocess.run(
                    ["kubectl", "delete", "-n", self.namespace, "-f", out],
                    capture_output=True,
                    text=True,
                )
                if delete.returncode != 0:
                    raise RuntimeError(delete.stderr)
                self.log_action("compose_down", params, delete.stdout)
                return delete.stdout
        except Exception as e:
            self.log_action("compose_down", params, error=e)
            raise RuntimeError("Failed to delete manifests") from e

    def compose_ps(self, compose_file: str) -> str:
        params = {"compose_file": compose_file}
        try:
            result = subprocess.run(
                ["kubectl", "get", "all", "-n", self.namespace],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_ps", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_ps", params, error=e)
            raise RuntimeError("Failed to list resources") from e

    def compose_logs(self, compose_file: str, service: str | None = None) -> str:
        params = {"compose_file": compose_file, "service": service}
        try:
            cmd = ["kubectl", "logs", "-n", self.namespace]
            cmd += [f"deploy/{service}"] if service else ["--all-containers"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            self.log_action("compose_logs", params, result.stdout)
            return result.stdout
        except Exception as e:
            self.log_action("compose_logs", params, error=e)
            raise RuntimeError("Failed to get logs") from e

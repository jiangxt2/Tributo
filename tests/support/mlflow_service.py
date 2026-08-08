"""Owned real MLflow service for registry integration tests."""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from contextlib import suppress
from dataclasses import dataclass

MLFLOW_IMAGE = os.environ.get(
    "TRIBUTO_MLFLOW_IMAGE",
    "ghcr.io/mlflow/mlflow@sha256:09d25b0c80efa2c1dd4b5e4834a1a0ed861d6b1c2b43017ff99493bd5c6bb4ee",
)


@dataclass
class MLflowService:
    """A test-owned MLflow tracking and model-registry server."""

    tracking_uri: str
    container_name: str

    @classmethod
    def start(cls) -> "MLflowService":
        """Start a pinned disposable server on an ephemeral host port."""
        container_name = f"tributo-mlflow-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            subprocess.run(
                ["docker", "info"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--rm",
                    "--name",
                    container_name,
                    "--publish",
                    "127.0.0.1::5000",
                    MLFLOW_IMAGE,
                    "mlflow",
                    "server",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "5000",
                    "--backend-store-uri",
                    "sqlite:////tmp/tributo-mlflow.db",
                    "--artifacts-destination",
                    "file:///tmp/tributo-artifacts",
                    "--serve-artifacts",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            port = cls._host_port(container_name)
            service = cls(
                tracking_uri=f"http://127.0.0.1:{port}",
                container_name=container_name,
            )
            service._wait_until_healthy()
            return service
        except Exception:
            cls._logs(container_name)
            cls._stop(container_name)
            raise

    def close(self) -> None:
        """Stop only the container owned by this fixture."""
        container_name = self.container_name
        self.container_name = ""
        if container_name:
            self._stop(container_name)

    @staticmethod
    def _host_port(container_name: str) -> str:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                '{{(index (index .NetworkSettings.Ports "5000/tcp") 0).HostPort}}',
                container_name,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        port = result.stdout.strip()
        if not port:
            raise RuntimeError("Docker did not publish the MLflow test port")
        return port

    def _wait_until_healthy(self) -> None:
        health_url = f"{self.tracking_uri}/health"
        deadline = time.monotonic() + 60
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
            time.sleep(1)
        raise RuntimeError(
            f"MLflow did not become healthy at {health_url}: {last_error}"
        )

    @staticmethod
    def _logs(container_name: str) -> None:
        with suppress(Exception):
            subprocess.run(
                ["docker", "logs", container_name],
                check=False,
                timeout=15,
            )

    @staticmethod
    def _stop(container_name: str) -> None:
        with suppress(Exception):
            subprocess.run(
                ["docker", "stop", "--time", "10", container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )


__all__ = ["MLflowService"]

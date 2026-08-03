"""Lifecycle helpers for portable object-storage tests.

The default S3 contract backend is an in-process Moto server. MinIO
compatibility tests use a pinned container image and an ephemeral host port,
so they never depend on a developer's container name, port, or network.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

MINIO_IMAGE = os.environ.get(
    "TRIBUTO_MINIO_IMAGE",
    "minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e",
)
MINIO_ACCESS_KEY_ID = "minioadmin"
MINIO_SECRET_ACCESS_KEY = "minioadmin123"


class S3InfrastructureUnavailable(RuntimeError):
    """Raised when the host cannot provide a requested test S3 service."""


@dataclass
class S3Service:
    """A test S3 endpoint and its owned lifecycle resources."""

    endpoint: str
    moto_server: Any | None = None
    container_name: str | None = None

    @classmethod
    def start_contract(cls) -> S3Service:
        """Start Moto unless an explicit endpoint was supplied."""
        endpoint = os.environ.get("TRIBUTO_S3_ENDPOINT")
        if endpoint:
            return cls(endpoint=endpoint)

        try:
            from moto.server import ThreadedMotoServer
        except ImportError as exc:  # pragma: no cover - locked dev dependency
            raise RuntimeError(
                "S3 contract tests require moto[server]. "
                "Run uv sync --extra dev --locked."
            ) from exc

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
        except OSError as exc:
            raise S3InfrastructureUnavailable(
                "The host does not permit binding an in-process S3 contract "
                "server; run this gate on a host with loopback sockets or "
                "set TRIBUTO_S3_ENDPOINT."
            ) from exc

        server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
        try:
            server.start()
        except OSError as exc:
            with suppress(Exception):
                server.stop()
            raise S3InfrastructureUnavailable(
                "The host does not permit binding an in-process S3 contract "
                "server; run this gate on a host with loopback sockets or "
                "set TRIBUTO_S3_ENDPOINT."
            ) from exc
        host, port = server.get_host_and_port()
        return cls(
            endpoint=f"http://{host}:{port}",
            moto_server=server,
        )

    @classmethod
    def start_minio(cls) -> S3Service:
        """Use an explicit endpoint or start an owned pinned MinIO container."""
        endpoint = os.environ.get("TRIBUTO_MINIO_ENDPOINT")
        if endpoint:
            return cls(endpoint=endpoint)

        container_name = f"tributo-minio-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            cls._run_docker_info()
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--rm",
                    "--name",
                    container_name,
                    "--publish",
                    "127.0.0.1::9000",
                    "--env",
                    f"MINIO_ROOT_USER={MINIO_ACCESS_KEY_ID}",
                    "--env",
                    f"MINIO_ROOT_PASSWORD={MINIO_SECRET_ACCESS_KEY}",
                    MINIO_IMAGE,
                    "server",
                    "/data",
                    "--console-address",
                    ":9001",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            host_port = cls._container_host_port(container_name)
            service = cls(
                endpoint=f"http://127.0.0.1:{host_port}",
                container_name=container_name,
            )
            service._wait_until_healthy()
            return service
        except FileNotFoundError as exc:
            raise S3InfrastructureUnavailable(
                "MinIO compatibility tests require the Docker CLI. "
                "Install Docker Desktop or set TRIBUTO_MINIO_ENDPOINT."
            ) from exc
        except S3InfrastructureUnavailable:
            cls._stop_container(container_name)
            raise
        except subprocess.CalledProcessError as exc:
            cls._print_container_logs(container_name)
            detail = (exc.stderr or exc.stdout or "").strip()
            cls._stop_container(container_name)
            raise RuntimeError(
                "Failed to start the pinned MinIO compatibility service"
                + (f": {detail}" if detail else ".")
            ) from exc
        except Exception:
            cls._print_container_logs(container_name)
            cls._stop_container(container_name)
            raise

    def close(self) -> None:
        """Release only resources owned by this test service."""
        moto_server = self.moto_server
        container_name = self.container_name
        self.moto_server = None
        self.container_name = None
        try:
            if moto_server is not None:
                moto_server.stop()
        finally:
            if container_name is not None:
                self._stop_container(container_name)

    @staticmethod
    def _run_docker_info() -> None:
        try:
            result = subprocess.run(
                ["docker", "info"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except FileNotFoundError as exc:
            raise S3InfrastructureUnavailable(
                "MinIO compatibility tests require the Docker CLI. "
                "Install Docker Desktop or set TRIBUTO_MINIO_ENDPOINT."
            ) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise S3InfrastructureUnavailable(
                "Docker is unavailable for MinIO compatibility tests"
                + (f": {detail}" if detail else ".")
            )

    @staticmethod
    def _container_host_port(container_name: str) -> str:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                '{{(index (index .NetworkSettings.Ports "9000/tcp") 0).HostPort}}',
                container_name,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        host_port = result.stdout.strip()
        if not host_port:
            raise RuntimeError(
                f"Docker did not publish MinIO port for {container_name}"
            )
        return host_port

    def _wait_until_healthy(self) -> None:
        health_url = f"{self.endpoint}/minio/health/live"
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
            f"MinIO did not become healthy at {health_url}"
            + (f": {last_error}" if last_error else ".")
        )

    @staticmethod
    def _print_container_logs(container_name: str) -> None:
        with suppress(FileNotFoundError, subprocess.TimeoutExpired):
            subprocess.run(
                ["docker", "logs", container_name],
                check=False,
                capture_output=False,
                timeout=15,
            )

    @staticmethod
    def _stop_container(container_name: str) -> None:
        with suppress(FileNotFoundError, subprocess.TimeoutExpired):
            subprocess.run(
                ["docker", "stop", "--time", "10", container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )

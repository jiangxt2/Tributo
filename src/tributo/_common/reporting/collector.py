"""Own exactly one pre-provisioned Collector and its private configuration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from tributo._common.reporting.config import (
    ReportingConfig,
    ReportingError,
    resolve_reference,
)


class ManagedCollector:
    """Only the creator may stop this subprocess; never manage an external PID."""

    def __init__(
        self,
        config: ReportingConfig,
        diagnose: Callable[[str], None],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self._diagnose = diagnose
        self._environment = environment
        self.process: subprocess.Popen[str] | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._ready = threading.Event()
        self._reader: threading.Thread | None = None

    def start(self, deadline: float) -> None:
        import requests

        config = self.config.collector
        profile = self.config.write_profile
        if config is None or profile is None:
            raise ReportingError("reporting.collector.config")
        binary = Path(config.binary)
        try:
            if (
                not binary.is_absolute()
                or not binary.is_file()
                or not os.access(binary, os.X_OK)
            ):
                raise ValueError
            digest = hashlib.sha256()
            with binary.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != config.sha256:
                raise ValueError
            for arguments, terms in (
                (["--version"], [config.version]),
                (
                    ["components"],
                    ["otlp", "doris", "health_check", "memory_limiter", "file", "env"],
                ),
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError
                result = subprocess.run(
                    [str(binary), *arguments],
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                    check=False,
                )
                if (
                    result.returncode
                    or len(result.stdout) > 1024 * 1024
                    or any(
                        not re.search(r"\b" + re.escape(t) + r"\b", result.stdout)
                        for t in terms
                    )
                ):
                    raise ValueError
            self._directory = tempfile.TemporaryDirectory(prefix="tributo-reporting-")
            path = Path(self._directory.name) / "collector.json"
            path.write_text(json.dumps(self._pipeline()), encoding="utf-8")
            path.chmod(0o600)
            child_env = {
                "PATH": os.defpath,
                "TRIBUTO_REPORTING_DORIS_USER": resolve_reference(
                    profile.username_ref, environment=self._environment
                ),
                "TRIBUTO_REPORTING_DORIS_PASSWORD": resolve_reference(
                    profile.password_ref, empty=True, environment=self._environment
                ),
            }
            self.process = subprocess.Popen(
                [str(binary), "--config", "file:" + str(path)],
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._reader = threading.Thread(
                target=self._read_diagnostics,
                name="tributo-collector-diagnostics",
                daemon=True,
            )
            self._reader.start()
            with requests.Session() as health:
                health.trust_env = False
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise ValueError
                    if self._ready.wait(timeout=0.05):
                        try:
                            response = health.get(
                                "http://" + config.health_bind,
                                timeout=min(1, max(0.01, deadline - time.monotonic())),
                                allow_redirects=False,
                            )
                            try:
                                if (
                                    response.status_code == 200
                                    and self.process.poll() is None
                                ):
                                    return
                            finally:
                                response.close()
                        except requests.RequestException:
                            pass
                raise ValueError
        except (OSError, ValueError, subprocess.SubprocessError):
            self.close()
            raise ReportingError("reporting.collector.start_failed") from None

    def _pipeline(self) -> dict[str, Any]:
        config, profile = self.config.collector, self.config.write_profile
        if config is None or profile is None:
            raise ReportingError("reporting.collector.config")
        receiver: dict[str, Any] = {
            "endpoint": config.otlp_bind,
            "max_request_body_size": self.config.sender.max_request_bytes,
        }
        if config.tls_certificate:
            tls = {"cert_file": config.tls_certificate, "key_file": config.tls_key}
            if config.client_ca_file:
                tls["client_ca_file"] = config.client_ca_file
            receiver["tls"] = tls
        exporter: dict[str, Any] = {
            "endpoint": profile.http_endpoint,
            "database": profile.database,
            "table": {"logs": profile.table},
            "username": "${env:TRIBUTO_REPORTING_DORIS_USER}",
            "password": "${env:TRIBUTO_REPORTING_DORIS_PASSWORD}",
            "create_schema": False,
            "log_response": False,
            "log_progress_interval": 0,
            "timeout": f"{self.config.sender.send_timeout}s",
            "retry_on_failure": {
                "enabled": True,
                "initial_interval": "1s",
                "max_interval": "2s",
                "max_elapsed_time": f"{self.config.sender.send_timeout}s",
            },
            "sending_queue": {
                "enabled": True,
                "num_consumers": 1,
                "queue_size": self.config.sender.queue_size,
                "batch": {
                    "flush_timeout": f"{self.config.sender.interval_seconds}s",
                    "min_size": self.config.sender.batch_size,
                    "max_size": self.config.sender.batch_size,
                },
            },
        }
        if profile.ca_file:
            exporter["tls"] = {"ca_file": profile.ca_file}
        return {
            "receivers": {"otlp": {"protocols": {"http": receiver}}},
            "processors": {
                "memory_limiter": {
                    "check_interval": "1s",
                    "limit_mib": config.memory_limit_mib,
                    "spike_limit_mib": max(8, config.memory_limit_mib // 4),
                }
            },
            "exporters": {"doris": exporter},
            "extensions": {"health_check": {"endpoint": config.health_bind}},
            "service": {
                "extensions": ["health_check"],
                "pipelines": {
                    "logs": {
                        "receivers": ["otlp"],
                        "processors": ["memory_limiter"],
                        "exporters": ["doris"],
                    }
                },
                "telemetry": {
                    "logs": {"encoding": "json", "level": "info"},
                    "metrics": {"level": "none"},
                },
            },
        }

    def _read_diagnostics(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        # Consume bounded lines; raw exporter errors may contain sensitive data.
        while line := process.stderr.readline(8192):
            if not line.endswith("\n"):
                while remainder := process.stderr.readline(8192):
                    if remainder.endswith("\n"):
                        break
                self._diagnose("collector.diagnostic_truncated")
                continue
            try:
                record = json.loads(line)
            except ValueError:
                self._diagnose("collector.diagnostic_redacted")
                continue
            if not isinstance(record, dict):
                self._diagnose("collector.diagnostic_redacted")
                continue
            if (
                record.get("msg")
                == "Everything is ready. Begin running and processing data."
            ):
                self._ready.set()
            elif record.get("level") in {"warn", "error", "fatal"}:
                self._diagnose(
                    "collector.warning"
                    if record["level"] == "warn"
                    else "collector.failure"
                )

    def close(self, timeout: float | None = None) -> bool:
        process = self.process
        drained = True
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(
                        timeout=self.config.sender.shutdown_timeout
                        if timeout is None
                        else max(0, timeout)
                    )
                except subprocess.TimeoutExpired:
                    drained = False
                    process.kill()
                    process.wait(timeout=2)
            if self._reader is not None:
                self._reader.join(timeout=1)
            if process.stderr is not None:
                process.stderr.close()
            self.process = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
        return drained

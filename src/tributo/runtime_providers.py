"""Ray runtime lifecycle providers.

Providers own only the lifecycle of a Ray runtime. They do not implement Ray
scheduling or any data-plane reader/writer behavior.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable
from urllib.parse import urlsplit

import requests
from ray.job_submission import JobSubmissionClient

from tributo.exceptions import JobConfigurationError, JobExecutionError
from tributo.runtime import RuntimeTarget
from tributo.util.annotations import DeveloperAPI, PublicAPI


@PublicAPI(stability="beta")
@dataclass
class RuntimeLease:
    """A provider-owned Ray endpoint and its idempotent release operation."""

    address: str
    _release: Callable[[], None]
    _closed: bool = False

    def close(self) -> None:
        """Release the provider-owned runtime exactly once."""
        if self._closed:
            return
        self._closed = True
        self._release()

    def __enter__(self) -> RuntimeLease:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, traceback
        try:
            self.close()
        except BaseException as cleanup_error:
            if isinstance(exc, BaseException):
                exc.add_note(
                    f"Ray runtime cleanup failed: {type(cleanup_error).__name__}"
                )
                return
            raise


@runtime_checkable
@DeveloperAPI
class RuntimeProvider(Protocol):
    """Provider protocol for explicitly owned Ray runtime lifecycles."""

    name: str

    def provision(self, target: RuntimeTarget) -> RuntimeLease:
        """Create a runtime and return its Ray Jobs endpoint."""


_PROVIDERS: dict[str, RuntimeProvider] = {}


@DeveloperAPI
def register_runtime_provider(name: str, provider: RuntimeProvider) -> None:
    """Register one named runtime provider."""
    normalized = name.strip()
    if not normalized:
        raise JobConfigurationError("runtime provider name must be non-empty")
    if not isinstance(provider, RuntimeProvider):
        raise JobConfigurationError("runtime provider must implement provision(target)")
    if normalized in _PROVIDERS:
        raise JobConfigurationError(
            f"runtime provider is already registered: {normalized}"
        )
    _PROVIDERS[normalized] = provider


@DeveloperAPI
def resolve_runtime_provider(name: str) -> RuntimeProvider:
    """Resolve one explicitly registered provider or fail closed."""
    try:
        return _PROVIDERS[name]
    except KeyError as exc:
        raise JobConfigurationError(
            f"runtime provider is not registered: {name}"
        ) from exc


def _run_provider_command(args: list[str], *, check: bool = True) -> None:
    """Run a provider command without copying its output into diagnostics."""
    try:
        subprocess.run(
            args,
            check=check,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        if check:
            raise JobExecutionError(
                f"Ray runtime provider command failed: {args[0]}"
            ) from exc


@DeveloperAPI
class LocalRayJobsProvider:
    """Own a local Ray runtime and expose its Jobs dashboard endpoint."""

    name = "local"

    def provision(self, target: RuntimeTarget) -> RuntimeLease:
        """Start one local Ray runtime for a synchronous Jobs submission."""
        if target.provider != self.name:
            raise JobConfigurationError("local provider received a non-local target")
        import ray

        if ray.is_initialized():
            raise JobConfigurationError(
                "local provider cannot reuse an already initialized Ray runtime"
            )
        try:
            context = ray.init(address="local", include_dashboard=True)
            dashboard_url = str(getattr(context, "dashboard_url", "") or "")
            if not dashboard_url:
                raise JobExecutionError(
                    "local Ray runtime did not expose a dashboard endpoint"
                )
            if not dashboard_url.startswith(("http://", "https://")):
                dashboard_url = f"http://{dashboard_url}"
        except BaseException:
            ray.shutdown()
            raise
        return RuntimeLease(dashboard_url.rstrip("/"), ray.shutdown)


@DeveloperAPI
class RayClusterLauncherProvider:
    """Provision a Ray cluster through the official ``ray up/down`` CLI.

    The provider config is a local JSON control file containing a
    ``cluster_config`` path for Ray Cluster Launcher and a reachable
    ``dashboard_url``. The referenced Ray cluster YAML belongs to the external
    provider boundary and is never parsed as a Tributo workload config.
    """

    name = "ray_cluster_launcher"

    def provision(self, target: RuntimeTarget) -> RuntimeLease:
        """Create the configured cluster and wait for its Jobs endpoint."""
        if target.provider != self.name or not target.provider_config:
            raise JobConfigurationError(
                f"{self.name} requires managed provider configuration"
            )
        config_path = Path(target.provider_config)
        payload = self._load_config(config_path)
        backend = self._required_string(payload, "backend").lower()
        if backend in {"kubernetes", "k8s"}:
            raise JobConfigurationError(
                "Kubernetes lifecycle must be managed by KubeRay externally"
            )
        cluster_config = self._required_string(payload, "cluster_config")
        cluster_config_path = Path(cluster_config)
        if not cluster_config_path.is_absolute():
            cluster_config = str(config_path.parent / cluster_config_path)
        dashboard_url = self._required_string(payload, "dashboard_url").rstrip("/")
        try:
            parsed_dashboard = urlsplit(dashboard_url)
        except ValueError as exc:
            raise JobConfigurationError(
                "runtime provider dashboard_url is not a valid URL"
            ) from exc
        if (
            parsed_dashboard.scheme not in {"http", "https"}
            or not parsed_dashboard.netloc
        ):
            raise JobConfigurationError(
                "runtime provider dashboard_url must be an http(s) URL"
            )
        if (
            parsed_dashboard.username is not None
            or parsed_dashboard.password is not None
        ):
            raise JobConfigurationError(
                "runtime provider dashboard_url must not embed credentials"
            )
        ray_command = self._optional_string(payload, "ray_command", default="ray")
        timeout_seconds = self._positive_float(
            payload, "ready_timeout_seconds", default=300
        )
        poll_seconds = self._positive_float(payload, "ready_poll_seconds", default=2)

        started = False
        try:
            _run_provider_command([ray_command, "up", "-y", cluster_config])
            started = True
            self._wait_for_dashboard(
                dashboard_url,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
            )
        except BaseException as primary_error:
            if started:
                try:
                    _run_provider_command([ray_command, "down", "-y", cluster_config])
                except BaseException as cleanup_error:
                    primary_error.add_note(
                        "Ray runtime provider cleanup failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            raise

        def release() -> None:
            _run_provider_command([ray_command, "down", "-y", cluster_config])

        return RuntimeLease(dashboard_url, release)

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise JobConfigurationError(
                f"unable to read runtime provider config: {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise JobConfigurationError("runtime provider config must be a JSON object")
        return payload

    @staticmethod
    def _required_string(payload: dict[str, Any], name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value.strip():
            raise JobConfigurationError(
                f"runtime provider config requires non-empty {name!r}"
            )
        return value.strip()

    @staticmethod
    def _optional_string(payload: dict[str, Any], name: str, *, default: str) -> str:
        value = payload.get(name, default)
        if not isinstance(value, str) or not value.strip():
            raise JobConfigurationError(
                f"runtime provider config field {name!r} must be a non-empty string"
            )
        return value.strip()

    @staticmethod
    def _positive_float(payload: dict[str, Any], name: str, *, default: float) -> float:
        value = payload.get(name, default)
        if isinstance(value, bool):
            raise JobConfigurationError(
                f"runtime provider config field {name!r} must be positive"
            )
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise JobConfigurationError(
                f"runtime provider config field {name!r} must be positive"
            ) from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise JobConfigurationError(
                f"runtime provider config field {name!r} must be positive"
            )
        return parsed

    @staticmethod
    def _wait_for_dashboard(
        dashboard_url: str,
        *,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        endpoint = f"{dashboard_url}/api/version"
        while time.monotonic() < deadline:
            try:
                response = requests.get(endpoint, timeout=min(poll_seconds, 5.0))
                if response.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(poll_seconds)
        raise JobExecutionError("Ray Jobs endpoint did not become ready in time")


@DeveloperAPI
def run_local_entrypoint(
    entrypoint: str,
    *,
    env_vars: dict[str, str] | None = None,
    num_cpus: float | None = None,
    num_gpus: float | None = None,
    timeout: float | None = None,
) -> None:
    """Run an entrypoint with an owned Ray local runtime.

    Local execution intentionally runs the application as a child process.
    The child sees ``RAY_ADDRESS=auto`` and can use Ray's native local-cluster
    discovery without Tributo implementing a second task scheduler.
    """
    command = shlex.split(entrypoint)
    if not command:
        raise JobConfigurationError("local entrypoint must be non-empty")
    if env_vars is not None and (
        not isinstance(env_vars, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in env_vars.items()
        )
    ):
        raise JobConfigurationError("local env_vars must map strings to strings")

    import ray

    if ray.is_initialized():
        raise JobConfigurationError(
            "local entrypoint cannot reuse an externally initialized Ray runtime"
        )
    ray_kwargs: dict[str, Any] = {"address": "local"}
    if num_cpus is not None:
        ray_kwargs["num_cpus"] = num_cpus
    if num_gpus is not None:
        ray_kwargs["num_gpus"] = num_gpus
    ray.init(**ray_kwargs)
    try:
        child_env = os.environ.copy()
        child_env["RAY_ADDRESS"] = "auto"
        child_env.update(env_vars or {})
        try:
            result = subprocess.run(
                command,
                check=False,
                env=child_env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise JobExecutionError("local Ray entrypoint timed out") from exc
        if result.returncode != 0:
            raise JobExecutionError(
                f"local Ray entrypoint failed with exit code {result.returncode}"
            )
    finally:
        ray.shutdown()


@DeveloperAPI
@contextmanager
def open_ray_client(target: RuntimeTarget) -> Iterator[Any]:
    """Open an explicitly selected Ray Client session for interactive work."""
    if target.submission_mode.value != "client" or not target.address:
        raise JobConfigurationError(
            "Ray Client requires a ray:// RuntimeTarget with an address"
        )
    import ray

    if ray.is_initialized():
        raise JobConfigurationError(
            "Ray Client cannot reuse an already initialized Ray runtime"
        )
    ray.init(address=target.address)
    try:
        yield ray
    finally:
        ray.shutdown()


register_runtime_provider(
    LocalRayJobsProvider.name,
    LocalRayJobsProvider(),
)
register_runtime_provider(
    RayClusterLauncherProvider.name,
    RayClusterLauncherProvider(),
)


@DeveloperAPI
@contextmanager
def open_job_submission_client(target: RuntimeTarget) -> Iterator[JobSubmissionClient]:
    """Open a Ray Jobs client, provisioning only for owned targets."""
    if target.execution_mode.value == "local":
        with LocalRayJobsProvider().provision(target) as lease:
            yield JobSubmissionClient(lease.address)
        return
    if target.submission_mode.value != "jobs":
        raise JobConfigurationError("Ray Jobs client requires submission_mode='jobs'")
    if not target.is_managed:
        yield JobSubmissionClient(target.require_jobs_address())
        return
    provider = resolve_runtime_provider(target.provider)
    with provider.provision(target) as lease:
        yield JobSubmissionClient(lease.address)


__all__ = [
    "LocalRayJobsProvider",
    "RayClusterLauncherProvider",
    "RuntimeLease",
    "RuntimeProvider",
    "open_job_submission_client",
    "open_ray_client",
    "register_runtime_provider",
    "resolve_runtime_provider",
    "run_local_entrypoint",
]

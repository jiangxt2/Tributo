"""Deployment-neutral Ray runtime and submission targets.

The target describes how a workload obtains a Ray runtime. It deliberately
does not select a data engine; ingestion and output engines remain owned by
the data Provider/Binding layers.

The composition root may depend on concrete integrations; domain modules do
not import this module at import time.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from pydantic import model_validator

from tributo._common.config import StrictConfigModel
from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import DeveloperAPI, PublicAPI

if TYPE_CHECKING:
    from tributo.inference.contracts import ResultSinkProvider


@DeveloperAPI
def default_result_sink_provider() -> ResultSinkProvider:
    """Build the default sink provider from installed integrations."""
    from tributo.integrations.sinks.registry import default_result_sink_registry

    return default_result_sink_registry()


@PublicAPI(stability="beta")
class RuntimeExecutionMode(StrEnum):
    """Ray execution topology selected for a workload."""

    LOCAL = "local"
    CLUSTER = "cluster"


@PublicAPI(stability="beta")
class RuntimeSubmissionMode(StrEnum):
    """Ray application submission transport."""

    LOCAL = "local"
    JOBS = "jobs"
    CLIENT = "client"


@PublicAPI(stability="beta")
class RuntimeLifecycle(StrEnum):
    """Whether Tributo owns the runtime lifecycle."""

    OWNED = "owned"
    ATTACHED = "attached"


@PublicAPI(stability="beta")
class RuntimeTarget(StrictConfigModel):
    """Deployment-neutral target for a Ray workload.

    ``master`` accepts ``local``, an HTTP(S) Ray Jobs endpoint, a ``ray://``
    Ray Client endpoint, or ``managed://<provider>/<config>``.  A managed
    target is only a declarative provider reference; provider execution is a
    separate lifecycle concern and must not be inferred from a missing address.
    """

    master: str
    execution_mode: RuntimeExecutionMode
    submission_mode: RuntimeSubmissionMode
    lifecycle: RuntimeLifecycle
    provider: str
    address: str | None = None
    provider_config: str | None = None

    @classmethod
    def from_master(cls, master: str) -> RuntimeTarget:
        """Parse one public master reference into a validated target."""
        if not isinstance(master, str) or not master.strip():
            raise JobConfigurationError("master must be a non-empty string")
        value = master.strip()
        if value == "local":
            return cls(
                master=value,
                execution_mode=RuntimeExecutionMode.LOCAL,
                submission_mode=RuntimeSubmissionMode.LOCAL,
                lifecycle=RuntimeLifecycle.OWNED,
                provider="local",
            )
        if value.startswith("managed://"):
            provider_ref = value.removeprefix("managed://")
            provider, separator, provider_config = provider_ref.partition("/")
            if not provider or not separator or not provider_config.strip("/"):
                raise JobConfigurationError(
                    "managed master must include a provider and config, for example "
                    "managed://ray_cluster_launcher/config.yaml"
                )
            return cls(
                master=value,
                execution_mode=RuntimeExecutionMode.CLUSTER,
                submission_mode=RuntimeSubmissionMode.JOBS,
                lifecycle=RuntimeLifecycle.OWNED,
                provider=provider,
                provider_config=provider_config if separator else None,
            )

        try:
            parsed = urlsplit(value)
        except ValueError as exc:
            raise JobConfigurationError("master is not a valid Ray endpoint") from exc
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            if parsed.username is not None or parsed.password is not None:
                raise JobConfigurationError(
                    "master URLs must not embed credentials; use Ray's supported "
                    "authentication configuration"
                )
            return cls(
                master=value,
                execution_mode=RuntimeExecutionMode.CLUSTER,
                submission_mode=RuntimeSubmissionMode.JOBS,
                lifecycle=RuntimeLifecycle.ATTACHED,
                provider="external",
                address=value.rstrip("/"),
            )
        if parsed.scheme == "ray" and parsed.netloc:
            if parsed.username is not None or parsed.password is not None:
                raise JobConfigurationError(
                    "master URLs must not embed credentials; use Ray's supported "
                    "authentication configuration"
                )
            return cls(
                master=value,
                execution_mode=RuntimeExecutionMode.CLUSTER,
                submission_mode=RuntimeSubmissionMode.CLIENT,
                lifecycle=RuntimeLifecycle.ATTACHED,
                provider="external",
                address=value.rstrip("/"),
            )
        raise JobConfigurationError(
            "unsupported master; use 'local', an http(s) Ray Jobs endpoint, "
            "a ray:// Ray Client endpoint, or managed://<provider>/<config>"
        )

    @model_validator(mode="after")
    def validate_target(self) -> RuntimeTarget:
        """Reject impossible lifecycle and submission combinations."""
        if self.execution_mode is RuntimeExecutionMode.LOCAL:
            if (
                self.submission_mode is not RuntimeSubmissionMode.LOCAL
                or self.lifecycle is not RuntimeLifecycle.OWNED
                or self.provider != "local"
                or self.address is not None
                or self.provider_config is not None
            ):
                raise JobConfigurationError(
                    "local runtime must be an owned local submission without address"
                )
            return self

        if self.submission_mode is RuntimeSubmissionMode.LOCAL:
            raise JobConfigurationError("cluster execution cannot use local submission")
        if self.lifecycle is RuntimeLifecycle.ATTACHED:
            if not self.address:
                raise JobConfigurationError(
                    "attached cluster execution requires a Ray endpoint"
                )
            if self.provider != "external":
                raise JobConfigurationError(
                    "attached cluster execution must use the external provider"
                )
            if self.provider_config is not None:
                raise JobConfigurationError(
                    "attached cluster execution must not carry provider config"
                )
            try:
                parsed_address = urlsplit(self.address)
            except ValueError as exc:
                raise JobConfigurationError(
                    "attached cluster address is not a valid Ray endpoint"
                ) from exc
            if (
                parsed_address.username is not None
                or parsed_address.password is not None
            ):
                raise JobConfigurationError("Ray endpoints must not embed credentials")
            expected_schemes = (
                {"http", "https"}
                if self.submission_mode is RuntimeSubmissionMode.JOBS
                else {"ray"}
            )
            if (
                parsed_address.scheme not in expected_schemes
                or not parsed_address.netloc
            ):
                expected = "http(s)" if "http" in expected_schemes else "ray"
                raise JobConfigurationError(
                    f"{self.submission_mode.value} submission requires a {expected} endpoint"
                )
        else:
            if self.provider in {"", "external", "local"} or not self.provider_config:
                raise JobConfigurationError(
                    "owned cluster execution requires an explicit provider and config"
                )
            if self.address is not None:
                raise JobConfigurationError(
                    "owned cluster execution resolves its address from the provider"
                )
        return self

    @property
    def is_managed(self) -> bool:
        """Return whether a provider must create the Ray cluster."""
        return (
            self.lifecycle is RuntimeLifecycle.OWNED
            and self.execution_mode is RuntimeExecutionMode.CLUSTER
        )

    def require_jobs_address(self) -> str:
        """Return the Ray Jobs endpoint or fail closed for non-Jobs targets."""
        if self.submission_mode is not RuntimeSubmissionMode.JOBS:
            raise JobConfigurationError(
                "Ray Jobs endpoint is unavailable for the selected submission mode"
            )
        if not self.address:
            raise JobConfigurationError(
                "Ray Jobs endpoint is unavailable until the managed provider is ready"
            )
        return self.address


__all__ = [
    "default_result_sink_provider",
    "RuntimeExecutionMode",
    "RuntimeLifecycle",
    "RuntimeSubmissionMode",
    "RuntimeTarget",
]

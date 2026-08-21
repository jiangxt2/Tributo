"""Workload-neutral Ray Jobs admission and control helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from ray.job_submission import JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL, build_runtime_env
from tributo._common.retry import retry_with_exponential_backoff
from tributo._common.submission_id import generate_submission_id
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

_RESERVED_ENV_KEYS = frozenset(
    {
        "TRIBUTO_RUN_ID",
        "TRIBUTO_ATTEMPT_ID",
        "TRIBUTO_SUBMISSION_ID",
    }
)
_REQUEST_DIGEST_METADATA_KEY = "tributo.request_digest"


def _require_submission_id(submission_id: str) -> None:
    if not isinstance(submission_id, str) or not submission_id.strip():
        raise ValueError("submission_id must not be empty")


@PublicAPI(stability="alpha")
class RayJobSubmission(BaseModel):
    """Identity returned after one Ray Jobs attempt is accepted or reconciled."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    submission_id: str = Field(min_length=1)
    ray_job_id: str | None = Field(default=None, min_length=1)
    request_digest: str | None = Field(default=None, min_length=1)


def _validate_inputs(
    operation_namespace: str,
    run_id: str,
    attempt_id: str,
    env_vars: dict[str, str] | None,
    metadata: dict[str, str] | None,
    request_digest: str | None,
) -> None:
    for name, value in (
        ("operation_namespace", operation_namespace),
        ("run_id", run_id),
        ("attempt_id", attempt_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must not be empty")
    conflicts = _RESERVED_ENV_KEYS.intersection(env_vars or {})
    if conflicts:
        raise ValueError(
            "env_vars must not override Ray Job identity: "
            + ", ".join(sorted(conflicts))
        )
    if metadata is not None and _REQUEST_DIGEST_METADATA_KEY in metadata:
        raise ValueError(
            f"metadata must not define reserved key {_REQUEST_DIGEST_METADATA_KEY!r}"
        )
    if request_digest is not None and not request_digest.strip():
        raise ValueError("request_digest must not be empty")


def _ray_job_id(client: JobSubmissionClient, submission_id: str) -> str | None:
    get_job_info = getattr(client, "get_job_info", None)
    if not callable(get_job_info):
        return None
    try:
        info = get_job_info(submission_id)
    except Exception as exc:
        logger.debug(
            "Ray JobDetails unavailable for submission %s (%s)",
            submission_id,
            type(exc).__name__,
        )
        return None
    value = getattr(info, "job_id", None)
    return value if isinstance(value, str) and value else None


def _reconcile_ambiguous_submission(
    client: JobSubmissionClient,
    submission_id: str,
    request_digest: str | None,
) -> str | None:
    """Accept an ambiguous response only for the exact submitted payload."""
    if request_digest is None:
        raise RuntimeError(
            "Ambiguous Ray admission cannot be reconciled without request_digest"
        )
    info = client.get_job_info(submission_id)
    metadata = getattr(info, "metadata", None)
    if not isinstance(metadata, dict):
        raise RuntimeError("Ray JobDetails metadata is unavailable for reconciliation")
    if metadata.get(_REQUEST_DIGEST_METADATA_KEY) != request_digest:
        raise RuntimeError("Ray JobDetails request_digest mismatch")
    value = getattr(info, "job_id", None)
    return value if isinstance(value, str) and value else None


def _submit_ray_job_with_client(
    client: JobSubmissionClient,
    *,
    entrypoint: str,
    run_id: str,
    attempt_id: str,
    submission_id: str,
    runtime_env: dict[str, Any] | None,
    metadata: dict[str, str] | None = None,
    request_digest: str | None = None,
    entrypoint_num_cpus: float | None = None,
    entrypoint_num_gpus: float | None = None,
    entrypoint_memory: int | None = None,
) -> RayJobSubmission:
    """Submit through an existing client; shared by Core workload adapters."""
    job_metadata = dict(metadata or {})
    if _REQUEST_DIGEST_METADATA_KEY in job_metadata:
        raise ValueError(
            f"metadata must not define reserved key {_REQUEST_DIGEST_METADATA_KEY!r}"
        )
    if request_digest is not None:
        if not request_digest.strip():
            raise ValueError("request_digest must not be empty")
        job_metadata[_REQUEST_DIGEST_METADATA_KEY] = request_digest

    reconciled_job_id: str | None = None
    try:
        client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            metadata=job_metadata or None,
            submission_id=submission_id,
            entrypoint_num_cpus=entrypoint_num_cpus,
            entrypoint_num_gpus=entrypoint_num_gpus,
            entrypoint_memory=entrypoint_memory,
        )
    except Exception as submit_error:
        try:
            status = client.get_job_status(submission_id)
        except Exception as query_error:
            raise submit_error from query_error
        if status is None:
            raise submit_error from None
        try:
            reconciled_job_id = _reconcile_ambiguous_submission(
                client, submission_id, request_digest
            )
        except Exception as reconcile_error:
            raise reconcile_error from submit_error
        logger.warning(
            "Reconciled Ray submission %s after an ambiguous submit response",
            submission_id,
        )

    return RayJobSubmission(
        run_id=run_id,
        attempt_id=attempt_id,
        submission_id=submission_id,
        ray_job_id=(
            reconciled_job_id
            if reconciled_job_id is not None
            else _ray_job_id(client, submission_id)
        ),
        request_digest=request_digest,
    )


@PublicAPI(stability="alpha")
def submit_ray_job(
    entrypoint: str,
    *,
    operation_namespace: str,
    run_id: str,
    attempt_id: str = "attempt-1",
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
    extra_excludes: list[str] | None = None,
    extra_py_modules: list[str | Path] | None = None,
    runtime_pip_packages: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    request_digest: str | None = None,
    entrypoint_num_cpus: float | None = None,
    entrypoint_num_gpus: float | None = None,
    entrypoint_memory: int | None = None,
    execution_context: Any | None = None,
) -> RayJobSubmission:
    """Submit one deterministic Ray Job and reconcile an ambiguous response.

    ``extra_py_modules`` and ``runtime_pip_packages`` are trusted deployment
    configuration, not broker task payload fields. Core forwards these values
    without discovering providers or resolving dependencies. Runtime pip
    packages cannot be combined with algorithm artifact pip distribution.
    """

    if not entrypoint.strip():
        raise ValueError("entrypoint must not be empty")
    _validate_inputs(
        operation_namespace,
        run_id,
        attempt_id,
        env_vars,
        metadata,
        request_digest,
    )
    submission_id = generate_submission_id(
        operation_namespace,
        run_id,
        attempt_id,
    )
    job_env = dict(env_vars or {})
    job_env.update(
        {
            "TRIBUTO_RUN_ID": run_id,
            "TRIBUTO_ATTEMPT_ID": attempt_id,
            "TRIBUTO_SUBMISSION_ID": submission_id,
        }
    )
    runtime_env_kwargs: dict[str, Any] = {
        "project_root": project_root,
        "env_vars": job_env,
        "extra_excludes": extra_excludes,
        "extra_py_modules": extra_py_modules,
        "runtime_pip_packages": runtime_pip_packages,
    }
    if execution_context is not None:
        runtime_env_kwargs["execution_context"] = execution_context
    runtime_env = build_runtime_env(**runtime_env_kwargs)
    client = _get_submission_client(dashboard_url)
    return _submit_ray_job_with_client(
        client,
        entrypoint=entrypoint,
        run_id=run_id,
        attempt_id=attempt_id,
        submission_id=submission_id,
        runtime_env=runtime_env,
        metadata=metadata,
        request_digest=request_digest,
        entrypoint_num_cpus=entrypoint_num_cpus,
        entrypoint_num_gpus=entrypoint_num_gpus,
        entrypoint_memory=entrypoint_memory,
    )


@PublicAPI(stability="alpha")
def get_ray_job_status(
    submission_id: str,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
) -> str:
    """Return the normalized Ray Jobs status for a submission identity."""

    _require_submission_id(submission_id)
    status = _get_submission_client(dashboard_url).get_job_status(submission_id)
    if status is None:
        raise LookupError(f"Unknown Ray submission {submission_id!r}")
    return str(getattr(status, "value", status)).upper()


@PublicAPI(stability="alpha")
def get_ray_job_logs(
    submission_id: str,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
) -> str:
    """Return logs for the Ray Job identified by ``submission_id``."""

    _require_submission_id(submission_id)
    return _get_submission_client(dashboard_url).get_job_logs(submission_id)


@PublicAPI(stability="alpha")
def stop_ray_job(
    submission_id: str,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
) -> bool:
    """Request that Ray stop the job identified by ``submission_id``."""

    _require_submission_id(submission_id)
    return bool(_get_submission_client(dashboard_url).stop_job(submission_id))


@retry_with_exponential_backoff(
    max_retries=3,
    base_delay=1.0,
    exceptions=(ConnectionError, TimeoutError, OSError),
)
def _get_submission_client(dashboard_url: str) -> JobSubmissionClient:
    return JobSubmissionClient(dashboard_url)


__all__ = [
    "RayJobSubmission",
    "get_ray_job_logs",
    "get_ray_job_status",
    "stop_ray_job",
    "submit_ray_job",
]

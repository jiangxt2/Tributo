"""Training job submission wrapper via Ray Jobs API.

Builds the Ray-managed code distribution environment. Package dependencies
remain owned by the cluster image or an explicit Ray runtime environment.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from ray.job_submission import JobStatus, JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL, build_runtime_env
from tributo._common.retry import retry_with_exponential_backoff
from tributo._common.submission_id import generate_submission_id
from tributo.algorithms.api import EnvironmentSpec
from tributo.algorithms.api.artifacts import AlgorithmArtifact, ImageProfile
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 180
JobAttemptStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "STOPPED"]
TerminalJobStatus = Literal["SUCCEEDED", "FAILED", "STOPPED"]
_RESERVED_ENV_KEYS = frozenset({"TRIBUTO_RUN_ID", "TRIBUTO_ATTEMPT_ID"})


def _resolve_algorithm_dependencies(
    declared_dependencies: tuple[str, ...],
    environment: EnvironmentSpec | None,
) -> tuple[str, ...]:
    """Reuse EnvironmentSpec dependencies without duplicating declarations."""
    values = set(declared_dependencies)
    if environment is not None:
        values.update(environment.dependencies)
    return tuple(sorted(values))


@PublicAPI(stability="beta")
class JobAttempt(BaseModel):
    """Immutable record of one Ray Jobs submission attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(..., min_length=1)
    attempt_id: str = Field(..., min_length=1)
    submission_id: str = Field(..., min_length=1)
    job_id: str = Field(..., min_length=1)
    attempt_number: int = Field(..., ge=1)
    status: JobAttemptStatus
    retryable: bool = False


@PublicAPI(stability="beta")
class TrainingJobResult(BaseModel):
    """Terminal state and attempt history for a submitted training run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(..., min_length=1)
    bundle_id: str = Field(..., min_length=1)
    job_id: str = Field(..., min_length=1)
    status: TerminalJobStatus
    logs: str = ""
    attempts: tuple[JobAttempt, ...] = ()
    retryable: bool = False


def _validate_metadata(metadata: dict[str, str] | None) -> None:
    """Reject job metadata keys that are reserved for runtime identity."""
    if metadata is None:
        return
    conflicts = _RESERVED_ENV_KEYS.intersection(metadata)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"metadata must not contain reserved keys: {names}")


def _status_name(status: JobStatus | str) -> JobAttemptStatus:
    """Normalize Ray status and fail fast on an unknown terminal state."""
    status_name = str(getattr(status, "value", status)).upper()
    if status_name not in {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "STOPPED"}:
        raise ValueError(f"Unsupported Ray job status: {status_name!r}")
    return cast(JobAttemptStatus, status_name)


def _resolve_run_id(
    entrypoint: str,
    env_vars: dict[str, str] | None,
    run_id: str | None,
) -> str:
    """Resolve the stable logical run identity used by all attempts."""
    return run_id or generate_submission_id(
        "run", entrypoint, str(sorted((env_vars or {}).items()))
    )


def _submit_training_job_attempt(
    client: JobSubmissionClient,
    *,
    entrypoint: str,
    runtime_env: dict[str, Any],
    run_id: str,
    attempt_id: str,
    metadata: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Submit one stable attempt and reconcile ambiguous server responses."""
    submission_id = generate_submission_id("train", run_id, attempt_id)
    try:
        job_id = client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            metadata=metadata,
            submission_id=submission_id,
        )
    except Exception as exc:
        # The request may have reached Ray before the client observed an
        # error.  Query the deterministic submission ID before considering a
        # retry; inventing another ID here could run the same attempt twice.
        try:
            status = client.get_job_status(submission_id)
        except Exception as query_exc:
            raise exc from query_exc
        if status is None:
            raise exc from None
        logger.warning(
            "Reconciled submission %s after ambiguous error (status=%s)",
            submission_id,
            _status_name(status),
        )
        return submission_id, submission_id
    logger.info("Submitted training job %s: %s", job_id, entrypoint)
    return str(job_id), submission_id


@PublicAPI(stability="beta")
def submit_training_job(
    entrypoint: str,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
    extra_excludes: list[str] | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
    metadata: dict[str, str] | None = None,
    algorithm_artifact: AlgorithmArtifact | None = None,
    image_profile: ImageProfile | None = None,
    declared_dependencies: tuple[str, ...] = (),
    environment: EnvironmentSpec | None = None,
) -> str:
    """Submit a training job via the Ray Jobs API.

    Automatically builds the correct ``runtime_env``, ensuring:
    - The latest ``tributo`` code is uploaded to the cluster via ``py_modules``;
    - The entrypoint script is provided via ``working_dir``;
    - an algorithm Wheel is either code-only with image-provided dependencies
      or installed from a preflighted offline Wheelhouse;
    - package dependencies are not implicitly mixed across Python environments.

    Args:
        entrypoint: Entrypoint command, e.g. ``"python examples/xgboost_s3_training.py"``.
        dashboard_url: Ray Dashboard address.
        env_vars: Additional environment variables (training params passed this way).
        project_root: Project root directory, auto-finds pyproject.toml if None.
        extra_excludes: Additional directories/files to exclude from working_dir.
        run_id: Stable logical run identifier shared by retries.
        attempt_id: Unique attempt identifier; defaults to ``attempt-1``.
        metadata: Metadata stored with the Ray Job, not exposed as worker
            environment variables.
        algorithm_artifact: Optional validated algorithm Wheel or offline
            Bundle to make available for this Job.
        image_profile: Immutable image compatibility record for the artifact.
        declared_dependencies: Additional PEP 508 constraints to preflight.
        environment: Optional formal or ``from_sklearn()`` EnvironmentSpec;
            its dependencies are merged into the same preflight.

    Returns:
        Submitted job ID on success.

    Raises:
        RuntimeError: Submission failed.
    """
    resolved_run_id = _resolve_run_id(entrypoint, env_vars, run_id)
    resolved_attempt_id = attempt_id or "attempt-1"
    _validate_metadata(metadata)
    job_env_vars = dict(env_vars or {})
    job_env_vars.update(
        {
            "TRIBUTO_RUN_ID": resolved_run_id,
            "TRIBUTO_ATTEMPT_ID": resolved_attempt_id,
        }
    )
    runtime_env = build_runtime_env(
        project_root=project_root,
        env_vars=job_env_vars,
        extra_excludes=extra_excludes,
        algorithm_artifact=algorithm_artifact,
        image_profile=image_profile,
        declared_dependencies=_resolve_algorithm_dependencies(
            declared_dependencies,
            environment,
        ),
    )

    client = _get_submission_client(dashboard_url)
    job_id, _submission_id = _submit_training_job_attempt(
        client,
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        run_id=resolved_run_id,
        attempt_id=resolved_attempt_id,
        metadata=metadata,
    )
    return job_id


@PublicAPI(stability="beta")
def submit_training_job_with_retry(
    entrypoint: str,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
    extra_excludes: list[str] | None = None,
    run_id: str | None = None,
    metadata: dict[str, str] | None = None,
    max_attempts: int = 1,
    timeout: int = DEFAULT_TIMEOUT,
    poll_interval: int = 2,
    retry_classifier: Callable[[JobStatus | str, str], bool] | None = None,
    algorithm_artifact: AlgorithmArtifact | None = None,
    image_profile: ImageProfile | None = None,
    declared_dependencies: tuple[str, ...] = (),
    environment: EnvironmentSpec | None = None,
) -> TrainingJobResult:
    """Submit, reconcile and optionally retry a training run.

    Each attempt has a unique deterministic ``attempt_id`` within the run.
    ``FAILED`` is eligible for another attempt only when an explicit
    classifier says so and the attempt budget remains.  Passing no classifier
    disables automatic retries.  ``STOPPED`` is treated as a user cancellation
    and is never retried automatically.

    Args:
        timeout: Maximum wait time per attempt. A timeout raises ``TimeoutError``
            and does not submit another attempt.
        retry_classifier: Function deciding whether a failed attempt is
            transient. ``None`` disables automatic retries.

    Raises:
        TimeoutError: If an attempt does not reach a terminal state in time.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    resolved_run_id = _resolve_run_id(entrypoint, env_vars, run_id)
    _validate_metadata(metadata)
    job_env_vars = dict(env_vars or {})
    job_env_vars["TRIBUTO_RUN_ID"] = resolved_run_id
    client = _get_submission_client(dashboard_url)
    attempts: list[JobAttempt] = []
    last_result: dict[str, Any] | None = None

    for attempt_number in range(1, max_attempts + 1):
        attempt_id = f"attempt-{attempt_number}"
        attempt_env_vars = dict(job_env_vars)
        attempt_env_vars["TRIBUTO_ATTEMPT_ID"] = attempt_id
        runtime_env = build_runtime_env(
            project_root=project_root,
            env_vars=attempt_env_vars,
            extra_excludes=extra_excludes,
            algorithm_artifact=algorithm_artifact,
            image_profile=image_profile,
            declared_dependencies=_resolve_algorithm_dependencies(
                declared_dependencies,
                environment,
            ),
        )
        job_id, submission_id = _submit_training_job_attempt(
            client,
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            run_id=resolved_run_id,
            attempt_id=attempt_id,
            metadata=metadata,
        )
        last_result = wait_for_job(
            client,
            job_id,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        status = last_result["status"]
        status_name = _status_name(status)
        logs = str(last_result.get("logs", ""))
        retryable = bool(
            status_name == "FAILED"
            and retry_classifier is not None
            and retry_classifier(status, logs)
        )
        attempts.append(
            JobAttempt(
                run_id=resolved_run_id,
                attempt_id=attempt_id,
                submission_id=submission_id,
                job_id=job_id,
                attempt_number=attempt_number,
                status=status_name,
                retryable=retryable,
            )
        )
        if status_name == "SUCCEEDED" or not retryable:
            break

    if last_result is None:
        raise RuntimeError("training job did not produce a terminal result")
    from tributo.exporting.service import bundle_id_for_request

    final_status = _status_name(last_result["status"])
    return TrainingJobResult(
        run_id=resolved_run_id,
        bundle_id=bundle_id_for_request(resolved_run_id),
        job_id=attempts[-1].job_id,
        status=cast(TerminalJobStatus, final_status),
        logs=str(last_result.get("logs", "")),
        attempts=tuple(attempts),
        retryable=attempts[-1].retryable,
    )


@retry_with_exponential_backoff(
    max_retries=3,
    base_delay=1.0,
    exceptions=(ConnectionError, TimeoutError, OSError),
)
def _get_submission_client(dashboard_url: str) -> JobSubmissionClient:
    """Create a JobSubmissionClient with retry on connection failures.

    Retries are limited to the connection establishment phase to avoid
    duplicate job submissions when the request has already reached the
    Ray server.
    """
    return JobSubmissionClient(dashboard_url)


@PublicAPI(stability="beta")
def wait_for_job(
    client: JobSubmissionClient,
    job_id: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    poll_interval: int = 2,
) -> dict[str, Any]:
    """Poll and wait for a job to complete.

    Args:
        client: Ray Jobs API client.
        job_id: Job ID to wait for.
        timeout: Maximum wait time in seconds.
        poll_interval: Polling interval in seconds.

    Returns:
        ``{"status": JobStatus, "logs": str}``

    Raises:
        TimeoutError: Timed out without completion.
    """
    start = time.time()
    while time.time() - start < timeout:
        status = client.get_job_status(job_id)
        if _status_name(status) in {"SUCCEEDED", "FAILED", "STOPPED"}:
            logs = client.get_job_logs(job_id)
            return {"status": status, "logs": logs}
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def submit_and_export_onnx(
    entrypoint: str,
    onnx_output_path: str,
    n_features: int,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, str]:
    """Submit a training job and export ONNX upon completion (Driver side).

    TODO: Implement checkpoint extraction from /tmp/ray_results and ONNX export.
    For now, use Ray Train API (trainer.fit()) for direct handling.

    Args:
        entrypoint: Training script entrypoint command.
        onnx_output_path: ONNX file save path.
        n_features: Number of features.
        dashboard_url: Ray Dashboard address.
        env_vars: Environment variables.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (job_id, onnx_path).

    Raises:
        RuntimeError: Training failed or ONNX export failed.
    """

    # TODO: Implement checkpoint extraction from /tmp/ray_results and ONNX export.
    # Requires submit_training_job + wait_for_job + checkpoint extraction logic.
    # For now, use Ray Train API (trainer.fit()) for direct handling.
    raise NotImplementedError(
        "Checkpoint extraction from Jobs API is not yet implemented. "
        "Use Ray Train API (trainer.fit()) for automatic checkpoint handling."
    )

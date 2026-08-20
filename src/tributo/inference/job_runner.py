"""Ray Jobs adapters for legacy configs and frozen inference plans."""

from __future__ import annotations

import base64
import logging
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from ray.job_submission import JobStatus, JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL, build_runtime_env
from tributo._common.retry import retry_with_exponential_backoff
from tributo._common.submission_id import generate_submission_id
from tributo.inference.contracts import InferenceRequest, ResolvedInference
from tributo.inference.resolver import InferenceResolver
from tributo.ray_jobs import RayJobSubmission, _submit_ray_job_with_client
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 180
InferenceJobStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]
TerminalInferenceJobStatus = Literal["succeeded", "failed", "cancelled"]
_TERMINAL_RAY_STATUSES = frozenset({"SUCCEEDED", "FAILED", "STOPPED"})
_MAX_RESOLVED_PLAN_B64_BYTES = 64 * 1024
_RESERVED_ENV_KEYS = frozenset(
    {
        "TRIBUTO_RUN_ID",
        "TRIBUTO_ATTEMPT_ID",
        "TRIBUTO_SUBMISSION_ID",
        "TRIBUTO_PARENT_RUN_ID",
        "TRIBUTO_JOB_KIND",
        "TRIBUTO_INFERENCE_PLAN_B64",
    }
)


@PublicAPI(stability="alpha")
class InferenceJobAttempt(BaseModel):
    """Immutable record of one Ray Jobs inference attempt."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    submission_id: str = Field(min_length=1)
    ray_job_id: str | None = Field(default=None, min_length=1)
    attempt_number: int = Field(ge=1)
    status: InferenceJobStatus
    retryable: bool = False


@PublicAPI(stability="alpha")
class InferenceJobResult(BaseModel):
    """Terminal Ray Jobs state with all inference attempts."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    run_id: str = Field(min_length=1)
    submission_id: str = Field(min_length=1)
    ray_job_id: str | None = Field(default=None, min_length=1)
    status: TerminalInferenceJobStatus
    logs: str = ""
    attempts: tuple[InferenceJobAttempt, ...] = ()
    retryable: bool = False


@PublicAPI(stability="alpha")
def map_ray_job_status(status: JobStatus | str) -> InferenceJobStatus:
    """Translate Ray Jobs state without inventing a ``partial`` terminal."""
    status_name = _ray_status_name(status)
    mapping: dict[str, InferenceJobStatus] = {
        "PENDING": "pending",
        "RUNNING": "running",
        "SUCCEEDED": "succeeded",
        "FAILED": "failed",
        "STOPPED": "cancelled",
    }
    return mapping[status_name]


@PublicAPI(stability="beta")
def submit_inference_job(
    config_path: str,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
    run_id: str | None = None,
    attempt_id: str = "attempt-1",
) -> str:
    """Submit the legacy strict-JSON entry point with stable job identity."""
    _validate_env_vars(env_vars)
    resolved_run_id = run_id or generate_submission_id(
        "infer-run", config_path, str(sorted((env_vars or {}).items()))
    )
    submission_id = generate_submission_id("infer", resolved_run_id, attempt_id)
    job_env = dict(env_vars or {})
    job_env.update(
        {
            "TRIBUTO_RUN_ID": resolved_run_id,
            "TRIBUTO_ATTEMPT_ID": attempt_id,
            "TRIBUTO_SUBMISSION_ID": submission_id,
            "TRIBUTO_JOB_KIND": "inference",
        }
    )
    runtime_env = build_runtime_env(
        project_root=project_root,
        env_vars=job_env,
    )
    entrypoint = (
        f"python -m tributo.inference.batch_job --config {shlex.quote(config_path)}"
    )
    client = _get_submission_client(dashboard_url)
    submission = _submit_attempt(
        client,
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        run_id=resolved_run_id,
        attempt_id=attempt_id,
        submission_id=submission_id,
    )
    logger.info(
        "Submitted inference job %s: config=%s",
        submission.submission_id,
        config_path,
    )
    return submission.submission_id


@PublicAPI(stability="alpha")
def submit_inference_request(
    request: InferenceRequest,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
    resolver: InferenceResolver | None = None,
) -> str:
    """Resolve once and return the accepted Ray Jobs submission identity."""
    return submit_inference_request_with_identity(
        request,
        dashboard_url=dashboard_url,
        env_vars=env_vars,
        project_root=project_root,
        resolver=resolver,
    ).submission_id


@PublicAPI(stability="alpha")
def submit_inference_request_with_identity(
    request: InferenceRequest,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
    resolver: InferenceResolver | None = None,
) -> RayJobSubmission:
    """Resolve once, freeze the plan, and return complete submission identity."""
    _validate_env_vars(env_vars)
    plan = (resolver or InferenceResolver()).resolve(request)
    client = _get_submission_client(dashboard_url)
    return _submit_resolved_plan(
        client,
        plan=plan,
        env_vars=env_vars,
        project_root=project_root,
    )


@PublicAPI(stability="alpha")
def submit_resolved_inference(
    plan: ResolvedInference,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
) -> str:
    """Submit an already-frozen plan and return its submission identity."""
    return submit_resolved_inference_with_identity(
        plan,
        dashboard_url=dashboard_url,
        env_vars=env_vars,
        project_root=project_root,
    ).submission_id


@PublicAPI(stability="alpha")
def submit_resolved_inference_with_identity(
    plan: ResolvedInference,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
) -> RayJobSubmission:
    """Submit a frozen plan and return workload-neutral Ray Jobs identity."""
    _validate_env_vars(env_vars)
    client = _get_submission_client(dashboard_url)
    return _submit_resolved_plan(
        client,
        plan=plan,
        env_vars=env_vars,
        project_root=project_root,
    )


@PublicAPI(stability="alpha")
def submit_inference_request_with_retry(
    request: InferenceRequest,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
    resolver: InferenceResolver | None = None,
    max_attempts: int = 1,
    timeout: int = DEFAULT_TIMEOUT,
    poll_interval: int = 2,
    retry_classifier: Callable[[JobStatus | str, str], bool] | None = None,
) -> InferenceJobResult:
    """Submit stable attempts; STOPPED maps to cancelled and is never retried."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    _validate_env_vars(env_vars)
    first_plan = (resolver or InferenceResolver()).resolve(request)
    client = _get_submission_client(dashboard_url)
    attempts: list[InferenceJobAttempt] = []
    last: dict[str, Any] | None = None

    for attempt_number in range(1, max_attempts + 1):
        plan = _plan_for_attempt(first_plan, attempt_number)
        submission = _submit_resolved_plan(
            client,
            plan=plan,
            env_vars=env_vars,
            project_root=project_root,
        )
        last = wait_for_job(
            client,
            submission.submission_id,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        ray_status = last["status"]
        status = map_ray_job_status(ray_status)
        logs = str(last.get("logs", ""))
        retryable = bool(
            status == "failed"
            and retry_classifier is not None
            and retry_classifier(ray_status, logs)
        )
        attempts.append(
            InferenceJobAttempt(
                run_id=plan.run_id,
                attempt_id=plan.attempt_id,
                submission_id=submission.submission_id,
                ray_job_id=submission.ray_job_id,
                attempt_number=attempt_number,
                status=status,
                retryable=retryable,
            )
        )
        if status == "succeeded" or not retryable:
            break

    if last is None:
        raise RuntimeError("inference job did not produce a terminal result")
    final_status = cast(TerminalInferenceJobStatus, attempts[-1].status)
    return InferenceJobResult(
        run_id=first_plan.run_id,
        submission_id=attempts[-1].submission_id,
        ray_job_id=attempts[-1].ray_job_id,
        status=final_status,
        logs=str(last.get("logs", "")),
        attempts=tuple(attempts),
        retryable=attempts[-1].retryable,
    )


def _submit_resolved_plan(
    client: JobSubmissionClient,
    *,
    plan: ResolvedInference,
    env_vars: dict[str, str] | None,
    project_root: Path | None,
) -> RayJobSubmission:
    plan = ResolvedInference.model_validate(plan.model_dump(mode="python"))
    encoded_plan = base64.urlsafe_b64encode(
        plan.model_dump_json().encode("utf-8")
    ).decode("ascii")
    encoded_plan_size = len(encoded_plan.encode("ascii"))
    if encoded_plan_size > _MAX_RESOLVED_PLAN_B64_BYTES:
        raise ValueError(
            "Resolved inference plan exceeds the 65536-byte Ray Jobs "
            "environment transport limit"
        )
    job_env = dict(env_vars or {})
    job_env.update(
        {
            "TRIBUTO_RUN_ID": plan.run_id,
            "TRIBUTO_ATTEMPT_ID": plan.attempt_id,
            "TRIBUTO_SUBMISSION_ID": plan.submission_id,
            "TRIBUTO_JOB_KIND": "inference",
            "TRIBUTO_INFERENCE_PLAN_B64": encoded_plan,
        }
    )
    if plan.parent_run_id is not None:
        job_env["TRIBUTO_PARENT_RUN_ID"] = plan.parent_run_id
    runtime_env = build_runtime_env(
        project_root=project_root,
        env_vars=job_env,
    )
    submission = _submit_attempt(
        client,
        entrypoint=(
            "python -m tributo.inference.batch_job "
            "--resolved-plan-env TRIBUTO_INFERENCE_PLAN_B64"
        ),
        runtime_env=runtime_env,
        run_id=plan.run_id,
        attempt_id=plan.attempt_id,
        submission_id=plan.submission_id,
        request_digest=plan.plan_digest,
    )
    logger.info(
        "Submitted inference attempt %s for run %s as submission %s",
        plan.attempt_id,
        plan.run_id,
        submission.submission_id,
    )
    return submission


def _submit_attempt(
    client: JobSubmissionClient,
    *,
    entrypoint: str,
    runtime_env: dict[str, Any],
    run_id: str,
    attempt_id: str,
    submission_id: str,
    request_digest: str | None = None,
) -> RayJobSubmission:
    return _submit_ray_job_with_client(
        client,
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        run_id=run_id,
        attempt_id=attempt_id,
        submission_id=submission_id,
        request_digest=request_digest,
    )


def _plan_for_attempt(
    first_plan: ResolvedInference, attempt_number: int
) -> ResolvedInference:
    attempt_id = f"attempt-{attempt_number}"
    submission_id = generate_submission_id("infer", first_plan.run_id, attempt_id)
    return first_plan.model_copy(
        update={"attempt_id": attempt_id, "submission_id": submission_id}
    )


def _validate_env_vars(env_vars: dict[str, str] | None) -> None:
    if env_vars is None:
        return
    conflicts = _RESERVED_ENV_KEYS.intersection(env_vars)
    if conflicts:
        raise ValueError(
            "env_vars must not contain reserved inference keys: "
            + ", ".join(sorted(conflicts))
        )


def _ray_status_name(status: JobStatus | str) -> str:
    status_name = str(getattr(status, "value", status)).upper()
    if status_name not in {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "STOPPED"}:
        raise ValueError(f"Unsupported Ray job status: {status_name!r}")
    return status_name


@PublicAPI(stability="alpha")
def wait_for_job(
    client: JobSubmissionClient,
    job_id: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    poll_interval: int = 2,
) -> dict[str, Any]:
    """Wait for one Ray job terminal state and retain its logs."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get_job_status(job_id)
        if _ray_status_name(status) in _TERMINAL_RAY_STATUSES:
            return {"status": status, "logs": _get_job_logs(client, job_id)}
        time.sleep(poll_interval)
    status = client.get_job_status(job_id)
    if _ray_status_name(status) in _TERMINAL_RAY_STATUSES:
        return {"status": status, "logs": _get_job_logs(client, job_id)}
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def _get_job_logs(client: JobSubmissionClient, job_id: str) -> str:
    try:
        return client.get_job_logs(job_id)
    except Exception as exc:
        logger.warning(
            "Could not retrieve logs for terminal inference job %s (%s)",
            job_id,
            type(exc).__name__,
        )
        return ""


@retry_with_exponential_backoff(
    max_retries=3,
    base_delay=1.0,
    exceptions=(ConnectionError, TimeoutError, OSError),
)
def _get_submission_client(dashboard_url: str) -> JobSubmissionClient:
    """Create the Ray client; retries stop before the submission boundary."""
    return JobSubmissionClient(dashboard_url)


__all__ = [
    "InferenceJobAttempt",
    "InferenceJobResult",
    "map_ray_job_status",
    "submit_inference_job",
    "submit_inference_request",
    "submit_inference_request_with_identity",
    "submit_inference_request_with_retry",
    "submit_resolved_inference",
    "submit_resolved_inference_with_identity",
    "wait_for_job",
]

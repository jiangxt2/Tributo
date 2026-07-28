"""Ray Jobs API submission wrapper: batch inference job.

Isomorphic with training/job_submitter.py, automatically handles runtime_env.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

from ray.job_submission import JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL, build_runtime_env
from tributo._common.retry import retry_with_exponential_backoff
from tributo._common.submission_id import generate_submission_id
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
def submit_inference_job(
    config_path: str,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
) -> str:
    """Submit a batch inference job via the Ray Jobs API.

    Args:
        config_path: Path to the YAML config file (relative to project root).
        dashboard_url: Ray Dashboard address.
        env_vars: Additional environment variables.
        project_root: Project root directory.

    Returns:
        The job ID of the successfully submitted job.
    """
    runtime_env = build_runtime_env(
        project_root=project_root,
        env_vars=env_vars,
    )

    entrypoint = (
        f"python -m tributo.inference.batch_job --config {shlex.quote(config_path)}"
    )

    submission_id = generate_submission_id(
        "infer", config_path, str(sorted((env_vars or {}).items()))
    )

    client = _get_submission_client(dashboard_url)
    job_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        submission_id=submission_id,
    )
    logger.info("Submitted inference job %s: config=%s", job_id, config_path)
    return job_id


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

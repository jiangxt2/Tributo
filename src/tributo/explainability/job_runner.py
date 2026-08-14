"""Ray Jobs API submission for explainability requests."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

from ray.job_submission import JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL, build_runtime_env
from tributo._common.submission_id import generate_submission_id
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="alpha")
def submit_explainability_job(
    config_path: str,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
) -> str:
    """Submit a strict-JSON explainability request to the Ray Jobs API."""
    if Path(config_path).suffix.lower() in {".yaml", ".yml"}:
        raise ValueError("YAML config is no longer supported; please use JSON")
    runtime_env = build_runtime_env(project_root=project_root, env_vars=env_vars)
    submission_id = generate_submission_id("explain", str(Path(config_path).resolve()))
    entrypoint = "python -m tributo.explainability.batch_job --config " + shlex.quote(
        config_path
    )
    client = JobSubmissionClient(dashboard_url)
    job_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        submission_id=submission_id,
    )
    logger.info("Submitted explainability job %s", job_id)
    return job_id


__all__ = ["submit_explainability_job"]

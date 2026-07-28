"""Ray Jobs API submission wrapper for batch embedding jobs.

Follows the same runtime_env strategy as ``training/job_submitter.py``:
all dependencies are pre-installed in the Docker image; runtime_env
only distributes code.
"""

from __future__ import annotations

import logging
import shlex

from ray.job_submission import JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL, build_runtime_env
from tributo._common.retry import retry_with_exponential_backoff
from tributo._common.submission_id import generate_submission_id
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
def submit_embedding_job(
    s3_input_path: str,
    s3_output_path: str,
    model_name: str = "bge-small-zh",
    text_column: str = "text",
    batch_size: int = 64,
    concurrency: int = 4,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
) -> str:
    """Submit a batch embedding job to the Ray cluster.

    The job runs inside the cluster and executes:
    ``python -m tributo.embeddings.batch_job <args>``.

    Args:
        s3_input_path: S3 URI to input Parquet files.
        s3_output_path: S3 URI for output (Lance or Parquet auto-selected).
        model_name: Registered short model name.
        text_column: Column containing text to embed.
        batch_size: Records per inference batch.
        concurrency: Number of Embedder actors.
        dashboard_url: Ray Dashboard address.
        env_vars: Extra environment variables passed to the job.

    Returns:
        Submitted job ID.
    """
    runtime_env = build_runtime_env(env_vars=env_vars)

    entrypoint = (
        f"python -m tributo.embeddings.batch_job "
        f"--input {shlex.quote(s3_input_path)} "
        f"--output {shlex.quote(s3_output_path)} "
        f"--model {shlex.quote(model_name)} "
        f"--text-column {shlex.quote(text_column)} "
        f"--batch-size {batch_size} "
        f"--concurrency {concurrency}"
    )

    submission_id = generate_submission_id(
        "embed",
        model_name,
        s3_input_path,
        s3_output_path,
        text_column,
        str(batch_size),
        str(concurrency),
    )

    client = _get_submission_client(dashboard_url)
    job_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        submission_id=submission_id,
    )
    logger.info("Submitted embedding job %s: %s", job_id, entrypoint)
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

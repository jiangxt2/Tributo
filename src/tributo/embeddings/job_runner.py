"""Ray Jobs API submission wrapper for batch embedding jobs.

Follows the same runtime_env strategy as ``training/job_submitter.py``:
all dependencies are pre-installed in the Docker image; runtime_env
only distributes code.
"""

from __future__ import annotations

import logging
import shlex
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import TypeAdapter, ValidationError
from ray.job_submission import JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL, build_runtime_env
from tributo._common.retry import retry_with_exponential_backoff
from tributo._common.submission_id import generate_submission_id
from tributo.data import CanonicalSourceInput, IngestionRequest, ProviderSourceConfig
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


def _credential_free_uri(uri: str) -> str:
    """Remove userinfo and query credentials from a logged URI."""
    parsed = urlsplit(uri)
    if parsed.hostname is None:
        return uri
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


@PublicAPI(stability="beta")
def submit_embedding_job(
    s3_input_path: str | None = None,
    s3_output_path: str | None = None,
    model_name: str = "bge-small-zh",
    text_column: str | None = None,
    batch_size: int = 64,
    concurrency: int = 4,
    *,
    source: CanonicalSourceInput | dict[str, Any] | None = None,
    engine: str = "ray",
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
) -> str:
    """Submit a batch embedding job to the Ray cluster.

    The job runs inside the cluster and executes:
    ``python -m tributo.embeddings.batch_job <args>``.

    Args:
        s3_input_path: Legacy S3 URI to input Parquet files.
        s3_output_path: S3 URI for output (Lance or Parquet auto-selected).
        model_name: Registered short model name.
        text_column: Column containing text to embed.
        batch_size: Records per inference batch.
        concurrency: Number of Embedder actors.
        source: Canonical source configuration. Mutually exclusive with
            ``s3_input_path``. Inline credentials are rejected; use cluster
            environment variables or IAM.
        engine: Explicit ingestion engine (``ray`` or ``daft``). Daft results
            are converted through the public Daft-to-Ray adapter in the job.
        dashboard_url: Ray Dashboard address.
        env_vars: Extra environment variables passed to the job.

    Returns:
        Submitted job ID.
    """
    if s3_output_path is None:
        raise ValueError("s3_output_path is required")
    if (source is None) == (s3_input_path is None):
        raise ValueError("provide exactly one of source or s3_input_path")

    validated_source: CanonicalSourceInput | None = None
    source_json: str | None = None
    provider_name = "tributo.parquet"
    if source is not None:
        try:
            validated_source = TypeAdapter(CanonicalSourceInput).validate_python(source)
        except ValidationError as exc:
            raise ValueError(f"invalid embedding source: {exc}") from exc
        ingestion_request = IngestionRequest(source=validated_source, engine=engine)
        source_json = ingestion_request.source_json_for_remote_transport()
        canonical_engine = ingestion_request.engine
        provider_name = (
            validated_source.provider
            if isinstance(validated_source, ProviderSourceConfig)
            else validated_source.type
        )
    else:
        assert s3_input_path is not None
        legacy_source = ProviderSourceConfig(
            provider="tributo.parquet",
            uri=s3_input_path,
        )
        ingestion_request = IngestionRequest(source=legacy_source, engine=engine)
        # Validate that the legacy URI is also safe to place in an entrypoint.
        ingestion_request.source_json_for_remote_transport()
        canonical_engine = ingestion_request.engine

    runtime_env = build_runtime_env(env_vars=env_vars)

    entrypoint_parts = ["python -m tributo.embeddings.batch_job"]
    if source_json is not None:
        entrypoint_parts.append(f"--source {shlex.quote(source_json)}")
    else:
        assert s3_input_path is not None
        entrypoint_parts.append(f"--input {shlex.quote(s3_input_path)}")
    entrypoint_parts.extend(
        [
            f"--output {shlex.quote(s3_output_path)}",
            f"--model {shlex.quote(model_name)}",
            f"--engine {shlex.quote(canonical_engine)}",
        ]
    )
    if text_column is not None:
        entrypoint_parts.append(f"--text-column {shlex.quote(text_column)}")
    entrypoint_parts.extend(
        [f"--batch-size {batch_size}", f"--concurrency {concurrency}"]
    )
    entrypoint = " ".join(entrypoint_parts)

    source_identity = source_json if source_json is not None else s3_input_path
    assert source_identity is not None

    submission_id = generate_submission_id(
        "embed",
        model_name,
        source_identity,
        canonical_engine,
        s3_output_path,
        text_column or "",
        str(batch_size),
        str(concurrency),
    )

    client = _get_submission_client(dashboard_url)
    job_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        submission_id=submission_id,
    )
    logger.info(
        "Submitted embedding job %s: provider=%s output=%s",
        job_id,
        provider_name,
        _credential_free_uri(s3_output_path),
    )
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

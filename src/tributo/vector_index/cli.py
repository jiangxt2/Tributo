"""CLI commands for Lance vector-index Ray Jobs."""

from __future__ import annotations

from pathlib import Path

import click
from pydantic import ValidationError

from tributo.job import TributoClient
from tributo.vector_index.contracts import (
    VectorCompactRequest,
    VectorIndexBuildRequest,
    VectorOptimizeRequest,
    VectorSearchRequest,
)
from tributo.vector_index.job import (
    VectorBuildJobRequest,
    VectorCompactJobRequest,
    VectorJobRequest,
    VectorOptimizeJobRequest,
    VectorSearchJobRequest,
    parse_job_result,
    submit_vector_job,
)


@click.group(name="vector")
def vector() -> None:
    """Build, search, and maintain distributed Lance vector indices."""


def _submit(address: str, job_request: VectorJobRequest) -> None:
    try:
        job_id = submit_vector_job(address=address, job_request=job_request)
    except Exception as exc:
        raise click.ClickException(
            f"vector job submission failed ({type(exc).__name__})"
        ) from exc
    click.echo(f"Vector job submitted: {job_id}")


@vector.command("build")
@click.option("--address", required=True, help="Ray dashboard URL")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="VectorIndexBuildRequest JSON file",
)
def vector_build(address: str, config: str) -> None:
    """Submit a distributed Lance vector-index build."""
    try:
        request = VectorIndexBuildRequest.model_validate_json(
            Path(config).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise click.ClickException(
            f"invalid vector build config ({type(exc).__name__})"
        ) from exc
    _submit(address, VectorBuildJobRequest(request=request))


@vector.command("search")
@click.option("--address", required=True, help="Ray dashboard URL")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="VectorSearchRequest JSON file",
)
def vector_search(address: str, config: str) -> None:
    """Submit a fixed-version distributed Top-K query."""
    try:
        request = VectorSearchRequest.model_validate_json(
            Path(config).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise click.ClickException(
            f"invalid vector search config ({type(exc).__name__})"
        ) from exc
    _submit(address, VectorSearchJobRequest(request=request))


@vector.command("optimize")
@click.option("--address", required=True, help="Ray dashboard URL")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="VectorOptimizeRequest JSON file",
)
def vector_optimize(address: str, config: str) -> None:
    """Submit incremental Lance index optimization."""
    try:
        request = VectorOptimizeRequest.model_validate_json(
            Path(config).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise click.ClickException(
            f"invalid vector optimize config ({type(exc).__name__})"
        ) from exc
    _submit(address, VectorOptimizeJobRequest(request=request))


@vector.command("compact")
@click.option("--address", required=True, help="Ray dashboard URL")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="VectorCompactRequest JSON file",
)
def vector_compact(address: str, config: str) -> None:
    """Submit distributed Lance file compaction."""
    try:
        request = VectorCompactRequest.model_validate_json(
            Path(config).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise click.ClickException(
            f"invalid vector compact config ({type(exc).__name__})"
        ) from exc
    _submit(address, VectorCompactJobRequest(request=request))


@vector.command("result")
@click.option("--address", required=True, help="Ray dashboard URL")
@click.argument("job_id")
def vector_result(address: str, job_id: str) -> None:
    """Print the validated structured receipt from a completed Ray Job."""
    try:
        logs = TributoClient(address).get_logs(job_id)
        result = parse_job_result(logs)
    except Exception as exc:
        raise click.ClickException(
            f"vector result retrieval failed ({type(exc).__name__})"
        ) from exc
    click.echo(result.model_dump_json(indent=2, exclude_none=True))

"""Ray Jobs transport for Lance vector-index operations."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from tributo._common.logging import configure_logging
from tributo._common.runtime_env import build_runtime_env
from tributo._common.submission_id import generate_submission_id
from tributo.job import TributoClient
from tributo.runtime import RuntimeExecutionMode, RuntimeTarget
from tributo.vector_index.contracts import (
    VectorCompactRequest,
    VectorIndexBuildReceipt,
    VectorIndexBuildRequest,
    VectorMaintenanceReceipt,
    VectorOptimizeRequest,
    VectorSearchReceipt,
    VectorSearchRequest,
)
from tributo.vector_index.errors import (
    VectorIndexConfigurationError,
    safe_vector_error_diagnostic,
)
from tributo.vector_index.index_job import build_vector_index
from tributo.vector_index.maintenance import (
    compact_vector_dataset,
    optimize_vector_indices,
)
from tributo.vector_index.search import search_vectors

REQUEST_ENV = "TRIBUTO_VECTOR_JOB_REQUEST_B64"
RESULT_MARKER = "TRIBUTO_VECTOR_RESULT="
FAILURE_MARKER = "TRIBUTO_VECTOR_FAILURE="
_MAX_REQUEST_BYTES = 65_536


class _JobModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


class VectorBuildJobRequest(_JobModel):
    operation: Literal["build"] = "build"
    request: VectorIndexBuildRequest


class VectorSearchJobRequest(_JobModel):
    operation: Literal["search"] = "search"
    request: VectorSearchRequest


class VectorOptimizeJobRequest(_JobModel):
    operation: Literal["optimize"] = "optimize"
    request: VectorOptimizeRequest


class VectorCompactJobRequest(_JobModel):
    operation: Literal["compact"] = "compact"
    request: VectorCompactRequest


VectorJobRequest: TypeAlias = Annotated[
    VectorBuildJobRequest
    | VectorSearchJobRequest
    | VectorOptimizeJobRequest
    | VectorCompactJobRequest,
    Field(discriminator="operation"),
]


class VectorBuildJobResult(_JobModel):
    operation: Literal["build"] = "build"
    receipt: VectorIndexBuildReceipt


class VectorSearchJobResult(_JobModel):
    operation: Literal["search"] = "search"
    receipt: VectorSearchReceipt


class VectorOptimizeJobResult(_JobModel):
    operation: Literal["optimize"] = "optimize"
    receipt: VectorMaintenanceReceipt


class VectorCompactJobResult(_JobModel):
    operation: Literal["compact"] = "compact"
    receipt: VectorMaintenanceReceipt


VectorJobResult: TypeAlias = Annotated[
    VectorBuildJobResult
    | VectorSearchJobResult
    | VectorOptimizeJobResult
    | VectorCompactJobResult,
    Field(discriminator="operation"),
]

_REQUEST_ADAPTER: TypeAdapter[VectorJobRequest] = TypeAdapter(VectorJobRequest)
_RESULT_ADAPTER: TypeAdapter[VectorJobResult] = TypeAdapter(VectorJobResult)


def _request_digest(job_request: VectorJobRequest) -> str:
    return job_request.request.request_digest


def _request_key(job_request: VectorJobRequest) -> str:
    return job_request.request.request_key or ""


def encode_job_request(job_request: VectorJobRequest) -> str:
    """Encode a request for the trusted, bounded Ray control-plane transport."""
    payload = job_request.model_dump_json(exclude_none=True).encode("utf-8")
    if len(payload) > _MAX_REQUEST_BYTES:
        raise VectorIndexConfigurationError(
            "Ray Job request exceeds the 64 KiB serialized transport limit"
        )
    return base64.b64encode(payload).decode("ascii")


def decode_job_request(value: str) -> VectorJobRequest:
    """Decode and validate a Ray Job request without logging its payload."""
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        raise VectorIndexConfigurationError(
            "vector job request is not valid base64"
        ) from None
    if len(raw) > _MAX_REQUEST_BYTES:
        raise VectorIndexConfigurationError(
            "Ray Job request exceeds the 64 KiB serialized transport limit"
        )
    try:
        return cast(VectorJobRequest, _REQUEST_ADAPTER.validate_json(raw))
    except Exception as exc:
        raise VectorIndexConfigurationError(
            f"vector job request validation failed ({type(exc).__name__})"
        ) from None


def submit_vector_job(
    *,
    address: str,
    runtime_target: RuntimeTarget | None = None,
    job_request: VectorJobRequest,
    project_root: Path | None = None,
    client: TributoClient | None = None,
) -> str:
    """Submit through TributoClient using the trusted Ray control-plane boundary."""
    if runtime_target is not None:
        if runtime_target.execution_mode is RuntimeExecutionMode.LOCAL:
            raise VectorIndexConfigurationError(
                "vector index Jobs require an attached or managed Ray Jobs runtime"
            )
        if runtime_target.is_managed:
            raise VectorIndexConfigurationError(
                "managed runtime targets require a lifecycle-aware vector runner"
            )
        address = runtime_target.require_jobs_address()
    encoded = encode_job_request(job_request)
    request_digest = _request_digest(job_request)
    submission_id = generate_submission_id(
        "vector",
        job_request.operation,
        _request_key(job_request),
        request_digest,
    )
    runtime_env = build_runtime_env(
        project_root=project_root,
        env_vars={REQUEST_ENV: encoded},
    )
    submitter = client or TributoClient(address)
    return submitter.submit(
        entrypoint="python -m tributo.vector_index.job",
        runtime_env=runtime_env,
        metadata={
            "tributo.operation": f"vector-{job_request.operation}",
            "tributo.request_digest": request_digest,
        },
        submission_id=submission_id,
    )


def run_job_request(job_request: VectorJobRequest) -> VectorJobResult:
    """Execute one validated operation inside an initialized Ray Job driver."""
    if isinstance(job_request, VectorBuildJobRequest):
        return VectorBuildJobResult(receipt=build_vector_index(job_request.request))
    if isinstance(job_request, VectorSearchJobRequest):
        return VectorSearchJobResult(receipt=search_vectors(job_request.request))
    if isinstance(job_request, VectorOptimizeJobRequest):
        return VectorOptimizeJobResult(
            receipt=optimize_vector_indices(job_request.request)
        )
    return VectorCompactJobResult(receipt=compact_vector_dataset(job_request.request))


def parse_job_result(logs: str) -> VectorJobResult:
    """Parse the last structured result marker from Ray Job logs."""
    marker_payloads = [
        line[len(RESULT_MARKER) :]
        for line in logs.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if not marker_payloads:
        raise VectorIndexConfigurationError("Ray Job logs contain no vector result")
    try:
        return cast(VectorJobResult, _RESULT_ADAPTER.validate_json(marker_payloads[-1]))
    except Exception as exc:
        raise VectorIndexConfigurationError(
            f"Ray Job result validation failed ({type(exc).__name__})"
        ) from None


def _failure_payload(exc: BaseException) -> str:
    diagnostic, cause_types = safe_vector_error_diagnostic(exc)
    return json.dumps(
        {
            "status": "failed",
            "error_type": type(exc).__name__,
            "diagnostic": diagnostic,
            "cause_types": cause_types,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: list[str] | None = None) -> int:
    """Cluster entrypoint with structured, credential-safe diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    configure_logging(log_format="json")
    try:
        encoded = os.environ.get(REQUEST_ENV)
        if encoded is None:
            raise VectorIndexConfigurationError(
                f"required environment variable {REQUEST_ENV} is not set"
            )
        job_request = decode_job_request(encoded)
        import ray

        ray.init(address="auto", ignore_reinit_error=True)
        result = run_job_request(job_request)
        print(RESULT_MARKER + result.model_dump_json(exclude_none=True), flush=True)
        return 0
    except Exception as exc:
        print(FAILURE_MARKER + _failure_payload(exc), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

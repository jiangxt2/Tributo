"""Distributed Lance vector-index build orchestration."""

from __future__ import annotations

import time
from typing import Any

from tributo.util.annotations import PublicAPI
from tributo.vector_index.contracts import (
    CoverageStatus,
    VectorIndexBuildReceipt,
    VectorIndexBuildRequest,
    VectorIndexType,
)
from tributo.vector_index.errors import VectorIndexConfigurationError
from tributo.vector_index.lance_ray_adapter import (
    LanceRayAdapter,
    summarize_index_coverage,
    validate_vector_schema,
)

_PQ_NUM_BITS = 8


def _dataset_version(dataset: Any) -> int:
    version = getattr(dataset, "version", None)
    if type(version) is not int:
        raise VectorIndexConfigurationError(
            "Lance dataset did not expose an integer version"
        )
    return version


def _validate_build_capacity(
    request: VectorIndexBuildRequest,
    *,
    dimension: int,
    row_count: int,
) -> None:
    if request.num_partitions is not None:
        required = request.num_partitions * request.sample_rate
        if required > row_count:
            raise VectorIndexConfigurationError(
                "dataset has too few rows for num_partitions * sample_rate"
            )
    if request.index_type is VectorIndexType.IVF_PQ:
        if request.num_sub_vectors is None:
            raise VectorIndexConfigurationError("IVF_PQ requires num_sub_vectors")
        if dimension % request.num_sub_vectors != 0:
            raise VectorIndexConfigurationError(
                "vector dimension must be divisible by num_sub_vectors"
            )
        # Lance-Ray 0.5.0 verifies PQ training capacity as
        # (2 ** num_bits) * sample_rate and fixes num_bits to its default of 8.
        required = (2**_PQ_NUM_BITS) * request.sample_rate
        if required > row_count:
            raise VectorIndexConfigurationError(
                "dataset has too few rows for IVF_PQ codebook training"
            )


@PublicAPI(stability="alpha")
def build_vector_index(
    request: VectorIndexBuildRequest,
    *,
    adapter: LanceRayAdapter | None = None,
) -> VectorIndexBuildReceipt:
    """Build an index and derive post-commit coverage from Lance metadata."""
    started = time.monotonic()
    backend = adapter or LanceRayAdapter()
    runtime = backend.runtime_evidence(
        num_workers=request.num_workers,
        resources=request.worker_resources,
    )
    current_access = backend.current_dataset(request.dataset)
    planning_dataset = current_access.dataset
    planning_version = _dataset_version(planning_dataset)

    if request.dataset.version is not None:
        requested_access = backend.requested_dataset(request.dataset)
        requested_version = _dataset_version(requested_access.dataset)
        if requested_version != planning_version:
            raise VectorIndexConfigurationError(
                "historical-version index builds are unsupported because "
                "Lance-Ray workers reopen the current dataset"
            )

    dimension = validate_vector_schema(planning_dataset, request.column)
    row_count = planning_dataset.count_rows()
    if type(row_count) is not int or row_count <= 0:
        raise VectorIndexConfigurationError("Lance dataset must contain vector rows")
    _validate_build_capacity(
        request,
        dimension=dimension,
        row_count=row_count,
    )

    existing = backend.index_by_name(planning_dataset, request.index_name)
    if existing is not None and not request.replace:
        raise VectorIndexConfigurationError(
            "an index with the requested name already exists; replacement is explicit"
        )

    planning_fragment_ids = backend.fragment_ids(planning_dataset)
    backend.create_index(request)

    output_access = backend.current_dataset(request.dataset)
    output_dataset = output_access.dataset
    output_version = _dataset_version(output_dataset)
    current_fragment_ids = backend.fragment_ids(output_dataset)
    index = backend.index_by_name(output_dataset, request.index_name)
    coverage = summarize_index_coverage(
        planning_fragment_ids=planning_fragment_ids,
        current_fragment_ids=current_fragment_ids,
        index=index,
    )

    warnings: list[str] = []
    if output_version != planning_version:
        warnings.append("dataset version advanced after build planning")
    if coverage.status is CoverageStatus.PARTIAL:
        warnings.append(
            "index coverage is partial after append; keep include_unindexed=true "
            "and run optimize_indices"
        )
    elif coverage.status in {CoverageStatus.STALE, CoverageStatus.INDETERMINATE}:
        warnings.append(
            "index coverage cannot be proven safe after a layout change; rebuild "
            "before using fast_search"
        )

    return VectorIndexBuildReceipt(
        request_digest=request.request_digest,
        dataset_ref=request.dataset.identity_digest,
        planning_base_version=planning_version,
        output_dataset_version=output_version,
        index_name=request.index_name,
        index_type=request.index_type,
        metric=request.metric,
        coverage=coverage,
        num_workers=request.num_workers,
        worker_resources=request.worker_resources,
        runtime=runtime,
        warnings=tuple(warnings),
        elapsed_seconds=time.monotonic() - started,
    )

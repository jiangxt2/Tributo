"""Lance-Ray index optimization and distributed compaction orchestration."""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from tributo.util.annotations import PublicAPI
from tributo.vector_index.contracts import (
    CoverageStatus,
    RayWorkerResources,
    VectorCompactRequest,
    VectorMaintenanceReceipt,
    VectorOptimizeRequest,
)
from tributo.vector_index.errors import VectorIndexConfigurationError
from tributo.vector_index.lance_ray_adapter import LanceRayAdapter


def _dataset_version(dataset: Any) -> int:
    version = getattr(dataset, "version", None)
    if type(version) is not int:
        raise VectorIndexConfigurationError(
            "Lance dataset did not expose an integer version"
        )
    return version


def _metrics_dict(metrics: Any) -> dict[str, Any] | None:
    if metrics is None:
        return None
    if is_dataclass(metrics) and not isinstance(metrics, type):
        return {str(key): value for key, value in asdict(metrics).items()}
    if isinstance(metrics, Mapping):
        return {str(key): value for key, value in metrics.items()}
    values = {
        name: getattr(metrics, name)
        for name in (
            "fragments_removed",
            "fragments_added",
            "files_removed",
            "files_added",
        )
        if hasattr(metrics, name)
    }
    return values or {"result_type": type(metrics).__name__}


def _coverage_warnings(coverages: tuple[Any, ...]) -> tuple[str, ...]:
    statuses = {coverage.status for coverage in coverages}
    if CoverageStatus.STALE in statuses:
        return ("index coverage is stale after maintenance; rebuild is required",)
    if CoverageStatus.INDETERMINATE in statuses:
        return (
            "index coverage is indeterminate after maintenance; rebuild is required",
        )
    if CoverageStatus.PARTIAL in statuses:
        return ("one or more indices still have partial fragment coverage",)
    return ()


@PublicAPI(stability="alpha")
def optimize_vector_indices(
    request: VectorOptimizeRequest,
    *,
    adapter: LanceRayAdapter | None = None,
) -> VectorMaintenanceReceipt:
    """Incrementally add unindexed fragments using Lance-Ray."""
    started = time.monotonic()
    backend = adapter or LanceRayAdapter()
    runtime = backend.runtime_evidence(
        num_workers=1,
        resources=RayWorkerResources(),
    )
    before = backend.current_dataset(request.dataset).dataset
    input_version = _dataset_version(before)
    planning_ids = backend.fragment_ids(before)
    index_names = request.indices or backend.index_names(before)
    if not index_names:
        raise VectorIndexConfigurationError("dataset has no indices to optimize")

    backend.optimize_indices(request)
    after = backend.current_dataset(request.dataset).dataset
    coverages = backend.coverage_for_indices(
        planning_fragment_ids=planning_ids,
        current_dataset=after,
        index_names=index_names,
    )
    return VectorMaintenanceReceipt(
        operation="optimize",
        request_digest=request.request_digest,
        dataset_ref=request.dataset.identity_digest,
        input_dataset_version=input_version,
        output_dataset_version=_dataset_version(after),
        coverage=coverages,
        metrics=None,
        num_workers=1,
        worker_resources=RayWorkerResources(),
        runtime=runtime,
        warnings=_coverage_warnings(coverages),
        elapsed_seconds=time.monotonic() - started,
    )


@PublicAPI(stability="alpha")
def compact_vector_dataset(
    request: VectorCompactRequest,
    *,
    adapter: LanceRayAdapter | None = None,
) -> VectorMaintenanceReceipt:
    """Compact Lance files and verify every existing index after commit."""
    started = time.monotonic()
    backend = adapter or LanceRayAdapter()
    runtime = backend.runtime_evidence(
        num_workers=request.num_workers,
        resources=request.worker_resources,
    )
    before = backend.current_dataset(request.dataset).dataset
    input_version = _dataset_version(before)
    planning_ids = backend.fragment_ids(before)
    index_names = backend.index_names(before)

    metrics = backend.compact_files(request)
    after = backend.current_dataset(request.dataset).dataset
    coverages = backend.coverage_for_indices(
        planning_fragment_ids=planning_ids,
        current_dataset=after,
        index_names=index_names,
    )
    return VectorMaintenanceReceipt(
        operation="compact",
        request_digest=request.request_digest,
        dataset_ref=request.dataset.identity_digest,
        input_dataset_version=input_version,
        output_dataset_version=_dataset_version(after),
        coverage=coverages,
        metrics=_metrics_dict(metrics),
        num_workers=request.num_workers,
        worker_resources=request.worker_resources,
        runtime=runtime,
        warnings=_coverage_warnings(coverages),
        elapsed_seconds=time.monotonic() - started,
    )

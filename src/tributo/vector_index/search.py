"""Fixed-version distributed vector-search orchestration."""

from __future__ import annotations

import time
from typing import Any

from tributo.util.annotations import PublicAPI
from tributo.vector_index.contracts import (
    ResultDeliveryMode,
    VectorSearchReceipt,
    VectorSearchRequest,
)
from tributo.vector_index.errors import VectorIndexConfigurationError
from tributo.vector_index.lance_ray_adapter import (
    LanceRayAdapter,
    validate_search_index_coverage,
    validate_vector_index_target,
    validate_vector_schema,
)
from tributo.vector_index.result_writer import inline_rows, write_parquet_result


def _dataset_version(dataset: Any) -> int:
    version = getattr(dataset, "version", None)
    if type(version) is not int:
        raise VectorIndexConfigurationError(
            "Lance dataset did not expose an integer version"
        )
    return version


@PublicAPI(stability="alpha")
def search_vectors(
    request: VectorSearchRequest,
    *,
    adapter: LanceRayAdapter | None = None,
) -> VectorSearchReceipt:
    """Execute a global Top-K query on one pre-opened Lance version."""
    started = time.monotonic()
    backend = adapter or LanceRayAdapter()
    runtime = backend.runtime_evidence(
        num_workers=request.num_workers,
        resources=request.worker_resources,
    )
    access = backend.requested_dataset(request.dataset)
    dataset = access.dataset
    dataset_version = _dataset_version(dataset)
    dimension = validate_vector_schema(dataset, request.column)
    if len(request.query_vector) != dimension:
        raise VectorIndexConfigurationError(
            "query vector dimension does not match the Lance vector column"
        )
    index = backend.index_by_name(dataset, request.index_name)
    if index is None:
        raise VectorIndexConfigurationError("requested vector index does not exist")
    validate_vector_index_target(
        index,
        column=request.column,
        metric=request.metric.value,
    )
    validate_search_index_coverage(dataset, index)

    table = backend.vector_search(request, dataset)
    if request.result.mode is ResultDeliveryMode.INLINE:
        rows = inline_rows(
            table,
            limit=request.result.inline_max_rows,
            max_bytes=request.result.inline_max_bytes,
        )
        output_uri = None
        output_format = None
    else:
        rows = ()
        output_uri = write_parquet_result(
            table,
            output=request.result,
            storage_profile=request.dataset.storage_profile,
        )
        output_format = request.result.format

    return VectorSearchReceipt(
        request_digest=request.request_digest,
        dataset_ref=request.dataset.identity_digest,
        dataset_version=dataset_version,
        index_name=request.index_name,
        metric=request.metric,
        k=request.k,
        row_count=table.num_rows,
        include_unindexed=request.include_unindexed,
        fast_search=request.fast_search,
        oversample_factor=request.oversample_factor,
        num_workers=request.num_workers,
        worker_resources=request.worker_resources,
        delivery_mode=request.result.mode,
        inline_rows=rows,
        output_uri=output_uri,
        output_format=output_format,
        runtime=runtime,
        elapsed_seconds=time.monotonic() - started,
    )

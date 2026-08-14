"""Distributed Lance vector-index construction, maintenance, and search."""

from __future__ import annotations

from tributo.vector_index.contracts import (
    CoverageStatus,
    LanceCompactionOptions,
    LanceDatasetRef,
    LanceScannerOptions,
    RayWorkerResources,
    ResultDeliveryMode,
    SearchResultOutput,
    VectorCompactRequest,
    VectorIndexBuildReceipt,
    VectorIndexBuildRequest,
    VectorIndexType,
    VectorMaintenanceReceipt,
    VectorMetric,
    VectorOptimizeRequest,
    VectorSearchReceipt,
    VectorSearchRequest,
)
from tributo.vector_index.index_job import build_vector_index
from tributo.vector_index.maintenance import (
    compact_vector_dataset,
    optimize_vector_indices,
)
from tributo.vector_index.search import search_vectors

__all__ = [
    "CoverageStatus",
    "LanceCompactionOptions",
    "LanceDatasetRef",
    "LanceScannerOptions",
    "RayWorkerResources",
    "ResultDeliveryMode",
    "SearchResultOutput",
    "VectorCompactRequest",
    "VectorIndexBuildReceipt",
    "VectorIndexBuildRequest",
    "VectorIndexType",
    "VectorMaintenanceReceipt",
    "VectorMetric",
    "VectorOptimizeRequest",
    "VectorSearchReceipt",
    "VectorSearchRequest",
    "build_vector_index",
    "compact_vector_dataset",
    "optimize_vector_indices",
    "search_vectors",
]

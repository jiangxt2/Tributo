"""Build and search request examples for an existing Lance dataset."""

from __future__ import annotations

from tributo.vector_index import (
    LanceDatasetRef,
    VectorIndexBuildReceipt,
    VectorIndexBuildRequest,
    VectorIndexType,
    VectorMetric,
    VectorSearchReceipt,
    VectorSearchRequest,
    build_vector_index,
    search_vectors,
)


def build(dataset_uri: str) -> VectorIndexBuildReceipt:
    """Build a cosine IVF_FLAT index through the active Ray context."""
    request = VectorIndexBuildRequest(
        dataset=LanceDatasetRef(uri=dataset_uri),
        column="embedding",
        index_name="embedding_idx",
        index_type=VectorIndexType.IVF_FLAT,
        metric=VectorMetric.COSINE,
        num_workers=4,
        num_partitions=64,
    )
    return build_vector_index(request)


def search(dataset_uri: str, version: int) -> VectorSearchReceipt:
    """Search a fixed dataset version through the active Ray context."""
    request = VectorSearchRequest(
        dataset=LanceDatasetRef(uri=dataset_uri, version=version),
        column="embedding",
        index_name="embedding_idx",
        query_vector=(0.2, 0.1, 0.4, 0.3),
        k=10,
        columns=("entity_id",),
    )
    return search_vectors(request)

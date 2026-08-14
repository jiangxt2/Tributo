"""Contract tests for distributed Lance vector operations."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from tributo.vector_index.contracts import (
    LanceCompactionOptions,
    LanceDatasetRef,
    LanceScannerOptions,
    RayWorkerResources,
    ResultDeliveryMode,
    SearchResultOutput,
    VectorCompactRequest,
    VectorIndexBuildRequest,
    VectorIndexType,
    VectorMetric,
    VectorOptimizeRequest,
    VectorSearchRequest,
)


@pytest.fixture
def dataset_ref(tmp_path) -> LanceDatasetRef:
    return LanceDatasetRef(uri=str(tmp_path / "vectors.lance"))


def test_dataset_ref_accepts_uri_or_namespace_but_not_both(tmp_path) -> None:
    uri = str(tmp_path / "vectors.lance")
    assert LanceDatasetRef(uri=uri).uri == uri
    namespace = LanceDatasetRef(namespace_impl="dir", table_id=("db", "vectors"))
    assert namespace.table_id == ("db", "vectors")

    with pytest.raises(ValidationError, match="exactly one"):
        LanceDatasetRef(
            uri=uri,
            namespace_impl="dir",
            table_id=("vectors",),
        )
    with pytest.raises(ValidationError, match="exactly one"):
        LanceDatasetRef()


@pytest.mark.parametrize(
    "uri",
    [
        "s3://user:password@bucket/table.lance",
        "s3://bucket/table.lance?X-Amz-Signature=secret",
        "https://example.test/table.lance",
        "relative/table.lance",
    ],
)
def test_dataset_ref_rejects_unsafe_or_ambiguous_uris(uri: str) -> None:
    with pytest.raises(ValidationError):
        LanceDatasetRef(uri=uri)


def test_dataset_ref_namespace_properties_are_environment_references_only() -> None:
    ref = LanceDatasetRef(
        namespace_impl="rest",
        namespace_properties_env="TRIBUTO_LANCE_NAMESPACE_PROD",
        table_id=("db", "vectors"),
    )
    assert ref.namespace_properties_env == "TRIBUTO_LANCE_NAMESPACE_PROD"
    with pytest.raises(ValidationError, match="TRIBUTO_LANCE_NAMESPACE"):
        LanceDatasetRef(
            namespace_impl="rest",
            namespace_properties_env="LANCE_PROPERTIES",
            table_id=("vectors",),
        )


def test_dataset_ref_versions_are_numeric_or_tags(tmp_path) -> None:
    uri = str(tmp_path / "vectors.lance")
    assert LanceDatasetRef(uri=uri, version=7).version == 7
    assert LanceDatasetRef(uri=uri, version="published").version == "published"
    with pytest.raises(ValidationError):
        LanceDatasetRef(uri=uri, version=0)
    with pytest.raises(ValidationError):
        LanceDatasetRef(uri=uri, version="invalid tag")


def test_build_contract_has_fail_closed_defaults(dataset_ref) -> None:
    request = VectorIndexBuildRequest(
        dataset=dataset_ref,
        column="vector",
        index_name="vector_idx",
        index_type=VectorIndexType.IVF_FLAT,
    )
    assert request.replace is False
    assert request.metric is VectorMetric.L2
    assert len(request.request_digest) == 64
    with pytest.raises(ValidationError):
        VectorIndexBuildRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "coverage_policy": "reject_stale_layout",
            }
        )


def test_build_contract_scopes_pq_parameters(dataset_ref) -> None:
    with pytest.raises(ValidationError, match="requires num_sub_vectors"):
        VectorIndexBuildRequest(
            dataset=dataset_ref,
            column="vector",
            index_name="vector_idx",
            index_type="IVF_PQ",
        )
    with pytest.raises(ValidationError, match="only valid for IVF_PQ"):
        VectorIndexBuildRequest(
            dataset=dataset_ref,
            column="vector",
            index_name="vector_idx",
            index_type="IVF_FLAT",
            num_sub_vectors=4,
        )


def test_unverified_index_types_and_metrics_are_rejected(dataset_ref) -> None:
    with pytest.raises(ValidationError):
        VectorIndexBuildRequest(
            dataset=dataset_ref,
            column="vector",
            index_name="vector_idx",
            index_type="IVF_RQ",
        )
    with pytest.raises(ValidationError):
        VectorIndexBuildRequest(
            dataset=dataset_ref,
            column="vector",
            index_name="vector_idx",
            index_type="IVF_FLAT",
            metric="hamming",
        )


def test_search_defaults_preserve_unindexed_data(dataset_ref) -> None:
    request = VectorSearchRequest(
        dataset=dataset_ref,
        column="vector",
        query_vector=(0.0, 1.0),
        index_name="vector_idx",
    )
    assert request.include_unindexed is True
    assert request.fast_search is False
    assert request.oversample_factor == 1.0
    assert "query_vector" not in repr(request)


def test_search_rejects_unsafe_bounds_and_nonfinite_vectors(dataset_ref) -> None:
    with pytest.raises(ValidationError, match="minimum_nprobes"):
        VectorSearchRequest(
            dataset=dataset_ref,
            column="vector",
            query_vector=(0.0, 1.0),
            index_name="vector_idx",
            minimum_nprobes=4,
            maximum_nprobes=2,
        )
    with pytest.raises(ValidationError, match="finite"):
        VectorSearchRequest(
            dataset=dataset_ref,
            column="vector",
            query_vector=(0.0, math.nan),
            index_name="vector_idx",
        )
    with pytest.raises(ValidationError, match="fast_search"):
        VectorSearchRequest(
            dataset=dataset_ref,
            column="vector",
            query_vector=(0.0, 1.0),
            index_name="vector_idx",
            fast_search=True,
        )


def test_search_rejects_oversized_inputs_and_duplicate_columns(dataset_ref) -> None:
    with pytest.raises(ValidationError, match="dimension bound"):
        VectorSearchRequest(
            dataset=dataset_ref,
            column="vector",
            query_vector=(0.0,) * 65_537,
            index_name="vector_idx",
        )
    with pytest.raises(ValidationError, match="at most 8192 characters"):
        VectorSearchRequest(
            dataset=dataset_ref,
            column="vector",
            query_vector=(0.0, 1.0),
            index_name="vector_idx",
            filter="x" * 8_193,
        )
    with pytest.raises(ValidationError, match="duplicates"):
        VectorSearchRequest(
            dataset=dataset_ref,
            column="vector",
            query_vector=(0.0, 1.0),
            index_name="vector_idx",
            columns=("id", "id"),
        )


def test_scanner_options_are_an_explicit_allowlist(dataset_ref) -> None:
    options = LanceScannerOptions(batch_size=128, prefilter=True)
    assert options.to_lance_options() == {"batch_size": 128, "prefilter": True}
    with pytest.raises(ValidationError):
        LanceScannerOptions.model_validate({"fragments": [1, 2]})
    with pytest.raises(ValidationError):
        VectorSearchRequest.model_validate(
            {
                "dataset": dataset_ref.model_dump(),
                "column": "vector",
                "query_vector": [0.0, 1.0],
                "index_name": "vector_idx",
                "scanner_options": {"nearest": {"k": 100}},
            }
        )


def test_result_delivery_is_bounded_and_explicit(tmp_path) -> None:
    inline = SearchResultOutput()
    assert inline.mode is ResultDeliveryMode.INLINE
    assert inline.inline_max_rows == 100
    assert inline.inline_max_bytes == 1_048_576
    output = SearchResultOutput(
        mode="materialized",
        output_uri=str(tmp_path / "result.parquet"),
    )
    assert output.format == "parquet"
    with pytest.raises(ValidationError, match="requires output_uri"):
        SearchResultOutput(mode="materialized")
    with pytest.raises(ValidationError, match="must not include"):
        SearchResultOutput(
            mode="inline",
            output_uri=str(tmp_path / "result.parquet"),
        )
    with pytest.raises(ValidationError, match="must be local"):
        SearchResultOutput(
            mode="materialized",
            output_uri="file://remote-host/tmp/result.parquet",
        )
    with pytest.raises(ValidationError):
        SearchResultOutput(inline_max_bytes=0)
    with pytest.raises(ValidationError):
        SearchResultOutput(inline_max_bytes=16_777_217)


def test_dataset_file_uri_rejects_remote_host() -> None:
    with pytest.raises(ValidationError, match="must be local"):
        LanceDatasetRef(uri="file://remote-host/tmp/vectors.lance")


def test_maintenance_contracts_reject_historical_mutation(dataset_ref) -> None:
    versioned = dataset_ref.model_copy(update={"version": 2})
    with pytest.raises(ValidationError, match="current version"):
        VectorOptimizeRequest(dataset=versioned)
    with pytest.raises(ValidationError, match="current version"):
        VectorCompactRequest(dataset=versioned)


def test_compaction_and_ray_resource_fields_are_reviewed(dataset_ref) -> None:
    options = LanceCompactionOptions(
        target_rows_per_fragment=1_000,
        compaction_mode="try_binary_copy",
    )
    assert options.to_lance_options()["target_rows_per_fragment"] == 1_000
    resources = RayWorkerResources(resources={"index_worker": 0.5})
    assert resources.to_ray_remote_args()["resources"] == {"index_worker": 0.5}
    with pytest.raises(ValidationError):
        RayWorkerResources(resources={"bad resource": 1.0})
    with pytest.raises(ValidationError):
        RayWorkerResources(memory=0)
    with pytest.raises(ValidationError):
        LanceDatasetRef(uri=dataset_ref.uri, block_size=0)


def test_request_digest_is_stable_but_query_sensitive(dataset_ref) -> None:
    first = VectorSearchRequest(
        dataset=dataset_ref,
        column="vector",
        query_vector=(0.0, 1.0),
        index_name="vector_idx",
    )
    same = VectorSearchRequest.model_validate_json(first.model_dump_json())
    changed = first.model_copy(update={"query_vector": (1.0, 0.0)})
    assert first.request_digest == same.request_digest
    assert first.request_digest != changed.request_digest

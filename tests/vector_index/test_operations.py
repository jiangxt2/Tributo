"""Unit orchestration tests for build, search, and maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import tributo.data.persistence.object_store as object_store_module
from tributo.vector_index.contracts import (
    CoverageStatus,
    LanceDatasetRef,
    ResultDeliveryMode,
    RuntimeVersionEvidence,
    SearchResultOutput,
    VectorCompactRequest,
    VectorIndexBuildRequest,
    VectorOptimizeRequest,
    VectorSearchRequest,
)
from tributo.vector_index.errors import (
    VectorIndexConfigurationError,
    VectorResultDeliveryError,
)
from tributo.vector_index.index_job import build_vector_index
from tributo.vector_index.lance_ray_adapter import summarize_index_coverage
from tributo.vector_index.maintenance import (
    compact_vector_dataset,
    optimize_vector_indices,
)
from tributo.vector_index.result_writer import inline_rows, write_parquet_result
from tributo.vector_index.search import search_vectors


@dataclass
class _Fragment:
    fragment_id: int


@dataclass
class _Segment:
    fragment_ids: set[int]


@dataclass
class _Index:
    name: str
    segments: list[_Segment]
    field_names: list[str] | None = None
    index_type: str | None = None
    details: dict[str, str] | None = None


class _Dataset:
    def __init__(
        self,
        *,
        version: int,
        fragment_ids: set[int],
        indexed_ids: set[int] | None,
        rows: int = 1_024,
        dimension: int = 8,
    ) -> None:
        self.version = version
        self.schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), dimension))])
        self._fragments = [_Fragment(item) for item in sorted(fragment_ids)]
        self._rows = rows
        self._indices = (
            []
            if indexed_ids is None
            else [
                _Index(
                    "vector_idx",
                    [_Segment(set(indexed_ids))],
                    field_names=["vector"],
                    index_type="IVF_FLAT",
                    details={"metric_type": "L2"},
                )
            ]
        )

    def get_fragments(self) -> list[_Fragment]:
        return list(self._fragments)

    def describe_indices(self) -> list[_Index]:
        return list(self._indices)

    def count_rows(self) -> int:
        return self._rows


def _runtime() -> RuntimeVersionEvidence:
    return RuntimeVersionEvidence(
        ray="2.55.1",
        pylance="9.0.0",
        lance_ray="0.5.0",
        pyarrow="19.0.1",
        worker_count=0,
        worker_versions=(),
        worker_validation_complete=True,
    )


class _BuildBackend:
    def __init__(
        self,
        before: _Dataset,
        after: _Dataset,
        requested: _Dataset | None = None,
    ) -> None:
        self.before = before
        self.after = after
        self.requested = requested or before
        self.current_calls = 0
        self.create_calls = 0

    def runtime_evidence(self, **kwargs: Any) -> RuntimeVersionEvidence:
        return _runtime()

    def current_dataset(self, ref: LanceDatasetRef) -> Any:
        self.current_calls += 1
        dataset = self.before if self.current_calls == 1 else self.after
        return SimpleNamespace(dataset=dataset)

    def requested_dataset(self, ref: LanceDatasetRef) -> Any:
        return SimpleNamespace(dataset=self.requested)

    @staticmethod
    def index_by_name(dataset: _Dataset, name: str) -> _Index | None:
        return next(
            (item for item in dataset.describe_indices() if item.name == name), None
        )

    @staticmethod
    def fragment_ids(dataset: _Dataset) -> set[int]:
        return {fragment.fragment_id for fragment in dataset.get_fragments()}

    def create_index(self, request: VectorIndexBuildRequest) -> _Dataset:
        self.create_calls += 1
        return self.after


def _build_request(tmp_path, **updates: Any) -> VectorIndexBuildRequest:
    values: dict[str, Any] = {
        "dataset": LanceDatasetRef(uri=str(tmp_path / "vectors.lance")),
        "column": "vector",
        "index_name": "vector_idx",
        "index_type": "IVF_FLAT",
        "num_workers": 2,
        "num_partitions": 2,
        "sample_rate": 2,
    }
    values.update(updates)
    return VectorIndexBuildRequest(**values)


def test_build_receipt_reports_complete_coverage(tmp_path) -> None:
    before = _Dataset(version=1, fragment_ids={1, 2}, indexed_ids=None)
    after = _Dataset(version=2, fragment_ids={1, 2}, indexed_ids={1, 2})
    backend = _BuildBackend(before, after)
    receipt = build_vector_index(_build_request(tmp_path), adapter=backend)
    assert receipt.planning_base_version == 1
    assert receipt.output_dataset_version == 2
    assert receipt.coverage.status is CoverageStatus.COMPLETE
    assert receipt.coverage.indexed.count == 2
    assert receipt.worker_resources.num_cpus == 1.0
    assert backend.create_calls == 1


def test_build_receipt_reports_append_as_partial(tmp_path) -> None:
    before = _Dataset(version=1, fragment_ids={1, 2}, indexed_ids=None)
    after = _Dataset(version=3, fragment_ids={1, 2, 3}, indexed_ids={1, 2})
    receipt = build_vector_index(
        _build_request(tmp_path),
        adapter=_BuildBackend(before, after),
    )
    assert receipt.coverage.status is CoverageStatus.PARTIAL
    assert receipt.coverage.unindexed.sample_ids == (3,)
    assert any("include_unindexed=true" in warning for warning in receipt.warnings)


def test_build_receipt_reports_stale_layout_change(tmp_path) -> None:
    before = _Dataset(version=1, fragment_ids={1, 2}, indexed_ids=None)
    after = _Dataset(version=3, fragment_ids={3, 4}, indexed_ids={1, 2})
    receipt = build_vector_index(
        _build_request(tmp_path),
        adapter=_BuildBackend(before, after),
    )
    assert receipt.coverage.status is CoverageStatus.STALE
    assert any("rebuild" in warning for warning in receipt.warnings)


def test_build_requires_explicit_replace_for_existing_index(tmp_path) -> None:
    before = _Dataset(version=2, fragment_ids={1, 2}, indexed_ids={1, 2})
    after = _Dataset(version=3, fragment_ids={1, 2}, indexed_ids={1, 2})
    backend = _BuildBackend(before, after)
    with pytest.raises(VectorIndexConfigurationError, match="replacement is explicit"):
        build_vector_index(_build_request(tmp_path), adapter=backend)
    assert backend.create_calls == 0

    replacement_backend = _BuildBackend(before, after)
    receipt = build_vector_index(
        _build_request(tmp_path, replace=True),
        adapter=replacement_backend,
    )
    assert receipt.coverage.status is CoverageStatus.COMPLETE
    assert replacement_backend.create_calls == 1


def test_build_rejects_historical_version_before_lance_ray(tmp_path) -> None:
    before = _Dataset(version=5, fragment_ids={1}, indexed_ids=None)
    historical = _Dataset(version=3, fragment_ids={1}, indexed_ids=None)
    backend = _BuildBackend(before, before, requested=historical)
    request = _build_request(
        tmp_path,
        dataset=LanceDatasetRef(
            uri=str(tmp_path / "vectors.lance"),
            version=3,
        ),
    )
    with pytest.raises(VectorIndexConfigurationError, match="historical-version"):
        build_vector_index(request, adapter=backend)
    assert backend.create_calls == 0


def test_build_validates_pq_dimension_and_training_rows(tmp_path) -> None:
    before = _Dataset(
        version=1,
        fragment_ids={1},
        indexed_ids=None,
        rows=128,
        dimension=10,
    )
    backend = _BuildBackend(before, before)
    request = _build_request(
        tmp_path,
        index_type="IVF_PQ",
        num_sub_vectors=4,
    )
    with pytest.raises(VectorIndexConfigurationError, match="divisible"):
        build_vector_index(request, adapter=backend)

    insufficient = _Dataset(
        version=1,
        fragment_ids={1},
        indexed_ids=None,
        rows=511,
        dimension=8,
    )
    with pytest.raises(VectorIndexConfigurationError, match="codebook training"):
        build_vector_index(
            _build_request(
                tmp_path,
                index_type="IVF_PQ",
                num_sub_vectors=2,
                num_partitions=None,
                sample_rate=2,
            ),
            adapter=_BuildBackend(insufficient, insufficient),
        )


def test_build_rejects_empty_dataset(tmp_path) -> None:
    empty = _Dataset(
        version=1,
        fragment_ids=set(),
        indexed_ids=None,
        rows=0,
    )
    with pytest.raises(VectorIndexConfigurationError, match="must contain vector rows"):
        build_vector_index(
            _build_request(tmp_path),
            adapter=_BuildBackend(empty, empty),
        )


class _SearchBackend:
    def __init__(self, dataset: _Dataset, table: pa.Table) -> None:
        self.dataset = dataset
        self.table = table
        self.dataset_seen: Any = None

    def runtime_evidence(self, **kwargs: Any) -> RuntimeVersionEvidence:
        return _runtime()

    def requested_dataset(self, ref: LanceDatasetRef) -> Any:
        return SimpleNamespace(dataset=self.dataset)

    @staticmethod
    def index_by_name(dataset: _Dataset, name: str) -> _Index | None:
        return next(
            (item for item in dataset.describe_indices() if item.name == name), None
        )

    def vector_search(self, request: VectorSearchRequest, dataset: Any) -> pa.Table:
        self.dataset_seen = dataset
        return self.table


def _search_request(tmp_path, **updates: Any) -> VectorSearchRequest:
    values: dict[str, Any] = {
        "dataset": LanceDatasetRef(uri=str(tmp_path / "vectors.lance")),
        "column": "vector",
        "query_vector": tuple(float(i) for i in range(8)),
        "k": 2,
        "index_name": "vector_idx",
        "num_workers": 2,
    }
    values.update(updates)
    return VectorSearchRequest(**values)


def test_search_uses_one_fixed_dataset_and_returns_bounded_inline_rows(
    tmp_path,
) -> None:
    dataset = _Dataset(version=7, fragment_ids={1, 2}, indexed_ids={1, 2})
    table = pa.table({"id": [3, 4], "_distance": [0.0, 1.0]})
    backend = _SearchBackend(dataset, table)
    receipt = search_vectors(_search_request(tmp_path), adapter=backend)
    assert backend.dataset_seen is dataset
    assert receipt.dataset_version == 7
    assert receipt.delivery_mode is ResultDeliveryMode.INLINE
    assert receipt.num_workers == 2
    assert receipt.inline_rows[0]["id"] == 3
    assert receipt.row_count == 2


def test_search_rejects_query_dimension_mismatch(tmp_path) -> None:
    dataset = _Dataset(version=7, fragment_ids={1}, indexed_ids={1})
    backend = _SearchBackend(dataset, pa.table({"id": []}))
    request = _search_request(tmp_path, query_vector=(0.0, 1.0))
    with pytest.raises(VectorIndexConfigurationError, match="dimension"):
        search_vectors(request, adapter=backend)


def test_search_rejects_missing_or_stale_index(tmp_path) -> None:
    missing = _Dataset(version=7, fragment_ids={1}, indexed_ids=None)
    with pytest.raises(VectorIndexConfigurationError, match="does not exist"):
        search_vectors(
            _search_request(tmp_path),
            adapter=_SearchBackend(missing, pa.table({"id": []})),
        )

    stale = _Dataset(version=8, fragment_ids={2}, indexed_ids={1})
    with pytest.raises(VectorIndexConfigurationError, match="stale fragment coverage"):
        search_vectors(
            _search_request(tmp_path),
            adapter=_SearchBackend(stale, pa.table({"id": []})),
        )


def test_search_materializes_parquet_without_inline_rows(tmp_path) -> None:
    dataset = _Dataset(version=7, fragment_ids={1}, indexed_ids={1})
    table = pa.table({"id": [1, 2], "_distance": [0.0, 1.0]})
    backend = _SearchBackend(dataset, table)
    output_path = tmp_path / "results" / "query.parquet"
    request = _search_request(
        tmp_path,
        result={"mode": "materialized", "output_uri": str(output_path)},
    )
    receipt = search_vectors(request, adapter=backend)
    assert receipt.inline_rows == ()
    assert receipt.output_uri == str(output_path)
    assert pq.read_table(output_path).to_pylist() == table.to_pylist()


def test_materialized_delivery_refuses_to_overwrite_local_file(tmp_path) -> None:
    output_path = tmp_path / "query.parquet"
    output_path.write_bytes(b"existing")
    with pytest.raises(VectorResultDeliveryError, match="already exists"):
        write_parquet_result(
            pa.table({"id": [1]}),
            output=SearchResultOutput(
                mode="materialized",
                output_uri=str(output_path),
            ),
            storage_profile=None,
        )


def test_materialized_delivery_writes_s3_object_atomically(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _Client:
        @staticmethod
        def put_object(**kwargs: Any) -> None:
            captured.update(kwargs)
            body = kwargs["Body"]
            captured["payload"] = body.read() if hasattr(body, "read") else body

    monkeypatch.setattr(
        object_store_module.StorageProfileResolver,
        "resolve",
        lambda self, name: SimpleNamespace(
            endpoint="http://minio:9000",
            access_key_id="test",
            secret_access_key="test-secret",
            region="us-east-1",
            use_ssl=False,
            path_style=True,
            profile_name=None,
        ),
    )
    monkeypatch.setattr(
        object_store_module, "get_boto3_client", lambda **kwargs: _Client()
    )
    uri = write_parquet_result(
        pa.table({"id": [1]}),
        output=SearchResultOutput(
            mode="materialized",
            output_uri="s3://bucket/result.parquet",
        ),
        storage_profile="vector-it",
    )
    assert uri == "s3://bucket/result.parquet"
    assert captured["Bucket"] == "bucket"
    assert captured["Key"] == "result.parquet"
    assert captured["IfNoneMatch"] == "*"
    assert captured["ContentType"] == "application/vnd.apache.parquet"
    assert pq.read_table(pa.BufferReader(captured["payload"])).to_pylist() == [
        {"id": 1}
    ]


def test_materialized_delivery_refuses_to_overwrite_s3_object(monkeypatch) -> None:
    class _Conflict(Exception):
        response = {"Error": {"Code": "PreconditionFailed"}}

    class _Client:
        @staticmethod
        def put_object(**kwargs: Any) -> None:
            raise _Conflict("secret endpoint details")

    monkeypatch.setattr(
        object_store_module.StorageProfileResolver,
        "resolve",
        lambda self, name: SimpleNamespace(
            endpoint=None,
            access_key_id=None,
            secret_access_key=None,
            region=None,
            use_ssl=True,
            path_style=False,
            profile_name=None,
        ),
    )
    monkeypatch.setattr(
        object_store_module, "get_boto3_client", lambda **kwargs: _Client()
    )
    with pytest.raises(VectorResultDeliveryError, match="already exists"):
        write_parquet_result(
            pa.table({"id": [1]}),
            output=SearchResultOutput(
                mode="materialized",
                output_uri="s3://bucket/result.parquet",
            ),
            storage_profile="vector-it",
        )


def test_materialized_delivery_reports_unsupported_conditional_put(
    monkeypatch,
) -> None:
    from botocore.exceptions import ParamValidationError

    class _Client:
        @staticmethod
        def put_object(**kwargs: Any) -> None:
            raise ParamValidationError(
                report='Unknown parameter in input: "IfNoneMatch"; secret detail'
            )

    monkeypatch.setattr(
        object_store_module.StorageProfileResolver,
        "resolve",
        lambda self, name: SimpleNamespace(
            endpoint=None,
            access_key_id=None,
            secret_access_key=None,
            region=None,
            use_ssl=True,
            path_style=False,
            profile_name=None,
        ),
    )
    monkeypatch.setattr(
        object_store_module, "get_boto3_client", lambda **kwargs: _Client()
    )
    with pytest.raises(
        VectorResultDeliveryError,
        match="does not support conditional PutObject",
    ) as raised:
        write_parquet_result(
            pa.table({"id": [1]}),
            output=SearchResultOutput(
                mode="materialized",
                output_uri="s3://bucket/result.parquet",
            ),
            storage_profile="vector-it",
        )
    assert "secret detail" not in str(raised.value)


def test_materialized_delivery_does_not_misclassify_parameter_errors(
    monkeypatch,
) -> None:
    from botocore.exceptions import ParamValidationError

    class _Client:
        @staticmethod
        def put_object(**kwargs: Any) -> None:
            raise ParamValidationError(
                report='Unknown parameter in input: "OtherParameter"; secret detail'
            )

    monkeypatch.setattr(
        object_store_module.StorageProfileResolver,
        "resolve",
        lambda self, name: SimpleNamespace(
            endpoint=None,
            access_key_id=None,
            secret_access_key=None,
            region=None,
            use_ssl=True,
            path_style=False,
            profile_name=None,
        ),
    )
    monkeypatch.setattr(
        object_store_module, "get_boto3_client", lambda **kwargs: _Client()
    )
    with pytest.raises(
        VectorResultDeliveryError,
        match=r"Parquet result delivery failed \(ParamValidationError\)",
    ) as raised:
        write_parquet_result(
            pa.table({"id": [1]}),
            output=SearchResultOutput(
                mode="materialized",
                output_uri="s3://bucket/result.parquet",
            ),
            storage_profile="vector-it",
        )
    assert "secret detail" not in str(raised.value)


def test_inline_delivery_never_silently_truncates() -> None:
    table = pa.table({"id": [1, 2]})
    with pytest.raises(VectorResultDeliveryError, match="materialized"):
        inline_rows(table, limit=1, max_bytes=1_024)


def test_inline_delivery_enforces_serialized_byte_limit() -> None:
    table = pa.table({"value": ["界" * 10]})
    with pytest.raises(VectorResultDeliveryError, match="inline_max_bytes"):
        inline_rows(table, limit=1, max_bytes=20)


class _MaintenanceBackend:
    def __init__(self, before: _Dataset, after: _Dataset, metrics: Any = None) -> None:
        self.before = before
        self.after = after
        self.metrics = metrics
        self.current_calls = 0

    def runtime_evidence(self, **kwargs: Any) -> RuntimeVersionEvidence:
        return _runtime()

    def current_dataset(self, ref: LanceDatasetRef) -> Any:
        self.current_calls += 1
        dataset = self.before if self.current_calls == 1 else self.after
        return SimpleNamespace(dataset=dataset)

    @staticmethod
    def fragment_ids(dataset: _Dataset) -> set[int]:
        return {fragment.fragment_id for fragment in dataset.get_fragments()}

    @staticmethod
    def index_names(dataset: _Dataset) -> tuple[str, ...]:
        return tuple(index.name for index in dataset.describe_indices())

    @staticmethod
    def coverage_for_indices(
        *,
        planning_fragment_ids: set[int],
        current_dataset: _Dataset,
        index_names: tuple[str, ...],
    ) -> tuple[Any, ...]:
        current = {fragment.fragment_id for fragment in current_dataset.get_fragments()}
        return tuple(
            summarize_index_coverage(
                planning_fragment_ids=planning_fragment_ids,
                current_fragment_ids=current,
                index=next(
                    (
                        index
                        for index in current_dataset.describe_indices()
                        if index.name == name
                    ),
                    None,
                ),
            )
            for name in index_names
        )

    def optimize_indices(self, request: VectorOptimizeRequest) -> _Dataset:
        return self.after

    def compact_files(self, request: VectorCompactRequest) -> Any:
        return self.metrics


def test_optimize_receipt_proves_new_fragment_coverage(tmp_path) -> None:
    before = _Dataset(version=2, fragment_ids={1, 2}, indexed_ids={1})
    after = _Dataset(version=3, fragment_ids={1, 2}, indexed_ids={1, 2})
    receipt = optimize_vector_indices(
        VectorOptimizeRequest(
            dataset=LanceDatasetRef(uri=str(tmp_path / "vectors.lance")),
            indices=("vector_idx",),
        ),
        adapter=_MaintenanceBackend(before, after),
    )
    assert receipt.operation == "optimize"
    assert receipt.coverage[0].status is CoverageStatus.COMPLETE


def test_compaction_receipt_exposes_stale_coverage_and_metrics(tmp_path) -> None:
    before = _Dataset(version=2, fragment_ids={1, 2}, indexed_ids={1, 2})
    after = _Dataset(version=3, fragment_ids={3}, indexed_ids={1, 2})
    metrics = SimpleNamespace(fragments_removed=2, fragments_added=1)
    receipt = compact_vector_dataset(
        VectorCompactRequest(
            dataset=LanceDatasetRef(uri=str(tmp_path / "vectors.lance")),
            num_workers=2,
        ),
        adapter=_MaintenanceBackend(before, after, metrics),
    )
    assert receipt.operation == "compact"
    assert receipt.coverage[0].status is CoverageStatus.STALE
    assert receipt.metrics == {"fragments_removed": 2, "fragments_added": 1}
    assert any("rebuild" in warning for warning in receipt.warnings)

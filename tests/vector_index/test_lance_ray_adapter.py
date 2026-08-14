"""Contract-double tests for the Lance-Ray public API adapter."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest

from tributo.vector_index.contracts import (
    CoverageStatus,
    LanceDatasetRef,
    RayWorkerResources,
    VectorCompactRequest,
    VectorIndexBuildRequest,
    VectorOptimizeRequest,
    VectorSearchRequest,
)
from tributo.vector_index.errors import (
    VectorIndexConfigurationError,
    VectorIndexDependencyError,
    VectorIndexExecutionError,
)
from tributo.vector_index.lance_ray_adapter import (
    LanceRayAdapter,
    _namespace_properties,
    summarize_index_coverage,
    validate_search_index_coverage,
    validate_vector_index_target,
    validate_vector_schema,
)


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
    def __init__(self, schema: pa.Schema, fragment_ids: set[int] | None = None) -> None:
        self.schema = schema
        self._fragment_ids = fragment_ids or set()

    def get_fragments(self) -> list[Any]:
        return [
            SimpleNamespace(fragment_id=fragment_id)
            for fragment_id in sorted(self._fragment_ids)
        ]


def _dataset_ref(tmp_path) -> LanceDatasetRef:
    return LanceDatasetRef(uri=str(tmp_path / "vectors.lance"))


def _adapter_or_skip() -> LanceRayAdapter:
    try:
        return LanceRayAdapter()
    except VectorIndexDependencyError as exc:
        if "requires 'tributo[vector-index]'" in str(exc):
            pytest.skip("requires the vector-index extra")
        raise


def test_selected_runtime_and_public_api_are_available() -> None:
    adapter = _adapter_or_skip()
    assert adapter is not None


def test_driver_runtime_mismatch_fails_closed(monkeypatch) -> None:
    from tributo.vector_index import lance_ray_adapter as module

    monkeypatch.setattr(
        module,
        "_distribution_versions",
        lambda: {
            "ray": "2.55.1",
            "pylance": "10.0.0",
            "lance-ray": "0.5.0",
            "pyarrow": "19.0.1",
        },
    )
    with pytest.raises(VectorIndexDependencyError, match="pylance expected 9.0.0"):
        _adapter_or_skip()


def test_vector_schema_requires_fixed_floating_vectors() -> None:
    vector_type = pa.list_(pa.float32(), 8)
    dataset = _Dataset(pa.schema([pa.field("vector", vector_type)]))
    assert validate_vector_schema(dataset, "vector") == 8

    variable = _Dataset(pa.schema([pa.field("vector", pa.list_(pa.float32()))]))
    with pytest.raises(VectorIndexConfigurationError, match="fixed-size"):
        validate_vector_schema(variable, "vector")

    integer = _Dataset(pa.schema([pa.field("vector", pa.list_(pa.int32(), 8))]))
    with pytest.raises(VectorIndexConfigurationError, match="floating"):
        validate_vector_schema(integer, "vector")


def test_vector_index_target_requires_matching_column_type_and_metric() -> None:
    index = _Index(
        "idx",
        [],
        field_names=["vector"],
        index_type="IVF_FLAT",
        details={"metric_type": "L2"},
    )
    validate_vector_index_target(index, column="vector", metric="l2")
    with pytest.raises(VectorIndexConfigurationError, match="column"):
        validate_vector_index_target(index, column="embedding", metric="l2")
    with pytest.raises(VectorIndexConfigurationError, match="metric"):
        validate_vector_index_target(index, column="vector", metric="cosine")
    index.index_type = "BTREE"
    with pytest.raises(VectorIndexConfigurationError, match="vector index"):
        validate_vector_index_target(index, column="vector", metric="l2")


def test_coverage_classifies_complete_partial_stale_and_indeterminate() -> None:
    complete = summarize_index_coverage(
        planning_fragment_ids={1, 2},
        current_fragment_ids={1, 2},
        index=_Index("idx", [_Segment({1}), _Segment({2})]),
    )
    assert complete.status is CoverageStatus.COMPLETE
    assert complete.segment_count == 2

    partial = summarize_index_coverage(
        planning_fragment_ids={1, 2},
        current_fragment_ids={1, 2, 3},
        index=_Index("idx", [_Segment({1, 2})]),
    )
    assert partial.status is CoverageStatus.PARTIAL
    assert partial.unindexed.sample_ids == (3,)

    stale = summarize_index_coverage(
        planning_fragment_ids={1, 2},
        current_fragment_ids={3, 4},
        index=_Index("idx", [_Segment({1, 2})]),
    )
    assert stale.status is CoverageStatus.STALE

    missing_planned = summarize_index_coverage(
        planning_fragment_ids={1, 2},
        current_fragment_ids={1, 2, 3},
        index=_Index("idx", [_Segment({1})]),
    )
    assert missing_planned.status is CoverageStatus.INDETERMINATE


def test_overlapping_segments_are_stale() -> None:
    coverage = summarize_index_coverage(
        planning_fragment_ids={1, 2},
        current_fragment_ids={1, 2},
        index=_Index("idx", [_Segment({1, 2}), _Segment({2})]),
    )
    assert coverage.status is CoverageStatus.STALE
    assert coverage.overlapping_fragment_count == 1


def test_search_coverage_accepts_complete_and_append_only_partial() -> None:
    schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), 8))])
    validate_search_index_coverage(
        _Dataset(schema, {1, 2}),
        _Index("idx", [_Segment({1, 2})]),
    )
    validate_search_index_coverage(
        _Dataset(schema, {1, 2, 3}),
        _Index("idx", [_Segment({1, 2})]),
    )


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (_Index("idx", [_Segment({1, 2}), _Segment({2})]), "overlapping"),
        (_Index("idx", [_Segment({1})]), "stale fragment coverage"),
        (_Index("idx", []), "cannot be verified"),
        (SimpleNamespace(name="idx", segments=None), "does not expose verifiable"),
        (
            SimpleNamespace(
                name="idx",
                segments=[SimpleNamespace(fragment_ids="not-a-sequence-of-ids")],
            ),
            "does not expose verifiable fragment coverage",
        ),
    ],
)
def test_search_coverage_rejects_unsafe_metadata(index: Any, message: str) -> None:
    schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), 8))])
    current = {2} if message == "stale fragment coverage" else {1, 2}
    with pytest.raises(VectorIndexConfigurationError, match=message):
        validate_search_index_coverage(_Dataset(schema, current), index)


def test_namespace_properties_resolve_only_at_runtime(monkeypatch) -> None:
    ref = LanceDatasetRef(
        namespace_impl="rest",
        namespace_properties_env="TRIBUTO_LANCE_NAMESPACE_TEST",
        table_id=("db", "vectors"),
    )
    monkeypatch.setenv(
        "TRIBUTO_LANCE_NAMESPACE_TEST",
        '{"endpoint":"https://catalog.test","token":"secret"}',
    )
    assert _namespace_properties(ref) == {
        "endpoint": "https://catalog.test",
        "token": "secret",
    }
    monkeypatch.setenv("TRIBUTO_LANCE_NAMESPACE_TEST", "not-json")
    with pytest.raises(VectorIndexConfigurationError) as error:
        _namespace_properties(ref)
    assert "not-json" not in str(error.value)


class _FakeRemoteFunction:
    def __init__(self, result: dict[str, str]) -> None:
        self.result = result
        self.options_seen: dict[str, Any] | None = None

    def options(self, **kwargs: Any) -> "_FakeRemoteFunction":
        self.options_seen = kwargs
        return self

    def remote(self) -> dict[str, str]:
        return dict(self.result)


class _FakeRay:
    def __init__(self, result: dict[str, str], *, initialized: bool = True) -> None:
        self.remote_function = _FakeRemoteFunction(result)
        self.initialized = initialized

    def is_initialized(self) -> bool:
        return self.initialized

    def remote(self, function: Any) -> _FakeRemoteFunction:
        return self.remote_function

    @staticmethod
    def get(values: list[dict[str, str]]) -> list[dict[str, str]]:
        return values


def test_runtime_evidence_checks_real_worker_payload(monkeypatch) -> None:
    from tributo.vector_index import lance_ray_adapter as module

    versions = {
        "ray": "2.55.1",
        "pylance": "9.0.0",
        "lance-ray": "0.5.0",
        "pyarrow": "19.0.1",
    }
    monkeypatch.setattr(module, "_distribution_versions", lambda: dict(versions))
    adapter = _adapter_or_skip()
    fake_ray = _FakeRay(versions)
    adapter._ray = fake_ray
    evidence = adapter.runtime_evidence(
        num_workers=2,
        resources=RayWorkerResources(num_cpus=0.5),
    )
    assert evidence.worker_validation_complete is True
    assert evidence.worker_count == 2
    assert evidence.worker_versions == (versions,)
    assert fake_ray.remote_function.options_seen == {
        "num_cpus": 0.5,
        "num_gpus": 0.0,
        "scheduling_strategy": "SPREAD",
    }


def test_runtime_evidence_rejects_worker_mismatch(monkeypatch) -> None:
    from tributo.vector_index import lance_ray_adapter as module

    versions = {
        "ray": "2.55.1",
        "pylance": "9.0.0",
        "lance-ray": "0.5.0",
        "pyarrow": "19.0.1",
    }
    monkeypatch.setattr(module, "_distribution_versions", lambda: dict(versions))
    adapter = _adapter_or_skip()
    adapter._ray = _FakeRay({**versions, "pylance": "10.0.0"})
    with pytest.raises(VectorIndexDependencyError, match="differ from driver"):
        adapter.runtime_evidence(
            num_workers=1,
            resources=RayWorkerResources(),
        )


def test_runtime_evidence_requires_initialized_ray(monkeypatch) -> None:
    from tributo.vector_index import lance_ray_adapter as module

    versions = {
        "ray": "2.55.1",
        "pylance": "9.0.0",
        "lance-ray": "0.5.0",
        "pyarrow": "19.0.1",
    }
    monkeypatch.setattr(module, "_distribution_versions", lambda: dict(versions))
    adapter = _adapter_or_skip()
    adapter._ray = _FakeRay(versions, initialized=False)
    with pytest.raises(VectorIndexConfigurationError, match="must be initialized"):
        adapter.runtime_evidence(
            num_workers=1,
            resources=RayWorkerResources(),
        )


def test_create_index_maps_only_reviewed_parameters(tmp_path, monkeypatch) -> None:
    adapter = _adapter_or_skip()
    captured: dict[str, Any] = {}

    def create_index(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(adapter._lance_ray, "create_index", create_index)
    request = VectorIndexBuildRequest(
        dataset=_dataset_ref(tmp_path),
        column="vector",
        index_name="vector_idx",
        index_type="IVF_PQ",
        num_workers=3,
        num_segments=2,
        num_partitions=4,
        num_sub_vectors=2,
        sample_rate=8,
    )
    adapter.create_index(request)
    assert set(captured) == {
        "block_size",
        "column",
        "index_type",
        "metric",
        "name",
        "num_partitions",
        "num_segments",
        "num_sub_vectors",
        "num_workers",
        "ray_remote_args",
        "replace",
        "sample_rate",
        "storage_options",
        "uri",
    }
    assert captured["uri"] == request.dataset.uri
    assert captured["replace"] is False
    assert captured["index_type"] == "IVF_PQ"
    assert captured["num_workers"] == 3
    assert captured["ray_remote_args"] == {"num_cpus": 1.0, "num_gpus": 0.0}


def test_vector_search_maps_fixed_dataset_and_scanner_allowlist(
    tmp_path, monkeypatch
) -> None:
    adapter = _adapter_or_skip()
    captured: dict[str, Any] = {}
    expected = pa.table({"id": [1], "_distance": [0.0]})

    def vector_search(**kwargs: Any) -> pa.Table:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(adapter._lance_ray, "vector_search", vector_search)
    request = VectorSearchRequest(
        dataset=_dataset_ref(tmp_path),
        column="vector",
        query_vector=(0.0, 1.0),
        index_name="vector_idx",
        minimum_nprobes=2,
        scanner_options={"batch_size": 64, "prefilter": True},
    )
    dataset = object()
    assert adapter.vector_search(request, dataset) is expected
    assert set(captured) == {
        "analyze_plan",
        "columns",
        "fast_search",
        "filter",
        "include_unindexed",
        "index_name",
        "nearest",
        "num_workers",
        "oversample_factor",
        "ray_remote_args",
        "scanner_options",
        "uri",
    }
    assert captured["uri"] is dataset
    assert captured["nearest"]["minimum_nprobes"] == 2
    assert captured["nearest"]["metric"] == "l2"
    assert captured["scanner_options"] == {"batch_size": 64, "prefilter": True}
    assert captured["include_unindexed"] is True


def test_maintenance_parameter_mapping(tmp_path, monkeypatch) -> None:
    adapter = _adapter_or_skip()
    optimized: dict[str, Any] = {}
    compacted: dict[str, Any] = {}
    monkeypatch.setattr(
        adapter._lance_ray,
        "optimize_indices",
        lambda **kwargs: optimized.update(kwargs),
    )
    monkeypatch.setattr(
        adapter._lance_ray,
        "compact_files",
        lambda **kwargs: compacted.update(kwargs),
    )
    ref = _dataset_ref(tmp_path)
    adapter.optimize_indices(
        VectorOptimizeRequest(
            dataset=ref,
            indices=("vector_idx",),
            num_indices_to_merge=2,
        )
    )
    adapter.compact_files(
        VectorCompactRequest(
            dataset=ref,
            num_workers=2,
            options={"target_rows_per_fragment": 1_000},
        )
    )
    assert optimized["indices"] == ["vector_idx"]
    assert set(optimized) == {
        "indices",
        "num_indices_to_merge",
        "retrain",
        "storage_options",
        "uri",
    }
    assert optimized["num_indices_to_merge"] == 2
    assert compacted["compaction_options"] == {"target_rows_per_fragment": 1_000}
    assert set(compacted) == {
        "compaction_options",
        "num_workers",
        "ray_remote_args",
        "storage_options",
        "uri",
    }
    assert compacted["num_workers"] == 2


def test_adapter_execution_failure_suppresses_dependency_traceback(
    tmp_path, monkeypatch
) -> None:
    adapter = _adapter_or_skip()

    def fail(**kwargs: Any) -> None:
        raise RuntimeError("secret_access_key=do-not-log")

    monkeypatch.setattr(adapter._lance_ray, "create_index", fail)
    request = VectorIndexBuildRequest(
        dataset=_dataset_ref(tmp_path),
        column="vector",
        index_name="vector_idx",
        index_type="IVF_FLAT",
    )
    with pytest.raises(VectorIndexExecutionError) as error:
        adapter.create_index(request)
    rendered = "".join(
        traceback.format_exception(
            type(error.value),
            error.value,
            error.value.__traceback__,
        )
    )
    assert "RuntimeError" in rendered
    assert "do-not-log" not in rendered

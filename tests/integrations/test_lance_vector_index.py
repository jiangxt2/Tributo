"""Real Ray/Lance-Ray/MinIO vector indexing and search integration gate."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - runtime image can omit pytest
    pytest = None

if os.environ.get("TRIBUTO_DOCKER_RAY_TEST") != "1":
    if pytest is not None:
        pytest.skip(
            "requires execution inside the lifecycle-owned Docker Ray cluster",
            allow_module_level=True,
        )
    raise RuntimeError("This script must run inside the Docker Ray cluster")

try:
    import lance
except ModuleNotFoundError as exc:
    if pytest is not None:
        pytest.skip(
            "requires the vector-index extra",
            allow_module_level=True,
        )
    raise RuntimeError("this integration module requires PyLance") from exc

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import ray

from tributo.inference.contracts import (
    LanceResultSinkRequest,
    LanceVectorColumnSpec,
)
from tributo.integrations.sinks.lance import LanceResultSink
from tributo.job import TributoClient
from tributo.vector_index.contracts import (
    CoverageStatus,
    LanceDatasetRef,
    VectorCompactRequest,
    VectorIndexBuildRequest,
    VectorOptimizeRequest,
    VectorSearchRequest,
)
from tributo.vector_index.errors import (
    VectorIndexConfigurationError,
    VectorIndexExecutionError,
)
from tributo.vector_index.index_job import build_vector_index
from tributo.vector_index.job import (
    VectorSearchJobRequest,
    parse_job_result,
    submit_vector_job,
)
from tributo.vector_index.maintenance import (
    compact_vector_dataset,
    optimize_vector_indices,
)
from tributo.vector_index.search import search_vectors

if pytest is not None:
    pytestmark = [pytest.mark.integration, pytest.mark.distributed]

_DIMENSION = 8
_ROWS_PER_FRAGMENT = 512
_INITIAL_FRAGMENTS = 4
_PROFILE = "vector_it"


class _EmbeddingPredictor:
    """Return a conventional two-dimensional NumPy embedding batch."""

    def __call__(self, batch: pa.Table) -> dict[str, np.ndarray]:
        row_ids = np.asarray(batch.column("id").to_pylist(), dtype=np.int64)
        return {
            "id": row_ids,
            "group": np.asarray(
                ["even" if row_id % 2 == 0 else "odd" for row_id in row_ids]
            ),
            "vector": np.asarray(
                [_vector_for(int(row_id)) for row_id in row_ids],
                dtype=np.float32,
            ),
        }


def _configure_cluster() -> None:
    ray.init(address="auto", ignore_reinit_error=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        alive = [node for node in ray.nodes() if node.get("Alive")]
        if len(alive) >= 2:
            break
        time.sleep(1)
    else:
        raise RuntimeError("vector gate requires a Ray head and one worker")

    @ray.remote(num_cpus=0)
    def runtime_versions() -> dict[str, str]:
        import importlib.metadata as metadata

        return {
            name: metadata.version(name)
            for name in ("ray", "pylance", "lance-ray", "pyarrow")
        }

    evidence = ray.get([runtime_versions.remote(), runtime_versions.remote()])
    assert all(
        item
        == {
            "ray": "2.55.1",
            "pylance": "9.0.0",
            "lance-ray": "0.5.0",
            "pyarrow": "19.0.1",
        }
        for item in evidence
    )


def _s3_filesystem() -> pafs.S3FileSystem:
    endpoint = os.environ["TRIBUTO_MINIO_ENDPOINT"]
    return pafs.S3FileSystem(
        access_key=os.environ["TRIBUTO_MINIO_ACCESS_KEY"],
        secret_key=os.environ["TRIBUTO_MINIO_SECRET_KEY"],
        allow_bucket_creation=True,
        allow_bucket_deletion=True,
        endpoint_override=endpoint.removeprefix("http://").removeprefix("https://"),
        scheme="http" if endpoint.startswith("http://") else "https",
        region="us-east-1",
    )


def _storage_options() -> dict[str, str]:
    return {
        "endpoint": os.environ["TRIBUTO_MINIO_ENDPOINT"],
        "access_key_id": os.environ["TRIBUTO_MINIO_ACCESS_KEY"],
        "secret_access_key": os.environ["TRIBUTO_MINIO_SECRET_KEY"],
        "region": "us-east-1",
        "allow_http": "true",
        "virtual_hosted_style_request": "false",
    }


def _vector_for(row_id: int) -> list[float]:
    return [
        row_id * 0.01,
        (row_id % 17) * 0.1,
        (row_id % 13) * 0.05,
        (row_id % 11) * 0.03,
        (row_id % 7) * 0.02,
        (row_id % 5) * 0.01,
        (row_id % 3) * 0.01,
        1.0,
    ]


def _table(row_ids: list[int], *, duplicate_query: bool = False) -> pa.Table:
    vectors = [
        _vector_for(42) if duplicate_query and row_id == 9_999 else _vector_for(row_id)
        for row_id in row_ids
    ]
    vector_values = pa.array(
        [value for vector in vectors for value in vector],
        type=pa.float32(),
    )
    return pa.table(
        {
            "id": pa.array(row_ids, type=pa.int64()),
            "group": pa.array(
                ["even" if row_id % 2 == 0 else "odd" for row_id in row_ids]
            ),
            "vector": pa.FixedSizeListArray.from_arrays(
                vector_values,
                _DIMENSION,
            ),
        }
    )


def _write_initial_dataset(uri: str, storage_options: dict[str, str]) -> list[int]:
    row_ids = list(range(_INITIAL_FRAGMENTS * _ROWS_PER_FRAGMENT))
    source = ray.data.from_items(
        [{"id": row_id} for row_id in row_ids],
        override_num_blocks=_INITIAL_FRAGMENTS,
    )
    embeddings = source.map_batches(
        _EmbeddingPredictor,
        batch_format="pyarrow",
        batch_size=128,
    )
    inferred_schema = embeddings.schema()
    arrow_schema = getattr(inferred_schema, "base_schema", inferred_schema)
    assert isinstance(arrow_schema, pa.Schema)
    ray_vector_type = arrow_schema.field("vector").type
    assert getattr(ray_vector_type, "extension_name", None) in {
        "ray.data.arrow_tensor",
        "ray.data.arrow_tensor_v2",
    }

    LanceResultSink().write(
        embeddings,
        LanceResultSinkRequest(
            uri=uri,
            storage_profile=_PROFILE,
            mode="create",
            min_rows_per_file=_ROWS_PER_FRAGMENT,
            max_rows_per_file=_ROWS_PER_FRAGMENT,
            vector_columns=(
                LanceVectorColumnSpec(name="vector", dimension=_DIMENSION),
            ),
        ),
        run_id="distributed-embedding-it",
        plan_digest="e" * 64,
    )

    dataset = lance.LanceDataset(uri, storage_options=storage_options)
    assert dataset.schema.field("vector").type == pa.list_(pa.float32(), _DIMENSION)
    persisted = dataset.to_table(columns=["id", "group", "vector"]).sort_by("id")
    assert persisted.num_rows == len(row_ids)
    assert [int(value) for value in persisted["id"].to_pylist()] == row_ids
    np.testing.assert_allclose(
        np.asarray(persisted["vector"].to_pylist(), dtype=np.float32),
        np.asarray([_vector_for(row_id) for row_id in row_ids], dtype=np.float32),
    )
    # Ray may split each input block into multiple batches, and Lance-Ray may
    # materialize those batches as separate fragments.
    assert len(dataset.get_fragments()) >= _INITIAL_FRAGMENTS
    return row_ids


def _build_with_concurrent_append(
    request: VectorIndexBuildRequest,
    *,
    uri: str,
    storage_options: dict[str, str],
) -> Any:
    """Coordinate append after Lance-Ray freezes its fragment batches."""
    import lance_ray.index as lance_ray_index

    original = lance_ray_index._distribute_fragments_balanced
    appended = False

    def append_after_planning(
        fragments: list[Any],
        num_segments: int,
        logger: Any,
    ) -> list[list[int]]:
        nonlocal appended
        batches = original(fragments, num_segments, logger)
        if not appended:
            lance.write_dataset(
                _table([9_999], duplicate_query=True),
                uri,
                mode="append",
                storage_options=storage_options,
            )
            appended = True
        return batches

    # This test-only seam coordinates real concurrent storage behavior. Production
    # code continues to call only lance_ray.create_index's public API.
    lance_ray_index._distribute_fragments_balanced = append_after_planning
    try:
        receipt = build_vector_index(request)
    finally:
        lance_ray_index._distribute_fragments_balanced = original
    assert appended
    return receipt


def _assert_worker_failure_after_concurrent_compaction(
    request: VectorIndexBuildRequest,
    *,
    uri: str,
    storage_options: dict[str, str],
) -> None:
    """Invalidate planned fragments before workers open the shared URI."""
    import lance_ray.index as lance_ray_index

    original = lance_ray_index._distribute_fragments_balanced
    compacted = False

    def compact_after_planning(
        fragments: list[Any],
        num_segments: int,
        logger: Any,
    ) -> list[list[int]]:
        nonlocal compacted
        batches = original(fragments, num_segments, logger)
        if not compacted:
            before = {
                fragment.fragment_id
                for fragment in lance.LanceDataset(
                    uri,
                    storage_options=storage_options,
                ).get_fragments()
            }
            dataset = lance.LanceDataset(uri, storage_options=storage_options)
            dataset.optimize.compact_files(
                target_rows_per_fragment=_ROWS_PER_FRAGMENT * _INITIAL_FRAGMENTS
            )
            after = {
                fragment.fragment_id
                for fragment in lance.LanceDataset(
                    uri,
                    storage_options=storage_options,
                ).get_fragments()
            }
            assert before.isdisjoint(after)
            compacted = True
        return batches

    lance_ray_index._distribute_fragments_balanced = compact_after_planning
    try:
        try:
            build_vector_index(request)
        except VectorIndexExecutionError:
            pass
        else:
            raise AssertionError("concurrent compaction must fail the worker build")
    finally:
        lance_ray_index._distribute_fragments_balanced = original
    assert compacted
    dataset = lance.LanceDataset(uri, storage_options=storage_options)
    assert all(index.name != request.index_name for index in dataset.describe_indices())


def _exact_top_ids(row_ids: list[int], query: list[float], k: int) -> set[int]:
    matrix = np.asarray(
        [
            _vector_for(42) if row_id == 9_999 else _vector_for(row_id)
            for row_id in row_ids
        ],
        dtype=np.float32,
    )
    distances = np.sum((matrix - np.asarray(query, dtype=np.float32)) ** 2, axis=1)
    return {row_ids[index] for index in np.argsort(distances, kind="stable")[:k]}


def _wait_for_job(client: TributoClient, job_id: str) -> str:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        status = client.get_status(job_id)
        if status in {"SUCCEEDED", "FAILED", "STOPPED"}:
            return status
        time.sleep(1)
    raise RuntimeError(f"Ray Job {job_id} did not finish within 180 seconds")


def test_distributed_lance_vector_index_and_search() -> None:
    _configure_cluster()
    filesystem = _s3_filesystem()
    bucket = f"tributo-lance-vector-{uuid.uuid4().hex}"
    bucket_created = False

    try:
        filesystem.create_dir(bucket)
        bucket_created = True
        uri = f"s3://{bucket}/vectors.lance"
        storage_options = _storage_options()
        dataset_ref = LanceDatasetRef(uri=uri, storage_profile=_PROFILE)
        all_ids = _write_initial_dataset(uri, storage_options)

        failure_uri = f"s3://{bucket}/compaction-failure.lance"
        _write_initial_dataset(failure_uri, storage_options)
        _assert_worker_failure_after_concurrent_compaction(
            VectorIndexBuildRequest(
                dataset=LanceDatasetRef(
                    uri=failure_uri,
                    storage_profile=_PROFILE,
                ),
                column="vector",
                index_name="must_not_commit_idx",
                index_type="IVF_FLAT",
                num_workers=2,
                num_segments=4,
                num_partitions=4,
                sample_rate=2,
            ),
            uri=failure_uri,
            storage_options=storage_options,
        )

        flat_request = VectorIndexBuildRequest(
            dataset=dataset_ref,
            column="vector",
            index_name="vector_flat_idx",
            index_type="IVF_FLAT",
            num_workers=2,
            num_segments=4,
            num_partitions=4,
            sample_rate=2,
        )
        flat_receipt = _build_with_concurrent_append(
            flat_request,
            uri=uri,
            storage_options=storage_options,
        )
        all_ids.append(9_999)
        assert flat_receipt.coverage.status is CoverageStatus.PARTIAL
        assert flat_receipt.coverage.unindexed.count == 1
        assert flat_receipt.coverage.segment_count == 4
        assert flat_receipt.coverage.overlapping_fragment_count == 0
        assert flat_receipt.runtime.worker_validation_complete
        assert flat_receipt.runtime.worker_count == 2

        query = _vector_for(42)
        complete_search = search_vectors(
            VectorSearchRequest(
                dataset=dataset_ref,
                column="vector",
                query_vector=tuple(query),
                k=10,
                index_name="vector_flat_idx",
                columns=("id", "group"),
                num_workers=2,
                include_unindexed=True,
            )
        )
        complete_ids = {int(row["id"]) for row in complete_search.inline_rows}
        assert 9_999 in complete_ids
        assert complete_ids == _exact_top_ids(all_ids, query, 10)
        complete_distances = [
            float(row["_distance"]) for row in complete_search.inline_rows
        ]
        assert complete_distances == sorted(complete_distances)

        repeated_search = search_vectors(
            VectorSearchRequest(
                dataset=dataset_ref.model_copy(
                    update={"version": complete_search.dataset_version}
                ),
                column="vector",
                query_vector=tuple(query),
                k=10,
                index_name="vector_flat_idx",
                columns=("id", "group"),
                num_workers=2,
                include_unindexed=True,
            )
        )
        assert sorted(repeated_search.inline_rows, key=lambda row: int(row["id"])) == (
            sorted(complete_search.inline_rows, key=lambda row: int(row["id"]))
        )

        fast_search = search_vectors(
            VectorSearchRequest(
                dataset=dataset_ref,
                column="vector",
                query_vector=tuple(query),
                k=10,
                index_name="vector_flat_idx",
                columns=("id",),
                num_workers=2,
                include_unindexed=False,
                fast_search=True,
            )
        )
        assert 9_999 not in {int(row["id"]) for row in fast_search.inline_rows}

        filtered = search_vectors(
            VectorSearchRequest(
                dataset=dataset_ref,
                column="vector",
                query_vector=tuple(query),
                k=5,
                index_name="vector_flat_idx",
                columns=("id", "group"),
                filter="group = 'odd'",
                num_workers=2,
            )
        )
        assert filtered.row_count == 5
        assert all(row["group"] == "odd" for row in filtered.inline_rows)

        optimize_receipt = optimize_vector_indices(
            VectorOptimizeRequest(
                dataset=dataset_ref,
                indices=("vector_flat_idx",),
            )
        )
        assert optimize_receipt.coverage[0].status is CoverageStatus.COMPLETE

        pq_receipt = build_vector_index(
            VectorIndexBuildRequest(
                dataset=dataset_ref,
                column="vector",
                index_name="vector_pq_idx",
                index_type="IVF_PQ",
                num_workers=2,
                num_segments=4,
                num_partitions=4,
                num_sub_vectors=2,
                sample_rate=2,
            )
        )
        assert pq_receipt.coverage.status is CoverageStatus.COMPLETE
        pq_search = search_vectors(
            VectorSearchRequest(
                dataset=dataset_ref,
                column="vector",
                query_vector=tuple(query),
                k=10,
                index_name="vector_pq_idx",
                columns=("id",),
                num_workers=2,
                refine_factor=2,
            )
        )
        pq_ids = {int(row["id"]) for row in pq_search.inline_rows}
        exact_ids = _exact_top_ids(all_ids, query, 10)
        assert len(pq_ids.intersection(exact_ids)) / 10 >= 0.6

        output_uri = f"s3://{bucket}/results/query-direct.parquet"
        materialized = search_vectors(
            VectorSearchRequest(
                dataset=dataset_ref,
                column="vector",
                query_vector=tuple(query),
                k=5,
                index_name="vector_flat_idx",
                columns=("id",),
                num_workers=2,
                result={"mode": "materialized", "output_uri": output_uri},
            )
        )
        assert materialized.output_uri == output_uri
        result_table = pq.read_table(
            f"{bucket}/results/query-direct.parquet",
            filesystem=filesystem,
        )
        assert result_table.num_rows == 5

        job_request = VectorSearchJobRequest(
            request=VectorSearchRequest(
                dataset=dataset_ref,
                column="vector",
                query_vector=tuple(query),
                k=3,
                index_name="vector_flat_idx",
                columns=("id",),
                num_workers=2,
                request_key=uuid.uuid4().hex,
                result={
                    "mode": "materialized",
                    "output_uri": f"s3://{bucket}/results/query-job.parquet",
                },
            )
        )
        dashboard = os.environ["TRIBUTO_RAY_DASHBOARD_URL"]
        client = TributoClient(dashboard)
        job_id = submit_vector_job(address=dashboard, job_request=job_request)
        status = _wait_for_job(client, job_id)
        logs = client.get_logs(job_id)
        assert status == "SUCCEEDED", logs
        job_result = parse_job_result(logs)
        assert job_result.operation == "search"
        assert job_result.receipt.dataset_version >= flat_receipt.output_dataset_version
        assert job_result.receipt.row_count == 3
        assert job_result.receipt.output_uri == (
            f"s3://{bucket}/results/query-job.parquet"
        )
        assert (
            pq.read_table(
                f"{bucket}/results/query-job.parquet",
                filesystem=filesystem,
            ).num_rows
            == 3
        )

        compaction = compact_vector_dataset(
            VectorCompactRequest(
                dataset=dataset_ref,
                num_workers=2,
                options={"target_rows_per_fragment": 1_024, "num_threads": 1},
            )
        )
        assert compaction.output_dataset_version >= compaction.input_dataset_version
        if compaction.output_dataset_version > compaction.input_dataset_version:
            assert any(
                coverage.status in {CoverageStatus.STALE, CoverageStatus.INDETERMINATE}
                for coverage in compaction.coverage
            )
            assert any("rebuild" in warning for warning in compaction.warnings)
            try:
                search_vectors(
                    VectorSearchRequest(
                        dataset=dataset_ref,
                        column="vector",
                        query_vector=tuple(query),
                        k=1,
                        index_name="vector_flat_idx",
                        columns=("id",),
                        num_workers=2,
                    )
                )
            except VectorIndexConfigurationError as exc:
                assert any(
                    message in str(exc)
                    for message in (
                        "does not exist",
                        "stale fragment coverage",
                        "cannot be verified",
                    )
                )
            else:
                raise AssertionError("search must reject stale index coverage")
    finally:
        if bucket_created:
            filesystem.delete_dir(bucket)


if __name__ == "__main__":
    test_distributed_lance_vector_index_and_search()
    print("distributed Lance vector indexing and search passed")

"""Local-Parquet dual-engine integration test for the Docker Ray cluster.

Run inside ``ray-head`` with ``TRIBUTO_DOCKER_RAY_TEST=1``. The test path is
under the cluster's shared ``/workspace`` mount so Ray and Daft workers see the
same Parquet file.
"""

from __future__ import annotations

import os
import uuid
from contextlib import ExitStack
from pathlib import Path

import daft
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import ray

try:
    import pytest
except ModuleNotFoundError:  # Docker image intentionally omits test dependencies.
    pytest = None

from tests.data.ingestion_conformance import assert_dual_engine_conformance
from tributo.data import (
    CastColumn,
    ColumnRename,
    DaftDataFrameHandle,
    DropColumns,
    FillNull,
    FilterComparison,
    FilterEq,
    FilterIsIn,
    FilterNotEq,
    FilterNotNull,
    FilterNull,
    FilterRange,
    IngestionRequest,
    IngestionRuntimeContext,
    Limit,
    ParquetSourceConfig,
    RayDataHandle,
    RenameColumns,
    S3Config,
    SelectColumns,
    TransformPipeline,
    open_ingestion,
    ray_worker_distribution_probe,
)

if pytest is not None:
    pytestmark = pytest.mark.integration
    if os.environ.get("TRIBUTO_DOCKER_RAY_TEST") != "1":
        pytest.skip(
            "requires execution inside the Docker Ray cluster",
            allow_module_level=True,
        )
elif os.environ.get("TRIBUTO_DOCKER_RAY_TEST") != "1":
    raise RuntimeError("This script must run inside the Docker Ray cluster")


def _configure_cluster() -> None:
    ray.init(address="auto", ignore_reinit_error=True)
    daft.set_runner_ray(address="auto", noop_if_initialized=True)


def _pipeline() -> TransformPipeline:
    return TransformPipeline(
        steps=(
            FillNull(column="category", value="unknown"),
            FillNull(column="active", value=False),
            FillNull(column="score", value=0.0),
            FilterNotEq(column="category", value="drop"),
            FilterComparison(column="id", operator="gte", value=2),
            FilterRange(column="score", low=0.5, high=1.0),
            FilterIsIn(column="id", values=(2, 3, 4)),
            FilterNotNull(column="active"),
            RenameColumns(renames=(ColumnRename(source="category", target="label"),)),
            FilterEq(column="label", value="keep"),
            CastColumn(column="numeric", target_type="int64"),
            DropColumns(columns=("active", "unused")),
            SelectColumns(columns=("id", "score", "label", "numeric")),
            Limit(count=10),
        )
    )


def _input_table() -> pa.Table:
    return pa.table(
        {
            "id": [1, 2, 3, 4],
            "score": [0.2, 0.7, 0.9, 0.8],
            "category": ["drop", "keep", "keep", None],
            "active": [True, True, None, True],
            "numeric": ["1", "2", "3", "4"],
            "unused": ["a", "b", "c", "d"],
        }
    )


def _assert_conformance(source: ParquetSourceConfig) -> None:
    pipeline = _pipeline()
    context = IngestionRuntimeContext(
        distribution_probe=ray_worker_distribution_probe,
        require_worker_validation=True,
    )
    with ExitStack() as stack:
        ray_result = stack.enter_context(
            open_ingestion(
                IngestionRequest(
                    source=source,
                    engine="tributo.ray_data",
                    transforms=pipeline,
                ),
                context,
            )
        )
        daft_result = stack.enter_context(
            open_ingestion(
                IngestionRequest(
                    source=source,
                    engine="tributo.daft",
                    transforms=pipeline,
                ),
                context,
            )
        )

        assert isinstance(ray_result.handle, RayDataHandle)
        assert isinstance(daft_result.handle, DaftDataFrameHandle)
        assert_dual_engine_conformance(
            ray_result,
            ray_result.handle.dataset.take_all(),
            daft_result,
            daft_result.handle.dataframe.to_pylist(),
            expected_rows=[
                {"id": 2, "score": 0.7, "label": "keep", "numeric": 2},
                {"id": 3, "score": 0.9, "label": "keep", "numeric": 3},
            ],
            limit=10,
        )
    assert daft.get_or_create_runner().name == "ray"

    null_pipeline = TransformPipeline(
        steps=(FilterNull(column="active"), SelectColumns(columns=("id",)))
    )
    with ExitStack() as stack:
        ray_null = stack.enter_context(
            open_ingestion(
                IngestionRequest(source=source, engine="ray", transforms=null_pipeline),
                context,
            )
        )
        daft_null = stack.enter_context(
            open_ingestion(
                IngestionRequest(
                    source=source, engine="daft", transforms=null_pipeline
                ),
                context,
            )
        )
        assert_dual_engine_conformance(
            ray_null,
            ray_null.handle.dataset.take_all(),
            daft_null,
            daft_null.handle.dataframe.to_pylist(),
            expected_rows=[{"id": 3}],
        )


def _assert_empty_conformance(source: ParquetSourceConfig) -> None:
    context = IngestionRuntimeContext(
        distribution_probe=ray_worker_distribution_probe,
        require_worker_validation=True,
    )
    with ExitStack() as stack:
        ray_result = stack.enter_context(
            open_ingestion(IngestionRequest(source=source, engine="ray"), context)
        )
        daft_result = stack.enter_context(
            open_ingestion(IngestionRequest(source=source, engine="daft"), context)
        )
        assert_dual_engine_conformance(
            ray_result,
            ray_result.handle.dataset.take_all(),
            daft_result,
            daft_result.handle.dataframe.to_pylist(),
            expected_rows=[],
        )


def _assert_not_equal_null_conformance(source: ParquetSourceConfig) -> None:
    pipeline = TransformPipeline(
        steps=(
            FilterNotEq(column="category", value="drop"),
            SelectColumns(columns=("id",)),
        )
    )
    context = IngestionRuntimeContext(
        distribution_probe=ray_worker_distribution_probe,
        require_worker_validation=True,
    )
    with ExitStack() as stack:
        ray_result = stack.enter_context(
            open_ingestion(
                IngestionRequest(source=source, engine="ray", transforms=pipeline),
                context,
            )
        )
        daft_result = stack.enter_context(
            open_ingestion(
                IngestionRequest(source=source, engine="daft", transforms=pipeline),
                context,
            )
        )
        assert_dual_engine_conformance(
            ray_result,
            ray_result.handle.dataset.take_all(),
            daft_result,
            daft_result.handle.dataframe.to_pylist(),
            expected_rows=[{"id": 2}, {"id": 3}],
        )


def _assert_empty_isin_conformance(source: ParquetSourceConfig) -> None:
    pipeline = TransformPipeline(
        steps=(
            FilterIsIn(column="id", values=()),
            SelectColumns(columns=("id",)),
        )
    )
    context = IngestionRuntimeContext(
        distribution_probe=ray_worker_distribution_probe,
        require_worker_validation=True,
    )
    with ExitStack() as stack:
        ray_result = stack.enter_context(
            open_ingestion(
                IngestionRequest(source=source, engine="ray", transforms=pipeline),
                context,
            )
        )
        daft_result = stack.enter_context(
            open_ingestion(
                IngestionRequest(source=source, engine="daft", transforms=pipeline),
                context,
            )
        )
        assert_dual_engine_conformance(
            ray_result,
            ray_result.handle.dataset.take_all(),
            daft_result,
            daft_result.handle.dataframe.to_pylist(),
            expected_rows=[],
        )


def test_local_parquet_conformance_on_ray_cluster() -> None:
    shared_dir = Path("/workspace/tributo-ingestion-tests")
    shared_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = shared_dir / f"{uuid.uuid4().hex}.parquet"
    empty_path = shared_dir / f"{uuid.uuid4().hex}.parquet"
    pq.write_table(_input_table(), parquet_path)
    pq.write_table(_input_table().slice(0, 0), empty_path)

    try:
        _configure_cluster()
        source = ParquetSourceConfig(path=str(parquet_path))
        _assert_conformance(source)
        _assert_not_equal_null_conformance(source)
        _assert_empty_isin_conformance(source)
        _assert_empty_conformance(ParquetSourceConfig(path=str(empty_path)))
    finally:
        parquet_path.unlink(missing_ok=True)
        empty_path.unlink(missing_ok=True)


def test_s3_parquet_conformance_on_ray_cluster_and_minio() -> None:
    endpoint = os.environ["TRIBUTO_MINIO_ENDPOINT"]
    access_key = os.environ["TRIBUTO_MINIO_ACCESS_KEY"]
    secret_key = os.environ["TRIBUTO_MINIO_SECRET_KEY"]
    bucket = f"tributo-ingestion-{uuid.uuid4().hex}"
    object_path = f"{bucket}/input.parquet"
    filesystem = pafs.S3FileSystem(
        access_key=access_key,
        secret_key=secret_key,
        allow_bucket_creation=True,
        allow_bucket_deletion=True,
        endpoint_override=endpoint.removeprefix("http://").removeprefix("https://"),
        scheme="http" if endpoint.startswith("http://") else "https",
        region="us-east-1",
    )
    filesystem.create_dir(bucket)
    pq.write_table(_input_table(), object_path, filesystem=filesystem)

    try:
        _configure_cluster()
        _assert_conformance(
            ParquetSourceConfig(
                path=f"s3://{object_path}",
                s3=S3Config(
                    endpoint=endpoint,
                    access_key_id=access_key,
                    secret_access_key=secret_key,
                    region="us-east-1",
                ),
            )
        )
    finally:
        filesystem.delete_dir(bucket)


if __name__ == "__main__":
    test_local_parquet_conformance_on_ray_cluster()
    test_s3_parquet_conformance_on_ray_cluster_and_minio()
    print("dual-engine ingestion conformance passed")

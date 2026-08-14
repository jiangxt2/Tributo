"""Dual-engine ingestion and native-write integration test for Docker Ray.

Run inside ``ray-head`` with ``TRIBUTO_DOCKER_RAY_TEST=1``. The test path is
under the cluster's shared writable work mount so Ray and Daft workers see the
same files, while every node imports from one read-only source snapshot. The
MinIO case also exercises native Parquet, CSV, Iceberg, and Lance writes over S3.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

import daft
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

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
    WriteMode,
    WriteRequest,
    default_write_gateway,
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
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        if len(alive_nodes) >= 2:
            break
        time.sleep(1)
    else:
        raise RuntimeError(
            "Docker ingestion gate requires a Ray head and at least one worker"
        )
    daft.set_runner_ray(address="auto", noop_if_initialized=True)
    _assert_source_snapshot_identity(alive_nodes)


def _assert_source_snapshot_identity(alive_nodes: list[dict[str, object]]) -> None:
    snapshot_root = Path(os.environ["TRIBUTO_SOURCE_SNAPSHOT_PATH"])
    expected = (snapshot_root / ".tributo-source-ready").read_text().strip()
    assert len(expected) == 64

    @ray.remote(num_cpus=0)
    def read_snapshot_digest() -> str:
        root = Path(os.environ["TRIBUTO_SOURCE_SNAPSHOT_PATH"])
        return (root / ".tributo-source-ready").read_text().strip()

    futures = {
        str(node["NodeID"]): read_snapshot_digest.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=str(node["NodeID"]), soft=False
            )
        ).remote()
        for node in alive_nodes
    }
    observed = {node_id: ray.get(future) for node_id, future in futures.items()}
    assert observed
    assert set(observed.values()) == {expected}
    print(f"source snapshot {expected} verified by Ray jobs on {len(observed)} nodes")


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


def _assert_native_write_conformance() -> None:
    """Exercise every built-in Ray/Daft writer on the shared Ray cluster."""
    from pyiceberg.catalog import load_catalog

    output_root = Path("/workspace/tributo-work") / (
        f"tributo-write-tests-{uuid.uuid4().hex}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    gateway = default_write_gateway()
    rows = [{"id": 1, "category": "a"}, {"id": 2, "category": "b"}]

    for engine in ("ray", "daft"):
        for target_kind in ("parquet", "csv"):
            target = output_root / f"{engine}-{target_kind}"
            handle = (
                RayDataHandle(ray.data.from_items(rows))
                if engine == "ray"
                else DaftDataFrameHandle(daft.from_pylist(rows))
            )
            receipt = gateway.execute(
                WriteRequest(
                    engine=engine,
                    target_kind=target_kind,
                    target=str(target),
                    mode=WriteMode.OVERWRITE,
                    options=(
                        {"compression": "zstd"} if target_kind == "parquet" else {}
                    ),
                ),
                handle,
            )
            assert receipt.committed is True
            assert list(target.rglob("*." + target_kind))

        lance_target = output_root / f"{engine}-lance"
        handle = (
            RayDataHandle(ray.data.from_items(rows))
            if engine == "ray"
            else DaftDataFrameHandle(daft.from_pylist(rows))
        )
        receipt = gateway.execute(
            WriteRequest(
                engine=engine,
                target_kind="lance",
                target=str(lance_target),
                mode=WriteMode.OVERWRITE,
            ),
            handle,
        )
        assert receipt.committed is True
        assert lance_target.exists()

        print(f"{engine} Iceberg native write")
        catalog_name = f"{engine}-write-catalog"
        catalog_uri = f"sqlite:///{output_root / f'{engine}-iceberg.db'}"
        warehouse = output_root / f"{engine}-iceberg-warehouse"
        catalog = load_catalog(
            catalog_name,
            type="sql",
            uri=catalog_uri,
            warehouse=warehouse.as_uri(),
        )
        catalog.create_namespace("default")
        table_identifier = "default.events"
        handle = (
            RayDataHandle(ray.data.from_items(rows))
            if engine == "ray"
            else DaftDataFrameHandle(daft.from_pylist(rows))
        )
        receipt = gateway.execute(
            WriteRequest(
                engine=engine,
                target_kind="iceberg",
                target=table_identifier,
                mode=WriteMode.OVERWRITE,
                runtime_options={
                    "catalog_name": catalog_name,
                    "catalog_properties": {
                        "type": "sql",
                        "uri": catalog_uri,
                        "warehouse": warehouse.as_uri(),
                    },
                    "table_identifier": table_identifier,
                },
            ),
            handle,
        )
        assert receipt.committed is True
        assert catalog.load_table(table_identifier).scan().to_arrow().num_rows == 2


def _assert_native_s3_write_conformance(s3: S3Config) -> None:
    """Exercise native Ray/Daft Parquet, CSV, Iceberg, and Lance on MinIO."""
    from pyiceberg.catalog import load_catalog

    from tributo.data._s3 import merge_iceberg_properties

    endpoint = s3.endpoint
    access_key = s3.access_key_id
    secret_key = s3.secret_access_key
    assert endpoint and access_key and secret_key
    parsed_endpoint = endpoint.removeprefix("http://").removeprefix("https://")
    filesystem = pafs.S3FileSystem(
        access_key=access_key,
        secret_key=secret_key,
        allow_bucket_creation=True,
        allow_bucket_deletion=True,
        endpoint_override=parsed_endpoint,
        scheme="http" if endpoint.startswith("http://") else "https",
        region=s3.region or "us-east-1",
    )
    bucket = f"tributo-write-{uuid.uuid4().hex}"
    filesystem.create_dir(bucket)
    gateway = default_write_gateway()
    rows = [{"id": 1, "category": "a"}, {"id": 2, "category": "b"}]

    def assert_files(prefix: str) -> None:
        infos = filesystem.get_file_info(pafs.FileSelector(prefix, recursive=True))
        assert any(info.type == pafs.FileType.File for info in infos), prefix

    try:
        for engine in ("ray", "daft"):
            for target_kind in ("parquet", "csv"):
                prefix = f"{bucket}/{engine}-{target_kind}"
                handle = (
                    RayDataHandle(ray.data.from_items(rows))
                    if engine == "ray"
                    else DaftDataFrameHandle(daft.from_pylist(rows))
                )
                receipt = gateway.execute(
                    WriteRequest(
                        engine=engine,
                        target_kind=target_kind,
                        target=f"s3://{prefix}",
                        mode=WriteMode.OVERWRITE,
                        options=(
                            {"compression": "zstd"} if target_kind == "parquet" else {}
                        ),
                        runtime_options={"s3": s3},
                    ),
                    handle,
                )
                assert receipt.committed is True
                assert_files(prefix)

            lance_prefix = f"{bucket}/{engine}-lance"
            handle = (
                RayDataHandle(ray.data.from_items(rows))
                if engine == "ray"
                else DaftDataFrameHandle(daft.from_pylist(rows))
            )
            receipt = gateway.execute(
                WriteRequest(
                    engine=engine,
                    target_kind="lance",
                    target=f"s3://{lance_prefix}",
                    mode=WriteMode.OVERWRITE,
                    runtime_options={"s3": s3},
                ),
                handle,
            )
            assert receipt.committed is True
            assert_files(lance_prefix)

            print(f"{engine} Iceberg native S3 write")
            catalog_name = f"{engine}-s3-write-catalog"
            catalog_uri = (
                f"sqlite:////workspace/tributo-work/{engine}-s3-{uuid.uuid4().hex}.db"
            )
            warehouse = f"s3://{bucket}/{engine}-iceberg-warehouse"
            catalog_properties = {
                "type": "sql",
                "uri": catalog_uri,
                "warehouse": warehouse,
            }
            catalog = load_catalog(
                catalog_name,
                **merge_iceberg_properties(catalog_properties, source=s3),
            )
            catalog.create_namespace("default")
            table_identifier = "default.events"
            handle = (
                RayDataHandle(ray.data.from_items(rows))
                if engine == "ray"
                else DaftDataFrameHandle(daft.from_pylist(rows))
            )
            receipt = gateway.execute(
                WriteRequest(
                    engine=engine,
                    target_kind="iceberg",
                    target=table_identifier,
                    mode=WriteMode.OVERWRITE,
                    runtime_options={
                        "catalog_name": catalog_name,
                        "catalog_properties": catalog_properties,
                        "table_identifier": table_identifier,
                        "s3": s3,
                    },
                ),
                handle,
            )
            assert receipt.committed is True
            assert catalog.load_table(table_identifier).scan().to_arrow().num_rows == 2
    finally:
        filesystem.delete_dir(bucket)


def test_local_parquet_conformance_on_ray_cluster() -> None:
    shared_dir = Path("/workspace/tributo-work/tributo-ingestion-tests")
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
        _assert_native_write_conformance()
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
        _assert_native_s3_write_conformance(
            S3Config(
                endpoint=endpoint,
                access_key_id=access_key,
                secret_access_key=secret_key,
                region="us-east-1",
            )
        )
    finally:
        filesystem.delete_dir(bucket)


if __name__ == "__main__":
    test_local_parquet_conformance_on_ray_cluster()
    test_s3_parquet_conformance_on_ray_cluster_and_minio()
    print("dual-engine ingestion conformance passed")

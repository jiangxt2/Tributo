"""Required dual-engine ingestion conformance on local files and MinIO."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import daft
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from tests.data.ingestion_conformance import assert_dual_engine_conformance
from tributo.data import (
    CsvSourceConfig,
    DaftDataFrameHandle,
    FilterComparison,
    IngestionRequest,
    IngestionRuntimeContext,
    ParquetSourceConfig,
    RayDataHandle,
    S3Config,
    SelectColumns,
    TransformPipeline,
    open_ingestion,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ingestion_conformance,
    pytest.mark.usefixtures("native_daft_ray_local_runtime"),
    pytest.mark.filterwarnings(
        "ignore:Tip.*future versions of Ray.*:FutureWarning",
        "ignore::pytest.PytestUnraisableExceptionWarning",
    ),
]


def _input_table() -> pa.Table:
    return pa.table(
        {
            "id": [1, 2, 3, 4],
            "category": ["drop", "keep", "keep", "drop"],
        }
    )


def _assert_source_conformance(source: Any) -> None:
    assert daft.get_or_create_runner().name == "native"
    context = IngestionRuntimeContext()
    transforms = TransformPipeline(
        steps=(
            FilterComparison(column="id", operator="gte", value=2),
            SelectColumns(columns=("id", "category")),
        )
    )
    ray_result = open_ingestion(
        IngestionRequest(source=source, engine="ray", transforms=transforms),
        context,
    )
    daft_result = open_ingestion(
        IngestionRequest(source=source, engine="daft", transforms=transforms),
        context,
    )
    try:
        assert isinstance(ray_result.handle, RayDataHandle)
        assert isinstance(daft_result.handle, DaftDataFrameHandle)
        assert_dual_engine_conformance(
            ray_result,
            ray_result.handle.dataset.take_all(),
            daft_result,
            daft_result.handle.dataframe.to_pylist(),
            expected_rows=[
                {"id": 2, "category": "keep"},
                {"id": 3, "category": "keep"},
                {"id": 4, "category": "drop"},
            ],
            require_worker_validation=False,
        )
    finally:
        ray_result.close()
        daft_result.close()


@pytest.mark.parametrize("file_format", ["parquet", "csv"])
def test_local_file_dual_engine_conformance(tmp_path: Path, file_format: str) -> None:
    path = tmp_path / f"input.{file_format}"
    if file_format == "parquet":
        pq.write_table(_input_table(), path)
        source: Any = ParquetSourceConfig(path=str(path))
    else:
        pacsv.write_csv(_input_table(), path)
        source = CsvSourceConfig(path=str(path))

    _assert_source_conformance(source)


@pytest.mark.minio_compat
@pytest.mark.usefixtures("s3_environment")
@pytest.mark.parametrize("file_format", ["parquet", "csv"])
def test_s3_file_dual_engine_conformance(file_format: str) -> None:
    endpoint = os.environ["S3_ENDPOINT"]
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    bucket = f"tributo-ingestion-{uuid.uuid4().hex}"
    object_path = f"{bucket}/input.{file_format}"
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
    if file_format == "parquet":
        pq.write_table(_input_table(), object_path, filesystem=filesystem)
    else:
        with filesystem.open_output_stream(object_path) as stream:
            pacsv.write_csv(_input_table(), stream)

    try:
        source_type = (
            ParquetSourceConfig if file_format == "parquet" else CsvSourceConfig
        )
        _assert_source_conformance(
            source_type(
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

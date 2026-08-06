"""Real local Iceberg and Lance conformance for Ray Data and Daft."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import daft
import lance
import pyarrow as pa
import pyarrow.fs as pafs
import pytest
from pyiceberg.catalog.sql import SqlCatalog

from tests.data.ingestion_conformance import assert_dual_engine_conformance
from tributo.data import (
    DaftDataFrameHandle,
    IcebergSourceConfig,
    IngestionRequest,
    IngestionRuntimeContext,
    ProviderSourceConfig,
    RayDataHandle,
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
            "id": [1, 2, 3],
            "category": ["drop", "keep", "keep"],
        }
    )


def _assert_table_conformance(source: Any) -> None:
    assert daft.get_or_create_runner().name == "native"
    context = IngestionRuntimeContext()
    ray_result = open_ingestion(IngestionRequest(source=source, engine="ray"), context)
    daft_result = open_ingestion(
        IngestionRequest(source=source, engine="daft"), context
    )
    try:
        assert isinstance(ray_result.handle, RayDataHandle)
        assert isinstance(daft_result.handle, DaftDataFrameHandle)
        assert_dual_engine_conformance(
            ray_result,
            ray_result.handle.dataset.take_all(),
            daft_result,
            daft_result.handle.dataframe.to_pylist(),
            expected_rows=_input_table().to_pylist(),
            require_worker_validation=False,
        )
    finally:
        ray_result.close()
        daft_result.close()


def test_local_iceberg_dual_engine_conformance(tmp_path: Path) -> None:
    warehouse = tmp_path / "iceberg-warehouse"
    catalog_uri = f"sqlite:///{tmp_path / 'iceberg.db'}"
    catalog = SqlCatalog(
        "integration",
        uri=catalog_uri,
        warehouse=warehouse.as_uri(),
    )
    catalog.create_namespace("default")
    table = catalog.create_table("default.events", _input_table().schema)
    table.append(_input_table())

    _assert_table_conformance(
        IcebergSourceConfig(
            catalog="integration",
            table="default.events",
            catalog_properties={
                "type": "sql",
                "uri": catalog_uri,
                "warehouse": warehouse.as_uri(),
            },
            selected_fields=["id", "category"],
        )
    )


@pytest.mark.minio_compat
@pytest.mark.usefixtures("s3_environment")
def test_s3_iceberg_dual_engine_conformance(tmp_path: Path) -> None:
    endpoint = os.environ["S3_ENDPOINT"]
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    bucket = f"tributo-iceberg-{uuid.uuid4().hex}"
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
    catalog_uri = f"sqlite:///{tmp_path / 'iceberg-s3.db'}"
    catalog_properties = {
        "type": "sql",
        "uri": catalog_uri,
        "warehouse": f"s3://{bucket}/warehouse",
        "s3.endpoint": endpoint,
        "s3.access-key-id": access_key,
        "s3.secret-access-key": secret_key,
        "s3.region": "us-east-1",
        "s3.force-virtual-addressing": "false",
    }
    catalog = SqlCatalog("integration-s3", **catalog_properties)
    catalog.create_namespace("default")
    table = catalog.create_table("default.events", _input_table().schema)
    table.append(_input_table())

    try:
        _assert_table_conformance(
            IcebergSourceConfig(
                catalog="integration-s3",
                table="default.events",
                catalog_properties=catalog_properties,
                s3={
                    "endpoint": endpoint,
                    "access_key_id": access_key,
                    "secret_access_key": secret_key,
                    "region": "us-east-1",
                },
                selected_fields=["id", "category"],
            )
        )
    finally:
        filesystem.delete_dir(bucket)


def test_local_lance_dual_engine_conformance(tmp_path: Path) -> None:
    path = tmp_path / "events.lance"
    dataset = lance.write_dataset(_input_table(), path)

    _assert_table_conformance(
        ProviderSourceConfig(
            provider="tributo.lance",
            uri=str(path),
            options={"version": dataset.version},
        )
    )


@pytest.mark.minio_compat
@pytest.mark.usefixtures("s3_environment")
def test_s3_lance_dual_engine_conformance() -> None:
    endpoint = os.environ["S3_ENDPOINT"]
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    bucket = f"tributo-lance-{uuid.uuid4().hex}"
    uri = f"s3://{bucket}/events.lance"
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
    storage_options = {
        "endpoint": endpoint,
        "access_key_id": access_key,
        "secret_access_key": secret_key,
        "region": "us-east-1",
        "allow_http": "true",
    }
    dataset = lance.write_dataset(_input_table(), uri, storage_options=storage_options)

    try:
        _assert_table_conformance(
            ProviderSourceConfig(
                provider="tributo.lance",
                uri=uri,
                options={
                    "version": dataset.version,
                    "s3": {
                        "endpoint": endpoint,
                        "access_key_id": access_key,
                        "secret_access_key": secret_key,
                        "region": "us-east-1",
                    },
                },
            )
        )
    finally:
        filesystem.delete_dir(bucket)

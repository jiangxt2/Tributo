"""Real local Iceberg and Lance conformance for Ray Data and Daft."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

# The CI unit-test matrix installs only the ``dev`` extra; skip collection
# when the optional engine/connector dependencies are absent. The dedicated
# integration jobs install them via their extras.
try:
    import daft
    import lance
    import pyarrow as pa
    import pyarrow.fs as pafs
    from pyiceberg.catalog.sql import SqlCatalog
except ModuleNotFoundError:
    pytest.skip("requires the data and data-daft extras", allow_module_level=True)

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


def _assert_table_conformance(
    source: Any,
    *,
    expected_rows: list[dict[str, Any]] | None = None,
    storage_profile: str | None = None,
) -> None:
    assert daft.get_or_create_runner().name == "native"
    context = IngestionRuntimeContext()
    ray_result = open_ingestion(
        IngestionRequest(
            source=source,
            engine="ray",
            storage_profile=storage_profile,
        ),
        context,
    )
    daft_result = open_ingestion(
        IngestionRequest(
            source=source,
            engine="daft",
            storage_profile=storage_profile,
        ),
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
            expected_rows=(
                _input_table().to_pylist() if expected_rows is None else expected_rows
            ),
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


def test_empty_local_iceberg_preserves_schema_for_both_engines(tmp_path: Path) -> None:
    warehouse = tmp_path / "empty-iceberg-warehouse"
    catalog_uri = f"sqlite:///{tmp_path / 'empty-iceberg.db'}"
    catalog = SqlCatalog(
        "empty-integration",
        uri=catalog_uri,
        warehouse=warehouse.as_uri(),
    )
    catalog.create_namespace("default")
    catalog.create_table("default.events", _input_table().schema)

    _assert_table_conformance(
        IcebergSourceConfig(
            catalog="empty-integration",
            table="default.events",
            catalog_properties={
                "type": "sql",
                "uri": catalog_uri,
                "warehouse": warehouse.as_uri(),
            },
            selected_fields=["id", "category"],
        ),
        expected_rows=[],
    )


def test_local_iceberg_row_filter_matches_across_engines(tmp_path: Path) -> None:
    warehouse = tmp_path / "filtered-iceberg-warehouse"
    catalog_uri = f"sqlite:///{tmp_path / 'filtered-iceberg.db'}"
    catalog = SqlCatalog(
        "filtered-integration",
        uri=catalog_uri,
        warehouse=warehouse.as_uri(),
    )
    catalog.create_namespace("default")
    table = catalog.create_table("default.events", _input_table().schema)
    table.append(_input_table())

    _assert_table_conformance(
        IcebergSourceConfig(
            catalog="filtered-integration",
            table="default.events",
            catalog_properties={
                "type": "sql",
                "uri": catalog_uri,
                "warehouse": warehouse.as_uri(),
            },
            selected_fields=["id", "category"],
            row_filter="category = 'keep'",
        ),
        expected_rows=[
            {"id": 2, "category": "keep"},
            {"id": 3, "category": "keep"},
        ],
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


@pytest.mark.minio_compat
@pytest.mark.usefixtures("s3_environment")
def test_s3_iceberg_named_profile_and_path_style_conformance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = os.environ["S3_ENDPOINT"]
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    bucket = f"tributo-iceberg-profile-{uuid.uuid4().hex}"
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
    catalog_uri = f"sqlite:///{tmp_path / 'iceberg-s3-profile.db'}"
    writer_properties = {
        "type": "sql",
        "uri": catalog_uri,
        "warehouse": f"s3://{bucket}/warehouse",
        "s3.endpoint": endpoint,
        "s3.access-key-id": access_key,
        "s3.secret-access-key": secret_key,
        "s3.region": "us-east-1",
        "s3.force-virtual-addressing": "false",
    }
    catalog = SqlCatalog("integration-s3-profile", **writer_properties)
    catalog.create_namespace("default")
    table = catalog.create_table("default.events", _input_table().schema)
    table.append(_input_table())
    credentials_file = tmp_path / "aws-credentials"
    credentials_file.write_text(
        "[minio-iceberg]\n"
        f"aws_access_key_id={access_key}\n"
        f"aws_secret_access_key={secret_key}\n"
    )
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_file))
    monkeypatch.setenv(
        "TRIBUTO_STORAGE_PROFILE_ICEBERG_MINIO",
        json.dumps(
            {
                "endpoint": endpoint,
                "region": "us-east-1",
                "use_ssl": not endpoint.startswith("http://"),
                "path_style": True,
                "profile_name": "minio-iceberg",
            }
        ),
    )

    try:
        _assert_table_conformance(
            IcebergSourceConfig(
                catalog="integration-s3-profile",
                table="default.events",
                catalog_properties={
                    "type": "sql",
                    "uri": catalog_uri,
                    "warehouse": f"s3://{bucket}/warehouse",
                },
                selected_fields=["id", "category"],
            ),
            storage_profile="iceberg_minio",
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


@pytest.mark.minio_compat
@pytest.mark.usefixtures("s3_environment")
def test_s3_lance_named_profile_and_path_style_conformance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = os.environ["S3_ENDPOINT"]
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    bucket = f"tributo-lance-profile-{uuid.uuid4().hex}"
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
    credentials_file = tmp_path / "aws-credentials"
    credentials_file.write_text(
        "[minio-lance]\n"
        f"aws_access_key_id={access_key}\n"
        f"aws_secret_access_key={secret_key}\n"
    )
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_file))
    monkeypatch.setenv(
        "TRIBUTO_STORAGE_PROFILE_LANCE_MINIO",
        json.dumps(
            {
                "endpoint": endpoint,
                "region": "us-east-1",
                "use_ssl": not endpoint.startswith("http://"),
                "path_style": True,
                "profile_name": "minio-lance",
            }
        ),
    )

    try:
        _assert_table_conformance(
            ProviderSourceConfig(
                provider="tributo.lance",
                uri=uri,
                options={"version": dataset.version},
            ),
            storage_profile="lance_minio",
        )
    finally:
        filesystem.delete_dir(bucket)

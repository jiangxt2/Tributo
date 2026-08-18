"""Unit tests for training/data_loader.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from tributo.data import CsvSourceConfig, ParquetSourceConfig
from tributo.data.ingestion import RayDataHandle
from tributo.exceptions import EngineNotAvailableError, JobConfigurationError
from tributo.training.data_loader import (
    load_ray_dataset_from_config,
    load_ray_dataset_from_source,
)


def test_s3_parquet_uses_ray_ingestion_binding() -> None:
    dataset = MagicMock()
    result = MagicMock(handle=RayDataHandle(dataset))
    with patch(
        "tributo.training.data_loader.open_ingestion", return_value=result
    ) as open_:
        actual = load_ray_dataset_from_source(
            {"type": "parquet", "path": "s3://bucket/data.parquet"}
        )
    request = open_.call_args.args[0]
    source = request.source
    assert isinstance(source, ParquetSourceConfig)
    assert request.engine == "tributo.ray_data"
    assert actual is dataset
    result.close.assert_called_once_with()


def test_s3_csv_uses_ray_ingestion_binding() -> None:
    dataset = MagicMock()
    result = MagicMock(handle=RayDataHandle(dataset))
    with patch(
        "tributo.training.data_loader.open_ingestion", return_value=result
    ) as open_:
        actual = load_ray_dataset_from_source(
            {
                "type": "csv",
                "path": "s3://bucket/data.csv",
                "s3": {"region": "us-east-1"},
            }
        )
    request = open_.call_args.args[0]
    source = request.source
    assert isinstance(source, CsvSourceConfig)
    assert request.engine == "tributo.ray_data"
    assert actual is dataset
    result.close.assert_called_once_with()


def test_training_loader_rejects_non_ray_handle_and_closes_result() -> None:
    result = MagicMock(handle=MagicMock())
    with (
        patch("tributo.training.data_loader.open_ingestion", return_value=result),
        pytest.raises(
            JobConfigurationError,
            match="requires a RayDataHandle",
        ),
    ):
        load_ray_dataset_from_source(
            {"type": "parquet", "path": "s3://bucket/data.parquet"}
        )

    result.close.assert_called_once_with()


def test_legacy_flat_config_uses_ray_ingestion_gateway(tmp_path: Path) -> None:
    source_path = tmp_path / "data.parquet"
    source_path.touch()
    dataset = MagicMock()
    result = MagicMock(handle=RayDataHandle(dataset))
    with (
        patch(
            "tributo.training.data_loader.open_ingestion", return_value=result
        ) as open_,
        pytest.warns(
            FutureWarning,
            match=r"load_ray_dataset_from_config\(\) is deprecated",
        ),
    ):
        actual = load_ray_dataset_from_config(
            {"type": "parquet", "path": str(source_path)}
        )

    request = open_.call_args.args[0]
    assert isinstance(request.source, ParquetSourceConfig)
    assert request.engine == "tributo.ray_data"
    assert actual is dataset
    result.close.assert_called_once_with()


def test_s3_unsupported_format_raises():
    """Unknown source type is rejected by TypeAdapter discriminator."""
    try:
        load_ray_dataset_from_source({"type": "bogus", "path": "s3://bucket/data.orc"})
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_doris_requires_independent_ray_doris_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tributo never falls back to its former in-process MySQL reader."""
    import tributo.data.bindings as builtin_bindings

    installed_version = builtin_bindings._distribution_version
    monkeypatch.setattr(builtin_bindings, "_DEFAULT_BINDINGS", None)
    monkeypatch.setattr(
        builtin_bindings,
        "_distribution_version",
        lambda name: None if name == "ray-doris" else installed_version(name),
    )
    with pytest.raises(
        EngineNotAvailableError, match=r"ray-doris==1\.0.*tributo\[mysql\]"
    ):
        load_ray_dataset_from_source(
            {
                "type": "sql",
                "dialect": "doris",
                "table": "events",
                "host": "localhost",
                "port": 9030,
                "database": "db",
                "user": "user",
                "password": "pass",
            }
        )


@pytest.mark.parametrize("dialect", ["clickhouse", "doris", "postgresql"])
def test_legacy_raw_sql_has_structured_source_migration_error(dialect: str) -> None:
    with pytest.raises(JobConfigurationError, match="structured 'table' source"):
        load_ray_dataset_from_source(
            {
                "type": "sql",
                "dialect": dialect,
                "sql": "SELECT * FROM events",
            }
        )


def test_mysql_has_explicit_migration_error() -> None:
    with pytest.raises(JobConfigurationError, match="MySQL is unsupported"):
        load_ray_dataset_from_source(
            {
                "type": "sql",
                "dialect": "mysql",
                "sql": "SELECT * FROM events",
            }
        )

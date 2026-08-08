"""Unit tests for training/data_loader.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from tributo.data import CsvSourceConfig, ParquetSourceConfig
from tributo.exceptions import EngineNotAvailableError, JobConfigurationError
from tributo.training.data_loader import load_ray_dataset_from_source


def test_s3_parquet_uses_ray_ingestion_binding() -> None:
    dataset = MagicMock()
    with patch(
        "tributo.training.data_loader.open_ray_compat", return_value=dataset
    ) as open_:
        actual = load_ray_dataset_from_source(
            {"type": "parquet", "path": "s3://bucket/data.parquet"}
        )
    source = open_.call_args.args[0]
    assert isinstance(source, ParquetSourceConfig)
    assert actual is dataset


def test_s3_csv_uses_ray_ingestion_binding() -> None:
    dataset = MagicMock()
    with patch(
        "tributo.training.data_loader.open_ray_compat", return_value=dataset
    ) as open_:
        actual = load_ray_dataset_from_source(
            {
                "type": "csv",
                "path": "s3://bucket/data.csv",
                "s3": {"region": "us-east-1"},
            }
        )
    source = open_.call_args.args[0]
    assert isinstance(source, CsvSourceConfig)
    assert actual is dataset


def test_s3_unsupported_format_raises():
    """Unknown source type is rejected by TypeAdapter discriminator."""
    try:
        load_ray_dataset_from_source({"type": "bogus", "path": "s3://bucket/data.orc"})
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_doris_requires_independent_ray_doris_binding() -> None:
    """Tributo never falls back to its former in-process MySQL reader."""
    with pytest.raises(EngineNotAvailableError, match=r"ray-doris\[mysql,flight\]"):
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

"""Unit tests for training/data_loader.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from tributo.data import CsvSourceConfig, ParquetSourceConfig, RayDataHandle
from tributo.exceptions import EngineNotAvailableError
from tributo.training.data_loader import load_ray_dataset_from_source


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
    assert request.engine == "tributo.ray_data"
    assert isinstance(request.source, ParquetSourceConfig)
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
    assert request.engine == "tributo.ray_data"
    assert isinstance(request.source, CsvSourceConfig)
    assert actual is dataset
    result.close.assert_called_once_with()


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

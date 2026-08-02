"""Unit tests for training/data_loader.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from tributo.data.source_config import SqlSourceConfig
from tributo.training.data_loader import load_ray_dataset_from_source


def test_s3_parquet_uses_parquet_connector():
    with patch("tributo.data.get_connector") as mock_get_connector:
        mock_connector = MagicMock()
        mock_connector.read.return_value = MagicMock()
        mock_get_connector.return_value = mock_connector
        load_ray_dataset_from_source(
            {"type": "parquet", "path": "s3://bucket/data.parquet"}
        )
        mock_get_connector.assert_called_once_with("parquet")


def test_s3_csv_uses_csv_connector():
    with patch("tributo.data.get_connector") as mock_get_connector:
        mock_connector = MagicMock()
        mock_connector.read.return_value = MagicMock()
        mock_get_connector.return_value = mock_connector
        load_ray_dataset_from_source(
            {
                "type": "csv",
                "path": "s3://bucket/data.csv",
                "s3": {"region": "us-east-1"},
            }
        )
        mock_get_connector.assert_called_once_with("csv")


def test_s3_unsupported_format_raises():
    """Unknown source type is rejected by TypeAdapter discriminator."""
    from pydantic import ValidationError

    try:
        load_ray_dataset_from_source({"type": "bogus", "path": "s3://bucket/data.orc"})
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_doris_missing_mysql_extra_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Doris sources fail fast with an install hint when the mysql extra is absent."""
    from tributo.training.data_loader import _load_doris_mysql

    monkeypatch.setitem(sys.modules, "pymysql", None)  # import fails

    source = SqlSourceConfig(
        dialect="doris",
        sql="SELECT 1",
        host="localhost",
        port=9030,
        database="db",
        user="user",
        password="pass",
    )
    with pytest.raises(ImportError, match=r"tributo\[mysql\]"):
        _load_doris_mysql(source)

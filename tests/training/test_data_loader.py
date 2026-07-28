"""Unit tests for training/data_loader.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tributo.training.data_loader import load_ray_dataset_from_config


def test_s3_parquet_uses_parquet_connector():
    with patch("tributo.data.get_connector") as mock_get_connector:
        mock_connector = MagicMock()
        mock_connector.read.return_value = MagicMock()
        mock_get_connector.return_value = mock_connector
        load_ray_dataset_from_config(
            {
                "type": "s3",
                "uri": "s3://bucket/data.parquet",
                "format": "parquet",
            }
        )
        mock_get_connector.assert_called_once_with("parquet")


def test_s3_csv_uses_csv_connector():
    with patch("tributo.data.get_connector") as mock_get_connector:
        mock_connector = MagicMock()
        mock_connector.read.return_value = MagicMock()
        mock_get_connector.return_value = mock_connector
        load_ray_dataset_from_config(
            {
                "type": "s3",
                "uri": "s3://bucket/data.csv",
                "format": "csv",
                "s3": {"region": "us-east-1"},
            }
        )
        mock_get_connector.assert_called_once_with("csv")


def test_s3_unsupported_format_raises():
    with patch("tributo.data.get_connector"):
        try:
            load_ray_dataset_from_config(
                {
                    "type": "s3",
                    "uri": "s3://bucket/data.orc",
                    "format": "orc",
                }
            )
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "unsupported s3 format" in str(exc)

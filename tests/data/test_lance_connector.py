"""LanceDataConnector 单元测试。"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from pydantic import ValidationError

from tributo.data.base import S3Config, WriteMode
from tributo.data.lance import (
    LanceDataConnector,
    LanceReadConfig,
    LanceWriteConfig,
)


class TestLanceReadConfig:
    """LanceReadConfig Pydantic 验证测试。"""

    def test_valid_config(self):
        cfg = LanceReadConfig(path="/data/dataset.lance")
        assert cfg.path == "/data/dataset.lance"
        assert cfg.s3 is None

    def test_with_s3_config(self):
        s3 = S3Config(endpoint="http://minio:9000")
        cfg = LanceReadConfig(path="s3://bucket/dataset.lance", s3=s3)
        assert cfg.s3 is not None
        assert cfg.s3.endpoint == "http://minio:9000"


class TestLanceWriteConfig:
    """LanceWriteConfig Pydantic 验证测试。"""

    def test_defaults(self):
        cfg = LanceWriteConfig(path="/data/output")
        assert cfg.mode == WriteMode.OVERWRITE
        assert cfg.s3 is None


class TestLanceDataConnector:
    """LanceDataConnector 行为测试。"""

    def test_read_empty_path_raises(self):
        connector = LanceDataConnector()
        with pytest.raises(ValidationError):
            connector.read(path="")

    @patch("tributo.data.lance.ray.data.from_arrow")
    @patch("lance.dataset")
    def test_read_calls_lance_dataset(self, mock_lance_dataset, mock_from_arrow):
        """read() 应调用 lance.dataset().to_table() 并转换为 Ray Dataset。"""
        arrow_table = pa.table({"id": [1, 2], "name": ["a", "b"]})

        mock_ds = MagicMock()
        mock_ds.to_table.return_value = arrow_table
        mock_lance_dataset.return_value = mock_ds

        mock_ray_ds = MagicMock()
        mock_from_arrow.return_value = mock_ray_ds

        connector = LanceDataConnector()
        result = connector.read(path="/data/test.lance")

        mock_lance_dataset.assert_called_once_with(
            "/data/test.lance", storage_options=None
        )
        mock_ds.to_table.assert_called_once()
        mock_from_arrow.assert_called_once_with(arrow_table)
        assert result is mock_ray_ds

    @patch("tributo.data.lance.ray.data.from_arrow")
    @patch("lance.dataset")
    def test_read_with_s3(self, mock_lance_dataset, mock_from_arrow):
        """read() S3 路径应传递 storage_options。"""
        arrow_table = pa.table({"id": [1]})

        mock_ds = MagicMock()
        mock_ds.to_table.return_value = arrow_table
        mock_lance_dataset.return_value = mock_ds

        mock_from_arrow.return_value = MagicMock()

        connector = LanceDataConnector()
        connector.read(
            path="s3://bucket/dataset.lance",
            s3=S3Config(endpoint="http://minio:9000"),
        )

        mock_lance_dataset.assert_called_once()
        call_kwargs = mock_lance_dataset.call_args
        assert call_kwargs[1]["storage_options"] is not None
        assert call_kwargs[1]["storage_options"]["endpoint"] == "http://minio:9000"


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))

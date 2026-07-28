"""IcebergDataConnector 单元测试。"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from pyiceberg.exceptions import NoSuchTableError

from tributo.data.base import S3Config, WriteMode
from tributo.data.iceberg import (
    IcebergDataConnector,
    IcebergReadConfig,
    IcebergWriteConfig,
)


class TestIcebergReadConfig:
    """IcebergReadConfig Pydantic 验证测试。"""

    def test_valid_config(self):
        cfg = IcebergReadConfig(
            table_identifier="db.table",
            catalog_properties={"type": "rest", "uri": "http://localhost:8181"},
        )
        assert cfg.table_identifier == "db.table"
        assert cfg.catalog_name == "default"
        assert cfg.s3 is None
        assert cfg.snapshot_id is None
        assert cfg.row_filter is None
        assert cfg.selected_fields is None

    def test_empty_table_identifier_raises(self):
        with pytest.raises(ValidationError):
            IcebergReadConfig(table_identifier="", catalog_properties={"type": "rest"})

    def test_with_s3_config(self):
        s3 = S3Config(endpoint="http://minio:9000")
        cfg = IcebergReadConfig(
            table_identifier="db.table",
            catalog_properties={"type": "rest"},
            s3=s3,
        )
        assert cfg.s3 is not None
        assert cfg.s3.endpoint == "http://minio:9000"

    def test_with_optional_fields(self):
        cfg = IcebergReadConfig(
            table_identifier="db.table",
            catalog_properties={"type": "rest"},
            snapshot_id=12345,
            row_filter="id > 100",
            selected_fields=["id", "name"],
        )
        assert cfg.snapshot_id == 12345
        assert cfg.row_filter == "id > 100"
        assert cfg.selected_fields == ["id", "name"]


class TestIcebergWriteConfig:
    """IcebergWriteConfig Pydantic 验证测试。"""

    def test_defaults(self):
        cfg = IcebergWriteConfig(
            table_identifier="db.table",
            catalog_properties={"type": "rest"},
        )
        assert cfg.mode == WriteMode.OVERWRITE
        assert cfg.catalog_name == "default"

    def test_empty_table_identifier_raises(self):
        with pytest.raises(ValidationError):
            IcebergWriteConfig(table_identifier="", catalog_properties={"type": "rest"})

    def test_append_mode(self):
        cfg = IcebergWriteConfig(
            table_identifier="db.table",
            catalog_properties={"type": "rest"},
            mode=WriteMode.APPEND,
        )
        assert cfg.mode == WriteMode.APPEND


class TestIcebergDataConnector:
    """IcebergDataConnector 行为测试。"""

    @patch("tributo.data.iceberg.ray.data.read_parquet")
    @patch("tributo.data.iceberg._load_catalog")
    def test_read_calls_plan_files(self, mock_load_catalog, mock_read_parquet):
        """read() 应调用 plan_files() 获取文件列表，再由 ray.data.read_parquet 分布式读取。"""
        # 模拟 FileScanTask
        mock_task = MagicMock()
        mock_task.file.file_path = "s3://bucket/data/part-0.parquet"

        mock_scan = MagicMock()
        mock_scan.plan_files.return_value = [mock_task]

        mock_table = MagicMock()
        mock_table.scan.return_value = mock_scan

        mock_catalog = MagicMock()
        mock_catalog.load_table.return_value = mock_table
        mock_load_catalog.return_value = mock_catalog

        mock_ds = MagicMock()
        mock_read_parquet.return_value = mock_ds

        connector = IcebergDataConnector()
        result = connector.read(
            table_identifier="db.test",
            catalog_properties={"type": "rest"},
        )

        mock_catalog.load_table.assert_called_once_with("db.test")
        mock_scan.plan_files.assert_called_once()
        mock_read_parquet.assert_called_once()
        # 验证传入的是去掉 s3:// 前缀的路径
        call_args = mock_read_parquet.call_args
        assert call_args[0][0] == ["bucket/data/part-0.parquet"]
        assert result is mock_ds

    @patch("tributo.data.iceberg.ray.data.read_parquet")
    @patch("tributo.data.iceberg._load_catalog")
    def test_read_passes_scan_kwargs(self, mock_load_catalog, mock_read_parquet):
        """read() 应正确传递 snapshot_id、row_filter、selected_fields 到 scan()。"""
        mock_task = MagicMock()
        mock_task.file.file_path = "/local/data/part-0.parquet"

        mock_scan = MagicMock()
        mock_scan.plan_files.return_value = [mock_task]

        mock_table = MagicMock()
        mock_table.scan.return_value = mock_scan

        mock_catalog = MagicMock()
        mock_catalog.load_table.return_value = mock_table
        mock_load_catalog.return_value = mock_catalog

        mock_read_parquet.return_value = MagicMock()

        connector = IcebergDataConnector()
        connector.read(
            table_identifier="db.test",
            catalog_properties={"type": "rest"},
            snapshot_id=123,
            row_filter="id > 1",
            selected_fields=["id", "name"],
        )

        mock_table.scan.assert_called_once_with(
            row_filter="id > 1",
            selected_fields=("id", "name"),
            snapshot_id=123,
        )
        # 验证 selected_fields 传递给 read_parquet 的 columns 参数
        call_kwargs = mock_read_parquet.call_args[1]
        assert call_kwargs["columns"] == ["id", "name"]

    @patch("tributo.data.iceberg._load_catalog")
    def test_write_overwrite(self, mock_load_catalog):
        """write() OVERWRITE 模式应调用 table.overwrite(arrow_table)。"""
        mock_table = MagicMock()
        mock_catalog = MagicMock()
        mock_catalog.load_table.return_value = mock_table
        mock_load_catalog.return_value = mock_catalog

        mock_ds = MagicMock()
        mock_arrow = MagicMock()
        mock_ds.to_arrow.return_value = mock_arrow

        connector = IcebergDataConnector()
        connector.write(
            mock_ds, table_identifier="db.test", catalog_properties={"type": "rest"}
        )

        mock_table.overwrite.assert_called_once_with(mock_arrow)

    @patch("tributo.data.iceberg._load_catalog")
    def test_write_creates_table_if_not_exists(self, mock_load_catalog):
        """write() 表不存在时应自动建表。"""
        mock_table = MagicMock()
        mock_catalog = MagicMock()
        mock_catalog.load_table.side_effect = NoSuchTableError("db.test")
        mock_catalog.create_table.return_value = mock_table
        mock_load_catalog.return_value = mock_catalog

        mock_ds = MagicMock()
        mock_arrow = MagicMock()
        mock_ds.to_arrow.return_value = mock_arrow

        connector = IcebergDataConnector()
        connector.write(
            mock_ds, table_identifier="db.test", catalog_properties={"type": "rest"}
        )

        mock_catalog.create_table.assert_called_once_with(
            identifier="db.test",
            schema=mock_arrow.schema,
            location=None,
        )
        mock_table.overwrite.assert_called_once_with(mock_arrow)

    @patch("tributo.data.iceberg._load_catalog")
    def test_write_append(self, mock_load_catalog):
        """write() APPEND 模式应调用 table.append(arrow_table)。"""
        mock_table = MagicMock()
        mock_catalog = MagicMock()
        mock_catalog.load_table.return_value = mock_table
        mock_load_catalog.return_value = mock_catalog

        mock_ds = MagicMock()
        mock_arrow = MagicMock()
        mock_ds.to_arrow.return_value = mock_arrow

        connector = IcebergDataConnector()
        connector.write(
            mock_ds,
            table_identifier="db.test",
            catalog_properties={"type": "rest"},
            mode=WriteMode.APPEND,
        )

        mock_table.append.assert_called_once_with(mock_arrow)

    @patch("tributo.data.iceberg._load_catalog")
    def test_exists_returns_true(self, mock_load_catalog):
        """exists() 表存在时应返回 True。"""
        mock_catalog = MagicMock()
        mock_load_catalog.return_value = mock_catalog

        connector = IcebergDataConnector()
        result = connector.exists(
            table_identifier="db.test", catalog_properties={"type": "rest"}
        )

        assert result is True

    @patch("tributo.data.iceberg._load_catalog")
    def test_exists_returns_false(self, mock_load_catalog):
        """exists() 表不存在时应返回 False。"""
        mock_catalog = MagicMock()
        mock_catalog.load_table.side_effect = NoSuchTableError("db.test")
        mock_load_catalog.return_value = mock_catalog

        connector = IcebergDataConnector()
        result = connector.exists(
            table_identifier="db.test", catalog_properties={"type": "rest"}
        )

        assert result is False


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))

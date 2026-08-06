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

    @patch("tributo.data._compat_read.open_ray_compat")
    def test_read_delegates_to_gateway(self, mock_open_ray_compat):
        """read() 只构造逻辑请求并委托 Ray Gateway。"""
        mock_ds = MagicMock()
        mock_open_ray_compat.return_value = mock_ds

        connector = IcebergDataConnector()
        result = connector.read(
            table_identifier="db.test",
            catalog_properties={"type": "rest"},
        )

        source = mock_open_ray_compat.call_args.args[0]
        assert source.catalog == "default"
        assert source.table == "db.test"
        assert source.catalog_properties == {"type": "rest"}
        assert result is mock_ds

    @patch("tributo.data._compat_read.open_ray_compat")
    def test_read_preserves_table_options(self, mock_open_ray_compat):
        """read() 将快照、过滤和投影原样映射到规范 SourceConfig。"""
        connector = IcebergDataConnector()
        connector.read(
            table_identifier="db.test",
            catalog_properties={"type": "rest"},
            snapshot_id=123,
            row_filter="id > 1",
            selected_fields=["id", "name"],
        )

        source = mock_open_ray_compat.call_args.args[0]
        assert source.snapshot_id == 123
        assert source.row_filter == "id > 1"
        assert source.selected_fields == ["id", "name"]

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

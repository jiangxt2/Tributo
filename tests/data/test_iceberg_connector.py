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

    @patch("tributo.data.writing.compatibility.execute_ray_connector_write")
    def test_write_delegates_to_native_gateway(self, mock_execute):
        """write() preserves configuration while delegating to Ray native API."""
        connector = IcebergDataConnector()
        connector.write(
            MagicMock(),
            table_identifier="db.test",
            catalog_properties={"type": "rest"},
        )

        mock_execute.assert_called_once()
        call = mock_execute.call_args.kwargs
        assert call["target_kind"] == "iceberg"
        assert call["target"] == "db.test"
        assert call["runtime_options"]["catalog_properties"] == {"type": "rest"}
        assert call["runtime_options"]["table_identifier"] == "db.test"
        assert call["mode"] == WriteMode.OVERWRITE

    @patch("tributo.data.writing.compatibility.execute_ray_connector_write")
    def test_write_preserves_append_mode_in_request(self, mock_execute):
        connector = IcebergDataConnector()
        connector.write(
            MagicMock(),
            table_identifier="db.test",
            catalog_properties={"type": "rest"},
            mode=WriteMode.APPEND,
        )

        assert mock_execute.call_args.kwargs["mode"] == WriteMode.APPEND

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

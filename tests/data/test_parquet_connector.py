"""ParquetDataConnector 单元测试。"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tributo.data.base import DataConnector, WriteMode
from tributo.data.csv import CsvDataConnector
from tributo.data.parquet import (
    ParquetDataConnector,
    ParquetReadConfig,
    ParquetWriteConfig,
)


class TestParquetReadConfig:
    """ParquetReadConfig Pydantic 验证测试。"""

    def test_valid_config(self):
        cfg = ParquetReadConfig(path="/data/train.parquet")
        assert cfg.path == "/data/train.parquet"
        assert cfg.columns is None
        assert cfg.s3 is None

    def test_with_columns(self):
        cfg = ParquetReadConfig(path="/data/train.parquet", columns=["a", "b"])
        assert cfg.columns == ["a", "b"]


class TestParquetWriteConfig:
    """ParquetWriteConfig Pydantic 验证测试。"""

    def test_defaults(self):
        cfg = ParquetWriteConfig(path="/data/output")
        assert cfg.mode == WriteMode.OVERWRITE
        assert cfg.compression == "zstd"


class TestParquetDataConnector:
    """ParquetDataConnector 行为测试。"""

    def test_read_empty_path_raises(self):
        connector = ParquetDataConnector()
        with pytest.raises(ValueError, match="path must not be empty"):
            connector.read(path="")

    def test_exists_returns_false(self):
        connector = ParquetDataConnector()
        assert connector.exists(path="/nonexistent") is False


class TestCompatibilityConnectorModes:
    """Legacy format facades preserve their mode restrictions."""

    @pytest.mark.parametrize("connector_type", [ParquetDataConnector, CsvDataConnector])
    def test_file_connectors_reject_append(
        self, connector_type: type[DataConnector]
    ) -> None:
        """APPEND is rejected before the compatibility facade reaches Gateway."""
        with pytest.raises(ValueError, match="does not support APPEND"):
            connector_type().write(MagicMock(), path="/tmp/out", mode=WriteMode.APPEND)


class TestParquetS3Glob:
    """The compatibility connector delegates glob semantics to Ray Data."""

    @patch("tributo.data._compat_read.open_ingestion")
    def test_glob_pattern_is_preserved(self, mock_open):
        result = MagicMock()
        result.handle.dataset = object()
        from tributo.data.ingestion import RayDataHandle

        result.handle = RayDataHandle(result.handle.dataset)
        mock_open.return_value = result
        try:
            connector = ParquetDataConnector()
            connector.read(path="s3://bucket/data/*.parquet")
        finally:
            result.close.assert_called_once()
        request = mock_open.call_args.args[0]
        assert request.source.path == "s3://bucket/data/*.parquet"


class TestBackwardCompatibility:
    """向后兼容：验证业务模块通过 data/ 读取，不依赖已删除的 _common/io。"""

    @staticmethod
    def _get_import_from_targets(source: str) -> set[str]:
        tree = ast.parse(source)
        return {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }

    def test_embeddings_uses_data_module(self):
        src_root = Path(__file__).resolve().parents[2] / "src"
        source = (src_root / "tributo" / "embeddings" / "batch_job.py").read_text()
        modules = self._get_import_from_targets(source)
        assert "tributo._common.io" not in modules
        assert "tributo.data" in modules

    def test_inference_uses_data_module(self):
        src_root = Path(__file__).resolve().parents[2] / "src"
        source = (src_root / "tributo" / "inference" / "pipeline.py").read_text()
        modules = self._get_import_from_targets(source)
        assert "tributo._common.io" not in modules
        assert "tributo.data" in modules


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))

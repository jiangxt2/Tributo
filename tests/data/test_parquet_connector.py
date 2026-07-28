"""ParquetDataConnector 单元测试。"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tributo.data.base import WriteMode
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

    def test_write_append_raises(self):
        """APPEND 模式应被 ParquetDataConnector.write 拒绝。"""
        connector = ParquetDataConnector()
        # 用 mock 避免创建真实 Ray Dataset
        from unittest.mock import MagicMock

        mock_ds = MagicMock()
        with pytest.raises(ValueError, match="does not support APPEND"):
            connector.write(mock_ds, path="/tmp/out", mode=WriteMode.APPEND)

    def test_exists_returns_false(self):
        connector = ParquetDataConnector()
        assert connector.exists(path="/nonexistent") is False


class TestParquetS3Glob:
    """ParquetDataConnector S3 glob 模式测试。"""

    @patch("tributo.data.parquet.pafs.S3FileSystem")
    def test_glob_pattern(self, mock_s3fs_cls):
        """Glob 模式应匹配文件列表并传递给 ray.data.read_parquet。"""
        mock_fs = MagicMock()
        mock_s3fs_cls.return_value = mock_fs

        file_info_1 = MagicMock()
        file_info_1.is_file = True
        file_info_1.path = "bucket/data/part-001.parquet"
        file_info_2 = MagicMock()
        file_info_2.is_file = True
        file_info_2.path = "bucket/data/part-002.parquet"
        file_info_3 = MagicMock()
        file_info_3.is_file = False
        file_info_3.path = "bucket/data/_temporary"
        mock_fs.get_file_info.return_value = [file_info_1, file_info_2, file_info_3]

        with patch("ray.data.read_parquet") as mock_read:
            connector = ParquetDataConnector()
            connector.read(path="s3://bucket/data/*.parquet")

        args, _ = mock_read.call_args
        assert isinstance(args[0], list)
        assert len(args[0]) == 2

    @patch("tributo.data.parquet.pafs.S3FileSystem")
    def test_glob_no_match_raises(self, mock_s3fs_cls):
        """Glob 无匹配时应抛出 FileNotFoundError。"""
        mock_fs = MagicMock()
        mock_s3fs_cls.return_value = mock_fs
        mock_fs.get_file_info.return_value = []

        connector = ParquetDataConnector()
        with pytest.raises(FileNotFoundError, match="No files matched"):
            connector.read(path="s3://bucket/data/*.parquet")


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

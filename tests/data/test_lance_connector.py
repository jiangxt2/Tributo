"""LanceDataConnector 单元测试。"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

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

    @patch("tributo.data._compat_read.open_ray_compat")
    def test_read_delegates_to_gateway(self, mock_open_ray_compat):
        """read() 只构造逻辑请求并委托 Ray Gateway。"""
        mock_ray_ds = MagicMock()
        mock_open_ray_compat.return_value = mock_ray_ds

        connector = LanceDataConnector()
        result = connector.read(path="/data/test.lance")

        source = mock_open_ray_compat.call_args.args[0]
        assert source.provider == "tributo.lance"
        assert source.uri == "/data/test.lance"
        assert source.options == {}
        assert result is mock_ray_ds

    @patch("tributo.data._compat_read.open_ray_compat")
    def test_read_with_s3(self, mock_open_ray_compat):
        """read() 将 S3 配置封装进 Gateway runtime options。"""
        connector = LanceDataConnector()
        connector.read(
            path="s3://bucket/dataset.lance",
            s3=S3Config(endpoint="http://minio:9000"),
        )

        source = mock_open_ray_compat.call_args.args[0]
        assert source.provider == "tributo.lance"
        assert source.uri == "s3://bucket/dataset.lance"
        assert source.options["s3"]["endpoint"] == "http://minio:9000"


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))

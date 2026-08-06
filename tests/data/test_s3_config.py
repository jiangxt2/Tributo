"""S3 配置解析测试。"""

from __future__ import annotations

import sys

import pytest

from tributo._common.storage_profiles import StorageProfile
from tributo.data._s3 import (
    resolve_endpoint,
    resolve_region,
    to_daft_s3_kwargs,
    to_iceberg_properties,
    to_lance_storage_options,
    to_pyarrow_s3_kwargs,
)
from tributo.data.base import S3Config


def test_s3_configuration_reprs_hide_credentials_and_endpoint_userinfo() -> None:
    config = S3Config(
        access_key_id="access-secret",
        secret_access_key="top-secret",
        endpoint="http://user:password@minio:9000",
        region="us-east-1",
    )
    profile = StorageProfile(
        access_key_id="profile-key",
        secret_access_key="profile-secret",
        endpoint="http://user:password@minio:9000",
        region="us-east-1",
    )

    for rendered in (repr(config), repr(profile)):
        assert "access-secret" not in rendered
        assert "top-secret" not in rendered
        assert "profile-key" not in rendered
        assert "profile-secret" not in rendered
        assert "password" not in rendered


class TestResolveEndpoint:
    """resolve_endpoint 环境变量优先级测试。"""

    def test_explicit_over_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("S3_ENDPOINT", "http://env:9000")
        assert resolve_endpoint("http://explicit:9000") == "http://explicit:9000"

    def test_s3_endpoint_over_aws_endpoint_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("S3_ENDPOINT", "http://s3:9000")
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://aws:9000")
        assert resolve_endpoint() == "http://s3:9000"

    def test_fallback_to_aws_endpoint_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("S3_ENDPOINT", raising=False)
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://aws:9000")
        assert resolve_endpoint() == "http://aws:9000"

    def test_none_when_no_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("S3_ENDPOINT", raising=False)
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        assert resolve_endpoint() is None


class TestResolveRegion:
    """resolve_region 返回值测试。"""

    def test_explicit_value(self):
        assert resolve_region("cn-north-1") == "cn-north-1"

    def test_default_us_east_1(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        assert resolve_region() == "us-east-1"

    def test_always_returns_str(self):
        """resolve_region 应永远返回 str，不返回 None。"""
        result = resolve_region()
        assert isinstance(result, str)


class TestToPyarrowS3Kwargs:
    """to_pyarrow_s3_kwargs 参数映射测试。"""

    def test_maps_to_pyarrow_names(self):
        cfg = S3Config(
            access_key_id="AK",
            secret_access_key="SK",
            endpoint="http://minio:9000",
            region="us-west-2",
        )
        kwargs = to_pyarrow_s3_kwargs(cfg)
        assert kwargs["access_key"] == "AK"
        assert kwargs["secret_key"] == "SK"
        assert kwargs["endpoint_override"] == "minio:9000"
        assert kwargs["scheme"] == "http"
        assert kwargs["region"] == "us-west-2"

    def test_none_config_uses_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ENV_KEY")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ENV_SECRET")
        kwargs = to_pyarrow_s3_kwargs(None)
        assert kwargs["access_key"] == "ENV_KEY"
        assert kwargs["secret_key"] == "ENV_SECRET"

    def test_http_scheme_detected(self):
        cfg = S3Config(endpoint="http://minio:9000")
        kwargs = to_pyarrow_s3_kwargs(cfg)
        assert kwargs["scheme"] == "http"

    def test_https_scheme_not_set(self):
        cfg = S3Config(endpoint="https://s3.amazonaws.com")
        kwargs = to_pyarrow_s3_kwargs(cfg)
        assert "scheme" not in kwargs


class TestToDaftS3Kwargs:
    """Daft S3 public configuration mapping tests."""

    def test_maps_config_names(self):
        cfg = S3Config(
            access_key_id="AK",
            secret_access_key="SK",
            endpoint="http://minio:9000",
            region="us-west-2",
        )
        assert to_daft_s3_kwargs(cfg) == {
            "key_id": "AK",
            "access_key": "SK",
            "endpoint_url": "http://minio:9000",
            "region_name": "us-west-2",
            "use_ssl": False,
        }

    def test_https_endpoint_keeps_default_ssl(self):
        kwargs = to_daft_s3_kwargs(S3Config(endpoint="https://s3.amazonaws.com"))

        assert "use_ssl" not in kwargs

    def test_none_config_uses_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ENV_KEY")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ENV_SECRET")
        assert to_daft_s3_kwargs(None)["key_id"] == "ENV_KEY"
        assert to_daft_s3_kwargs(None)["access_key"] == "ENV_SECRET"

    def test_explicit_credentials_override_named_profile(self):
        profile = StorageProfile(
            access_key_id="explicit-key",
            secret_access_key="explicit-secret",
            profile_name="fallback-profile",
        )

        assert to_daft_s3_kwargs(profile) == {
            "key_id": "explicit-key",
            "access_key": "explicit-secret",
        }


class TestToLanceStorageOptions:
    """to_lance_storage_options 参数映射测试。"""

    def test_maps_to_lance_names(self):
        cfg = S3Config(
            access_key_id="AK",
            secret_access_key="SK",
            endpoint="http://minio:9000",
            region="us-west-2",
        )
        opts = to_lance_storage_options(cfg)
        assert opts["access_key_id"] == "AK"
        assert opts["secret_access_key"] == "SK"
        assert opts["endpoint"] == "http://minio:9000"
        assert opts["allow_http"] == "true"
        assert opts["region"] == "us-west-2"

    def test_none_config_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("S3_ENDPOINT", raising=False)
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        assert to_lance_storage_options(None) is None


class TestToIcebergProperties:
    """to_iceberg_properties 参数映射测试。"""

    def test_maps_to_iceberg_names(self):
        cfg = S3Config(
            access_key_id="AK",
            secret_access_key="SK",
            endpoint="http://minio:9000",
            region="us-west-2",
        )
        props = to_iceberg_properties(cfg)
        assert props["s3.access-key-id"] == "AK"
        assert props["s3.secret-access-key"] == "SK"
        assert props["s3.endpoint"] == "http://minio:9000"
        assert props["s3.region"] == "us-west-2"

    def test_no_default_region(self, monkeypatch: pytest.MonkeyPatch):
        """无显式 region 时不应设置 s3.region（与 to_lance_storage_options 一致）。"""
        monkeypatch.delenv("AWS_REGION", raising=False)
        cfg = S3Config(endpoint="http://minio:9000")
        props = to_iceberg_properties(cfg)
        assert "s3.region" not in props

    def test_empty_config_returns_empty_dict(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("S3_ENDPOINT", raising=False)
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        assert to_iceberg_properties(None) == {}


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))

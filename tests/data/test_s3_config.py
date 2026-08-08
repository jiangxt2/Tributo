"""S3 配置解析测试。"""

from __future__ import annotations

import sys

import pytest
from botocore.exceptions import ProfileNotFound

from tributo._common.storage_profiles import StorageProfile, StorageProfileResolver
from tributo.data._s3 import (
    merge_iceberg_properties,
    resolve_endpoint,
    resolve_region,
    to_daft_s3_kwargs,
    to_iceberg_properties,
    to_lance_storage_options,
    to_pyarrow_s3_kwargs,
)
from tributo.data.base import S3Config
from tributo.data.bindings._shared import runtime_s3_profile
from tributo.exceptions import JobConfigurationError


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


class TestStorageProfileResolver:
    def test_resolves_json_object_from_named_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "TRIBUTO_STORAGE_PROFILE_MODEL",
            '{"endpoint":"http://minio:9000","region":"us-east-1"}',
        )

        profile = StorageProfileResolver().resolve("model")

        assert profile.endpoint == "http://minio:9000"
        assert profile.region == "us-east-1"

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ("not-json-secret", "must contain valid JSON"),
            ('["embedded-secret"]', "must contain a JSON object"),
            ('{"unknown_field":"embedded-secret"}', "invalid storage profile fields"),
        ],
    )
    def test_rejects_invalid_environment_payload_without_leaking_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        payload: str,
        message: str,
    ) -> None:
        monkeypatch.setenv("TRIBUTO_STORAGE_PROFILE_MODEL", payload)

        with pytest.raises(JobConfigurationError, match=message) as error:
            StorageProfileResolver().resolve("model")

        rendered = str(error.value)
        assert "TRIBUTO_STORAGE_PROFILE_MODEL" in rendered
        assert "embedded-secret" not in rendered
        assert "not-json-secret" not in rendered


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
        monkeypatch.setenv("AWS_SESSION_TOKEN", "ENV_TOKEN")
        kwargs = to_pyarrow_s3_kwargs(None)
        assert kwargs["access_key"] == "ENV_KEY"
        assert kwargs["secret_key"] == "ENV_SECRET"
        assert kwargs["session_token"] == "ENV_TOKEN"

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
        monkeypatch.setenv("AWS_SESSION_TOKEN", "ENV_TOKEN")
        assert to_daft_s3_kwargs(None)["key_id"] == "ENV_KEY"
        assert to_daft_s3_kwargs(None)["access_key"] == "ENV_SECRET"
        assert to_daft_s3_kwargs(None)["session_token"] == "ENV_TOKEN"

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

    def test_named_profile_overrides_environment_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ENV_KEY")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ENV_SECRET")

        assert to_daft_s3_kwargs(StorageProfile(profile_name="analytics")) == {
            "profile_name": "analytics"
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

    def test_named_profile_and_transport_flags_are_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ENV_KEY")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ENV_SECRET")
        monkeypatch.setattr(
            "tributo.data._s3._named_profile_credentials",
            lambda name: ("PROFILE_KEY", "PROFILE_SECRET", "SESSION_TOKEN"),
        )
        profile = StorageProfile(
            endpoint="https://minio:9000",
            profile_name="analytics",
            use_ssl=False,
            path_style=True,
        )

        assert to_lance_storage_options(profile) == {
            "access_key_id": "PROFILE_KEY",
            "secret_access_key": "PROFILE_SECRET",
            "session_token": "SESSION_TOKEN",
            "endpoint": "http://minio:9000",
            "allow_http": "true",
            "virtual_hosted_style_request": "false",
        }


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
        monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
        assert to_iceberg_properties(None) == {}

    def test_environment_session_token_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ENV_KEY")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ENV_SECRET")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "ENV_TOKEN")

        assert to_iceberg_properties(None)["s3.session-token"] == "ENV_TOKEN"
        assert to_lance_storage_options(None)["session_token"] == "ENV_TOKEN"

    def test_named_profile_and_path_style_are_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ENV_KEY")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ENV_SECRET")
        monkeypatch.setattr(
            "tributo.data._s3._named_profile_credentials",
            lambda name: ("PROFILE_KEY", "PROFILE_SECRET", "PROFILE_TOKEN"),
        )
        props = to_iceberg_properties(
            StorageProfile(profile_name="analytics", path_style=True)
        )

        assert props == {
            "s3.access-key-id": "PROFILE_KEY",
            "s3.secret-access-key": "PROFILE_SECRET",
            "s3.session-token": "PROFILE_TOKEN",
            "s3.profile-name": "analytics",
            "s3.force-virtual-addressing": "false",
        }

    def test_missing_named_profile_is_normalized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing_profile(*args: object, **kwargs: object) -> object:
            raise ProfileNotFound(profile="missing")

        monkeypatch.setattr("boto3.Session", missing_profile)

        with pytest.raises(
            JobConfigurationError,
            match="Configured S3 profile could not be resolved",
        ) as exc_info:
            to_iceberg_properties(StorageProfile(profile_name="missing"))

        assert "missing" not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

    def test_named_profile_without_credentials_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class EmptySession:
            @staticmethod
            def get_credentials() -> None:
                return None

        monkeypatch.setattr("boto3.Session", lambda **kwargs: EmptySession())

        with pytest.raises(
            JobConfigurationError,
            match="Configured S3 profile could not be resolved",
        ) as exc_info:
            to_lance_storage_options(StorageProfile(profile_name="empty"))

        assert "empty" not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

    def test_named_profile_without_s3_extra_has_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "boto3", None)

        with pytest.raises(
            JobConfigurationError,
            match=r"pip install 'tributo\[s3\]'",
        ) as exc_info:
            to_lance_storage_options(StorageProfile(profile_name="analytics"))

        assert "analytics" not in str(exc_info.value)
        assert exc_info.value.__cause__ is None

    def test_catalog_named_profile_is_materialized_for_pyarrow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "tributo.data._s3._named_profile_credentials",
            lambda name: ("CATALOG_KEY", "CATALOG_SECRET", "CATALOG_TOKEN"),
        )

        merged = merge_iceberg_properties({"s3.profile-name": "catalog-profile"})

        assert merged["s3.access-key-id"] == "CATALOG_KEY"
        assert merged["s3.secret-access-key"] == "CATALOG_SECRET"
        assert merged["s3.session-token"] == "CATALOG_TOKEN"
        assert merged["py-io-impl"] == "pyiceberg.io.pyarrow.PyArrowFileIO"

    def test_non_pyarrow_iceberg_file_io_fails_closed(self) -> None:
        with pytest.raises(JobConfigurationError, match="require PyArrowFileIO"):
            merge_iceberg_properties({"py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO"})

    def test_merge_precedence_is_source_catalog_profile_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://environment:9000")
        merged = merge_iceberg_properties(
            {
                "s3.endpoint": "http://catalog:9000",
                "s3.region": "catalog-region",
            },
            profile=StorageProfile(
                endpoint="http://profile:9000",
                region="profile-region",
            ),
            source=S3Config(endpoint="http://source:9000"),
        )

        assert merged["s3.endpoint"] == "http://source:9000"
        assert merged["s3.region"] == "catalog-region"

    def test_credential_layers_are_not_partially_mixed(self) -> None:
        merged = merge_iceberg_properties(
            {"s3.access-key-id": "catalog-key"},
            profile=StorageProfile(
                access_key_id="profile-key",
                secret_access_key="profile-secret",
            ),
        )

        assert merged["s3.access-key-id"] == "catalog-key"
        assert "s3.secret-access-key" not in merged

    def test_source_credentials_do_not_mix_with_profile(self) -> None:
        merged = runtime_s3_profile(
            {
                "s3_profile": StorageProfile(
                    access_key_id="profile-key",
                    secret_access_key="profile-secret",
                    profile_name="profile-name",
                ),
                "s3": S3Config(access_key_id="source-key"),
            }
        )

        assert merged.access_key_id == "source-key"
        assert merged.secret_access_key is None
        assert merged.profile_name is None


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))

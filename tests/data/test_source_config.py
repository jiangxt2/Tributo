"""Tests for SourceConfig discriminated union and LegacyConfigNormalizer.

Covers every legacy format documented in the original ``data_loader.py``:
type=s3, type=csv, type=clickhouse, type=iceberg.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tributo.data.provider import DataSourceProvider, ResolvedSource
from tributo.data.provider_registry import register_provider, unregister_provider
from tributo.data.source_config import (
    CanonicalSourceInput,
    CsvSourceConfig,
    IcebergSourceConfig,
    LegacyConfigNormalizer,
    ParquetSourceConfig,
    ProviderSourceConfig,
    RawSourceConfig,
    RayReadTaskOptions,
    SqlPartitioning,
    SqlSourceConfig,
    apply_source_projection,
    normalize_legacy_inference_json_source,
    normalize_legacy_inference_source,
    source_projection,
)


class _ProjectionPluginProvider(DataSourceProvider):
    provider_id = "example.hive"
    projection_option_name = "projected_columns"

    def normalize(self, source: CanonicalSourceInput) -> ResolvedSource:
        return ResolvedSource(
            provider_id=self.provider_id,
            canonical_uri="hive://catalog/analytics/events",
        )


def test_source_configuration_reprs_hide_runtime_payloads() -> None:
    sources = (
        SqlSourceConfig(
            dialect="clickhouse",
            user="sensitive-user",
            password="top-secret",
            sql="SELECT 'business-secret'",
        ),
        IcebergSourceConfig(
            catalog="catalog",
            table="analytics.events",
            catalog_properties={"rest.token": "catalog-secret"},
            s3={"secret_access_key": "iceberg-secret"},
        ),
        ProviderSourceConfig(
            provider="third.party",
            uri="custom://source",
            options={"password": "provider-secret"},
        ),
        RawSourceConfig(type="third-party", raw={"token": "raw-secret"}),
    )

    rendered = " ".join(repr(source) for source in sources)
    for secret in (
        "sensitive-user",
        "top-secret",
        "business-secret",
        "catalog-secret",
        "iceberg-secret",
        "provider-secret",
        "raw-secret",
    ):
        assert secret not in rendered


class TestLegacyS3:
    """Legacy ``type: s3`` config normalization."""

    def test_parquet_format(self) -> None:
        result = LegacyConfigNormalizer.normalize(
            {"type": "s3", "uri": "s3://bkt/data.parquet", "format": "parquet"}
        )
        assert isinstance(result, ParquetSourceConfig)
        assert result.path == "s3://bkt/data.parquet"

    def test_parquet_format_is_default(self) -> None:
        result = LegacyConfigNormalizer.normalize(
            {"type": "s3", "uri": "s3://bkt/data.parquet"}
        )
        assert isinstance(result, ParquetSourceConfig)

    def test_csv_format(self) -> None:
        result = LegacyConfigNormalizer.normalize(
            {"type": "s3", "uri": "s3://bkt/data.csv", "format": "csv"}
        )
        assert isinstance(result, CsvSourceConfig)
        assert result.path == "s3://bkt/data.csv"

    def test_unsupported_format_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported s3 format"):
            LegacyConfigNormalizer.normalize(
                {"type": "s3", "uri": "s3://bkt/data.orc", "format": "orc"}
            )

    def test_with_s3_config(self) -> None:
        s3_cfg = {"endpoint": "http://minio:9000", "region": "cn-north-1"}
        result = LegacyConfigNormalizer.normalize(
            {"type": "s3", "uri": "s3://bkt/data.parquet", "s3": s3_cfg}
        )
        assert isinstance(result, ParquetSourceConfig)
        assert result.s3 is not None
        assert result.s3.endpoint == s3_cfg["endpoint"]
        assert result.s3.region == s3_cfg["region"]

    def test_with_columns(self) -> None:
        result = LegacyConfigNormalizer.normalize(
            {
                "type": "s3",
                "uri": "s3://bkt/data.parquet",
                "columns": ["a", "b", "c"],
            }
        )
        assert isinstance(result, ParquetSourceConfig)
        assert result.columns == ["a", "b", "c"]


class TestLegacyCsv:
    """Legacy ``type: csv`` config normalization.

    ⚠️ The historical behaviour is that ``type=csv`` WITHOUT an explicit
    ``format`` field defaults to reading Parquet.  Only ``format=csv``
    actually reads CSV.
    """

    def test_no_format_defaults_to_parquet(self) -> None:
        """Regression test: historical default reads Parquet."""
        result = LegacyConfigNormalizer.normalize(
            {"type": "csv", "path": "/tmp/data.parquet"}
        )
        assert isinstance(result, ParquetSourceConfig)

    def test_empty_format_defaults_to_parquet(self) -> None:
        """Empty string ``format`` never matches ``"csv"``."""
        result = LegacyConfigNormalizer.normalize(
            {"type": "csv", "path": "/tmp/data.parquet", "format": ""}
        )
        assert isinstance(result, ParquetSourceConfig)

    def test_format_csv_reads_csv(self) -> None:
        result = LegacyConfigNormalizer.normalize(
            {"type": "csv", "path": "/tmp/data.csv", "format": "csv"}
        )
        assert isinstance(result, CsvSourceConfig)
        assert result.path == "/tmp/data.csv"


class TestLegacyClickHouse:
    """Legacy ``type: clickhouse`` config normalization."""

    def test_minimal(self) -> None:
        result = LegacyConfigNormalizer.normalize(
            {"type": "clickhouse", "ch_sql": "SELECT 1"}
        )
        assert isinstance(result, SqlSourceConfig)
        assert result.dialect == "clickhouse"
        assert result.sql == "SELECT 1"

    def test_full(self) -> None:
        result = LegacyConfigNormalizer.normalize(
            {
                "type": "clickhouse",
                "ch_host": "127.0.0.1",
                "ch_port": 9000,
                "ch_database": "analytics",
                "ch_user": "reader",
                "ch_password": "secret",
                "ch_sql": "SELECT * FROM events",
            }
        )
        assert result.host == "127.0.0.1"
        assert result.port == 9000
        assert result.database == "analytics"
        assert result.user == "reader"
        assert result.password == "secret"

    def test_none_fields_for_env_fallback(self) -> None:
        """Missing config keys → None (env fallback at connection time)."""
        result = LegacyConfigNormalizer.normalize(
            {"type": "clickhouse", "ch_sql": "SELECT 1"}
        )
        assert result.host is None
        assert result.port is None
        assert result.database is None
        assert result.user is None
        assert result.password is None


class TestLegacyIceberg:
    """Legacy ``type: iceberg`` config normalization."""

    def test_full(self) -> None:
        result = LegacyConfigNormalizer.normalize(
            {"type": "iceberg", "catalog": "gravitino", "table": "telecom.users"}
        )
        assert isinstance(result, IcebergSourceConfig)
        assert result.catalog == "gravitino"
        assert result.table == "telecom.users"


class TestUnsupportedType:
    """Unrecognised ``type`` returns RawSourceConfig for plugin passthrough."""

    def test_unsupported(self) -> None:
        from tributo.data.source_config import RawSourceConfig

        result = LegacyConfigNormalizer.normalize({"type": "hive", "key": "val"})
        assert isinstance(result, RawSourceConfig)
        assert result.type == "hive"
        assert result.raw == {"type": "hive", "key": "val"}

    def test_builtin_type_rejected_from_raw(self) -> None:
        from tributo.data.source_config import RawSourceConfig

        with pytest.raises(
            ValueError, match="cannot be constructed as RawSourceConfig"
        ):
            RawSourceConfig(type="parquet", raw={})


class TestResolveEnv:
    """Environment variable fallback for SqlSourceConfig."""

    def test_explicit_values_preserved(self) -> None:
        source = SqlSourceConfig(
            dialect="clickhouse",
            host="10.0.0.1",
            port=9000,
            user="admin",
            password="pw",
            database="db",
            sql="SELECT 1",
        )
        resolved = LegacyConfigNormalizer.resolve_env(source)
        assert resolved.host == "10.0.0.1"
        assert resolved.user == "admin"

    def test_none_fields_resolved_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRIBUTO_CLICKHOUSE_HOST", "env-host")
        monkeypatch.setenv("TRIBUTO_CLICKHOUSE_PORT", "9999")
        monkeypatch.setenv("TRIBUTO_CLICKHOUSE_USER", "env-user")

        source = SqlSourceConfig(
            dialect="clickhouse",
            host=None,
            port=None,
            user=None,
            sql="SELECT 1",
        )
        resolved = LegacyConfigNormalizer.resolve_env(source)
        assert resolved.host == "env-host"
        assert resolved.port == 9999
        assert resolved.user == "env-user"


def test_legacy_inference_flat_source_is_normalized_by_data_module() -> None:
    source = normalize_legacy_inference_source(
        data_type="clickhouse",
        input_uri="legacy-input",
        s3_config={},
        ch_host="clickhouse.example",
        ch_port=8123,
        ch_database="analytics",
        ch_user="reader",
        ch_password="secret",
        ch_sql="SELECT * FROM events",
    )

    assert isinstance(source, SqlSourceConfig)
    assert source.dialect == "clickhouse"
    assert source.host == "clickhouse.example"
    assert source.database == "analytics"
    assert source.sql == "SELECT * FROM events"


def test_legacy_inference_json_source_preserves_parquet_compatibility(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = normalize_legacy_inference_json_source(
        {
            "type": "doris",
            "uri": "/legacy/input",
        }
    )

    assert isinstance(source, ParquetSourceConfig)
    assert source.path == "/legacy/input"
    assert "preserves historical Parquet semantics" in caplog.text


class TestSqlPartitioning:
    def test_existing_column_shape_defaults_to_parallel(self) -> None:
        partitioning = SqlPartitioning(column="id", num_partitions=4)

        assert partitioning.mode == "parallel"

    @pytest.mark.parametrize(
        "value",
        [
            {"mode": "parallel"},
            {"mode": "auto", "column": "id"},
            {"mode": "single", "num_partitions": 2},
        ],
    )
    def test_inconsistent_partitioning_mode_fails_closed(
        self, value: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError):
            SqlPartitioning.model_validate(value)


class TestRayReadTaskOptions:
    def test_doris_options_accept_the_supported_strict_subset(self) -> None:
        options = RayReadTaskOptions(
            num_cpus=1.5,
            scheduling_strategy="SPREAD",
            max_retries=3,
        )

        assert options.model_dump() == {
            "num_cpus": 1.5,
            "scheduling_strategy": "SPREAD",
            "max_retries": 3,
        }

    @pytest.mark.parametrize(
        "value",
        [
            {"num_cpus": True},
            {"num_cpus": 0},
            {"num_cpus": "1"},
            {"num_cpus": float("inf")},
            {"num_cpus": float("-inf")},
            {"num_cpus": float("nan")},
            {"scheduling_strategy": "DEFAULT"},
            {"max_retries": -1},
            {"max_retries": "3"},
            {"resources": {"olap_worker": 1}},
        ],
    )
    def test_unsupported_values_fail_closed(self, value: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            RayReadTaskOptions.model_validate(value)

    def test_doris_only_fields_are_rejected_for_other_sql_dialects(self) -> None:
        with pytest.raises(ValidationError, match="only valid for Doris"):
            SqlSourceConfig(
                dialect="clickhouse",
                table="events",
                tablet_size=100,
            )

    def test_doris_source_config_serializes_task_options_without_credentials(
        self,
    ) -> None:
        source = SqlSourceConfig(
            dialect="doris",
            table="events",
            tablet_size=100,
            on_query_plan_error="error",
            ray_remote_args=RayReadTaskOptions(
                num_cpus=1,
                scheduling_strategy="SPREAD",
                max_retries=0,
            ),
        )

        assert source.ray_remote_args is not None
        assert source.ray_remote_args.scheduling_strategy == "SPREAD"
        assert "password" not in repr(source)

    def test_doris_task_options_are_immutable_after_validation(self) -> None:
        options = RayReadTaskOptions(max_retries=0)

        with pytest.raises(ValidationError):
            options.max_retries = 1


class TestPydanticValidation:
    """Pydantic model validation for SourceConfig types."""

    def test_parquet_requires_path(self) -> None:
        with pytest.raises(ValidationError):
            ParquetSourceConfig(path="")  # type: ignore[arg-type]

    def test_sql_requires_dialect(self) -> None:
        with pytest.raises(ValidationError):
            SqlSourceConfig(dialect="unknown")  # type: ignore[arg-type]

    def test_csv_requires_path(self) -> None:
        with pytest.raises(ValidationError):
            CsvSourceConfig(path="")  # type: ignore[arg-type]


class TestSourceProjection:
    """Provider-native projection helpers."""

    def test_sql_projection_is_preserved(self) -> None:
        source = SqlSourceConfig(
            dialect="clickhouse",
            sql="SELECT * FROM events",
            columns=["id", "score"],
        )
        assert source_projection(source) == ["id", "score"]
        narrowed = apply_source_projection(source, ["score"])
        assert isinstance(narrowed, SqlSourceConfig)
        assert narrowed.columns == ["score"]

    def test_projection_outside_existing_columns_fails(self) -> None:
        source = ParquetSourceConfig(path="data.parquet", columns=["id"])
        with pytest.raises(ValueError, match="outside the configured"):
            apply_source_projection(source, ["missing"])

    def test_provider_projection_uses_native_option(self) -> None:
        source = ProviderSourceConfig(
            provider="tributo.parquet",
            uri="data.parquet",
        )
        projected = apply_source_projection(source, ["text"])
        assert isinstance(projected, ProviderSourceConfig)
        assert projected.options == {"columns": ["text"]}

    def test_third_party_projection_metadata_avoids_consumer_changes(self) -> None:
        register_provider(_ProjectionPluginProvider)
        try:
            source = ProviderSourceConfig(
                provider="example.hive",
                uri="hive://catalog/analytics/events",
            )

            projected = apply_source_projection(source, ["id", "score"])

            assert source_projection(projected) == ["id", "score"]
            assert projected.options == {"projected_columns": ["id", "score"]}
        finally:
            unregister_provider("example.hive")

    @pytest.mark.parametrize(
        "provider",
        ["tributo.postgresql", "postgresql", "tributo.lance", "lance"],
    )
    def test_provider_projection_covers_sql_and_table_bindings(
        self, provider: str
    ) -> None:
        source = ProviderSourceConfig(provider=provider, uri="source://target")

        projected = apply_source_projection(source, ["id"])

        assert isinstance(projected, ProviderSourceConfig)
        assert projected.options == {"columns": ["id"]}

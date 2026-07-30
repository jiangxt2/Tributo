"""Tests for SourceConfig discriminated union and LegacyConfigNormalizer.

Covers every legacy format documented in the original ``data_loader.py``:
type=s3, type=csv, type=clickhouse, type=iceberg.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tributo.data.source_config import (
    CsvSourceConfig,
    IcebergSourceConfig,
    LegacyConfigNormalizer,
    ParquetSourceConfig,
    SqlSourceConfig,
)


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
        assert result.s3 == s3_cfg

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

        with pytest.raises(ValueError, match="cannot be constructed as RawSourceConfig"):
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

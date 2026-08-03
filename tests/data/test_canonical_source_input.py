"""Tests for D1+D2 canonical input models.

Covers the target ``provider/uri`` shape (``ProviderSourceConfig``), the
structural discrimination of ``CanonicalSourceInput`` (type-shaped vs
provider-shaped, never guessed from a bare dict), and the explicit legacy
wrapper (``LegacySourceInput``).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import TypeAdapter, ValidationError

from tributo.data.source_config import (
    CanonicalSourceInput,
    CsvSourceConfig,
    LegacyConfigNormalizer,
    LegacySourceInput,
    ParquetSourceConfig,
    ProviderSourceConfig,
    SqlSourceConfig,
)


class TestProviderSourceConfig:
    """Target ``provider/uri`` shape validation."""

    def test_valid(self) -> None:
        cfg = ProviderSourceConfig(
            provider="tributo.parquet",
            uri="s3://bucket/data.parquet",
            options={"columns": ["a", "b"]},
        )
        assert cfg.provider == "tributo.parquet"
        assert cfg.uri == "s3://bucket/data.parquet"
        assert cfg.options == {"columns": ["a", "b"]}

    def test_empty_provider_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProviderSourceConfig(provider="", uri="file:///tmp/x.parquet")

    def test_missing_uri_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProviderSourceConfig(provider="tributo.parquet")

    def test_unknown_field_rejected(self) -> None:
        # StrictConfigModel: extra fields fail fast.
        with pytest.raises(ValidationError):
            ProviderSourceConfig(
                provider="tributo.parquet",
                uri="file:///tmp/x.parquet",
                path="typo-field",  # not a declared field
            )

    def test_options_accept_any_dict(self) -> None:
        cfg = ProviderSourceConfig(
            provider="tributo.clickhouse",
            uri="clickhouse://db.example:8443/analytics",
            options={"sql": "SELECT 1", "params": {"x": 1}},
        )
        assert cfg.options["params"] == {"x": 1}


class TestCanonicalSourceInput:
    """Structural discrimination between type-shaped and provider-shaped."""

    def test_type_shape_resolves_to_builtin(self) -> None:
        cfg = TypeAdapter(CanonicalSourceInput).validate_python(
            {"type": "parquet", "path": "s3://bkt/data.parquet"}
        )
        assert isinstance(cfg, ParquetSourceConfig)

    def test_type_shape_csv(self) -> None:
        cfg = TypeAdapter(CanonicalSourceInput).validate_python(
            {"type": "csv", "path": "local/data.csv"}
        )
        assert isinstance(cfg, CsvSourceConfig)

    def test_provider_shape_resolves_to_provider_config(self) -> None:
        cfg = TypeAdapter(CanonicalSourceInput).validate_python(
            {"provider": "tributo.parquet", "uri": "s3://bkt/data.parquet"}
        )
        assert isinstance(cfg, ProviderSourceConfig)
        assert cfg.provider == "tributo.parquet"

    def test_sql_type_shape(self) -> None:
        cfg = TypeAdapter(CanonicalSourceInput).validate_python(
            {"type": "sql", "dialect": "clickhouse", "sql": "SELECT 1"}
        )
        assert isinstance(cfg, SqlSourceConfig)

    def test_mixed_shape_rejected(self) -> None:
        # Both provider and type present: fails against every member —
        # the shape must never be guessed.
        with pytest.raises(ValidationError):
            TypeAdapter(CanonicalSourceInput).validate_python(
                {
                    "type": "parquet",
                    "path": "s3://bkt/data.parquet",
                    "provider": "tributo.parquet",
                    "uri": "s3://bkt/data.parquet",
                }
            )

    def test_provider_shape_missing_uri_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(CanonicalSourceInput).validate_python(
                {"provider": "tributo.parquet"}
            )

    def test_unknown_type_shape_rejected(self) -> None:
        # Unknown types do not silently pass through the canonical path;
        # RawSourceConfig passthrough lives in the legacy layer only.
        with pytest.raises(ValidationError):
            TypeAdapter(CanonicalSourceInput).validate_python(
                {"type": "kafka", "bootstrap": "localhost:9092"}
            )


class TestLegacySourceInput:
    """Explicit legacy wrapper — historical semantics live only here."""

    def test_default_mode(self) -> None:
        entry = LegacySourceInput(raw={"type": "csv"})
        assert entry.mode == "legacy"
        assert entry.raw == {"type": "csv"}

    def test_frozen(self) -> None:
        entry = LegacySourceInput(raw={"type": "parquet"})
        with pytest.raises(FrozenInstanceError):
            entry.__setattr__("raw", {"type": "s3"})

    def test_canonical_provider_key_is_rejected_by_legacy_normalizer(self) -> None:
        with pytest.raises(ValueError, match="canonical provider/uri"):
            LegacyConfigNormalizer.normalize(
                {"provider": "tributo.parquet", "uri": "data.parquet"}
            )

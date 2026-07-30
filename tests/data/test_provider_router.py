"""A1: Plugin discovery, Raw config passthrough, routing, error diagnostics."""

from __future__ import annotations

from typing import Any

import pytest

from tributo.data.provider import (
    Availability,
    ProviderIdentity,
    SourceProvider,
    SourceRouter,
    TransformBackend,
)
from tributo.data.source_config import LegacyConfigNormalizer, RawSourceConfig


# ---------------------------------------------------------------------------
# Mock providers for testing
# ---------------------------------------------------------------------------


class MockDaftProvider(SourceProvider):
    @classmethod
    def identity(cls) -> ProviderIdentity:
        return ProviderIdentity("tributo.daft", "daft", TransformBackend.DAFT)

    def open(self, config: Any) -> Any:
        return f"daft:{config}"

    @classmethod
    def can_handle(cls, config: Any) -> bool:
        from tributo.data.source_config import ParquetSourceConfig, CsvSourceConfig

        return isinstance(config, (ParquetSourceConfig, CsvSourceConfig))

    @classmethod
    def availability(cls, config: Any) -> Availability:
        # Mock: always ready — availability is an environment concern,
        # not a capability concern.  Routing unit tests should not
        # depend on whether daft is actually installed.
        return Availability.ready()


class MockLegacyClickHouseProvider(SourceProvider):
    @classmethod
    def identity(cls) -> ProviderIdentity:
        return ProviderIdentity(
            "tributo.legacy.clickhouse", "legacy", TransformBackend.RAY
        )

    def open(self, config: Any) -> Any:
        return f"legacy-ch:{config}"

    @classmethod
    def can_handle(cls, config: Any) -> bool:
        from tributo.data.source_config import SqlSourceConfig

        return isinstance(config, SqlSourceConfig) and config.dialect == "clickhouse"


class MockRawProvider(SourceProvider):
    @classmethod
    def identity(cls) -> ProviderIdentity:
        return ProviderIdentity("my.plugin", None, TransformBackend.DAFT)

    def open(self, config: Any) -> Any:
        assert isinstance(config, RawSourceConfig)
        return f"raw:{config.raw}"

    @classmethod
    def can_handle(cls, config: Any) -> bool:
        return isinstance(config, RawSourceConfig) and config.type == "my_custom"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPluginDiscovery:
    def test_register_provider(self) -> None:
        router = SourceRouter()
        router.register(MockDaftProvider)
        assert "tributo.daft" in router.list_providers()

    def test_duplicate_provider_id_rejected(self) -> None:
        router = SourceRouter()
        router.register(MockDaftProvider)
        with pytest.raises(ValueError, match="Duplicate provider_id"):
            router.register(MockDaftProvider)

    def test_alias_ambiguity_detected(self) -> None:
        """Two providers with same alias → explicit select must fail."""

        class AnotherDaft(SourceProvider):
            @classmethod
            def identity(cls) -> ProviderIdentity:
                return ProviderIdentity("other.daft", "daft", TransformBackend.DAFT)

            def open(self, config: Any) -> Any:
                return "other"

            @classmethod
            def can_handle(cls, config: Any) -> bool:
                from tributo.data.source_config import ParquetSourceConfig

                return isinstance(config, ParquetSourceConfig)

        router = SourceRouter()
        router.register(MockDaftProvider)
        router.register(AnotherDaft)

        from tributo.data.source_config import ParquetSourceConfig

        config = ParquetSourceConfig(path="/tmp/test.parquet")
        with pytest.raises(ValueError, match="Ambiguous alias"):
            router._explicit_select(config, "daft")


class TestRawConfigPassthrough:
    def test_unknown_type_becomes_raw(self) -> None:
        result = LegacyConfigNormalizer.normalize({"type": "hive", "key": "val"})
        assert isinstance(result, RawSourceConfig)
        assert result.type == "hive"
        assert result.raw == {"type": "hive", "key": "val"}

    def test_raw_config_routed_to_plugin(self) -> None:
        router = SourceRouter(strategy={("my_custom", None): ("my.plugin",)})
        router.register(MockRawProvider)
        router.register(MockDaftProvider)

        config = RawSourceConfig(type="my_custom", raw={"key": "val"})
        plan = router.open(config, engine="auto")
        assert plan == "raw:{'key': 'val'}"

    def test_builtin_type_rejected_from_raw(self) -> None:
        with pytest.raises(ValueError, match="cannot be constructed as RawSourceConfig"):
            RawSourceConfig(type="parquet", raw={})


class TestRouting:
    def setup_method(self) -> None:
        self.router = SourceRouter(
            strategy={
                ("parquet", None): ("tributo.daft",),
                ("sql_clickhouse", None): ("tributo.legacy.clickhouse",),
            }
        )
        self.router.register(MockDaftProvider)
        self.router.register(MockLegacyClickHouseProvider)

    def test_auto_parquet_selects_daft(self) -> None:
        from tributo.data.source_config import ParquetSourceConfig

        config = ParquetSourceConfig(path="/tmp/test.parquet")
        plan = self.router.open(config)
        assert plan.startswith("daft:")

    def test_auto_clickhouse_selects_legacy(self) -> None:
        from tributo.data.source_config import SqlSourceConfig

        config = SqlSourceConfig(dialect="clickhouse", sql="SELECT 1")
        plan = self.router.open(config)
        assert plan.startswith("legacy-ch:")

    def test_explicit_engine_by_alias(self) -> None:
        from tributo.data.source_config import ParquetSourceConfig

        config = ParquetSourceConfig(path="/tmp/test.parquet")
        plan = self.router.open(config, engine="daft")
        assert plan.startswith("daft:")

    def test_explicit_engine_by_provider_id(self) -> None:
        from tributo.data.source_config import ParquetSourceConfig

        config = ParquetSourceConfig(path="/tmp/test.parquet")
        plan = self.router.open(config, engine="tributo.daft")
        assert plan.startswith("daft:")

    def test_no_strategy_entry_raises(self) -> None:
        from tributo.data.source_config import ParquetSourceConfig

        empty_router = SourceRouter(strategy={})
        empty_router.register(MockDaftProvider)
        config = ParquetSourceConfig(path="/tmp/test.parquet")
        with pytest.raises(ValueError, match="No strategy entry"):
            empty_router.open(config)

    def test_explicit_engine_not_available_raises(self) -> None:
        """Explicit engine that is unavailable → raise, don't fallback."""

        class AlwaysUnavailable(SourceProvider):
            @classmethod
            def identity(cls) -> ProviderIdentity:
                return ProviderIdentity("tributo.broken", None, TransformBackend.DAFT)

            def open(self, config: Any) -> Any:
                return "broken"

            @classmethod
            def can_handle(cls, config: Any) -> bool:
                return True

            @classmethod
            def availability(cls, config: Any) -> Availability:
                return Availability.unavailable(reason="missing dep")

        router = SourceRouter()
        router.register(AlwaysUnavailable)
        from tributo.data.source_config import ParquetSourceConfig

        config = ParquetSourceConfig(path="/tmp/test.parquet")
        with pytest.raises(ValueError, match="not available"):
            router.open(config, engine="tributo.broken")


class TestErrorDiagnostics:
    def test_engine_not_found(self) -> None:
        router = SourceRouter()
        router.register(MockDaftProvider)
        from tributo.data.source_config import ParquetSourceConfig

        config = ParquetSourceConfig(path="/tmp/test.parquet")
        with pytest.raises(ValueError, match="No provider found"):
            router.open(config, engine="nonexistent")

    def test_sql_params_routes_to_legacy(self) -> None:
        """SQL with params → Daft can_handle must be False."""
        from tributo.data.source_config import SqlSourceConfig

        config = SqlSourceConfig(
            dialect="clickhouse", sql="SELECT * FROM t WHERE x = :p", params={"p": 42}
        )
        # Daft can't handle params
        assert not MockDaftProvider.can_handle(config)
        # ClickHouse legacy can
        assert MockLegacyClickHouseProvider.can_handle(config)


class TestAvailability:
    def test_ready(self) -> None:
        a = Availability.ready()
        assert a.is_available
        assert a.missing_extra is None
        assert a.reason is None

    def test_unavailable(self) -> None:
        a = Availability.unavailable(reason="no daft", missing_extra="daft[sql]")
        assert not a.is_available
        assert a.reason == "no daft"
        assert a.missing_extra == "daft[sql]"

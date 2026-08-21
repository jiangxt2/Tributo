"""ProviderRegistry contract tests: atomic registration, three-layer resolution,
explicit legacy semantics, third-party manual registration."""

from __future__ import annotations

import threading
import time
from typing import Any, cast

import pytest

from tributo.data import provider_builtins  # noqa: F401  (registers built-ins)
from tributo.data.provider import DatasetHandle, DataSourceProvider, ResolvedSource
from tributo.data.provider_builtins import ParquetProvider
from tributo.data.provider_registry import (
    ProviderRegistry,
    list_providers,
    register_provider,
    resolve_provider,
    unregister_provider,
)
from tributo.data.source_config import (
    LegacySourceInput,
    ProviderSourceConfig,
)
from tributo.exceptions import JobConfigurationError


def make_provider(
    pid: str, alias_set: frozenset[str] = frozenset()
) -> type[DataSourceProvider]:
    class _P(DataSourceProvider):
        provider_id = pid
        aliases = alias_set

        def normalize(self, source: object) -> ResolvedSource:
            return ResolvedSource(provider_id=pid, canonical_uri="mock://x")

        def open(self, resolved: ResolvedSource) -> DatasetHandle:
            raise NotImplementedError

    return _P


@pytest.fixture
def mock_provider():
    cls = make_provider("tributo.mock", frozenset({"mock"}))
    register_provider(cls)
    yield cls
    unregister_provider("tributo.mock")


class TestRegisterValidation:
    """Type/format validation — TypeError, distinct from conflicts."""

    def test_not_a_class(self) -> None:
        with pytest.raises(TypeError, match="DataSourceProvider subclass"):
            register_provider(cast(Any, "not-a-class"))

    def test_not_a_provider_subclass(self) -> None:
        class NotAProvider:
            provider_id = "tributo.nope"

        with pytest.raises(TypeError, match="DataSourceProvider subclass"):
            register_provider(cast(Any, NotAProvider))

    @pytest.mark.parametrize(
        "bad_id",
        ["parquet", "Tributo.parquet", "tributo..parquet", "tributo.parquet!", ""],
    )
    def test_invalid_provider_id_format(self, bad_id: str) -> None:
        cls = make_provider(bad_id)
        with pytest.raises(TypeError, match="Invalid provider_id"):
            register_provider(cls)

    @pytest.mark.parametrize("bad_alias", ["", "has.dot", "Upper"])
    def test_invalid_alias_format(self, bad_alias: str) -> None:
        cls = make_provider("tributo.ok", frozenset({bad_alias}))
        with pytest.raises(TypeError, match="Invalid alias"):
            register_provider(cls)

    def test_invalid_projection_option_metadata(self) -> None:
        cls = make_provider("tributo.ok")
        cls.projection_option_name = "selected fields"
        with pytest.raises(TypeError, match="projection_option_name"):
            register_provider(cls)

    @pytest.mark.parametrize("option_name", ["selected.fields", "selected-fields"])
    def test_projection_option_metadata_accepts_safe_external_keys(
        self, option_name: str
    ) -> None:
        cls = make_provider("tributo.ok")
        cls.projection_option_name = option_name
        register_provider(cls)
        try:
            assert "tributo.ok" in list_providers()
        finally:
            unregister_provider("tributo.ok")

    def test_invalid_relative_uri_metadata(self) -> None:
        cls = make_provider("tributo.ok")
        cls.relative_uri_is_path = cast(Any, "yes")
        with pytest.raises(TypeError, match="relative_uri_is_path"):
            register_provider(cls)


class TestAtomicRegistration:
    """Conflicts fail fast — JobConfigurationError, no partial entry."""

    def test_duplicate_provider_id(self, mock_provider: object) -> None:
        with pytest.raises(JobConfigurationError, match="already registered"):
            register_provider(make_provider("tributo.mock"))

    def test_alias_collides_with_existing_alias(self, mock_provider: object) -> None:
        other = make_provider("tributo.other", frozenset({"mock"}))
        with pytest.raises(JobConfigurationError, match="already maps to provider"):
            register_provider(other)
        assert "tributo.other" not in list_providers()
        assert (
            resolve_provider(
                ProviderSourceConfig(provider="mock", uri="mock://x")
            ).provider_id
            == "tributo.mock"
        )

    def test_failed_registration_leaves_no_partial_entry(
        self, mock_provider: object
    ) -> None:
        # A provider whose alias collides must not register its ID either.
        with pytest.raises(JobConfigurationError):
            register_provider(make_provider("tributo.partial", frozenset({"mock"})))
        with pytest.raises(JobConfigurationError, match="Unknown provider"):
            resolve_provider(
                ProviderSourceConfig(provider="tributo.partial", uri="mock://x")
            )


class TestResolveExactAndAlias:
    """Layers 1-2: exact ID wins, alias resolves, unknown fails loudly."""

    def test_exact_provider_id(self, mock_provider: object) -> None:
        provider = resolve_provider(
            ProviderSourceConfig(provider="tributo.mock", uri="mock://x")
        )
        assert provider.provider_id == "tributo.mock"

    def test_alias_resolution(self, mock_provider: object) -> None:
        provider = resolve_provider(
            ProviderSourceConfig(provider="mock", uri="mock://x")
        )
        assert provider.provider_id == "tributo.mock"

    def test_unknown_provider_lists_available(self, mock_provider: object) -> None:
        with pytest.raises(JobConfigurationError, match="tributo.mock"):
            resolve_provider(
                ProviderSourceConfig(provider="tributo.nope", uri="mock://x")
            )

    def test_alias_duplicate_fails_at_registration(self, mock_provider: object) -> None:
        # Under the ID/alias format split (IDs contain a dot, aliases never
        # do) an alias can never collide with a provider ID; alias↔alias
        # collisions fail at registration time, which also prevents any
        # ambiguity at resolution time.
        assert "mock" not in {p.split(".")[0] for p in list_providers()}


class TestResolveBuiltin:
    """Canonical type/path/dialect shapes → built-in provider routes."""

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ({"type": "parquet", "path": "x"}, "tributo.parquet"),
            ({"type": "csv", "path": "x"}, "tributo.csv"),
            (
                {"type": "sql", "dialect": "clickhouse", "sql": "SELECT 1"},
                "tributo.clickhouse",
            ),
            (
                {"type": "sql", "dialect": "doris", "sql": "SELECT 1"},
                "tributo.doris",
            ),
            (
                {"type": "sql", "dialect": "hive", "sql": "SELECT 1"},
                "tributo.hive",
            ),
            (
                {
                    "type": "sql",
                    "dialect": "postgresql",
                    "table": "events",
                },
                "tributo.postgresql",
            ),
            ({"type": "iceberg", "catalog": "c", "table": "t"}, "tributo.iceberg"),
        ],
    )
    def test_canonical_route(self, source: dict, expected: str) -> None:
        from pydantic import TypeAdapter

        from tributo.data.source_config import CanonicalSourceInput

        cfg = TypeAdapter(CanonicalSourceInput).validate_python(source)
        provider = resolve_provider(cfg)
        assert provider.provider_id == expected

    @pytest.mark.parametrize("dialect", ["mysql"])
    def test_unsupported_sql_dialects_fail(self, dialect: str) -> None:
        from pydantic import TypeAdapter

        from tributo.data.source_config import CanonicalSourceInput

        cfg = TypeAdapter(CanonicalSourceInput).validate_python(
            {"type": "sql", "dialect": dialect, "sql": "SELECT 1"}
        )
        with pytest.raises(JobConfigurationError, match=r"unsupported"):
            resolve_provider(cfg)

    def test_builtin_route_missing_provider(self) -> None:
        from pydantic import TypeAdapter

        from tributo.data.source_config import CanonicalSourceInput

        cfg = TypeAdapter(CanonicalSourceInput).validate_python(
            {"type": "parquet", "path": "x"}
        )
        unregister_provider("tributo.parquet")
        try:
            with pytest.raises(JobConfigurationError, match="not registered"):
                resolve_provider(cfg)
        finally:
            register_provider(ParquetProvider)


class TestResolveLegacy:
    """LegacySourceInput: historical semantics live only in this branch."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ({"type": "parquet", "path": "x"}, "tributo.parquet"),
            ({"type": "s3", "uri": "s3://b/x"}, "tributo.parquet"),
            ({"type": "s3", "uri": "s3://b/x", "format": "parquet"}, "tributo.parquet"),
            ({"type": "s3", "uri": "s3://b/x", "format": "csv"}, "tributo.csv"),
            # type=csv without format reads Parquet (historical default).
            ({"type": "csv", "path": "x"}, "tributo.parquet"),
            ({"type": "csv", "path": "x", "format": "csv"}, "tributo.csv"),
            ({"type": "clickhouse", "ch_sql": "SELECT 1"}, "tributo.clickhouse"),
            ({"type": "doris", "sql": "SELECT 1"}, "tributo.doris"),
            ({"type": "postgresql", "table": "events"}, "tributo.postgresql"),
            ({"type": "iceberg", "catalog": "c", "table": "t"}, "tributo.iceberg"),
        ],
    )
    def test_legacy_route(self, raw: dict, expected: str) -> None:
        provider = resolve_provider(LegacySourceInput(raw=raw))
        assert provider.provider_id == expected

    @pytest.mark.parametrize("dialect", ["mysql"])
    def test_legacy_unsupported_sql(self, dialect: str) -> None:
        with pytest.raises(JobConfigurationError, match=r"unsupported"):
            resolve_provider(
                LegacySourceInput(raw={"type": dialect, "sql": "SELECT 1"})
            )

    def test_legacy_unknown_type(self, mock_provider: object) -> None:
        with pytest.raises(JobConfigurationError, match="Unknown legacy source type"):
            resolve_provider(LegacySourceInput(raw={"type": "kafka"}))

    def test_legacy_bad_s3_format(self) -> None:
        with pytest.raises(JobConfigurationError, match="Unsupported s3 format"):
            resolve_provider(
                LegacySourceInput(
                    raw={"type": "s3", "uri": "s3://b/x", "format": "orc"}
                )
            )

    def test_lance_routes_to_native_engine_provider(self) -> None:
        assert (
            resolve_provider(
                ProviderSourceConfig(provider="tributo.lance", uri="s3://bkt/table")
            ).provider_id
            == "tributo.lance"
        )
        assert (
            resolve_provider(
                LegacySourceInput(raw={"type": "lance", "path": "x"})
            ).provider_id
            == "tributo.lance"
        )


class TestThirdPartyRegistration:
    """Explicit manual registration for third-party providers."""

    def test_third_party_provider(self) -> None:
        cls = make_provider("myorg.mysql", frozenset({"mymysql"}))
        register_provider(cls)
        try:
            provider = resolve_provider(
                ProviderSourceConfig(provider="myorg.mysql", uri="jdbc:mysql://db/x")
            )
            assert provider.provider_id == "myorg.mysql"
            assert "myorg.mysql" in list_providers()
        finally:
            unregister_provider("myorg.mysql")

    def test_default_resolution_lazily_loads_provider_plugins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tributo.data.provider_plugins as provider_plugins
        import tributo.data.provider_registry as provider_registry

        cls = make_provider("myorg.hive")

        def register_plugin(registry: ProviderRegistry) -> None:
            registry.register(cls)

        monkeypatch.setattr(provider_registry, "_provider_plugins_loaded", False)
        monkeypatch.setattr(provider_registry, "_provider_plugins_loading", False)
        monkeypatch.setattr(
            provider_plugins, "register_discovered_providers", register_plugin
        )
        try:
            provider = resolve_provider(
                ProviderSourceConfig(provider="myorg.hive", uri="hive://catalog/db/t")
            )
            assert provider.provider_id == "myorg.hive"
            assert provider_registry._provider_plugins_loaded is True
        finally:
            unregister_provider("myorg.hive")

    def test_concurrent_first_resolution_waits_for_provider_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tributo.data.provider_plugins as provider_plugins
        import tributo.data.provider_registry as provider_registry

        cls = make_provider("myorg.concurrent")
        started = threading.Event()
        release = threading.Event()
        results: list[str] = []
        failures: list[BaseException] = []

        def register_plugin(registry: ProviderRegistry) -> None:
            started.set()
            assert release.wait(timeout=5)
            registry.register(cls)

        def resolve_plugin() -> None:
            try:
                provider = resolve_provider(
                    ProviderSourceConfig(
                        provider="myorg.concurrent", uri="mock://source"
                    )
                )
                results.append(provider.provider_id)
            except BaseException as exc:  # pragma: no cover - asserted below
                failures.append(exc)

        monkeypatch.setattr(provider_registry, "_provider_plugins_loaded", False)
        monkeypatch.setattr(provider_registry, "_provider_plugins_loading", False)
        monkeypatch.setattr(
            provider_plugins, "register_discovered_providers", register_plugin
        )
        first = threading.Thread(target=resolve_plugin)
        second = threading.Thread(target=resolve_plugin)
        try:
            first.start()
            assert started.wait(timeout=5)
            second.start()
            time.sleep(0.02)
            assert second.is_alive()
            release.set()
            first.join(timeout=5)
            second.join(timeout=5)
            assert not first.is_alive()
            assert not second.is_alive()
            assert failures == []
            assert results == ["myorg.concurrent", "myorg.concurrent"]
        finally:
            release.set()
            first.join(timeout=5)
            second.join(timeout=5)
            unregister_provider("myorg.concurrent")

    def test_provider_discovery_failure_remains_retryable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tributo.data.provider_plugins as provider_plugins
        import tributo.data.provider_registry as provider_registry

        calls = 0

        def register_plugin(registry: ProviderRegistry) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("broken entry-point metadata")

        monkeypatch.setattr(provider_registry, "_provider_plugins_loaded", False)
        monkeypatch.setattr(provider_registry, "_provider_plugins_loading", False)
        monkeypatch.setattr(
            provider_plugins, "register_discovered_providers", register_plugin
        )

        with pytest.raises(RuntimeError, match="broken entry-point metadata"):
            list_providers()

        assert provider_registry._provider_plugins_loaded is False
        list_providers()
        assert calls == 2
        assert provider_registry._provider_plugins_loaded is True


class TestListAndCleanup:
    def test_list_sorted(self, mock_provider: object) -> None:
        register_provider(make_provider("tributo.aaa"))
        register_provider(make_provider("tributo.zzz"))
        try:
            providers = list_providers()
            assert "tributo.aaa" in providers
            assert "tributo.mock" in providers
            assert "tributo.zzz" in providers
            assert providers.index("tributo.aaa") < providers.index("tributo.zzz")
        finally:
            unregister_provider("tributo.aaa")
            unregister_provider("tributo.zzz")

    def test_unregister_removes_aliases(self, mock_provider: object) -> None:
        unregister_provider("tributo.mock")
        assert "tributo.mock" not in list_providers()
        with pytest.raises(JobConfigurationError, match="Unknown provider"):
            resolve_provider(ProviderSourceConfig(provider="mock", uri="mock://x"))

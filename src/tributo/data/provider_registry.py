"""ProviderRegistry — atomic registration + explicit three-layer resolution.

Resolution order (ADR 001): exact provider ID → alias → built-in legacy
mapping.  Once a layer matches, resolution stops — providers are never
probed by availability, and the bare-dict legacy semantics live only in the
``LegacySourceInput`` branch.  Registration is atomic: every ID/alias
conflict is checked up front, so a failed registration leaves no partial
entry behind.

Third-party providers register explicitly (``register_provider``); entry
point discovery, version negotiation and lifecycle management belong to
PL1+PL2.
"""

from __future__ import annotations

import re
import threading
from typing import Any

from tributo.data.provider import DataSourceProvider
from tributo.data.source_config import (
    BuiltinSourceConfig,
    CanonicalSourceInput,
    CsvSourceConfig,
    IcebergSourceConfig,
    LegacySourceInput,
    ParquetSourceConfig,
    ProviderSourceConfig,
    SqlSourceConfig,
)
from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import PublicAPI

# Provider ID format (ADR 001): "<domain>.<name>", lowercase dot-separated,
# each segment matches [a-z][a-z0-9_]*.
_PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Frozen canonical routes: builtin source type → logical provider ID.
_CANONICAL_TYPE_ROUTES: dict[str, str] = {
    "parquet": "tributo.parquet",
    "csv": "tributo.csv",
    "sql_clickhouse": "tributo.clickhouse",
    "sql_doris": "tributo.doris",
    "iceberg": "tributo.iceberg",
}

# SQL dialects without a canonical provider — unsupported/experimental, with
# explicit diagnostics instead of silent routing.
_UNSUPPORTED_SQL_DIALECTS: dict[str, str] = {
    "postgresql": "ConnectorX path is experimental and unsupported",
    "mysql": "MySQL is unsupported; use tributo.doris (MySQL protocol)",
}
_DEFERRED_PROVIDER_DIAGNOSTICS: dict[str, str] = {
    "tributo.lance": "Lance is deferred; use a supported file provider",
    "lance": "Lance is deferred; use a supported file provider",
}


class ProviderRegistry:
    """Thread-safe registry of ``DataSourceProvider`` classes.

    Registration maps the provider ID and every alias to the class; a
    conflict on any key fails the whole registration atomically.
    """

    def __init__(self) -> None:
        self._providers: dict[str, type[DataSourceProvider]] = {}
        self._aliases: dict[str, str] = {}
        self._lock = threading.Lock()

    # -- write path -----------------------------------------------------------

    def register(self, cls: type[DataSourceProvider]) -> None:
        """Register a provider class.

        Raises:
            TypeError: If *cls* is not a ``DataSourceProvider`` subclass, or
                its ``provider_id``/aliases violate the ID format.
            JobConfigurationError: If the provider ID or any alias collides
                with an already registered ID/alias.  On failure nothing is
                registered (atomic).
        """
        if not isinstance(cls, type):
            raise TypeError(
                f"Provider must be a DataSourceProvider subclass, got {type(cls).__name__!r}"
            )
        if not issubclass(cls, DataSourceProvider):
            raise TypeError(
                f"Provider {cls.__name__!r} must be a DataSourceProvider subclass."
            )
        provider_id = cls.provider_id
        if not _PROVIDER_ID_RE.fullmatch(provider_id):
            raise TypeError(
                f"Invalid provider_id {provider_id!r}: expected '<domain>.<name>' "
                f"with lowercase segments [a-z][a-z0-9_]*"
            )
        aliases = frozenset(cls.aliases)
        for alias in aliases:
            if not _ALIAS_RE.fullmatch(alias):
                raise TypeError(f"Invalid alias {alias!r}: expected [a-z][a-z0-9_]*")

        # Atomicity: validate every collision before writing anything.
        with self._lock:
            if provider_id in self._providers:
                raise JobConfigurationError(
                    f"Provider {provider_id!r} already registered. "
                    f"Available: {self._sorted_ids()}"
                )
            if provider_id in self._aliases:
                raise JobConfigurationError(
                    f"Provider ID {provider_id!r} collides with an existing alias "
                    f"of provider {self._aliases[provider_id]!r}."
                )
            for alias in aliases:
                if alias in self._providers:
                    raise JobConfigurationError(
                        f"Alias {alias!r} collides with registered provider ID "
                        f"{alias!r}."
                    )
                if alias in self._aliases and self._aliases[alias] != provider_id:
                    raise JobConfigurationError(
                        f"Alias {alias!r} already maps to provider "
                        f"{self._aliases[alias]!r}; conflicts are fail-fast."
                    )
            self._providers[provider_id] = cls
            for alias in aliases:
                self._aliases[alias] = provider_id

    def unregister(self, provider_id: str) -> None:
        """Remove a provider and its aliases (idempotent; test cleanup)."""
        with self._lock:
            cls = self._providers.pop(provider_id, None)
            if cls is not None:
                for alias in cls.aliases:
                    if self._aliases.get(alias) == provider_id:
                        del self._aliases[alias]

    # -- read path ------------------------------------------------------------

    def resolve(
        self, source: CanonicalSourceInput | LegacySourceInput
    ) -> DataSourceProvider:
        """Select the provider for *source* (exact ID → alias → built-in route).

        Raises:
            JobConfigurationError: If no provider can be resolved, or the
                source is an unsupported/unknown type.  Never silently
                falls back to another provider.
        """
        if isinstance(source, ProviderSourceConfig):
            return self._resolve_by_id_or_alias(source.provider)
        if isinstance(source, LegacySourceInput):
            return self._resolve_legacy(source.raw)
        if isinstance(
            source,
            (
                ParquetSourceConfig,
                CsvSourceConfig,
                IcebergSourceConfig,
                SqlSourceConfig,
            ),
        ):
            return self._resolve_builtin(source)
        raise JobConfigurationError(
            f"Unsupported source input: {type(source).__name__!r}"
        )

    def get(self, provider_id: str) -> DataSourceProvider:
        """Return an instance of the provider registered under *provider_id*.

        Raises:
            JobConfigurationError: If the ID is unknown (aliases are not
                accepted here — use ``resolve``).
        """
        with self._lock:
            cls = self._providers.get(provider_id)
        if cls is None:
            raise JobConfigurationError(
                f"Unknown provider: {provider_id!r}. Available: {self.list_providers()}"
            )
        return cls()

    def list_providers(self) -> list[str]:
        """Return the sorted provider IDs (not aliases)."""
        with self._lock:
            return sorted(self._providers)

    def _sorted_ids(self) -> list[str]:
        return sorted(self._providers)

    # -- resolution internals --------------------------------------------------

    def _resolve_by_id_or_alias(self, provider_id: str) -> DataSourceProvider:
        # Layer 1: exact provider ID.  Layer 2: alias.
        if provider_id in _DEFERRED_PROVIDER_DIAGNOSTICS:
            raise JobConfigurationError(
                f"Provider {provider_id!r} is not available: "
                f"{_DEFERRED_PROVIDER_DIAGNOSTICS[provider_id]}"
            )
        with self._lock:
            cls = self._providers.get(provider_id)
            if cls is None:
                mapped = self._aliases.get(provider_id)
                if mapped is not None:
                    cls = self._providers.get(mapped)
        if cls is None:
            raise JobConfigurationError(
                f"Unknown provider: {provider_id!r}. Available: {self.list_providers()}"
            )
        return cls()

    def _resolve_builtin(self, source: BuiltinSourceConfig) -> DataSourceProvider:
        if isinstance(source, SqlSourceConfig):
            route_key = f"sql_{source.dialect}"
            if source.dialect in _UNSUPPORTED_SQL_DIALECTS:
                raise JobConfigurationError(
                    f"SQL dialect {source.dialect!r} is unsupported: "
                    f"{_UNSUPPORTED_SQL_DIALECTS[source.dialect]}"
                )
        else:
            route_key = source.type
        provider_id = _CANONICAL_TYPE_ROUTES.get(route_key)
        if provider_id is None:
            raise JobConfigurationError(
                f"No built-in provider route for {route_key!r}. "
                f"Available: {self.list_providers()}"
            )
        with self._lock:
            cls = self._providers.get(provider_id)
        if cls is None:
            raise JobConfigurationError(
                f"Built-in provider {provider_id!r} is not registered. "
                f"Registered: {self.list_providers()}"
            )
        return cls()

    def _resolve_legacy(self, raw: dict[str, Any]) -> DataSourceProvider:
        # Historical semantics live only here (LegacySourceInput): type=csv
        # without format reads Parquet, type=s3 defaults to Parquet.
        data_type = raw.get("type", "csv")
        if data_type == "s3":
            fmt = raw.get("format", "parquet")
            if fmt == "parquet":
                provider_id = "tributo.parquet"
            elif fmt == "csv":
                provider_id = "tributo.csv"
            else:
                raise JobConfigurationError(
                    f"Unsupported s3 format: {fmt!r} (expected parquet or csv)"
                )
        elif data_type == "csv":
            fmt = raw.get("format", "")
            # Legacy default: format="" never matched "csv" → Parquet.
            provider_id = "tributo.csv" if fmt == "csv" else "tributo.parquet"
        elif data_type == "parquet":
            provider_id = "tributo.parquet"
        elif data_type == "clickhouse":
            provider_id = "tributo.clickhouse"
        elif data_type == "doris":
            provider_id = "tributo.doris"
        elif data_type == "iceberg":
            provider_id = "tributo.iceberg"
        elif data_type in _UNSUPPORTED_SQL_DIALECTS:
            raise JobConfigurationError(
                f"SQL dialect {data_type!r} is unsupported: "
                f"{_UNSUPPORTED_SQL_DIALECTS[data_type]}"
            )
        elif data_type == "lance":
            raise JobConfigurationError(
                f"Legacy source type {data_type!r} is not available: "
                f"{_DEFERRED_PROVIDER_DIAGNOSTICS['lance']}"
            )
        else:
            raise JobConfigurationError(
                f"Unknown legacy source type: {data_type!r}. "
                f"Available: {self.list_providers()}"
            )
        with self._lock:
            cls = self._providers.get(provider_id)
        if cls is None:
            raise JobConfigurationError(
                f"Legacy route {data_type!r} resolves to unregistered provider "
                f"{provider_id!r}. Registered: {self.list_providers()}"
            )
        return cls()


_registry = ProviderRegistry()


@PublicAPI(stability="beta")
def register_provider(cls: type[DataSourceProvider]) -> None:
    """Register a provider class (explicit/manual registration; PL1+PL2 adds
    entry-point discovery).

    Raises:
        TypeError: Invalid provider class or ID format.
        JobConfigurationError: ID/alias conflict (atomic — nothing registered).
    """
    _registry.register(cls)


@PublicAPI(stability="beta")
def resolve_provider(
    source: CanonicalSourceInput | LegacySourceInput,
) -> DataSourceProvider:
    """Resolve a canonical or legacy source to a provider instance.

    Raises:
        JobConfigurationError: Unknown provider, unsupported dialect, or
            unknown legacy type — with an explicit diagnostic.
    """
    return _registry.resolve(source)


@PublicAPI(stability="beta")
def unregister_provider(provider_id: str) -> None:
    """Remove a provider and its aliases (test cleanup / hot removal)."""
    _registry.unregister(provider_id)


@PublicAPI(stability="beta")
def list_providers() -> list[str]:
    """Return the sorted registered provider IDs."""
    return _registry.list_providers()

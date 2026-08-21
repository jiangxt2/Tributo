"""Built-in logical providers for canonical bounded ingestion.

Parquet, CSV, Iceberg, Lance, ClickHouse, Doris, and PostgreSQL normalize input
and build logical plans. Execution is delegated directly to Ray Data, Daft, or
an installed engine Binding. Provider IDs identify logical sources, not
execution engines.

Credential handling: passwords/access keys live in ``runtime_options`` only
(their values never appear in ``repr``/logs/errors), while ``identity_options``
carry only result-affecting fields (columns, snapshot, SQL/params digests)
used by the ``DatasetRef`` ref_id algorithm.  SQL text itself is not a
credential; the digest used for identity never leaks the raw query.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Mapping
from urllib.parse import unquote, urlsplit, urlunsplit

from pydantic import ValidationError

from tributo.data._s3 import resolve_endpoint
from tributo.data.base import S3Config
from tributo.data.provider import DataSourceProvider, ResolvedSource
from tributo.data.refs import (
    SENSITIVE_QUERY_KEYS,
    SENSITIVE_QUERY_PREFIXES,
    digest,
)
from tributo.data.source_config import (
    CanonicalSourceInput,
    CsvSourceConfig,
    IcebergSourceConfig,
    ParquetSourceConfig,
    ProviderSourceConfig,
    SqlPartitioning,
    SqlSourceConfig,
)
from tributo.exceptions import JobConfigurationError

if TYPE_CHECKING:
    from tributo.data.scan_plan import LogicalScanPlan

logger = logging.getLogger(__name__)

# Result-affecting option keys recognized on the provider/uri shape.  Unknown
# keys stay in runtime_options: a key we cannot classify never risks leaking
# credentials into a ref_id.
_FILE_IDENTITY_KEYS: frozenset[str] = frozenset({"columns"})
_ICEBERG_IDENTITY_KEYS: frozenset[str] = frozenset(
    {"selected_fields", "snapshot_id", "row_filter"}
)
# SQL text and parameters are represented by digests; projection is safe to
# retain directly because it is a bounded list of result column names.
_SQL_IDENTITY_KEYS: frozenset[str] = frozenset({"columns"})
_LANCE_IDENTITY_KEYS: frozenset[str] = frozenset(
    {"columns", "filter", "version", "asof"}
)

# Option-key whitelists for the provider/uri shape: an unrecognized key is a
# configuration error (never silently ignored — that would read the wrong
# data instead of failing loudly).
_FILE_OPTION_KEYS: frozenset[str] = frozenset({"columns", "s3"})
_SQL_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "sql",
        "table",
        "params",
        "host",
        "port",
        "http_port",
        "flight_port",
        "database",
        "schema",
        "user",
        "password",
        "columns",
        "partitioning",
        "protocol",
        "auth",
        "batch_size",
        "shard_mode",
        "hash_column",
        "hash_shards",
        "parallelism",
        "sort_key",
    }
)
_LANCE_OPTION_KEYS: frozenset[str] = frozenset(
    {"columns", "filter", "version", "asof", "s3"}
)
_ICEBERG_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "selected_fields",
        "snapshot_id",
        "row_filter",
        "catalog_name",
        "table_identifier",
        "catalog_properties",
        "s3",
    }
)

# Dialect connection defaults (mirror of _DIALECT_DEFAULTS in source_config).
_DIALECT_DEFAULTS: dict[str, dict[str, int | str]] = {
    "clickhouse": {"port": 8123, "user": "default"},
    "doris": {"port": 9030, "user": "root"},
    "postgresql": {"port": 5432, "user": "postgres"},
    "hive": {"port": 10000, "user": "default"},
}


def _split_options(
    options: Mapping[str, Any], identity_keys: frozenset[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split provider options into identity (result-affecting) and runtime.

    ``None`` values are treated as absent on the identity side: an explicit
    ``columns=None`` and an omitted ``columns`` read the same data, so they
    must produce the same ref_id.
    """
    identity = {
        k: v
        for k, v in options.items()
        if k in identity_keys and v is not None and v != []
    }
    runtime = {k: v for k, v in options.items() if k not in identity_keys}
    return identity, runtime


def _check_provider(
    actual: str, expected_id: str, expected_aliases: frozenset[str]
) -> None:
    if actual != expected_id and actual not in expected_aliases:
        raise JobConfigurationError(
            f"Provider {actual!r} cannot be normalized by {expected_id!r}"
        )


# S3Config field set — used both for identity extraction and for fail-fast
# rejection of unknown ``s3`` dict keys (a typo must not silently read the
# wrong endpoint with the wrong region).
_S3_CONFIG_FIELDS: frozenset[str] = frozenset(
    {"access_key_id", "secret_access_key", "endpoint", "region"}
)


def _identity_s3(s3: Any, *, include_environment: bool = False) -> dict[str, Any]:
    """Result-affecting subset of an S3 config for the identity.

    ``endpoint``/``region`` select *which* storage endpoint the data comes
    from — they belong in the ref_id.  Credentials (access key id / secret)
    never do; they stay in ``runtime_options`` for the actual read.  The
    endpoint value itself is stripped of userinfo/credential query — an
    endpoint like ``https://user:pass@host`` must not leak into the identity.
    """
    # An absent S3 configuration normally means that this source did not opt
    # into an S3 storage profile.  Do not resolve process environment here:
    # doing so would make local-file identities depend on unrelated S3
    # variables.  File providers opt in explicitly when their URI is s3://.
    if s3 is None and not include_environment:
        return {}
    if s3 is None:
        endpoint = None
        region = None
    elif isinstance(s3, S3Config):
        endpoint = s3.endpoint
        region = s3.region
    elif isinstance(s3, Mapping):
        endpoint = s3.get("endpoint")
        region = s3.get("region")
    else:
        return {}  # unreachable — type-checked in _require_option_types
    if not endpoint:
        endpoint = resolve_endpoint()
    if not region:
        region = os.getenv("AWS_REGION")
    identity: dict[str, Any] = {}
    if endpoint is not None:
        identity["endpoint"] = _strip_uri_credentials(endpoint)
    if region is not None:
        identity["region"] = region
    return identity


def _build_s3_config(provider_id: str, value: Any) -> S3Config | None:
    """Validate a dict-shaped S3 option without exposing invalid values."""
    if value is None or isinstance(value, S3Config):
        return value
    if not isinstance(value, Mapping):
        raise JobConfigurationError(
            f"{provider_id}: option 's3' must be an S3Config or mapping"
        )
    unknown = sorted(set(value) - _S3_CONFIG_FIELDS)
    if unknown:
        raise JobConfigurationError(
            f"{provider_id}: unknown s3 option(s) {unknown}; "
            f"supported: {sorted(_S3_CONFIG_FIELDS)}"
        )
    invalid = False
    try:
        config = S3Config(**dict(value))
    except ValidationError:
        invalid = True
    if invalid:
        # Raise outside the native handler so a ValidationError containing the
        # rejected input cannot survive as ``__context__``.
        raise JobConfigurationError(f"{provider_id}: invalid s3 configuration")
    return config


# Credential detection for catalog properties is substring-based: PyIceberg
# catalogs use namespaced keys (rest.token, s3.session-token,
# aws.secret-access-key, client_password, api_key, ...) that a bare
# exact-name denylist misses.  Every CREDENTIAL_KEYS name is covered by
# these tokens; values are additionally stripped of userinfo.
_SENSITIVE_PROPERTY_TOKENS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "credential",
    "signature",
    "api_key",
    "api-key",
    "access_key",
    "access-key",
    "session_token",
    "session-token",
)


def _is_sensitive_property_key(key: str) -> bool:
    """Recognize credential-like catalog keys without dropping safe settings.

    Matching both separator-preserving and compact forms catches names such
    as ``rest.token``, ``api_key`` and ``apiKey`` while keeping unrelated
    properties such as ``api_version`` and ``session_timeout`` in identity.
    """
    lowered = key.lower()
    compact = lowered.replace(".", "").replace("-", "").replace("_", "")
    return any(
        token in lowered or token.replace("-", "").replace("_", "") in compact
        for token in _SENSITIVE_PROPERTY_TOKENS
    )


def _identity_catalog_properties(
    properties: Mapping[str, str],
) -> dict[str, str]:
    """Result-affecting subset of Iceberg catalog properties for identity.

    Every non-credential property (type, uri, warehouse, catalog-impl, ...)
    selects which catalog/table is read and belongs in the ref_id.  Values
    are stripped of userinfo so ``uri=https://user:pass@host`` cannot leak
    a credential into the identity; credential-keyed properties (password,
    token, api_key, session_token, ...) stay in ``runtime_options``.
    """
    identity: dict[str, str] = {}
    for key, value in properties.items():
        if _is_sensitive_property_key(key):
            continue
        identity[key] = _strip_uri_credentials(value)
    return identity


def _params_digest(provider_id: str, params: Mapping[str, Any]) -> str:
    """SHA-256 digest of bound query parameters with a clear error.

    datetime/date/time/Decimal are serialized deterministically by
    ``refs._jsonable`` — they are legal bound-parameter values.  Anything
    else non-JSON-serializable is a configuration error surfaced here
    (ref_id time), not a raw ``TypeError``.
    """
    invalid = False
    try:
        params_digest = digest(params)
    except TypeError:
        invalid = True
    if invalid:
        raise JobConfigurationError(
            f"{provider_id}: options['params'] must contain JSON-serializable values"
        )
    return params_digest


# ---------------------------------------------------------------------------
# File providers (Parquet / CSV)
# ---------------------------------------------------------------------------


def _require_option_types(
    provider_id: str,
    options: Mapping[str, Any],
    expected: Mapping[str, type],
) -> None:
    """Validate option *value* types — a wrong type must fail during
    provider normalization (e.g. ``columns="id"`` would silently read one
    column).

    Stricter than a bare ``isinstance``:
    * ``bool`` never satisfies an ``int`` requirement (it is an int subclass
      but meaningless as a port);
    * list-valued options are checked element-wise (``columns=[1]`` would
      read columns named ``"1"`` / crash downstream);
    * ``s3`` dict keys must be known S3Config fields (fail-fast instead of
      a late binding error);
    * ``catalog_properties`` keys and values must be ``str`` (they are
      identity material — a non-str value would corrupt the ref_id).
    """
    for key, expected_type in expected.items():
        value = options.get(key)
        if value is None:
            continue
        if isinstance(value, bool) and expected_type is int:
            raise JobConfigurationError(
                f"{provider_id}: option {key!r} must be "
                f"{expected_type.__name__}, got bool"
            )
        if not isinstance(value, expected_type):
            raise JobConfigurationError(
                f"{provider_id}: option {key!r} must be "
                f"{expected_type.__name__}, got {type(value).__name__}"
            )
        # Element-level checks.  ``isinstance(value, list/dict)`` narrows for
        # mypy; the earlier ``isinstance(value, expected_type)`` already
        # guarantees the expected type matches.
        if isinstance(value, list):
            bad = [item for item in value if not isinstance(item, str)]
            if bad:
                raise JobConfigurationError(
                    f"{provider_id}: option {key!r} entries must be str, got {bad!r}"
                )
        elif isinstance(value, dict):
            if key == "s3":
                unknown = sorted(set(value) - _S3_CONFIG_FIELDS)
                if unknown:
                    raise JobConfigurationError(
                        f"{provider_id}: unknown s3 option(s) {unknown}; "
                        f"supported: {sorted(_S3_CONFIG_FIELDS)}"
                    )
                invalid = [
                    f"{nested_key}={type(nested_value).__name__}"
                    for nested_key, nested_value in value.items()
                    if not isinstance(nested_value, (str, type(None)))
                ]
                if invalid:
                    raise JobConfigurationError(
                        f"{provider_id}: option 's3' values must be str or None; "
                        f"invalid: {invalid}"
                    )
            elif key == "catalog_properties":
                bad_entries = {
                    k: type(v).__name__
                    for k, v in value.items()
                    if not isinstance(k, str) or not isinstance(v, str)
                }
                if bad_entries:
                    raise JobConfigurationError(
                        f"{provider_id}: option {key!r} keys and values "
                        f"must be str; invalid entries: {list(bad_entries)!r}"
                    )


_FILE_OPTION_TYPES: dict[str, type] = {
    "columns": list,
    "s3": dict,
}
_SQL_OPTION_TYPES: dict[str, type] = {
    "sql": str,
    "table": str,
    "params": dict,
    "columns": list,
    "host": str,
    "port": int,
    "http_port": int,
    "flight_port": int,
    "database": str,
    "schema": str,
    "user": str,
    "password": str,
    "partitioning": dict,
    "protocol": str,
    "auth": str,
    "batch_size": int,
    "shard_mode": str,
    "hash_column": str,
    "hash_shards": int,
    "parallelism": int,
    "sort_key": str,
}
_LANCE_OPTION_TYPES: dict[str, type] = {
    "columns": list,
    "filter": str,
    "asof": str,
    "s3": dict,
}
_ICEBERG_OPTION_TYPES: dict[str, type] = {
    "selected_fields": list,
    "snapshot_id": int,
    "row_filter": str,
    "catalog_name": str,
    "table_identifier": str,
    "catalog_properties": dict,
    "s3": dict,
}


def _check_option_keys(
    provider_id: str, options: Mapping[str, Any], allowed: frozenset[str]
) -> None:
    """Reject unrecognized provider/uri option keys (fail-fast, no silent ignore)."""
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise JobConfigurationError(
            f"{provider_id}: unknown option(s) {unknown}; supported: {sorted(allowed)}"
        )


# Sensitive query parameter names/prefixes live in ``refs`` (shared with the
# ``compute_ref_id`` canonical_uri backstop check).  Values of any other
# query key (e.g. versionId) stay — they can be result-affecting.


def _is_sensitive_query_param(key: str) -> bool:
    lowered = key.lower()
    return lowered in SENSITIVE_QUERY_KEYS or lowered.startswith(
        SENSITIVE_QUERY_PREFIXES
    )


def _strip_uri_credentials(uri: str) -> str:
    """Remove credentials from a URI.

    Strips userinfo (``s3://user:pass@host/p`` → ``s3://host/p``) and
    credential query parameters (``?token=...&X-Amz-Signature=...`` →
    no query).  Non-credential query keys (e.g. ``versionId``) are kept.
    URIs without credentials (local paths, ``file://``, plain ``s3://host``)
    are returned unchanged.
    """
    parts = urlsplit(uri)
    if parts.username is None and not parts.query:
        # URI schemes are case-insensitive. Canonicalize S3 here so both the
        # identity and every engine binding receive the same executable URI.
        return urlunsplit(parts) if parts.scheme.lower() == "s3" else uri
    try:
        hostname = parts.hostname or ""
        port = parts.port
    except ValueError as exc:
        raise JobConfigurationError("URI contains an invalid authority") from exc
    netloc = hostname
    if port is not None:
        netloc = f"{netloc}:{port}"
    query = "&".join(
        kv
        for kv in parts.query.split("&")
        if kv and not _is_sensitive_query_param(kv.split("=", 1)[0])
    )
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _validate_file_uri(provider_id: str, uri: str) -> None:
    """Reject S3 URI features the current file providers cannot execute safely.

    The Parquet and CSV providers accept an object path plus an ``S3Config``;
    they do not implement URI userinfo, signed query URLs, object-version
    query parameters, or fragments.  Failing during normalization prevents a
    credential-free ref_id from describing a read that would use different
    object-key semantics at execution time.
    """
    try:
        parts = urlsplit(uri)
    except ValueError as exc:
        raise JobConfigurationError(
            f"{provider_id}: URI must be a valid file or S3 URI"
        ) from exc
    if parts.scheme.lower() != "s3":
        return
    if parts.username is not None:
        raise JobConfigurationError(
            f"{provider_id}: S3 URI userinfo is unsupported; use the 's3' option"
        )
    if parts.query or parts.fragment:
        raise JobConfigurationError(
            f"{provider_id}: S3 URI query parameters and fragments are unsupported; "
            "use S3 configuration/options instead"
        )


class _FileProvider(DataSourceProvider):
    """Shared file-source provider logic (Parquet/CSV)."""

    projection_option_name = "columns"
    relative_uri_is_path = True
    _config_cls: ClassVar[type[ParquetSourceConfig] | type[CsvSourceConfig]]
    _allowed_option_keys: ClassVar[frozenset[str]] = _FILE_OPTION_KEYS

    def normalize(self, source: CanonicalSourceInput) -> ResolvedSource:
        if isinstance(source, ProviderSourceConfig):
            _check_provider(source.provider, self.provider_id, self.aliases)
            _validate_file_uri(self.provider_id, source.uri)
            normalized_uri = _strip_uri_credentials(source.uri)
            _check_option_keys(
                self.provider_id, source.options, self._allowed_option_keys
            )
            _require_option_types(self.provider_id, source.options, _FILE_OPTION_TYPES)
            identity, runtime = _split_options(source.options, _FILE_IDENTITY_KEYS)
            # endpoint/region select the storage endpoint — part of the
            # identity; credentials (access key id / secret) stay runtime.
            s3_identity = _identity_s3(
                source.options.get("s3"),
                include_environment=urlsplit(normalized_uri).scheme.lower() == "s3",
            )
            if s3_identity:
                identity["s3"] = s3_identity
            # Runtime execution uses the same normalized URI as the logical
            # identity; credential-bearing/signed S3 URIs were rejected above.
            runtime["uri"] = normalized_uri
            return ResolvedSource(
                provider_id=self.provider_id,
                canonical_uri=normalized_uri,
                identity_options=identity,
                runtime_options=runtime,
            )
        if isinstance(source, self._config_cls):
            _validate_file_uri(self.provider_id, source.path)
            normalized_uri = _strip_uri_credentials(source.path)
            identity = {"columns": list(source.columns)} if source.columns else {}
            s3_identity = _identity_s3(
                source.s3,
                include_environment=urlsplit(normalized_uri).scheme.lower() == "s3",
            )
            if s3_identity:
                identity["s3"] = s3_identity
            return ResolvedSource(
                provider_id=self.provider_id,
                canonical_uri=normalized_uri,
                identity_options=identity,
                # Runtime and identity share one executable URI. Credentials
                # remain in the isolated S3 configuration object instead.
                runtime_options={"uri": normalized_uri, "s3": source.s3},
            )
        raise JobConfigurationError(
            f"{self.provider_id}: unsupported source {type(source).__name__!r}"
        )


class ParquetProvider(_FileProvider):
    """Logical parquet data source (local or S3)."""

    provider_id = "tributo.parquet"
    aliases = frozenset({"parquet"})
    _config_cls = ParquetSourceConfig

    def plan(self, resolved: ResolvedSource) -> "LogicalScanPlan":
        """Describe a Parquet read for the selected native Binding."""
        from tributo.data.scan_plan import FileScan

        scheme = urlsplit(resolved.canonical_uri).scheme.lower()
        filesystem_id = scheme if scheme else "local"
        return FileScan(
            provider_id=self.provider_id,
            connector_id="parquet",
            uri=resolved.canonical_uri,
            filesystem_id=filesystem_id,
            options=resolved.identity_options,
        )


class CsvProvider(_FileProvider):
    """Logical csv data source (local or S3)."""

    provider_id = "tributo.csv"
    aliases = frozenset({"csv"})
    _config_cls = CsvSourceConfig

    def plan(self, resolved: ResolvedSource) -> "LogicalScanPlan":
        """Describe a CSV read for the selected native Binding."""
        from tributo.data.scan_plan import FileScan

        scheme = urlsplit(resolved.canonical_uri).scheme.lower()
        filesystem_id = scheme if scheme else "local"
        return FileScan(
            provider_id=self.provider_id,
            connector_id="csv",
            uri=resolved.canonical_uri,
            filesystem_id=filesystem_id,
            options=resolved.identity_options,
        )


# ---------------------------------------------------------------------------
# Iceberg provider
# ---------------------------------------------------------------------------


class IcebergProvider(DataSourceProvider):
    """Logical Iceberg table source."""

    provider_id = "tributo.iceberg"
    aliases = frozenset({"iceberg"})
    projection_option_name = "selected_fields"

    def normalize(self, source: CanonicalSourceInput) -> ResolvedSource:
        if isinstance(source, ProviderSourceConfig):
            _check_provider(source.provider, self.provider_id, self.aliases)
            _check_option_keys(self.provider_id, source.options, _ICEBERG_OPTION_KEYS)
            _require_option_types(
                self.provider_id, source.options, _ICEBERG_OPTION_TYPES
            )
            identity, runtime_extra = _split_options(
                source.options, _ICEBERG_IDENTITY_KEYS
            )
            # uri shape: <scheme>://<catalog>/<table_identifier> — the table
            # path is required; explicit options win over the uri parts.
            try:
                raw_uri = urlsplit(source.uri)
            except ValueError as exc:
                raise JobConfigurationError(
                    f"{self.provider_id}: uri must be a valid "
                    "<scheme>://<catalog>/<table> address"
                ) from exc
            if raw_uri.query or raw_uri.fragment:
                raise JobConfigurationError(
                    f"{self.provider_id}: catalog uri must not contain a "
                    "query or fragment"
                )
            if not raw_uri.scheme or raw_uri.hostname is None:
                raise JobConfigurationError(
                    f"{self.provider_id}: uri must use the form "
                    "<scheme>://<catalog>/<table>"
                )
            if raw_uri.username is not None:
                raise JobConfigurationError(
                    f"{self.provider_id}: catalog uri userinfo is unsupported; "
                    "use catalog_properties for credentials"
                )
            parsed = urlsplit(_strip_uri_credentials(source.uri))
            catalog = parsed.hostname
            table = parsed.path.lstrip("/") or ""
            if not table:
                raise JobConfigurationError(
                    f"{self.provider_id}: uri must include the table path, "
                    f"e.g. warehouse://catalog/db.table"
                )
            # The uri scheme selects the catalog type (warehouse/glue/rest/
            # ...) — part of the identity; non-credential catalog properties
            # enter it too, credential-keyed ones stay runtime.
            if parsed.scheme:
                identity["catalog_scheme"] = parsed.scheme
            properties = runtime_extra.get("catalog_properties") or {}
            catalog_identity = _identity_catalog_properties(properties)
            if catalog_identity:
                identity["catalog_properties"] = catalog_identity
            s3_identity = _identity_s3(runtime_extra.get("s3"))
            if s3_identity:
                identity["s3"] = s3_identity
            # None means "not provided" — fall back to the uri parts.
            catalog_name = runtime_extra.get("catalog_name")
            table_identifier = runtime_extra.get("table_identifier")
            runtime = {
                "catalog_name": catalog_name if catalog_name is not None else catalog,
                "table_identifier": (
                    table_identifier if table_identifier is not None else table
                ),
                # Full properties dict — execution needs the credentials too.
                "catalog_properties": dict(properties),
                "s3": runtime_extra.get("s3"),
            }
            return ResolvedSource(
                provider_id=self.provider_id,
                canonical_uri=(
                    f"{runtime['catalog_name']}/{runtime['table_identifier']}"
                ),
                identity_options=identity,
                runtime_options=runtime,
            )
        if isinstance(source, IcebergSourceConfig):
            builtin_identity: dict[str, Any] = {}
            if source.snapshot_id is not None:
                builtin_identity["snapshot_id"] = source.snapshot_id
            if source.row_filter:
                builtin_identity["row_filter"] = source.row_filter
            if source.selected_fields:
                builtin_identity["selected_fields"] = list(source.selected_fields)
            scheme = source.catalog_properties.get("type")
            if scheme:
                builtin_identity["catalog_scheme"] = scheme
            catalog_identity = _identity_catalog_properties(source.catalog_properties)
            if catalog_identity:
                builtin_identity["catalog_properties"] = catalog_identity
            s3 = _build_s3_config(self.provider_id, source.s3)
            s3_identity = _identity_s3(s3)
            if s3_identity:
                builtin_identity["s3"] = s3_identity
            return ResolvedSource(
                provider_id=self.provider_id,
                canonical_uri=f"{source.catalog}/{source.table}",
                identity_options=builtin_identity,
                runtime_options={
                    "catalog_name": source.catalog,
                    "table_identifier": source.table,
                    "catalog_properties": dict(source.catalog_properties),
                    "s3": s3,
                },
            )
        raise JobConfigurationError(
            f"{self.provider_id}: unsupported source {type(source).__name__!r}"
        )

    def plan(self, resolved: ResolvedSource) -> "LogicalScanPlan":
        """Describe a catalog-backed Iceberg table for native engine readers."""
        from tributo.data.scan_plan import (
            CatalogTableRef,
            SnapshotVersionRef,
            SourceCapability,
            TableScan,
        )

        catalog_name = str(resolved.runtime_options.get("catalog_name") or "default")
        table_identifier = str(resolved.runtime_options.get("table_identifier") or "")
        if not table_identifier:
            raise JobConfigurationError(
                "tributo.iceberg: resolved source is missing table_identifier"
            )
        parts = tuple(part for part in table_identifier.split(".") if part)
        if not parts:
            raise JobConfigurationError(
                "tributo.iceberg: table_identifier must be non-empty"
            )
        required: set[SourceCapability] = set()
        if resolved.identity_options.get("selected_fields"):
            required.add(SourceCapability.PROJECTION)
        snapshot_id = resolved.identity_options.get("snapshot_id")
        return TableScan(
            provider_id=self.provider_id,
            connector_id="iceberg",
            table=CatalogTableRef(
                catalog_id=catalog_name,
                namespace=parts[:-1],
                table=parts[-1],
            ),
            version_ref=(
                SnapshotVersionRef(snapshot_id=int(snapshot_id))
                if snapshot_id is not None
                else None
            ),
            required_capabilities=frozenset(required),
            options=resolved.identity_options,
        )


class LanceProvider(DataSourceProvider):
    """Logical Lance table source delegated to Ray Data or Daft."""

    provider_id = "tributo.lance"
    aliases = frozenset({"lance"})
    projection_option_name = "columns"
    relative_uri_is_path = True

    def normalize(self, source: CanonicalSourceInput) -> ResolvedSource:
        if not isinstance(source, ProviderSourceConfig):
            raise JobConfigurationError(
                "tributo.lance requires the provider/uri source shape"
            )
        _check_provider(source.provider, self.provider_id, self.aliases)
        _validate_file_uri(self.provider_id, source.uri)
        _check_option_keys(self.provider_id, source.options, _LANCE_OPTION_KEYS)
        _require_option_types(self.provider_id, source.options, _LANCE_OPTION_TYPES)
        version = source.options.get("version")
        if version is not None and (
            isinstance(version, bool) or not isinstance(version, (int, str))
        ):
            raise JobConfigurationError(
                "tributo.lance: option 'version' must be int or str"
            )
        if version is not None and source.options.get("asof") is not None:
            raise JobConfigurationError(
                "tributo.lance: version and asof are mutually exclusive"
            )
        identity, runtime = _split_options(source.options, _LANCE_IDENTITY_KEYS)
        normalized_uri = _strip_uri_credentials(source.uri)
        s3_identity = _identity_s3(
            source.options.get("s3"),
            include_environment=urlsplit(normalized_uri).scheme.lower() == "s3",
        )
        if s3_identity:
            identity["s3"] = s3_identity
        runtime["s3"] = source.options.get("s3")
        return ResolvedSource(
            provider_id=self.provider_id,
            canonical_uri=normalized_uri,
            identity_options=identity,
            runtime_options=runtime,
        )

    def plan(self, resolved: ResolvedSource) -> "LogicalScanPlan":
        from tributo.data.scan_plan import (
            AsOfVersionRef,
            NumericVersionRef,
            SourceCapability,
            TableScan,
            TagVersionRef,
            UriTableRef,
        )

        required: set[SourceCapability] = set()
        if resolved.identity_options.get("columns"):
            required.add(SourceCapability.PROJECTION)
        if resolved.identity_options.get("filter"):
            required.add(SourceCapability.PREDICATE_PUSHDOWN)
        version = resolved.identity_options.get("version")
        asof = resolved.identity_options.get("asof")
        version_ref: Any = None
        if isinstance(version, int):
            version_ref = NumericVersionRef(version=version)
        elif isinstance(version, str):
            version_ref = TagVersionRef(tag=version)
        elif isinstance(asof, str):
            try:
                timestamp = datetime.fromisoformat(asof)
            except ValueError as exc:
                raise JobConfigurationError(
                    "tributo.lance: option 'asof' must be an ISO-8601 timestamp"
                ) from exc
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise JobConfigurationError(
                    "tributo.lance: option 'asof' must include a timezone offset"
                )
            version_ref = AsOfVersionRef(timestamp=timestamp)
        return TableScan(
            provider_id=self.provider_id,
            connector_id="lance",
            table=UriTableRef(uri=resolved.canonical_uri),
            version_ref=version_ref,
            required_capabilities=frozenset(required),
            options=resolved.identity_options,
        )


# ---------------------------------------------------------------------------
# SQL readers (migrated from training/data_loader.py)
# ---------------------------------------------------------------------------


def _resolve_connection(
    dialect: str, rt: Mapping[str, Any]
) -> tuple[str, int, str, str, str]:
    """Fill connection fields: explicit value > env var > dialect default.

    Same semantics as ``LegacyConfigNormalizer.resolve_env`` — ``None``
    fields signal env fallback at connection time (never baked into the
    ResolvedSource).
    """
    prefix = f"TRIBUTO_{dialect.upper()}"
    defaults = _DIALECT_DEFAULTS.get(dialect, {})

    host = rt.get("host")
    if host is None:
        host = os.getenv(f"{prefix}_HOST", "localhost")
    port = rt.get("port")
    if port is None:
        port_env = os.getenv(f"{prefix}_PORT", "")
        port = port_env if port_env else defaults.get("port", 8123)
    try:
        port = int(port)
    except (TypeError, ValueError):
        raise JobConfigurationError(
            f"{dialect} source port must be an integer"
        ) from None
    user = rt.get("user")
    if user is None:
        user = os.getenv(f"{prefix}_USER", str(defaults.get("user", "")))
    password = rt.get("password")
    if password is None:
        password = os.getenv(f"{prefix}_PASSWORD", "")
    database = rt.get("database")
    if database is None:
        database = os.getenv(f"{prefix}_DB", "")
    return host, port, user, password, database


# ---------------------------------------------------------------------------
# SQL providers (logical control plane only)
# ---------------------------------------------------------------------------


class _SqlProvider(DataSourceProvider):
    """Shared SQL-source normalization and plan construction."""

    projection_option_name = "columns"
    _allowed_option_keys: ClassVar[frozenset[str]] = _SQL_OPTION_KEYS

    def _canonical_uri(
        self,
        source: SqlSourceConfig | None = None,
        runtime: Mapping[str, Any] | None = None,
    ) -> str:
        """Credential-free URI from the *effective* connection values.

        Port is result-relevant (it selects the endpoint) so it is part of
        the identity; explicit options override the uri, and the rebuilt
        URI reflects the effective endpoint.
        """
        if runtime is not None:
            host = runtime.get("host")
            port = runtime.get("port")
            database = runtime.get("database")
        else:
            assert source is not None
            host = source.host
            port = source.port
            database = source.database

        # Resolve non-credential environment fallbacks before computing the
        # identity.  The read path resolves the same values again, while
        # credentials remain runtime-only.
        host, port, _, _, database = _resolve_connection(
            self._dialect(),
            {"host": host, "port": port, "database": database},
        )
        if not host:
            raise JobConfigurationError(
                f"{self.provider_id}: SQL host must not be empty"
            )
        if "@" in host:
            raise JobConfigurationError(
                f"{self.provider_id}: SQL host must not contain userinfo; "
                "use separate user/password options"
            )
        # Bracket IPv6 literals so the canonical URI remains parseable.
        uri_host = host if ":" not in host or host.startswith("[") else f"[{host}]"
        default_port = int(_DIALECT_DEFAULTS[self._dialect()]["port"])
        port_str = f":{port}" if port is not None and port != default_port else ""
        return _strip_uri_credentials(
            f"{self._dialect()}://{uri_host}{port_str}/{database or ''}"
        )

    def normalize(self, source: CanonicalSourceInput) -> ResolvedSource:
        if isinstance(source, ProviderSourceConfig):
            _check_provider(source.provider, self.provider_id, self.aliases)
            _check_option_keys(
                self.provider_id, source.options, self._allowed_option_keys
            )
            _require_option_types(self.provider_id, source.options, _SQL_OPTION_TYPES)
            options = dict(source.options)
            sql = options.pop("sql", None)
            table = options.pop("table", None)
            params = options.pop("params", None)
            columns = options.pop("columns", None)
            partitioning_raw = options.pop("partitioning", None)
            protocol = options.pop("protocol", None)
            partitioning = (
                SqlPartitioning.model_validate(partitioning_raw)
                if partitioning_raw is not None
                else None
            )
            if params == {}:
                params = None
            has_sql = bool(sql and str(sql).strip())
            has_table = bool(table and str(table).strip())
            if has_sql == has_table:
                raise JobConfigurationError(
                    f"{self.provider_id}: exactly one of options['sql'] or "
                    "options['table'] is required"
                )
            if self._dialect() == "doris" and has_table and protocol is None:
                protocol = "mysql"
            if protocol is not None and (
                self._dialect() != "doris" or protocol not in {"mysql", "flight"}
            ):
                raise JobConfigurationError(
                    f"{self.provider_id}: protocol must be mysql or flight and "
                    "is only valid for Doris table reads"
                )
            # The uri is the connection address: host[:port]/database come
            # from it, explicit options win.
            try:
                parsed = urlsplit(source.uri)
                parsed_hostname = parsed.hostname
            except ValueError as exc:
                raise JobConfigurationError(
                    f"{self.provider_id}: uri must be a valid "
                    f"{self._dialect()}://host[:port]/database address"
                ) from exc
            if parsed.scheme != self._dialect():
                raise JobConfigurationError(
                    f"{self.provider_id}: uri scheme must be "
                    f"{self._dialect()!r}, got {parsed.scheme!r}"
                )
            if parsed_hostname is None:
                raise JobConfigurationError(
                    f"{self.provider_id}: uri must be a valid "
                    f"{self._dialect()}://host[:port]/database address"
                )
            if parsed.query or parsed.fragment:
                raise JobConfigurationError(
                    f"{self.provider_id}: SQL uri must not contain a query or "
                    "fragment; pass connection options explicitly"
                )
            try:
                parsed_port = parsed.port
            except ValueError as exc:
                raise JobConfigurationError(
                    f"{self.provider_id}: uri contains an invalid port"
                ) from exc
            identity: dict[str, Any] = (
                {"sql_digest": digest(str(sql))} if has_sql else {"table": str(table)}
            )
            if params is not None:
                identity["params_digest"] = _params_digest(self.provider_id, params)
            if columns:
                identity["columns"] = list(columns)
            schema = options.get("schema")
            if schema:
                identity["schema"] = schema
            if partitioning is not None:
                identity["partitioning"] = partitioning.model_dump(mode="json")
            runtime = {
                k: v
                for k, v in options.items()
                if k not in {"host", "port", "database"}
            }
            # None means "not provided" — fall back to the uri part, never
            # an env-var fallback that disagrees with the canonical_uri.
            host = options.get("host")
            runtime["host"] = host if host is not None else parsed_hostname
            port = options.get("port")
            runtime["port"] = port if port is not None else parsed_port
            database = options.get("database")
            runtime["database"] = (
                database if database is not None else parsed.path.lstrip("/") or None
            )
            # uri userinfo carries credentials too — explicit options win,
            # otherwise the uri's user/password must not be silently dropped
            # (canonical_uri strips them; runtime keeps them for the read).
            user = options.get("user")
            runtime["user"] = (
                user
                if user is not None
                else (unquote(parsed.username) if parsed.username is not None else None)
            )
            password = options.get("password")
            runtime["password"] = (
                password
                if password is not None
                else (unquote(parsed.password) if parsed.password is not None else None)
            )
            if has_sql:
                runtime["sql"] = str(sql)
            else:
                runtime["table"] = str(table)
            if protocol is not None:
                runtime["protocol"] = protocol
            if params is not None:
                runtime["params"] = params
            return ResolvedSource(
                provider_id=self.provider_id,
                canonical_uri=self._canonical_uri(runtime=runtime),
                identity_options=identity,
                runtime_options=runtime,
            )
        if isinstance(source, SqlSourceConfig):
            if source.dialect != self._dialect():
                raise JobConfigurationError(
                    f"{self.provider_id}: dialect {source.dialect!r} does not match"
                )
            builtin_identity: dict[str, Any] = (
                {"sql_digest": digest(source.sql)}
                if source.sql.strip()
                else {"table": str(source.table)}
            )
            if source.params is not None:
                builtin_identity["params_digest"] = _params_digest(
                    self.provider_id, source.params
                )
            if source.columns:
                builtin_identity["columns"] = list(source.columns)
            if source.database_schema:
                builtin_identity["schema"] = source.database_schema
            if source.partitioning is not None:
                builtin_identity["partitioning"] = source.partitioning.model_dump(
                    mode="json"
                )
            runtime = {
                "host": source.host,
                "port": source.port,
                "http_port": source.http_port,
                "flight_port": source.flight_port,
                "database": source.database,
                "schema": source.database_schema,
                "user": source.user,
                "password": source.password,
                "sql": source.sql,
                "table": source.table,
                "protocol": source.protocol,
                "params": source.params,
                "auth": source.auth,
                "batch_size": source.batch_size,
                "shard_mode": source.shard_mode,
                "hash_column": source.hash_column,
                "hash_shards": source.hash_shards,
                "parallelism": source.parallelism,
                "sort_key": source.sort_key,
            }
            return ResolvedSource(
                provider_id=self.provider_id,
                canonical_uri=self._canonical_uri(source),
                identity_options=builtin_identity,
                runtime_options=runtime,
            )
        raise JobConfigurationError(
            f"{self.provider_id}: unsupported source {type(source).__name__!r}"
        )

    def plan(self, resolved: ResolvedSource) -> "LogicalScanPlan":
        """Describe a bounded SQL query without exposing SQL text or credentials."""
        from tributo.data.scan_plan import (
            ParameterizedQuery,
            SourceCapability,
            SqlScan,
            SqlShardMode,
            SqlShardRequirement,
            SqlTableRead,
        )

        query_digest = resolved.identity_options.get("sql_digest")
        table = resolved.identity_options.get("table")
        if isinstance(query_digest, str):
            if self._dialect() not in {"clickhouse", "hive"}:
                raise JobConfigurationError(
                    f"{self.provider_id}: raw SQL ingestion is not supported by the "
                    "engine-neutral read contract; configure a structured 'table' "
                    "source with columns and partitioning"
                )
            query_target = ParameterizedQuery(query_digest)
            return SqlScan(
                provider_id=self.provider_id,
                connector_id=self._dialect(),
                target=query_target,
                sharding=SqlShardRequirement(mode=SqlShardMode.AUTO),
            )
        if not isinstance(table, str):
            raise JobConfigurationError(
                f"{self.provider_id}: resolved source has no SQL read target"
            )
        required: set[SourceCapability] = set()
        if resolved.identity_options.get("columns"):
            required.add(SourceCapability.PROJECTION)
        raw_partitioning = resolved.identity_options.get("partitioning")
        if isinstance(raw_partitioning, Mapping):
            partitioning = SqlPartitioning.model_validate(dict(raw_partitioning))
            if partitioning.mode == "parallel":
                assert partitioning.column is not None
                sharding = SqlShardRequirement(
                    mode=SqlShardMode.PARALLEL,
                    columns=(partitioning.column,),
                    target_partitions=partitioning.num_partitions,
                )
            elif partitioning.mode == "auto":
                sharding = SqlShardRequirement(
                    mode=SqlShardMode.AUTO,
                    target_partitions=partitioning.num_partitions,
                )
            else:
                sharding = SqlShardRequirement()
        else:
            sharding = SqlShardRequirement(
                mode=(
                    SqlShardMode.AUTO
                    if self._dialect() in {"clickhouse", "doris"}
                    else SqlShardMode.SINGLE
                )
            )
        if self._dialect() == "postgresql":
            schema = resolved.runtime_options.get("schema") or "public"
        else:
            schema = resolved.runtime_options.get("database")
        table_target = SqlTableRead(
            table=table,
            schema=str(schema or "") or None,
            projection=tuple(resolved.identity_options.get("columns", ())),
        )
        plan_options = {
            key: value
            for key, value in {
                "params_digest": resolved.identity_options.get("params_digest"),
                "partition_bound_strategy": (
                    raw_partitioning.get("bound_strategy")
                    if isinstance(raw_partitioning, Mapping)
                    else None
                ),
            }.items()
            if value is not None
        }
        return SqlScan(
            provider_id=self.provider_id,
            connector_id=self._dialect(),
            target=table_target,
            sharding=sharding,
            required_capabilities=frozenset(required),
            options=plan_options,
        )

    @classmethod
    def _dialect(cls) -> str:
        return cls.provider_id.rsplit(".", 1)[-1]


class ClickHouseProvider(_SqlProvider):
    """Logical ClickHouse source for an installed OLAP Binding."""

    provider_id = "tributo.clickhouse"
    aliases = frozenset({"clickhouse"})


class DorisProvider(_SqlProvider):
    """Logical Doris source for an installed OLAP Binding."""

    provider_id = "tributo.doris"
    aliases = frozenset({"doris"})


class PostgreSqlProvider(_SqlProvider):
    """Logical PostgreSQL source for Ray Data and Daft SQL readers."""

    provider_id = "tributo.postgresql"
    aliases = frozenset({"postgresql"})


class HiveProvider(_SqlProvider):
    """Logical HiveServer2 source for the distributed Ray Binding."""

    provider_id = "tributo.hive"
    aliases = frozenset({"hive"})


# ── Built-in registration (module import triggers registration) ──

from tributo.data.provider_registry import register_provider  # noqa: E402

for _provider_cls in (
    ParquetProvider,
    CsvProvider,
    IcebergProvider,
    LanceProvider,
    ClickHouseProvider,
    DorisProvider,
    PostgreSqlProvider,
    HiveProvider,
):
    register_provider(_provider_cls)

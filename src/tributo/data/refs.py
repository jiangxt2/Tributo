"""DatasetRef — credential-free record of what data was used.

Lives in ``tributo.data.refs`` so that both ``tributo.data`` and
``tributo.exporting`` can consume it (see ADR 001).  The Bundle Manifest v1
stores only ``DatasetRef.ref_id`` in ``ManifestSourceInfo.source_fingerprint``.

The ref_id algorithm is versioned and deterministic::

    ref_id = sha256(canonical_json({
        "version": 1,
        "provider_id": ...,
        "canonical_uri": ...,
        "result_affecting_options": ...,   # credentials must never appear here
    }))

``result_affecting_options`` covers every input that changes the data
(columns, SQL query/params digest, Iceberg snapshot, partition/filter, ...).
Providers strip credentials before building this mapping, and the framework
validates the result recursively. ``repr``/logs/errors must never expose
them either; ``compute_ref_id`` defensively rejects credential material at
any nesting level and rejects URI userinfo/sensitive query parameters.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import urlsplit

from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import pyarrow as pa

# Bump when the identity semantics change (e.g. new result-affecting field).
_REF_ID_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Credential field names — Providers must exclude these from
# result_affecting_options before computing a ref_id.  The set includes both
# the historical S3 names and the common names used by catalog/SQL clients.
CREDENTIAL_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "secret",
        "secret_access_key",
        "secret-access-key",
        "access_key_id",
        "access-key-id",
        "access_key",
        "access-key",
        "secret_key",
        "secret-key",
        "token",
        "session_token",
        "session-token",
        "api_key",
        "api-key",
        "signature",
        "client_secret",
        "client-secret",
    }
)
_NORMALIZED_CREDENTIAL_KEYS: frozenset[str] = frozenset(
    item.replace("-", "_") for item in CREDENTIAL_KEYS
)
_COMPACT_CREDENTIAL_KEYS: frozenset[str] = frozenset(
    item.replace("-", "").replace("_", "") for item in CREDENTIAL_KEYS
)

# Query parameter names that carry credentials (signed URLs, token auth).
# Shared by ``_strip_uri_credentials`` (provider_builtins) and the
# ``compute_ref_id`` canonical_uri backstop check.
SENSITIVE_QUERY_KEYS: frozenset[str] = frozenset(
    {
        "token",
        "secret",
        "signature",
        "key",
        "access_key",
        "accesskey",
        "password",
        "session_token",
        "api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        "client_secret",
        "oauth_token",
    }
)
SENSITIVE_QUERY_PREFIXES: tuple[str, ...] = (
    "x-amz-",
    "awsaccesskeyid",
    "awsaccesskey",
    "awssecretaccesskey",
)

_CREDENTIAL_KEY_MARKERS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "credential",
    "signature",
    "api_key",
    "api-key",
    "access_key",
    "access-key",
)
_COMPACT_CREDENTIAL_MARKERS: tuple[str, ...] = tuple(
    marker.replace(".", "").replace("-", "").replace("_", "")
    for marker in _CREDENTIAL_KEY_MARKERS
)
_CREDENTIAL_TEXT_RE = re.compile(
    r"(?i)(?:aws[_-]?access[_-]?key[_-]?id|"
    r"aws[_-]?secret[_-]?access[_-]?key|"
    r"password|secret|token|credential|signature|"
    r"api[_-]?key|access[_-]?key(?:[_-]?id)?|session[_-]?token)"
    r"\s*[:=]\s*[^\s,;]+"
)


def _jsonable(value: Any) -> Any:
    """Recursively convert structures into JSON-serializable types.

    ``ResolvedSource`` freezes nested options into ``MappingProxyType`` /
    tuples; ``json.dumps`` cannot serialize those directly.  Database
    parameter types (datetime/date/time/Decimal) get deterministic string
    representations so ref_id computation never raises on them — they are
    legal bound-parameter values, not credentials.
    """
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, ASCII only."""
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


@PublicAPI(stability="beta")
def digest(value: Any) -> str:
    """SHA-256 of the canonical JSON representation of *value*.

    Deterministic across processes: dict key order, nesting and value types
    are normalized by ``_canonical_json``.  The caller supplies credential-free
    *value*; this function does not silently hash credential-looking fields.
    """
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@PublicAPI(stability="beta")
def compute_ref_id(
    *,
    provider_id: str,
    canonical_uri: str,
    result_affecting_options: Mapping[str, Any],
    version: int = _REF_ID_VERSION,
) -> str:
    """Compute the versioned SHA-256 identity of a bounded data source.

    Args:
        provider_id: Full logical provider ID (e.g. ``"tributo.parquet"``).
        canonical_uri: Credential-free canonical URI of the source.
        result_affecting_options: Options that change the data (columns,
            snapshot, SQL digest, partition/filter, ...).  Must not contain
            credentials — strip them before calling.
        version: Identity schema version.  Changing the semantics of what
            counts as result-affecting bumps this.

    Returns:
        A 64-char hex SHA-256 digest.

    Raises:
        ValueError: If *result_affecting_options* contains a credential key
            at any nesting level, or *canonical_uri* still carries userinfo
            or a sensitive query parameter.  Defensive backstop — a Provider
            that leaks a credential into the identity is a bug and must fail
            loudly.
    """
    leaked = sorted(_credential_paths(result_affecting_options))
    if leaked:
        raise ValueError(
            f"result_affecting_options contains credential key/field(s) {leaked}; "
            "strip credentials before computing a ref_id"
        )
    _ensure_credential_free_uri(canonical_uri)
    payload = {
        "version": version,
        "provider_id": provider_id,
        "canonical_uri": canonical_uri,
        "result_affecting_options": dict(result_affecting_options),
    }
    return digest(payload)


def _is_credential_key(key: str) -> bool:
    """Return whether a mapping key identifies credential material."""
    lowered = key.lower()
    compact = lowered.replace(".", "").replace("-", "").replace("_", "")
    normalized = lowered.replace("-", "_")
    if (
        lowered in CREDENTIAL_KEYS
        or normalized in _NORMALIZED_CREDENTIAL_KEYS
        or compact in _COMPACT_CREDENTIAL_KEYS
    ):
        return True
    return any(
        marker in lowered or compact_marker in compact
        for marker, compact_marker in zip(
            _CREDENTIAL_KEY_MARKERS, _COMPACT_CREDENTIAL_MARKERS
        )
    )


def _credential_paths(
    value: Any,
    prefix: str = "",
    *,
    text_exempt_keys: frozenset[str] = frozenset(),
    _skip_text_scan: bool = False,
) -> list[str]:
    """Find credential-looking keys or URI values recursively.

    ``text_exempt_keys`` identifies fields whose free-form strings are
    execution payloads rather than credential configuration. Credential keys
    nested inside those fields and URI credential checks remain enforced.

    The returned paths contain field names/positions only, never the values
    that triggered the check.
    """
    leaked: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if _is_credential_key(key_text):
                leaked.append(path)
            else:
                leaked.extend(
                    _credential_paths(
                        nested,
                        path,
                        text_exempt_keys=text_exempt_keys,
                        _skip_text_scan=(
                            _skip_text_scan or key_text.lower() in text_exempt_keys
                        ),
                    )
                )
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            leaked.extend(
                _credential_paths(
                    nested,
                    f"{prefix}[{index}]",
                    text_exempt_keys=text_exempt_keys,
                    _skip_text_scan=_skip_text_scan,
                )
            )
    elif isinstance(value, str) and (
        _uri_has_credentials(value)
        or (not _skip_text_scan and _CREDENTIAL_TEXT_RE.search(value))
    ):
        leaked.append(prefix or "<value>")
    return leaked


def _ensure_credential_free_uri(uri: str, field_name: str = "canonical_uri") -> None:
    """Reject userinfo and credential query parameters in a URI-like field."""
    if _uri_has_credentials(uri):
        raise ValueError(
            f"{field_name} must not contain credentials (userinfo or a "
            "sensitive query parameter); strip them before computing a ref_id"
        )


def _ensure_credential_free_text(value: str, field_name: str) -> None:
    """Reject credential-looking key/value material in free-form fields."""
    if _CREDENTIAL_TEXT_RE.search(value):
        raise ValueError(
            f"{field_name} must not contain credential-like key/value material"
        )


def _uri_has_credentials(uri: str) -> bool:
    """True if a URI still carries credential material.

    Checks userinfo (``s3://user:pass@host/p``) and sensitive query
    parameters (``?token=...&X-Amz-Signature=...``).  Non-credential query
    keys (e.g. ``versionId``) are fine — they can be result-affecting.
    """
    parts = urlsplit(uri)
    if parts.username is not None:
        return True
    for param in parts.query.split("&"):
        if not param:
            continue
        key = param.split("=", 1)[0].lower()
        if key in SENSITIVE_QUERY_KEYS or key.startswith(SENSITIVE_QUERY_PREFIXES):
            return True
    return False


@PublicAPI(stability="beta")
def schema_fingerprint(schema: "pa.Schema") -> str:
    """SHA-256 of the canonical Arrow schema (field structure + metadata).

    Used to detect schema drift between runs without embedding the full
    schema in a manifest.  Field order matters (reordering columns changes
    the fingerprint); metadata is normalized by key order.
    """
    import pyarrow as pa

    if not isinstance(schema, pa.Schema):
        raise TypeError(f"Expected pa.Schema, got {type(schema).__name__}")
    payload = {
        "fields": [
            {
                "name": f.name,
                "type": str(f.type),
                "nullable": f.nullable,
            }
            for f in schema
        ],
        "metadata": {
            _bytes_to_str(k): _bytes_to_str(v)
            for k, v in (schema.metadata or {}).items()
        },
    }
    return digest(payload)


def _bytes_to_str(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class DatasetRef:
    """Credential-free record of what data was used.

    Attributes:
        ref_id: SHA-256 of the versioned identity payload.
        provider_id: Full logical provider ID.
        uri: Canonical URI (``s3://``, ``file://``, local path, ...).
        schema_fingerprint: SHA-256 of the canonical Arrow schema.
        row_count: Number of rows, ``None`` if not computed.
        provenance: Free-form version/timestamp string (not parsed).
    """

    ref_id: str
    provider_id: str
    uri: str
    schema_fingerprint: str
    row_count: int | None = None
    provenance: str = ""

    def __post_init__(self) -> None:
        """Keep the persisted URI credential-free at construction time."""
        if not _SHA256_RE.fullmatch(self.ref_id):
            raise ValueError("ref_id must be a 64-character lowercase SHA-256 digest")
        if not _SHA256_RE.fullmatch(self.schema_fingerprint):
            raise ValueError(
                "schema_fingerprint must be a 64-character lowercase SHA-256 digest"
            )
        _ensure_credential_free_text(self.provider_id, "provider_id")
        _ensure_credential_free_uri(self.uri)
        if self.provenance:
            _ensure_credential_free_uri(self.provenance, "provenance")
            _ensure_credential_free_text(self.provenance, "provenance")

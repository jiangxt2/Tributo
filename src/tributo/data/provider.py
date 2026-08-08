"""Logical DataSourceProvider contract for bounded ingestion.

The canonical runtime boundary (ADR 001)::

    ProviderRegistry.resolve(source) -> DataSourceProvider
    provider.normalize(source) -> ResolvedSource
    provider.plan(resolved) -> LogicalScanPlan
    IngestionGateway.open(request) -> IngestionOpenResult
    EngineBinding -> Ray Data / Daft / installed connector

``ResolvedSource`` separates *identity* options (everything that changes the
data — columns, snapshot, SQL digest, partition/filter) from *runtime*
options (connection credentials etc.). Credentials never appear in ``repr``,
logs, errors, ``DatasetRef`` or benchmark output. ``open()`` and
``DatasetHandle`` remain conversion-only Ray compatibility surfaces; built-in
providers no longer own an independent reader implementation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

from pydantic import BaseModel

from tributo.data.refs import (
    _credential_paths,
    _ensure_credential_free_uri,
    compute_ref_id,
)
from tributo.data.source_config import CanonicalSourceInput
from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data

    from tributo.data.scan_plan import LogicalScanPlan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deeply frozen mappings for ResolvedSource
# ---------------------------------------------------------------------------


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze mappings and lists into immutable structures."""
    if isinstance(value, BaseModel):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.model_dump().items()}
        )
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


# ---------------------------------------------------------------------------
# ResolvedSource — normalized, credential-safe source identity
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class ResolvedSource:
    """Canonical, credential-safe description of a bounded data source.

    Produced by ``DataSourceProvider.normalize()`` from either the
    ``type/path/dialect`` shape or the ``provider/uri`` shape.  Both option
    mappings are deeply frozen — the identity is immutable by construction.

    Attributes:
        provider_id: Full logical provider ID (e.g. ``"tributo.parquet"``).
        canonical_uri: Credential-free canonical URI of the source.
        identity_options: Result-affecting options (columns, snapshot,
            SQL digest, partition/filter).  Fed into the ``DatasetRef``
            ref_id algorithm.
        runtime_options: Connection/credential options used only at read
            time.  Never serialized; only their keys appear in ``repr``.
    """

    provider_id: str
    canonical_uri: str
    identity_options: Mapping[str, Any] = field(default_factory=dict)
    runtime_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        leaked = _credential_paths(self.identity_options)
        if leaked:
            raise ValueError(
                "identity_options must not contain credential field(s): "
                f"{sorted(leaked)}"
            )
        _ensure_credential_free_uri(self.canonical_uri)
        object.__setattr__(
            self, "identity_options", _deep_freeze(self.identity_options)
        )
        object.__setattr__(self, "runtime_options", _deep_freeze(self.runtime_options))

    def __repr__(self) -> str:
        # Runtime option *values* may carry credentials — show keys only.
        return (
            f"ResolvedSource(provider_id={self.provider_id!r}, "
            f"canonical_uri={self.canonical_uri!r}, "
            f"identity_options={dict(self.identity_options)!r}, "
            f"runtime_options={sorted(self.runtime_options)!r})"
        )

    def ref_id(self) -> str:
        """Versioned SHA-256 identity of this source (credential-free).

        Uses only the identity options (columns, snapshot, digests, ...);
        credentials in ``runtime_options`` never participate.
        """
        return compute_ref_id(
            provider_id=self.provider_id,
            canonical_uri=self.canonical_uri,
            result_affecting_options=dict(self.identity_options),
        )


# ---------------------------------------------------------------------------
# DataSourceProvider — stable bounded-read contract
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class DataSourceProvider(ABC):
    """Logical data-source provider (stable contract).

    Subclasses declare a unique ``provider_id`` and optional default
    ``aliases``. ``projection_option_name`` identifies the provider option
    used by the consumer-neutral projection helper. ``relative_uri_is_path``
    declares whether a scheme-less ``ProviderSourceConfig.uri`` is a local
    path resolved against the configured project root. Conflict checks,
    metadata validation, and alias resolution live in the ProviderRegistry,
    not here.
    """

    provider_id: ClassVar[str] = ""
    aliases: ClassVar[frozenset[str]] = frozenset()
    projection_option_name: ClassVar[str | None] = None
    relative_uri_is_path: ClassVar[bool] = False

    @abstractmethod
    def normalize(self, source: CanonicalSourceInput) -> ResolvedSource:
        """Normalize a canonical source into a credential-safe ResolvedSource.

        Providers accept both input shapes (builtin ``type/path/dialect`` and
        target ``provider/uri``) and validate that the resolved provider ID
        matches theirs.
        """

    def open(self, resolved: ResolvedSource) -> DatasetHandle:
        """Open the legacy Ray-only compatibility path over *resolved*.

        Plan-only providers need not implement this method.  The default
        fails explicitly so the ingestion path can never fall back to an
        obsolete in-process Reader.
        """
        raise JobConfigurationError(
            f"Provider {self.provider_id!r} has no legacy Ray-only reader; "
            "use IngestionGateway with an installed engine Binding"
        )

    def plan(self, resolved: ResolvedSource) -> LogicalScanPlan:
        """Build an engine-neutral scan plan for canonical ingestion.

        Providers opt in by overriding this method. The default fails
        explicitly and never falls back to a legacy Reader.
        """
        raise JobConfigurationError(
            f"Provider {self.provider_id!r} does not support planned ingestion"
        )


# ---------------------------------------------------------------------------
# DatasetHandle — bounded read lifecycle
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class DatasetHandle(ABC):
    """A bounded finite read with an idempotent close lifecycle.

    ``to_ray_dataset()`` may be called at most once; afterwards (or after
    ``close()``) provider resources are released and further reads raise
    ``RuntimeError``.  ``close()`` is idempotent and safe to call in
    ``finally`` / context-manager exit even after a failed read.
    """

    def __init__(self) -> None:
        self._released = False
        # Keep only the exception type: release errors must remain observable
        # without retaining a message that may contain a DSN or credential.
        self._release_error: str | None = None

    @abstractmethod
    def _read(self) -> "ray.data.Dataset":
        """Execute the bounded read (implementation detail)."""

    @abstractmethod
    def _release(self) -> None:
        """Release connections/temporary resources (implementation detail)."""

    def to_ray_dataset(self) -> "ray.data.Dataset":
        if self._released:
            raise RuntimeError(
                f"{type(self).__name__} resources already released; cannot read again"
            )
        try:
            dataset = self._read()
        finally:
            # Release even when the read fails — a failed read must not
            # leak provider resources or leave the handle reusable.
            self._release_safely()
        return dataset

    def close(self) -> None:
        if self._released:
            return
        self._release_safely()

    def _release_safely(self) -> None:
        """Release resources and mark the handle closed, unconditionally.

        A failing ``_release()`` is logged (never propagated — it must not
        mask the read's original exception nor leave the handle reusable)
        and recorded on ``_release_error`` for observability; the handle is
        closed either way and release is not retried.
        """
        try:
            self._release()
        except Exception as exc:
            self._release_error = type(exc).__name__
            # Exception messages can contain connection details.  Retain only
            # the exception type and log only that type so the credential-free
            # observability boundary remains intact.
            logger.warning(
                "%s._release() failed with %s; handle marked closed",
                type(self).__name__,
                type(exc).__name__,
            )
        finally:
            self._released = True

    def __enter__(self) -> "DatasetHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

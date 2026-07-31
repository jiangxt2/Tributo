"""Bundle repository protocol — storage abstraction for published bundles.

``BundleRepository`` is the domain-level port: it defines the contract
that every storage backend must satisfy without exposing implementation
details (lease, ETag, atomic rename, multipart upload).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from tributo.exporting.models import FailureInfo, LogicalArtifact
from tributo.util.annotations import DeveloperAPI, PublicAPI

# ── Data models ────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BundleRef(BaseModel):
    """Immutable reference to a committed bundle.

    This is the stable handle that callers keep — it never changes after
    commit and can be passed to ``BundleRepository.get()`` or
    ``load_bundle()`` without needing to know the storage backend.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_uri: str
    bundle_id: str
    manifest_sha256: str


@PublicAPI(stability="beta")
class UncommittedBundle(BaseModel):
    """A bundle that has been assembled but not yet committed to storage.

    Carries the manifest, the staging directory with artifact files,
    and execution metadata needed for provenance recording.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    bundle_id: str
    manifest: dict[str, Any]  # manifest as a JSON-serialisable dict
    manifest_bytes: bytes
    manifest_sha256: str
    artifacts: tuple[LogicalArtifact, ...] = ()
    staging_root: Path
    roles: dict[str, str] = Field(default_factory=dict)


@PublicAPI(stability="beta")
class CommitResult(BaseModel):
    """Result of a successful (or idempotent) bundle commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_uri: str
    manifest_uri: str
    manifest_sha256: str
    commit_status: str = Field(pattern=r"^(committed|idempotent)$")
    alias_uri: str | None = None
    alias_status: str = Field(
        default="not_requested",
        pattern=r"^(not_requested|updated|unchanged|failed)$",
    )
    alias_failure: FailureInfo | None = None


@PublicAPI(stability="beta")
class AliasUpdateResult(BaseModel):
    """Result of an alias update operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alias: str
    status: str = Field(pattern=r"^(updated|unchanged|failed)$")
    revision: str | None = None
    failure: FailureInfo | None = None


# ── Protocol ────────────────────────────────────────────────────────────────────


@runtime_checkable
@PublicAPI(stability="beta")
class BundleRepository(Protocol):
    """Storage backend contract for bundle persistence.

    Implementations guarantee:

    - **Atomicity**: ``commit()`` either succeeds (Bundle is fully readable)
      or fails (no partial content visible).
    - **Idempotency**: committing the same ``UncommittedBundle`` twice
      returns ``commit_status="idempotent"`` without data duplication.
    - **Alias consistency**: ``update_alias()`` uses CAS semantics where
      the backend supports it.

    The protocol hides implementation details: lease acquisition, ETag
    matching, multipart upload, and atomic rename are all backend-internal.
    """

    def commit(self, bundle: UncommittedBundle) -> CommitResult:
        """Atomically persist *bundle* and return a durable reference.

        Implementations must:
        - Upload artifact files first, then write the manifest last.
        - Detect pre-existing identical content and return idempotent.
        - Detect pre-existing different content and raise an error.
        """
        ...

    def get(self, ref: BundleRef) -> dict[str, Any]:
        """Read the manifest for *ref* as a JSON-serialisable dict."""
        ...

    def update_alias(
        self,
        alias: str,
        new_ref: BundleRef,
        expected_revision: str | None = None,
    ) -> AliasUpdateResult:
        """Create or update an alias to point at *new_ref*.

        When *expected_revision* is set, the update only succeeds if the
        current alias revision matches (CAS).
        """
        ...


# ── Factory ─────────────────────────────────────────────────────────────────────


@DeveloperAPI
def resolve_repository(
    uri: str,
    storage_profile: str | None = None,
) -> BundleRepository:
    """Return a ``BundleRepository`` for *uri*.

    Dispatches by URI scheme:
    - ``s3://`` → ``S3BundleRepository``
    - everything else → ``LocalBundleRepository``
    """
    if uri.startswith("s3://"):
        raise NotImplementedError(
            "S3BundleRepository is not yet implemented. "
            "Use a local path for now."
        )

    from tributo.integrations.storage.local import LocalBundleRepository

    return LocalBundleRepository()

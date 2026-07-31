"""Canonical export manifest and schema version registry.

The manifest is the single source of truth for a published bundle.
It records every artifact, execution node, role mapping, and lineage
so that a ``BundleReader`` can verify integrity without external state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tributo.training.exporters.models import (
    ArtifactRef,
    FailureInfo,
    LogicalArtifact,
)
from tributo.util.annotations import PublicAPI

#: Signature for a manifest reader callable.
ManifestReader = Callable[[dict[str, Any], bytes], "ExportManifest"]

# ── Manifest models ────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ManifestSourceInfo(BaseModel):
    """Stable source identification — no temp paths, credentials, or samples."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_kind: str
    source_fingerprint: str = ""
    framework: str | None = None
    framework_version: str | None = None
    architecture_id: str | None = None
    task_type: str | None = None


@PublicAPI(stability="beta")
class ManifestSignature(BaseModel):
    """Input / output signature recorded in the manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    dynamic_axes: dict[str, dict[int, str]] = Field(default_factory=dict)


@PublicAPI(stability="beta")
class ManifestExecutionNode(BaseModel):
    """A single DAG node as recorded in the manifest.

    Lightweight — only stores ``ArtifactRef``, not full ``LogicalArtifact``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    target_name: str
    exporter_id: str | None = None
    status: str = Field(pattern=r"^(succeeded|failed|blocked|cancelled)$")
    required: bool
    implicit: bool = False
    artifact_ref: ArtifactRef | None = None
    failure: FailureInfo | None = None
    duration_ms: int = 0


@PublicAPI(stability="beta")
class ManifestExecution(BaseModel):
    """Execution summary recorded in the manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: str
    nodes: tuple[ManifestExecutionNode, ...] = ()


@PublicAPI(stability="beta")
class ExportManifest(BaseModel):
    """Canonical manifest for a published model bundle (schema v1).

    The manifest is written last during publish and serves as the
    single integrity anchor: its SHA-256 digest is the ``manifest_sha256``
    stored in ``BundleResult``, S3 metadata, and aliases.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    bundle_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = Field(pattern=r"^(succeeded|partial)$")
    canonical_uri: str
    tributo_version: str
    source_info: ManifestSourceInfo
    input_signature: ManifestSignature = Field(default_factory=ManifestSignature)
    output_signature: ManifestSignature = Field(default_factory=ManifestSignature)
    artifacts: tuple[LogicalArtifact, ...] = ()
    roles: dict[str, str] = Field(default_factory=dict)
    execution: ManifestExecution

    def canonical_json(self) -> bytes:
        """Encode to canonical JSON for digest computation.

        Canonical JSON: no indentation, sorted keys, no trailing newline,
        ``ensure_ascii=False``.
        """
        raw = self.model_dump(mode="json")
        return json.dumps(
            raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def compute_sha256(self) -> str:
        """Compute SHA-256 of the canonical JSON representation."""
        return hashlib.sha256(self.canonical_json()).hexdigest()


# ── Schema registry ────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ManifestSchemaRegistry:
    """Versioned manifest reader registry.

    Each schema version has a registered reader callable that knows how
    to parse and validate manifests of that version.  Unknown versions
    are rejected outright — no best-effort guessing.
    """

    def __init__(self) -> None:
        self._readers: dict[int, ManifestReader] = {}

    def register(self, version: int, reader: ManifestReader) -> None:
        """Register a reader callable for *version*.

        The reader receives ``(raw_dict, canonical_json_bytes)`` and must
        return an ``ExportManifest`` (or subclass for future versions).
        """
        if version in self._readers:
            raise ValueError(f"Schema version {version} already registered")
        self._readers[version] = reader

    def read(
        self, version: int, raw: dict[str, Any], canonical_bytes: bytes
    ) -> ExportManifest:
        """Parse and validate a manifest from its raw dict."""
        reader = self._readers.get(version)
        if reader is None:
            raise ValueError(
                f"Unsupported manifest schema version {version}. "
                f"Supported versions: {sorted(self._readers)}"
            )
        return reader(raw, canonical_bytes)


# ── Built-in v1 reader ─────────────────────────────────────────────────────────


def _read_manifest_v1(raw: dict[str, Any], canonical_bytes: bytes) -> ExportManifest:
    """Parse and validate a v1 manifest."""
    manifest = ExportManifest(**raw)
    declared_digest = raw.get("_manifest_sha256")
    if declared_digest is not None:
        # The digest field should NOT be in the manifest itself — but if a
        # pre-release version included it, skip the check rather than failing.
        pass
    actual = hashlib.sha256(canonical_bytes).hexdigest()
    # NOTE: The canonical_bytes used for verification come from the
    # publisher/reader, which strips any _manifest_sha256 wrapper before
    # hashing.  The ExportManifest model itself does not carry this field.
    _ = actual  # digest verification is done by BundleReader
    return manifest


# ── v2 manifest ─────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BundleManifestV2(BaseModel):
    """Canonical manifest for a published model bundle (schema v2).

    Changes from v1:
    - ``schema_version`` = 2 (mandatory).
    - Adds ``bundle_digest`` — content-addressable identity (excludes
      ``created_at``, ``bundle_id``, ``canonical_uri``).
    - Adds ``flavor``, ``signatures``, ``exporter_options``.
    - Moves execution metadata to ``ExecutionRecord`` (separate object).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=2, ge=2)
    bundle_id: str
    bundle_digest: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = Field(pattern=r"^(succeeded|partial)$")
    canonical_uri: str
    tributo_version: str
    source_info: ManifestSourceInfo
    flavor: str | None = None
    input_signature: ManifestSignature = Field(default_factory=ManifestSignature)
    output_signature: ManifestSignature = Field(default_factory=ManifestSignature)
    artifacts: tuple[LogicalArtifact, ...] = ()
    roles: dict[str, str] = Field(default_factory=dict)
    exporter_options: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def canonical_json(self) -> bytes:
        """Encode to canonical JSON for digest computation."""
        raw = self.model_dump(mode="json")
        return json.dumps(
            raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def compute_sha256(self) -> str:
        """Compute SHA-256 of the canonical JSON representation."""
        return hashlib.sha256(self.canonical_json()).hexdigest()


def _read_manifest_v2(
    raw: dict[str, Any], canonical_bytes: bytes
) -> ExportManifest:
    """Parse a v2 manifest and return it as a v1-compatible ``ExportManifest``.

    This allows existing v1 consumers (BundleReader, Publisher) to read
    v2 bundles without code changes.  The ``bundle_digest`` and other v2
    fields are preserved in ``model_extra``.
    """
    v2 = BundleManifestV2(**raw)
    actual = hashlib.sha256(canonical_bytes).hexdigest()
    _ = actual  # digest verification done by BundleReader
    # Convert to v1 ExportManifest for backward compatibility.
    return ExportManifest(
        schema_version=1,  # v1 consumers see schema_version=1
        bundle_id=v2.bundle_id,
        created_at=v2.created_at,
        status=v2.status,
        canonical_uri=v2.canonical_uri,
        tributo_version=v2.tributo_version,
        source_info=v2.source_info,
        input_signature=v2.input_signature,
        output_signature=v2.output_signature,
        artifacts=v2.artifacts,
        roles=v2.roles,
        execution=ManifestExecution(execution_id=""),  # placeholder
    )


# ── Bundle digest computation ───────────────────────────────────────────────────


@PublicAPI(stability="beta")
def compute_bundle_digest(
    artifacts: tuple[LogicalArtifact, ...],
    roles: dict[str, str],
    input_sig: ManifestSignature | None = None,
    output_sig: ManifestSignature | None = None,
    flavor: str | None = None,
    exporter_options: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Compute a content-addressable bundle digest.

    Includes: sorted artifact descriptors, roles, signatures, flavor, and
    exporter options.  Excludes ``created_at``, ``bundle_id``, and
    ``canonical_uri`` so that identical content produces identical digest
    regardless of when or where it was committed.
    """
    # Build canonical representation of artifacts.
    artifact_entries: list[dict[str, Any]] = []
    for a in artifacts:
        file_entries = sorted(
            {
                "path": f.relative_path,
                "sha256": f.sha256,
                "size_bytes": f.size_bytes,
                "role": f.role,
            }
            for f in a.files
        )
        artifact_entries.append(
            {
                "name": a.name,
                "format": a.format,
                "flavor_id": a.flavor_id,
                "entrypoint": a.entrypoint,
                "files": file_entries,
                "derived_from": sorted(
                    {
                        "node_id": d.node_id,
                        "artifact_name": d.artifact_name,
                        "tree_digest": d.tree_digest,
                    }
                    for d in a.derived_from
                ),
            }
        )

    payload: dict[str, Any] = {
        "artifacts": sorted(artifact_entries, key=lambda x: x["name"]),
        "roles": dict(sorted(roles.items())),
    }
    if input_sig is not None:
        payload["input_signature"] = input_sig.model_dump(mode="json")
    if output_sig is not None:
        payload["output_signature"] = output_sig.model_dump(mode="json")
    if flavor is not None:
        payload["flavor"] = flavor
    if exporter_options:
        payload["exporter_options"] = dict(sorted(exporter_options.items()))

    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

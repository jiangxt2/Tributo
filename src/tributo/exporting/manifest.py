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
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Strict,
    field_validator,
    model_validator,
)

from tributo.exporting.models import (
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
class SignatureField(BaseModel):
    """A single input/output field with its data type and shape.

    ``dtype`` is a framework-neutral canonical string (e.g. ``"float32"``,
    ``"int64"``) — never a numpy/scipy object, so the manifest stays
    JSON-serialisable.  ``shape`` entries are positive integers for fixed
    dimensions or non-empty strings for dynamic axes (e.g. ``"batch"``).
    Strict types: floats and booleans are rejected, never coerced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    dtype: str = Field(..., min_length=1)
    shape: tuple[Annotated[int, Strict()] | Annotated[str, Strict()], ...] = ()

    @field_validator("name", "dtype")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if v != v.strip():
            raise ValueError(f"must not contain leading/trailing whitespace, got {v!r}")
        return v

    @field_validator("shape")
    @classmethod
    def _check_shape(cls, v: tuple[Any, ...]) -> tuple[Any, ...]:
        for dim in v:
            if isinstance(dim, int) and dim <= 0:
                raise ValueError(f"shape dimensions must be positive, got {dim!r}")
            if isinstance(dim, str) and not dim.strip():
                raise ValueError("dynamic shape axes must be non-empty strings")
        return v


@PublicAPI(stability="beta")
class ManifestSignature(BaseModel):
    """Input / output signature recorded in the manifest.

    Resolution order: ``input_fields``/``output_fields`` take precedence;
    the v1 ``input_names``/``output_names`` fields serve as fallback for
    manifests written before fields existed, and ``dynamic_axes`` remains
    the legacy axis representation.  When both representations are present
    but disagree on the set of names, the manifest is rejected (fail-fast)
    rather than silently preferring one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    dynamic_axes: dict[str, dict[int, str]] = Field(default_factory=dict)
    input_fields: tuple[SignatureField, ...] = ()
    output_fields: tuple[SignatureField, ...] = ()

    @model_validator(mode="after")
    def _check_representation_consistency(self) -> ManifestSignature:
        for legacy_attr, field_attr in (
            ("input_names", "input_fields"),
            ("output_names", "output_fields"),
        ):
            legacy_names = getattr(self, legacy_attr)
            fields = getattr(self, field_attr)
            if fields:
                self._check_unique_and_ordered(legacy_attr, legacy_names, fields)
        # dynamic_axes is a single shared map covering both sides — validate
        # it once against input_fields + output_fields together.
        all_fields = self.input_fields + self.output_fields
        if all_fields:
            self._check_dynamic_axes_consistency(all_fields)
        return self

    @staticmethod
    def _check_unique_and_ordered(
        legacy_attr: str,
        legacy_names: tuple[str, ...],
        fields: tuple[SignatureField, ...],
    ) -> None:
        field_names = tuple(f.name for f in fields)
        if len(set(field_names)) != len(field_names):
            raise ValueError(
                f"{legacy_attr.replace('names', 'fields')} contains duplicate "
                f"field names: {field_names!r}"
            )
        if legacy_names and legacy_names != field_names:
            raise ValueError(
                f"{legacy_attr} and {legacy_attr.replace('names', 'fields')} "
                f"disagree on field order: {legacy_names!r} vs {field_names!r}"
            )

    def _check_dynamic_axes_consistency(
        self, fields: tuple[SignatureField, ...]
    ) -> None:
        declared = {f.name for f in fields}
        for name in self.dynamic_axes:
            if name not in declared:
                raise ValueError(
                    f"dynamic_axes references undeclared field {name!r}; "
                    f"declared: {sorted(declared)}"
                )
        for f in fields:
            if f.name not in self.dynamic_axes:
                continue
            axes = self.dynamic_axes[f.name]
            for idx, dim in enumerate(f.shape):
                if isinstance(dim, str):
                    if idx not in axes or axes[idx] != dim:
                        raise ValueError(
                            f"dynamic axis {idx} of {f.name!r}: shape declares "
                            f"{dim!r} but dynamic_axes declares {axes!r}"
                        )
                elif idx in axes:
                    raise ValueError(
                        f"dimension {idx} of {f.name!r} is fixed in shape but "
                        f"declared dynamic in dynamic_axes: {axes!r}"
                    )
            for idx in axes:
                if idx < 0 or idx >= len(f.shape):
                    raise ValueError(
                        f"dynamic axis {idx} of {f.name!r} is out of range for "
                        f"shape {f.shape!r}"
                    )


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

    Artifacts carry ``artifact_kind`` so that consumers can distinguish
    model files from reports, diagnostics, and graph snapshots without
    inspecting file contents.

    The manifest is written last during publish and serves as the
    single integrity anchor: its SHA-256 digest is the ``manifest_sha256``
    stored in ``BundleResult``, S3 metadata, and aliases.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    bundle_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = Field(
        pattern=r"^(succeeded|partial)$",
        description="Published state. 'failed' is an execution-time state — "
        "a failed bundle is never published (BundleExportService raises "
        "BundleExportError before publish). Ordinary optional-node failures "
        "produce 'partial'; session-fatal integrity failures fail the whole "
        "execution and are never published either.",
    )
    canonical_uri: str
    tributo_version: str
    source_info: ManifestSourceInfo
    input_signature: ManifestSignature = Field(default_factory=ManifestSignature)
    output_signature: ManifestSignature = Field(default_factory=ManifestSignature)
    artifacts: tuple[LogicalArtifact, ...] = ()
    roles: dict[str, str] = Field(default_factory=dict)
    execution: ManifestExecution

    @field_validator("created_at")
    @classmethod
    def _require_timezone_aware_created_at(cls, value: datetime) -> datetime:
        """Reject ambiguous timestamps before a bundle can be committed."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone offset")
        return value

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
    """Parse and validate a v1 manifest.

    v1 artifacts lack ``artifact_kind`` — the reader defaults them to
    ``"model"`` for backward compatibility.  It also maps the former
    XGBoost format-plus-variant representation to its canonical format while
    preserving the shared native runtime flavor.  Both adaptations affect only
    the parsed object; callers still perform integrity checks against
    ``canonical_bytes`` unchanged.
    """
    del canonical_bytes

    def normalise_artifact(artifact: Any) -> Any:
        if not isinstance(artifact, dict):
            return artifact
        normalised = {
            **artifact,
            "artifact_kind": artifact.get("artifact_kind", "model"),
        }
        if (
            artifact.get("format") == "xgboost"
            and artifact.get("flavor_id") == "xgboost-native-v1"
        ):
            legacy_variant = artifact.get("variant") or "ubj"
            replacements = {"ubj": "ubj", "json": "xgboost-json"}
            replacement = replacements.get(legacy_variant)
            if replacement is not None:
                normalised["format"] = replacement
        return normalised

    # Normalise only the parsed in-memory view; never re-serialize it for the
    # manifest digest or mutate the persisted v1 payload.
    raw = dict(raw)
    artifacts = raw.get("artifacts", ())
    if artifacts:
        raw["artifacts"] = tuple(normalise_artifact(a) for a in artifacts)
    manifest = ExportManifest(**raw)
    return manifest


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
            (
                {
                    "path": f.relative_path,
                    "sha256": f.sha256,
                    "size_bytes": f.size_bytes,
                    "role": f.role,
                }
                for f in a.files
            ),
            key=lambda d: str(d["path"]),
        )
        derived_entries = sorted(
            (
                {
                    "node_id": d.node_id,
                    "artifact_name": d.artifact_name,
                    "tree_digest": d.tree_digest,
                }
                for d in a.derived_from
            ),
            key=lambda d: (str(d["node_id"]), str(d["artifact_name"])),
        )
        artifact_entries.append(
            {
                "name": a.name,
                "format": a.format,
                "flavor_id": a.flavor_id,
                "entrypoint": a.entrypoint,
                "files": file_entries,
                "derived_from": derived_entries,
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


# ── Schema registry singleton ──────────────────────────────────────────────────

_SCHEMA_REGISTRY = ManifestSchemaRegistry()
_SCHEMA_REGISTRY.register(1, _read_manifest_v1)


def get_schema_registry() -> ManifestSchemaRegistry:
    """Return the shared manifest schema registry."""
    return _SCHEMA_REGISTRY

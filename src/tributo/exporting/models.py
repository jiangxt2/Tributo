"""Bundle export data models — configuration, artifacts, and execution results.

All public config models use Pydantic v2 with ``ConfigDict(extra="forbid")``.
Runtime immutable values use frozen Pydantic models.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Strict,
    field_validator,
    model_validator,
)

from tributo.exporting.formats import validate_format_id
from tributo.util.annotations import DeveloperAPI, PublicAPI

if TYPE_CHECKING:
    from tributo.exporting.manifest import ManifestSignature

# ── Name validation ──────────────────────────────────────────────────────────

_TARGET_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
_RESERVED_NAMES: frozenset[str] = frozenset({".leases", "aliases", "trials"})


def _validate_safe_name(v: str, label: str) -> str:
    """Reject names that don't match the allowlist or shadow reserved prefixes."""
    if not _TARGET_NAME_RE.match(v):
        raise ValueError(
            f"{label} {v!r} does not match pattern {_TARGET_NAME_RE.pattern}"
        )
    if v.lower() in _RESERVED_NAMES:
        raise ValueError(f"{label} {v!r} is a reserved name")
    return v


def _validate_posix_relative(v: str, label: str) -> str:
    """Reject absolute paths, backslashes, NUL, and traversal components."""
    if not v or "\\" in v or "\x00" in v:
        raise ValueError(f"{label} {v!r} must be a non-empty POSIX relative path")
    p = PurePosixPath(v)
    if p.is_absolute():
        raise ValueError(f"{label} {v!r} must be relative")
    parts = p.parts
    if ".." in parts or "." in parts:
        raise ValueError(f"{label} {v!r} must not contain '.' or '..'")
    return v


# ── Configuration models ─────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class HookStatus(StrEnum):
    """Terminal and future asynchronous hook delivery states."""

    ACCEPTED = "accepted"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    RETRYABLE_FAILED = "retryable_failed"
    TERMINAL_FAILED = "terminal_failed"


@PublicAPI(stability="beta")
class HookBinding(BaseModel):
    """Explicit post-publish hook configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hook_id: str = Field(..., min_length=1, max_length=128)
    required: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


@PublicAPI(stability="beta")
class HookReceipt(BaseModel):
    """Immutable result of a single post-publish hook delivery."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(..., min_length=1)
    hook_id: str = Field(..., min_length=1)
    delivery_id: str = Field(..., min_length=1)
    idempotency_key: str = Field(..., min_length=1)
    status: HookStatus
    started_at: datetime
    completed_at: datetime
    error_code: str | None = None
    error_summary: str | None = Field(default=None, max_length=4096)
    external_references: dict[str, str] = Field(default_factory=dict)


@PublicAPI(stability="beta")
class AliasConfig(BaseModel):
    """Optional stable alias for a published bundle."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=128)
    policy: str = Field(default="newer", pattern=r"^(newer|compare_and_swap)$")
    expected_manifest_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="Required for compare_and_swap; empty means create-only",
    )

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if "/" in v:
            raise ValueError(f"alias name must not contain '/': {v!r}")
        return _validate_safe_name(v, "alias name")


@PublicAPI(stability="beta")
class ExportTarget(BaseModel):
    """A single export target within a bundle output configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=128)
    format: str = Field(..., min_length=1)
    exporter_id: str | None = Field(default=None, min_length=1)
    required: bool = True
    depends_on: tuple[str, ...] = ()
    options: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalise_legacy_xgboost_target(cls, value: Any) -> Any:
        """Map the former format-plus-option selector to one format id.

        The compatibility rule runs before Planner candidate selection, so
        every downstream component sees the same canonical contract.  An
        explicitly selected third-party exporter is left untouched because
        Tributo cannot reinterpret another plugin's private compatibility
        surface.
        """
        if not isinstance(value, dict) or value.get("format") != "xgboost":
            return value
        exporter_id = value.get("exporter_id")
        if exporter_id not in (None, "xgboost-native-v1"):
            return value

        options = dict(value.get("options") or {})
        legacy_format = options.pop("fmt", "ubj")
        replacements = {
            "ubj": ("ubj", "xgboost-ubj-v1"),
            "json": ("xgboost-json", "xgboost-json-v1"),
        }
        if not isinstance(legacy_format, str) or legacy_format not in replacements:
            raise ValueError(
                "legacy XGBoost target option 'fmt' must be 'ubj' or 'json'"
            )

        canonical_format, canonical_exporter = replacements[legacy_format]
        normalised = dict(value)
        normalised["format"] = canonical_format
        normalised["exporter_id"] = canonical_exporter
        normalised["options"] = options
        warnings.warn(
            "ExportTarget(format='xgboost', options={'fmt': ...}) is deprecated; "
            f"use format={canonical_format!r} with no secondary format selector",
            DeprecationWarning,
            stacklevel=2,
        )
        return normalised

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        return _validate_safe_name(v, "target name")

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        return validate_format_id(v)


@PublicAPI(stability="beta")
class BundleOutputConfig(BaseModel):
    """Top-level configuration for multi-format bundle export.

    When ``targets`` is ``None`` the system operates in legacy single-path
    mode.  When ``targets`` is non-empty the system operates in bundle mode
    and ``bundle_uri`` is required.
    """

    model_config = ConfigDict(extra="forbid")

    bundle_uri: str | None = Field(
        default=None,
        min_length=1,
        description="Only s3://, file://, or bare local paths",
    )
    storage_profile: str | None = None
    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    run_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Stable logical run identifier used for bundle idempotency",
    )
    alias: AliasConfig | None = None
    roles: dict[str, str] = Field(default_factory=dict)
    hooks: tuple[HookBinding, ...] = ()
    targets: list[ExportTarget] | None = Field(
        default=None, min_length=1, description="None = legacy mode"
    )

    @model_validator(mode="after")
    def _validate_bundle_mode(self) -> BundleOutputConfig:
        if self.request_id is not None and self.run_id is not None:
            if self.request_id != self.run_id:
                raise ValueError("request_id and run_id must identify the same run")
        hook_ids = [hook.hook_id for hook in self.hooks]
        if len(hook_ids) != len(set(hook_ids)):
            raise ValueError("hook_id values must be unique")
        if self.targets is not None:
            if not self.bundle_uri:
                raise ValueError("bundle_uri is required when targets are specified")
            names = {t.name for t in self.targets}
            if len(names) != len(self.targets):
                raise ValueError("target names must be unique")
            for t in self.targets:
                if t.name.startswith("_"):
                    raise ValueError(
                        f"target name {t.name!r} must not start with '_' "
                        "(reserved for implicit planner-generated nodes)"
                    )
            for t in self.targets:
                for d in t.depends_on:
                    if d == t.name:
                        raise ValueError(f"target {t.name!r} cannot depend on itself")
            for role_name, _target_name in self.roles.items():
                _validate_safe_name(role_name, "role name")
                # Roles are validated by the planner: references
                # to implicit targets are rejected there, not here.
        if self.alias is not None:
            if (
                self.alias.policy == "newer"
                and self.alias.expected_manifest_sha256 is not None
            ):
                raise ValueError(
                    "policy='newer' must not specify expected_manifest_sha256"
                )
            # Alias relies on S3 CAS semantics — reject local/file targets
            # explicitly instead of silently dropping the alias.
            if self.bundle_uri is not None and not self.bundle_uri.startswith("s3://"):
                raise ValueError(
                    f"alias is only supported with s3:// bundle URIs "
                    f"(got {self.bundle_uri!r})"
                )
        return self

    @field_validator("bundle_uri")
    @classmethod
    def _check_uri(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v.startswith("s3://") and len(v) > 5:
            # Reject persisted URIs carrying credentials or query/fragment
            # (plan: "拒绝带凭证或签名 query 的持久化 URI").
            rest = v[5:]
            if "@" in rest:
                raise ValueError("s3:// URI must not contain credentials")
            if "?" in rest or "#" in rest:
                raise ValueError("s3:// URI must not contain query or fragment")
            return v
        if v.startswith("file://"):
            path = v[7:]
            if not path or path == "/":
                raise ValueError("file:// URI must not point to filesystem root")
            if not Path(v[7:]).is_absolute():
                raise ValueError(f"file:// URI must be an absolute path, got {v!r}")
            return v
        if v.startswith("/") or v.startswith("./") or v.startswith("../"):
            # Reject paths that resolve to filesystem root.
            from pathlib import Path as _Path

            resolved = _Path(v).resolve()
            if resolved == _Path("/"):
                raise ValueError("bundle_uri must not point to filesystem root")
            return v
        raise ValueError(
            f"bundle_uri must be s3://, file://, or a local path, got {v!r}"
        )


# ── Export checkpoint contract ──────────────────────────────────────────────


@PublicAPI(stability="beta")
class CheckpointField(BaseModel):
    """Framework-neutral typed field stored in an export checkpoint.

    This intentionally mirrors ``ManifestSignature`` field structure while
    remaining separate: checkpoint fields describe producer metadata at the
    export boundary, whereas manifest fields describe the published artifact
    contract.
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
class ExportCheckpointV1(BaseModel):
    """Contract for checkpoint metadata needed by the Bundle exporter.

    The model-specific fields are intentionally allowed as extra keys.  The
    required envelope below is stable, while DNN/PU configurations need to
    retain their architecture-specific reconstruction parameters in the same
    ``model_config.json`` artifact.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    schema_version: Literal[1] = 1
    trainer_type: str = Field(..., min_length=1)
    architecture_id: str = Field(..., min_length=1)
    input_schema: tuple[CheckpointField, ...] = Field(..., min_length=1)
    output_schema: tuple[CheckpointField, ...] = Field(..., min_length=1)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    task_type: str = Field(..., min_length=1)
    framework: str = Field(..., min_length=1)
    framework_version: str = Field(..., min_length=1)
    checkpoint_format_version: int = Field(default=1, ge=1)
    required_artifacts: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reject_resume_only_fields(self) -> ExportCheckpointV1:
        """Keep optimizer and progress state out of export metadata."""
        resume_only = {
            "optimizer",
            "optimizer_state",
            "epoch",
            "step",
            "rng",
            "rng_state",
            "early_stopping",
        }
        leaked = sorted(resume_only.intersection(self.model_extra or {}))
        if leaked:
            raise ValueError(
                f"ExportCheckpointV1 cannot contain resume-only fields: {leaked!r}"
            )
        return self

    def to_manifest_signatures(
        self,
    ) -> tuple["ManifestSignature", "ManifestSignature"]:
        """Convert typed checkpoint fields to E1 Manifest signatures."""
        # Import lazily because ``manifest`` imports the artifact models from
        # this module.
        from tributo.exporting.manifest import ManifestSignature, SignatureField

        input_fields = tuple(
            SignatureField(
                name=field.name,
                dtype=field.dtype,
                shape=field.shape,
            )
            for field in self.input_schema
        )
        output_fields = tuple(
            SignatureField(
                name=field.name,
                dtype=field.dtype,
                shape=field.shape,
            )
            for field in self.output_schema
        )
        return (
            ManifestSignature(input_fields=input_fields),
            ManifestSignature(output_fields=output_fields),
        )


# ── Artifact models ──────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ArtifactFile(BaseModel):
    """A single file within a logical artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_path: str
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    role: str = Field(
        default="model", pattern=r"^(model|config|tokenizer|preprocessor|aux)$"
    )

    @field_validator("relative_path")
    @classmethod
    def _check_path(cls, v: str) -> str:
        return _validate_posix_relative(v, "relative_path")


@PublicAPI(stability="beta")
class DraftFile(BaseModel):
    """Exporter-provided file descriptor — no trusted hash yet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_path: str
    role: str = Field(
        default="model", pattern=r"^(model|config|tokenizer|preprocessor|aux)$"
    )

    @field_validator("relative_path")
    @classmethod
    def _check_path(cls, v: str) -> str:
        return _validate_posix_relative(v, "relative_path")


@PublicAPI(stability="beta")
class ProducerInfo(BaseModel):
    """Exporter identity and version information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exporter_id: str
    exporter_version: str | None = None
    framework_versions: dict[str, str] = Field(default_factory=dict)
    runtime_versions: dict[str, str] = Field(default_factory=dict)
    effective_options: dict[str, Any] = Field(default_factory=dict)
    effective_options_digest: str = Field(default="", min_length=0)


@PublicAPI(stability="beta")
class ArtifactRef(BaseModel):
    """A reference to a published artifact by node_id and tree digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    artifact_name: str
    tree_digest: str = Field(min_length=64, max_length=64)

    @field_validator("artifact_name")
    @classmethod
    def _check_artifact_name(cls, v: str) -> str:
        return _validate_safe_name(v, "artifact name")


@PublicAPI(stability="beta")
class ArtifactDraft(BaseModel):
    """Exporter return value — untrusted file list."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    format: str
    flavor_id: str
    variant: str | None = None
    files: tuple[DraftFile, ...]
    entrypoint: str
    producer: ProducerInfo
    derived_from: tuple[ArtifactRef, ...] = ()
    artifact_kind: str = Field(
        default="model",
        pattern=r"^(model|report|diagnostics|graph_snapshot)$",
        description="Artifact kind — model, report, diagnostics, or graph_snapshot.",
    )

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        return _validate_safe_name(v, "artifact name")

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        return validate_format_id(v)

    @model_validator(mode="after")
    def _check_entrypoint(self) -> ArtifactDraft:
        paths = {f.relative_path for f in self.files}
        if self.entrypoint not in paths:
            raise ValueError(
                f"entrypoint {self.entrypoint!r} not in files {sorted(paths)}"
            )
        return self


@PublicAPI(stability="beta")
class LogicalArtifact(BaseModel):
    """A verified, hash-materialized artifact ready for publishing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    format: str
    flavor_id: str
    variant: str | None = None
    files: tuple[ArtifactFile, ...]
    entrypoint: str
    tree_digest: str = Field(min_length=64, max_length=64)
    producer: ProducerInfo
    derived_from: tuple[ArtifactRef, ...] = ()
    validation: tuple["ValidationResult", ...] = ()
    artifact_kind: str = Field(
        default="model",
        pattern=r"^(model|report|diagnostics|graph_snapshot)$",
        description="Artifact kind — model, report, diagnostics, or graph_snapshot.",
    )

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        return _validate_safe_name(v, "artifact name")

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        return validate_format_id(v)

    @classmethod
    def compute_tree_digest(cls, files: tuple[ArtifactFile, ...]) -> str:
        """Canonical tree digest across file set.

        Algorithm: sort by path bytes, JSON-encode ``{path, sha256, size, role}``,
        SHA-256 the result.
        """
        records = sorted(
            [
                {
                    "path": f.relative_path,
                    "sha256": f.sha256,
                    "size_bytes": f.size_bytes,
                    "role": f.role,
                }
                for f in files
            ],
            key=lambda r: str(r["path"]).encode("utf-8"),
        )
        payload = json.dumps(
            records, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@DeveloperAPI
class ResolvedArtifact:
    """In-process view of an artifact with a local root directory.

    Not serialised — only lives within a BundleReader / ExportManager context.
    """

    def __init__(self, descriptor: LogicalArtifact, root_dir: Path) -> None:
        self.descriptor = descriptor
        self.root_dir = root_dir
        self.entrypoint_path = root_dir / descriptor.entrypoint

    def path_for(self, relative_path: str) -> Path:
        """Resolve a file within this artifact, guarding against traversal."""
        resolved = (self.root_dir / relative_path).resolve()
        if not resolved.is_relative_to(self.root_dir.resolve()):
            raise ValueError(f"Path traversal detected: {relative_path!r}")
        return resolved


# ── Execution results ────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class FailureInfo(BaseModel):
    """Structured, sanitised failure detail — no tracebacks or credentials."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    category: str = Field(
        pattern=r"^(configuration|unsupported|export|validation|publish)$"
    )
    message: str = Field(max_length=4096)
    retryable: bool = False


@PublicAPI(stability="beta")
class NodeResult(BaseModel):
    """Result of a single DAG node execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    target_name: str
    status: str = Field(pattern=r"^(succeeded|failed|blocked|cancelled)$")
    required: bool
    publish: bool
    exporter_id: str | None = None
    output_format: str | None = None
    flavor_id: str | None = None
    artifact_ref: ArtifactRef | None = None
    failure: FailureInfo | None = None
    duration_ms: int = Field(default=0, ge=0)


@PublicAPI(stability="beta")
class ExportExecutionResult(BaseModel):
    """Complete result of an export DAG execution (before publishing)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: str
    status: str = Field(pattern=r"^(succeeded|partial|failed)$")
    node_results: tuple[NodeResult, ...]
    staged_artifacts: dict[str, LogicalArtifact] = Field(default_factory=dict)
    roles: dict[str, str] = Field(default_factory=dict)

    @property
    def succeeded_artifacts(self) -> dict[str, ArtifactRef]:
        """Node ID → ArtifactRef for all succeeded nodes."""
        return {
            nr.node_id: nr.artifact_ref
            for nr in self.node_results
            if nr.status == "succeeded" and nr.artifact_ref is not None
        }


@PublicAPI(stability="beta")
class BundleResult(BaseModel):
    """Immutable result returned to the caller after a successful publish."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle_id: str
    canonical_uri: str
    manifest_uri: str
    manifest_sha256: str = Field(min_length=64, max_length=64)
    status: str = Field(pattern=r"^(succeeded|partial)$")
    artifacts: tuple[LogicalArtifact, ...] = ()
    node_results: tuple[NodeResult, ...] = ()
    roles: dict[str, str] = Field(default_factory=dict)
    alias_uri: str | None = None
    alias_status: str = Field(
        default="not_requested",
        pattern=r"^(not_requested|updated|unchanged|failed)$",
    )
    alias_failure: FailureInfo | None = None
    hook_receipts: tuple[HookReceipt, ...] = ()


@PublicAPI(stability="beta")
class BundleRef(BaseModel):
    """Immutable reference to a committed bundle.

    This is the stable handle that callers keep — it never changes after
    commit and can be passed to ``load_bundle()`` without needing to know
    the storage backend.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_uri: str
    bundle_id: str
    manifest_sha256: str


@DeveloperAPI
class PublishedBundle:
    """Transient handle returned within BundleExportService context.

    ``local_bundle_dir`` is only valid during the callback window.
    """

    def __init__(
        self,
        result: BundleResult,
        manifest_bytes: bytes,
        local_bundle_dir: Path,
        local_dir_ephemeral: bool = True,
    ) -> None:
        self.result = result
        self.manifest_bytes = manifest_bytes
        self.local_bundle_dir = local_bundle_dir
        self.local_dir_ephemeral = local_dir_ephemeral


# ── Validation models ────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ValidationResult(BaseModel):
    """Result of a single validator run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    validator_id: str
    status: str = Field(pattern=r"^(passed|failed|advisory_failed)$")
    metrics: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    failure: FailureInfo | None = None


@PublicAPI(stability="beta")
class ValidatorBinding(BaseModel):
    """Exporter declares which validators to run and their defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    validator_id: str
    required: bool = True
    default_options: dict[str, Any] = Field(default_factory=dict)


@PublicAPI(stability="beta")
class UpstreamRequirement(BaseModel):
    """Declares that an exporter needs an upstream artifact of a given format.

    Used by artifact-to-artifact transforms (e.g. ONNX quantizer needs
    an FP32 ONNX artifact).  The *name* must match a ``depends_on`` entry
    in the dependent target's ``ExportTarget``.  The planner reads this to
    inject implicit intermediate nodes into the DAG.

    Example::

        # ONNX quantizer declares it needs an upstream FP32 ONNX artifact.
        UpstreamRequirement(
            name="model",
            format="onnx",
            options={"quantization": None},
        )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    format: str = Field(..., min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        return validate_format_id(v)


# ── Planning models ──────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class SupportRequest(BaseModel):
    """Input to ModelExporter.supports()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_kind: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    upstream_formats: tuple[str, ...] = ()


@PublicAPI(stability="beta")
class SupportResult(BaseModel):
    """Return value from ModelExporter.supports()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    supported: bool
    code: str = ""
    reason: str = ""
    missing_dependencies: tuple[str, ...] = ()
    environment_constraints: dict[str, str] = Field(default_factory=dict)


@PublicAPI(stability="beta")
class PlannedTarget(BaseModel):
    """A target that has been matched to a concrete exporter and options."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: ExportTarget
    exporter_id: str
    typed_options: dict[str, Any] = Field(default_factory=dict)
    validator_bindings: tuple[ValidatorBinding, ...] = ()
    implicit: bool = False
    publish: bool = True


# ── Environment ──────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ExportContext(BaseModel):
    """Per-node export execution context."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    execution_id: str
    node_id: str
    artifact_dir: Path
    environment: dict[str, str] = Field(default_factory=dict)


@PublicAPI(stability="beta")
class ExportSource(BaseModel):
    """Read-only snapshot passed to every exporter and validator."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    source_kind: str
    model_object: Any = Field(default=None, exclude=True)
    architecture_id: str | None = None
    model_config_data: dict[str, Any] = Field(default_factory=dict)
    feature_schema: dict[str, Any] = Field(default_factory=dict)
    preprocessing_state: dict[str, Any] = Field(default_factory=dict)
    sample_inputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_fingerprint: str = ""
    checkpoint_contract: ExportCheckpointV1 | None = None


# ── Plugin diagnostics ───────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class PluginLoadDiagnostic(BaseModel):
    """Non-fatal plugin loading issue."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group: str
    entry_point_name: str
    reason: str
    error_type: str | None = None

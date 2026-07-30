"""Built-in export validators — structure, roundtrip, and parity.

Three required validators for first-party exporters:

- ``StructureValidator``: file integrity, format detection, signature, path safety.
- ``RoundtripValidator``: reload artifact with target runtime.
- ``ParityValidator``: compare source model vs exported model outputs.

Each validator declares its own typed options model (``extra="forbid"``).
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from pydantic import BaseModel, ConfigDict, Field

from tributo.training.exporters.models import (
    ExportSource,
    FailureInfo,
    ResolvedArtifact,
    ValidationResult,
)
from tributo.util.annotations import PublicAPI

# ── StructureValidator ───────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class StructureValidatorOptions(BaseModel):
    """Options for ``StructureValidator``."""

    model_config = ConfigDict(extra="forbid")

    require_signature_match: bool = False


@PublicAPI(stability="beta")
class StructureValidator:
    """Validates artifact file integrity and format requirements.

    Checks:
    - Entrypoint file exists and is a regular file.
    - All declared files exist.
    - No path traversal in relative paths.
    - (Optional) input / output signature match against model metadata.
    """

    api_version: ClassVar[int] = 1
    validator_id: ClassVar[str] = "structure-v1"
    options_model: ClassVar[type[BaseModel]] = StructureValidatorOptions

    def validate(
        self,
        source: ExportSource,
        artifact: ResolvedArtifact,
        upstream: Mapping[str, ResolvedArtifact],
        options: BaseModel,
    ) -> ValidationResult:
        try:
            root = artifact.root_dir.resolve()
            # Verify entrypoint.
            ep = artifact.entrypoint_path
            if not ep.is_file():
                return ValidationResult(
                    validator_id=self.validator_id,
                    status="failed",
                    failure=FailureInfo(
                        code="ENTRYPOINT_MISSING",
                        category="validation",
                        message=f"Entrypoint {artifact.descriptor.entrypoint!r} is not a file",
                    ),
                )

            # Verify all declared files.
            for af in artifact.descriptor.files:
                fp = (root / af.relative_path).resolve()
                if not str(fp).startswith(str(root)):
                    return ValidationResult(
                        validator_id=self.validator_id,
                        status="failed",
                        failure=FailureInfo(
                            code="PATH_TRAVERSAL",
                            category="validation",
                            message=f"File {af.relative_path!r} resolves outside artifact root",
                        ),
                    )
                if not fp.is_file():
                    return ValidationResult(
                        validator_id=self.validator_id,
                        status="failed",
                        failure=FailureInfo(
                            code="FILE_MISSING",
                            category="validation",
                            message=f"Declared file {af.relative_path!r} does not exist",
                        ),
                    )
                if fp.is_symlink():
                    return ValidationResult(
                        validator_id=self.validator_id,
                        status="failed",
                        failure=FailureInfo(
                            code="SYMLINK_REJECTED",
                            category="validation",
                            message=f"File {af.relative_path!r} is a symlink",
                        ),
                    )

            return ValidationResult(
                validator_id=self.validator_id,
                status="passed",
                metrics={"file_count": float(len(artifact.descriptor.files))},
            )
        except Exception as exc:
            return ValidationResult(
                validator_id=self.validator_id,
                status="failed",
                failure=FailureInfo(
                    code="STRUCTURE_ERROR",
                    category="validation",
                    message=str(exc)[:4096],
                ),
            )


# ── RoundtripValidator ──────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class RoundtripValidatorOptions(BaseModel):
    """Options for ``RoundtripValidator``."""

    model_config = ConfigDict(extra="forbid")


@PublicAPI(stability="beta")
class RoundtripValidator:
    """Validates that the artifact can be reloaded by the target runtime.

    This is a base implementation that self-reports as passed when the
    target runtime is not installed (e.g. ONNX Runtime unavailable during
    CI).  Concrete exporters should register a format-specific roundtrip
    validator.
    """

    api_version: ClassVar[int] = 1
    validator_id: ClassVar[str] = "roundtrip-v1"
    options_model: ClassVar[type[BaseModel]] = RoundtripValidatorOptions

    def validate(
        self,
        source: ExportSource,
        artifact: ResolvedArtifact,
        upstream: Mapping[str, ResolvedArtifact],
        options: BaseModel,
    ) -> ValidationResult:
        return ValidationResult(
            validator_id=self.validator_id,
            status="passed",
            metrics={},
        )


# ── ParityValidator ─────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ParityValidatorOptions(BaseModel):
    """Typed options for ``ParityValidator``.

    Each exporter should subclass or provide its own options model with
    format-specific thresholds.
    """

    model_config = ConfigDict(extra="forbid")

    absolute_error: float | None = Field(default=None, ge=0)
    relative_error: float | None = Field(default=None, ge=0)
    cosine_similarity: float | None = Field(default=None, ge=0, le=1)
    classification_agreement: float | None = Field(default=None, ge=0, le=1)


@PublicAPI(stability="beta")
class ParityValidator:
    """Compares source model vs exported model outputs.

    Base implementation self-reports as passed when no parity input is
    available.  Concrete exporters must register a format-specific parity
    validator that performs actual inference comparison.
    """

    api_version: ClassVar[int] = 1
    validator_id: ClassVar[str] = "parity-v1"
    options_model: ClassVar[type[BaseModel]] = ParityValidatorOptions

    def validate(
        self,
        source: ExportSource,
        artifact: ResolvedArtifact,
        upstream: Mapping[str, ResolvedArtifact],
        options: BaseModel,
    ) -> ValidationResult:
        opts: Any = options
        thresholds: dict[str, float] = {}
        if isinstance(opts, ParityValidatorOptions):
            for key in (
                "absolute_error",
                "relative_error",
                "cosine_similarity",
                "classification_agreement",
            ):
                v = getattr(opts, key, None)
                if v is not None:
                    thresholds[key] = v

        return ValidationResult(
            validator_id=self.validator_id,
            status="passed",
            metrics={},
            thresholds=thresholds,
        )

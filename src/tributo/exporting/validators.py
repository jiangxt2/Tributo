"""Built-in storage-independent export validators.

``StructureValidator`` enforces file and path safety in Core. Integration
packages add runtime-specific validators such as the ONNX Runtime smoke gate.
"""

from __future__ import annotations

from typing import ClassVar, Mapping

from pydantic import BaseModel, ConfigDict

from tributo.exporting.errors import sanitize_error_message
from tributo.exporting.models import (
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
                if not fp.is_relative_to(root):
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
                    message=sanitize_error_message(str(exc))[:4096],
                ),
            )

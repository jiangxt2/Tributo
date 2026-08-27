"""Framework-neutral explainability contracts.

The contracts intentionally contain no SHAP, Ray, ONNX Runtime, or model
objects.  They are safe to validate and serialize at the driver boundary.
Concrete adapters build their process-local runtime objects in ``prepare``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tributo.util.annotations import PublicAPI

Exactness = Literal["exact", "approximate", "conditional"]
ReferencePolicy = Literal["none", "optional", "required"]


class _FrozenContract(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )


@PublicAPI(stability="alpha")
class ReferenceBinding(_FrozenContract):
    """Credential-free reference/background data binding."""

    uri: str = Field(min_length=1)
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rows: int | None = Field(default=None, ge=1)
    privacy_level: str = Field(default="restricted", min_length=1)
    ttl_seconds: int | None = Field(default=None, ge=1)

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("reference URI must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("reference URI must not contain query or fragment")
        if parsed.scheme and parsed.scheme not in {"file", "s3"}:
            raise ValueError("reference URI must be local/file:// or s3://")
        return value


@PublicAPI(stability="alpha")
class ExplainabilityLimits(_FrozenContract):
    """Safety and output-size limits for one explainability operation."""

    max_rows: int | None = Field(default=None, ge=1)
    max_background_rows: int | None = Field(default=None, ge=1)
    max_features: int | None = Field(default=None, ge=1)
    max_explanation_bytes: int | None = Field(default=None, ge=1)
    max_explanation_rows: int | None = Field(default=None, ge=1)
    top_k: int | None = Field(default=None, ge=1)


@PublicAPI(stability="alpha")
class ResourcePolicy(_FrozenContract):
    """Resource policy passed to the batch execution planner."""

    batch_size: int = Field(default=1024, ge=1)
    concurrency: int = Field(default=1, ge=1)
    num_cpus_per_actor: float = Field(default=1.0, ge=0)
    num_gpus_per_actor: float = Field(default=0.0, ge=0)


@PublicAPI(stability="alpha")
class ResultPolicy(_FrozenContract):
    """Access, sensitivity, and retention policy for explanation results."""

    access_scope: Literal["private", "project", "restricted"] = "restricted"
    privacy_level: str = Field(default="restricted", min_length=1)
    allow_sensitive_features: bool = False
    retention_seconds: int | None = Field(default=None, ge=1)


@PublicAPI(stability="alpha")
class ExplainabilityConfig(BaseModel):
    """Bundle-time explainability preparation configuration.

    ``enabled=False`` is deliberately represented by the default model and
    must not cause optional explainability dependencies to be imported.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    explainer: str = Field(default="shap", min_length=1)
    backend: Literal["auto", "tree", "model_agnostic", "deep", "gradient"] = "auto"
    scope: Literal["batch"] = "batch"
    feature_view: Literal["raw", "transformed", "model_input"] = "raw"
    output_target: str = Field(default="model_output", min_length=1)
    label_column: str | None = Field(default=None, min_length=1)
    allow_approximate: bool = False
    reference: ReferenceBinding | None = None
    limits: ExplainabilityLimits = Field(default_factory=ExplainabilityLimits)
    resource_policy: ResourcePolicy = Field(default_factory=ResourcePolicy)
    model_role: str = Field(default="inference", min_length=1)

    @model_validator(mode="after")
    def _validate_backend_gate(self) -> ExplainabilityConfig:
        if (
            self.enabled
            and self.explainer == "shap"
            and self.backend
            in {
                "deep",
                "gradient",
            }
        ):
            raise ValueError(
                f"SHAP backend {self.backend!r} is not available in this release"
            )
        if (
            self.enabled
            and self.backend == "model_agnostic"
            and not self.allow_approximate
        ):
            raise ValueError("backend='model_agnostic' requires allow_approximate=true")
        if self.enabled and self.backend == "model_agnostic" and self.reference is None:
            raise ValueError(
                "model_agnostic SHAP requires an explicit reference binding"
            )
        if (
            self.enabled
            and self.backend in {"auto", "tree"}
            and self.output_target in {"probability", "log_loss"}
            and self.reference is None
        ):
            raise ValueError(
                "Tree SHAP probability/log_loss output requires a reference binding"
            )
        if (
            self.enabled
            and self.output_target == "log_loss"
            and self.backend not in {"auto", "tree"}
        ):
            raise ValueError("log_loss output is only supported by the tree backend")
        if (
            self.enabled
            and self.output_target == "log_loss"
            and self.label_column is None
        ):
            raise ValueError("Tree SHAP log_loss output requires label_column")
        return self

    def to_descriptor(
        self,
        *,
        required_artifacts: tuple[str, ...] = (),
        model_roles: tuple[str, ...] | None = None,
        backend: str | None = None,
        exactness: Exactness | None = None,
    ) -> ExplainabilityDescriptor | None:
        """Build the manifest descriptor for an enabled configuration."""
        if not self.enabled:
            return None
        selected_backend = backend or self.backend
        selected_exactness: Exactness = exactness or _default_exactness(
            selected_backend
        )
        reference_policy: ReferencePolicy = (
            "required" if selected_backend != "tree" else "optional"
        )
        if self.reference is not None or self.output_target in {
            "probability",
            "log_loss",
        }:
            reference_policy = "required"
        dependencies = {}
        if selected_backend == "model_agnostic":
            dependencies = {
                "onnxruntime": ">=1.16.0,<2.0.0",
                "shap": ">=0.52.0,<0.53.0",
            }
        return ExplainabilityDescriptor(
            adapter_id=f"{self.explainer}-v1",
            backend=selected_backend,
            exactness=selected_exactness,
            model_roles=model_roles or (self.model_role,),
            required_artifacts=required_artifacts,
            feature_view=self.feature_view,
            output_target=self.output_target,
            reference_policy=reference_policy,
            reference_digest=(self.reference.digest if self.reference else None),
            reference_rows=(self.reference.rows if self.reference else None),
            reference_privacy_level=(
                self.reference.privacy_level if self.reference else None
            ),
            reference_ttl_seconds=(
                self.reference.ttl_seconds if self.reference else None
            ),
            dependencies=dependencies,
        )


@PublicAPI(stability="alpha")
class ExplainabilityDescriptor(_FrozenContract):
    """Immutable capability promise written to manifest schema v2."""

    descriptor_version: Literal[1] = 1
    adapter_id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    exactness: Exactness
    model_roles: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    feature_view: Literal["raw", "transformed", "model_input"]
    output_target: str = Field(min_length=1)
    reference_policy: ReferencePolicy
    reference_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reference_rows: int | None = Field(default=None, ge=1)
    reference_privacy_level: str | None = Field(default=None, min_length=1)
    reference_ttl_seconds: int | None = Field(default=None, ge=1)
    dependencies: dict[str, str] = Field(default_factory=dict)
    conformance_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("model_roles", "required_artifacts")
    @classmethod
    def _unique_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("descriptor artifact names must be unique")
        return value


@PublicAPI(stability="alpha")
class ExplainabilityRequest(_FrozenContract):
    """One explicit batch explanation operation."""

    bundle_uri: str = Field(min_length=1)
    input: Any
    storage_profile: str | None = None
    bundle_id: str | None = Field(default=None, min_length=1)
    expected_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    model_role: str | None = Field(default=None, min_length=1)
    feature_columns: tuple[str, ...] = ()
    input_id_column: str | None = Field(default=None, min_length=1)
    explainer: str = Field(default="shap", min_length=1)
    backend: Literal["auto", "tree", "model_agnostic", "deep", "gradient"] = "auto"
    feature_view: Literal["raw", "transformed", "model_input"] = "raw"
    output_target: str = Field(default="model_output", min_length=1)
    output_selection: Literal["all", "predicted"] = "all"
    label_column: str | None = Field(default=None, min_length=1)
    allow_approximate: bool = False
    reference: ReferenceBinding | None = None
    limits: ExplainabilityLimits = Field(default_factory=ExplainabilityLimits)
    resource_policy: ResourcePolicy = Field(default_factory=ResourcePolicy)
    result_uri: str = Field(min_length=1)
    result_storage_profile: str | None = None
    result_policy: ResultPolicy = Field(default_factory=ResultPolicy)
    operation_store_uri: str | None = Field(default=None, min_length=1)
    operation_lease_seconds: int = Field(default=300, ge=1)
    force_resume: bool = False
    request_id: str = Field(min_length=1)
    operation_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_request(self) -> ExplainabilityRequest:
        from tributo.data import IngestionRequest

        if not isinstance(self.input, IngestionRequest):
            object.__setattr__(
                self,
                "input",
                IngestionRequest.model_validate(self.input),
            )
        if self.backend == "model_agnostic" and not self.allow_approximate:
            raise ValueError("backend='model_agnostic' requires allow_approximate=true")
        if self.explainer == "shap" and self.backend in {"deep", "gradient"}:
            raise ValueError(
                f"SHAP backend {self.backend!r} is not available in this release"
            )
        if self.backend == "model_agnostic" and self.reference is None:
            raise ValueError(
                "model_agnostic SHAP requires an explicit reference binding"
            )
        if self.limits.top_k is not None and self.limits.max_features is not None:
            if self.limits.top_k > self.limits.max_features:
                raise ValueError("top_k must not exceed max_features")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be unique")
        if self.label_column is not None and self.label_column in self.feature_columns:
            raise ValueError("label_column must not be included in feature_columns")
        if self.output_target == "log_loss" and self.label_column is None:
            raise ValueError("Tree SHAP log_loss output requires label_column")
        if self.output_target == "log_loss" and self.backend not in {"auto", "tree"}:
            raise ValueError("log_loss output is only supported by the tree backend")
        if (
            self.backend in {"auto", "tree"}
            and self.output_target in {"probability", "log_loss"}
            and self.reference is None
        ):
            raise ValueError(
                "Tree SHAP probability/log_loss output requires a reference binding"
            )
        if self.operation_store_uri is not None:
            parsed = urlsplit(self.operation_store_uri)
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("operation_store_uri must not contain credentials")
            if parsed.query or parsed.fragment:
                raise ValueError(
                    "operation_store_uri must not contain query or fragment"
                )
            if parsed.scheme not in {"", "file"}:
                raise ValueError(
                    "operation_store_uri must be a local path or file:// URI"
                )
        return self


@PublicAPI(stability="alpha")
class FeatureAttribution(_FrozenContract):
    """One framework-neutral long-format attribution row."""

    input_id: str = Field(min_length=1)
    output_id: str = Field(min_length=1)
    feature_id: str = Field(min_length=1)
    feature_name: str = Field(min_length=1)
    feature_view: Literal["raw", "transformed", "model_input"]
    feature_value: float | int | str | bool | None = None
    contribution: float
    base_value: float | None = None
    model_output: float | int | str | bool | None = None
    output_target: str = Field(min_length=1)
    explainer: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    exactness: Exactness
    model_digest: str = Field(min_length=1)
    preprocessor_digest: str | None = None
    feature_map_digest: str | None = None


@PublicAPI(stability="alpha")
class ExplainabilityReceipt(_FrozenContract):
    """Immutable result provenance for one explainability operation."""

    request_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    bundle_id: str | None = None
    bundle_digest: str = Field(min_length=64, max_length=64)
    model_digest: str = Field(min_length=64, max_length=64)
    preprocessor_digest: str | None = None
    feature_map_digest: str | None = None
    reference_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reference_rows: int | None = Field(default=None, ge=1)
    reference_privacy_level: str | None = Field(default=None, min_length=1)
    reference_ttl_seconds: int | None = Field(default=None, ge=1)
    adapter_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    exactness: Literal["exact", "approximate", "conditional"]
    feature_view: Literal["raw", "transformed", "model_input"]
    output_target: str = Field(min_length=1)
    output_selection: Literal["all", "predicted"] = "all"
    execution_profile: str = Field(default="batch", min_length=1)
    input_rows: int = Field(default=0, ge=0)
    explanation_rows: int = Field(default=0, ge=0)
    failed_rows: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    result_uri: str = Field(min_length=1)
    result_digest: str = Field(min_length=64, max_length=64)
    result_bytes: int = Field(default=0, ge=0)
    result_access_scope: Literal["private", "project", "restricted"] = "restricted"
    result_privacy_level: str = Field(default="restricted", min_length=1)
    result_retention_seconds: int | None = Field(default=None, ge=1)
    reference_policy: ReferencePolicy = "none"
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    schema_signature: str = Field(min_length=1)
    status: Literal["succeeded", "partial", "failed"]
    warnings: tuple[str, ...] = ()
    resource_summary: dict[str, Any] = Field(default_factory=dict)
    failure_code: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@PublicAPI(stability="alpha")
class ExplainabilityOperationRecord(_FrozenContract):
    """Mutable-operation snapshot persisted independently from a receipt."""

    operation_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    bundle_id: str | None = None
    bundle_digest: str = Field(min_length=64, max_length=64)
    result_uri: str = Field(min_length=1)
    receipt_uri: str | None = None
    reference_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str = Field(
        default="0" * 64,
        min_length=64,
        max_length=64,
    )
    status: Literal["pending", "running", "succeeded", "partial", "failed"]
    failure_phase: str | None = None
    failure_code: str | None = None
    retryable: bool = False
    input_rows: int = Field(default=0, ge=0)
    explanation_rows: int = Field(default=0, ge=0)
    failed_rows: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lease_token: str = ""
    lease_expires_at: datetime | None = None
    completed_at: datetime | None = None


def _default_exactness(backend: str) -> Exactness:
    return cast(
        Exactness,
        {
            "tree": "exact",
            "model_agnostic": "approximate",
        }.get(backend, "conditional"),
    )


__all__ = [
    "ExplainabilityConfig",
    "ExplainabilityDescriptor",
    "ExplainabilityLimits",
    "ExplainabilityOperationRecord",
    "ExplainabilityReceipt",
    "ExplainabilityRequest",
    "FeatureAttribution",
    "ReferenceBinding",
    "ResultPolicy",
    "ResourcePolicy",
]

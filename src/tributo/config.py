"""Job configuration management.

Provides a Pydantic-based configuration system for Ray job submissions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from tributo._common.config import StrictConfigModel
from tributo.algorithms.api import AlgorithmOperation, ExecutionProfile
from tributo.data import IngestionRequest
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="stable")
class JobConfig(BaseModel):
    """Configuration for Ray job submission.

    Attributes:
        entrypoint: The command to run for the job.
        runtime_env: Runtime environment configuration.
        num_cpus: Number of CPUs to allocate.
        num_gpus: Number of GPUs to allocate.
        memory: Memory to allocate (in bytes).
        metadata: Additional metadata for the job.
        submission_id: Optional submission ID for the job.
    """

    entrypoint: str = Field(..., description="Command to run for the job")
    runtime_env: Dict[str, Any] = Field(
        default_factory=dict, description="Runtime environment configuration"
    )
    num_cpus: Optional[float] = Field(
        default=None, ge=0, description="Number of CPUs to allocate"
    )
    num_gpus: Optional[float] = Field(
        default=None, ge=0, description="Number of GPUs to allocate"
    )
    memory: Optional[int] = Field(
        default=None, ge=0, description="Memory to allocate in bytes"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Additional metadata"
    )
    submission_id: Optional[str] = Field(
        default=None, description="Optional submission ID"
    )

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, v: str) -> str:
        """Validate that entrypoint is not empty."""
        if not v or not v.strip():
            raise ValueError("Entrypoint cannot be empty")
        return v.strip()

    model_config = ConfigDict(frozen=False, extra="forbid")


@PublicAPI(stability="alpha")
class AlgorithmInputConfig(StrictConfigModel):
    """Canonical bounded input plus the tabular roles used by an algorithm."""

    ingestion: IngestionRequest
    features: list[str] = Field(min_length=1)
    label: str
    sample_weight: str | None = None

    @model_validator(mode="after")
    def validate_roles(self) -> AlgorithmInputConfig:
        """Reject duplicate or overlapping tabular roles before planning."""
        if len(set(self.features)) != len(self.features):
            raise ValueError("input.features must contain unique column names")
        if not self.label:
            raise ValueError("input.label must be non-empty")
        if self.label in self.features:
            raise ValueError("input.label must not also be an input feature")
        if self.sample_weight is not None and (
            not self.sample_weight
            or self.sample_weight in self.features
            or self.sample_weight == self.label
        ):
            raise ValueError(
                "input.sample_weight must be a distinct non-empty column name"
            )
        return self


@PublicAPI(stability="alpha")
class AlgorithmWorkerResourcesConfig(StrictConfigModel):
    """Optional exact per-worker override checked against DistributionSpec."""

    num_cpus: float = Field(ge=0)
    num_gpus: float = Field(default=0, ge=0)
    custom: dict[str, float] = Field(default_factory=dict)

    @field_validator("custom")
    @classmethod
    def validate_custom_resources(cls, value: dict[str, float]) -> dict[str, float]:
        """Require scheduler-safe custom resource declarations."""
        if any(not name or amount < 0 for name, amount in value.items()):
            raise ValueError(
                "custom resources require non-empty names and non-negative values"
            )
        return dict(value)


@PublicAPI(stability="alpha")
class LocalRayRuntimeConfig(StrictConfigModel):
    """Explicit local Ray registration override for tests or resource isolation."""

    num_cpus: int | None = Field(default=None, ge=0)
    num_gpus: int | None = Field(default=None, ge=0)


@PublicAPI(stability="alpha")
class AlgorithmExecutionConfig(StrictConfigModel):
    """One JSON envelope shared by local[*] and Kubernetes execution."""

    algorithm: str = Field(min_length=1)
    profile: ExecutionProfile
    worker_count: int = Field(ge=1)
    input: AlgorithmInputConfig
    algorithm_config: dict[str, Any]
    operation: AlgorithmOperation = AlgorithmOperation.FIT
    implementation_id: str | None = None
    resources_per_worker: AlgorithmWorkerResourcesConfig | None = None
    local_runtime: LocalRayRuntimeConfig | None = None
    resume_from: str | None = None

    @model_validator(mode="after")
    def validate_profile_options(self) -> AlgorithmExecutionConfig:
        """Keep local runtime registration separate from Kubernetes requests."""
        if self.profile is ExecutionProfile.KUBERNETES and self.local_runtime:
            raise ValueError("local_runtime is valid only when profile is 'local'")
        if self.resume_from is not None and not self.resume_from:
            raise ValueError("resume_from must be non-empty when configured")
        return self

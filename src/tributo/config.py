"""Job configuration management.

Provides a Pydantic-based configuration system for Ray job submissions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

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

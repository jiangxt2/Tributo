"""Pydantic model definitions: training run metrics, model version, experiment info."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class RunMetrics(BaseModel):
    """Training run metrics.

    Supports extra="allow" for recording arbitrary custom metrics.
    """

    model_config = ConfigDict(extra="allow")

    loss: float | None = None
    accuracy: float | None = None
    auc: float | None = None
    f1: float | None = None


@PublicAPI(stability="beta")
class ModelVersion(BaseModel):
    """Model version information."""

    name: str
    version: int
    stage: str  # "None" | "Staging" | "Production" | "Archived"
    run_id: str
    artifact_uri: str
    creation_timestamp: int
    description: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


@PublicAPI(stability="beta")
class ExperimentInfo(BaseModel):
    """Experiment information."""

    experiment_id: str
    name: str
    artifact_location: str

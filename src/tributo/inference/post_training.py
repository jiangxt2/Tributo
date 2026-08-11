"""Entry adapter for inference after a successfully published training Bundle.

This module deliberately imports no Training implementation.  The Training
domain publishes a ``BundleRef`` and parent run identity; this adapter binds
those immutable values to normal Inference contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tributo.data import IngestionRequest
from tributo.exporting.models import BundleRef
from tributo.inference._credential_safety import credential_paths
from tributo.inference.api import run_inference
from tributo.inference.contracts import (
    BundleModelReference,
    InferenceRequest,
    InferenceResult,
    InputBindingSpec,
    OutputBindingSpec,
    ParquetResultSinkRequest,
    RayExecutionPolicy,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class PostTrainingInferenceAction(BaseModel):
    """Inference intent waiting for a published training BundleRef."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    input: IngestionRequest
    input_binding: InputBindingSpec
    output_binding: OutputBindingSpec
    result_sink: ParquetResultSinkRequest
    execution: RayExecutionPolicy = Field(default_factory=RayExecutionPolicy)
    role: str = Field(default="inference", min_length=1)
    model_storage_profile: str | None = None
    unsafe_model: bool = False
    mode: Literal["inline", "detached"] = "inline"

    @model_validator(mode="after")
    def _reject_plaintext_credentials(self) -> "PostTrainingInferenceAction":
        paths = credential_paths(self.model_dump(mode="python"), "post_training_action")
        if paths:
            raise ValueError(
                "PostTrainingInferenceAction must not contain plaintext "
                f"credentials (fields: {sorted(paths)})"
            )
        return self

    def bind(self, bundle_ref: BundleRef, *, parent_run_id: str) -> InferenceRequest:
        """Create the same strict request accepted by standalone inference."""
        return InferenceRequest(
            model=BundleModelReference.from_bundle_ref(
                bundle_ref,
                role=self.role,
                storage_profile=self.model_storage_profile,
                unsafe=self.unsafe_model,
            ),
            input=self.input,
            input_binding=self.input_binding,
            output_binding=self.output_binding,
            result_sink=self.result_sink,
            execution=self.execution,
            parent_run_id=parent_run_id,
        )


@PublicAPI(stability="alpha")
def run_post_training_inference(
    action: PostTrainingInferenceAction,
    bundle_ref: BundleRef,
    *,
    parent_run_id: str,
) -> InferenceResult:
    """Run an inline action through the ordinary inference API."""
    if action.mode != "inline":
        raise ValueError(
            "run_post_training_inference requires mode='inline'; use "
            "submit_post_training_inference for detached execution"
        )
    return run_inference(action.bind(bundle_ref, parent_run_id=parent_run_id))


@PublicAPI(stability="alpha")
def submit_post_training_inference(
    action: PostTrainingInferenceAction,
    bundle_ref: BundleRef,
    *,
    parent_run_id: str,
    dashboard_url: str,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
) -> str:
    """Submit a detached action through the ordinary Ray Jobs adapter."""
    if action.mode != "detached":
        raise ValueError(
            "submit_post_training_inference requires mode='detached'; use "
            "run_post_training_inference for inline execution"
        )
    from tributo.inference.job_runner import submit_inference_request

    return submit_inference_request(
        action.bind(bundle_ref, parent_run_id=parent_run_id),
        dashboard_url=dashboard_url,
        env_vars=env_vars,
        project_root=project_root,
    )


__all__ = [
    "PostTrainingInferenceAction",
    "run_post_training_inference",
    "submit_post_training_inference",
]

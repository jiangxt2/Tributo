"""tributo.inference — Distributed batch inference module.

Streaming inference based on Ray Data + ONNX Runtime, supporting XGBoost and other models.

Extend BasePredictor::

    from tributo.inference import BasePredictor

    class MyPredictor(BasePredictor):
        def _load_model(self): ...
        def __call__(self, batch): ...
"""

from __future__ import annotations

from tributo.inference.api import (
    resolve_inference,
    run_inference,
    run_resolved_inference,
)
from tributo.inference.base import BasePredictor
from tributo.inference.batch_predictor import XGBoostONNXPredictor
from tributo.inference.contracts import (
    ArtifactModelReference,
    BundleModelReference,
    InferenceRequest,
    InferenceResult,
    InputBindingSpec,
    LanceResultSinkRequest,
    LanceVectorColumnSpec,
    OutputBindingSpec,
    ParquetResultSinkRequest,
    RayExecutionPolicy,
    RegistryModelReference,
    TensorInputBinding,
    TensorOutputBinding,
)
from tributo.inference.job_runner import (
    submit_inference_job,
    submit_inference_request,
    submit_resolved_inference,
)
from tributo.inference.pipeline import (
    InferenceConfig,
    run_batch_inference,
    run_inference_from_json,
)

__all__ = [
    "BasePredictor",
    "ArtifactModelReference",
    "BundleModelReference",
    "InferenceConfig",
    "InferenceRequest",
    "InferenceResult",
    "InputBindingSpec",
    "LanceResultSinkRequest",
    "LanceVectorColumnSpec",
    "OutputBindingSpec",
    "ParquetResultSinkRequest",
    "RayExecutionPolicy",
    "RegistryModelReference",
    "TensorInputBinding",
    "TensorOutputBinding",
    "resolve_inference",
    "run_batch_inference",
    "run_inference",
    "run_inference_from_json",
    "run_resolved_inference",
    "XGBoostONNXPredictor",
    "submit_inference_job",
    "submit_inference_request",
    "submit_resolved_inference",
]

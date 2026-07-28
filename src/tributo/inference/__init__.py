"""tributo.inference — Distributed batch inference module.

Streaming inference based on Ray Data + ONNX Runtime, supporting XGBoost and other models.

Extend BasePredictor::

    from tributo.inference import BasePredictor

    class MyPredictor(BasePredictor):
        def _load_model(self): ...
        def __call__(self, batch): ...
"""

from __future__ import annotations

from tributo.inference.base import BasePredictor
from tributo.inference.batch_predictor import XGBoostONNXPredictor
from tributo.inference.job_runner import submit_inference_job
from tributo.inference.pipeline import (
    InferenceConfig,
    run_batch_inference,
    run_inference_from_json,
)

__all__ = [
    "BasePredictor",
    "InferenceConfig",
    "run_batch_inference",
    "run_inference_from_json",
    "XGBoostONNXPredictor",
    "submit_inference_job",
]

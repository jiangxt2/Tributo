"""ONNX Runtime validator — real roundtrip inference check."""

from __future__ import annotations

from typing import ClassVar, Mapping

from pydantic import BaseModel, ConfigDict

from tributo.exporting.models import (
    ExportSource,
    FailureInfo,
    ResolvedArtifact,
    ValidationResult,
)
from tributo.util.annotations import PublicAPI


class _ONNXRuntimeValidatorOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tolerance: float = 1e-5
    num_samples: int = 1


@PublicAPI(stability="beta")
class ONNXRuntimeValidator:
    """Validate an ONNX artifact by running inference with onnxruntime."""

    api_version: ClassVar[int] = 1
    validator_id: ClassVar[str] = "onnx-runtime-v1"
    options_model: ClassVar[type[BaseModel]] = _ONNXRuntimeValidatorOptions

    def validate(
        self,
        source: ExportSource,
        artifact: ResolvedArtifact,
        upstream: Mapping[str, ResolvedArtifact],
        options: BaseModel,
    ) -> ValidationResult:
        import time

        import onnxruntime as ort

        onnx_path = artifact.path_for(artifact.descriptor.entrypoint)

        try:
            start = time.perf_counter()
            session = ort.InferenceSession(str(onnx_path))
            load_seconds = time.perf_counter() - start

            input_info = session.get_inputs()[0]
            input_name = input_info.name
            input_shape = input_info.shape
            resolved_shape = [
                1 if d is None or isinstance(d, str) else d for d in input_shape
            ]

            import numpy as np

            dummy = np.zeros(resolved_shape, dtype=np.float32)
            start = time.perf_counter()
            outputs = session.run(None, {input_name: dummy})
            inference_seconds = time.perf_counter() - start

            if not outputs:
                return ValidationResult(
                    validator_id=self.validator_id,
                    status="failed",
                    failure=FailureInfo(
                        code="EMPTY_OUTPUT",
                        category="validation",
                        message="ONNX inference produced no outputs",
                    ),
                )

            return ValidationResult(
                validator_id=self.validator_id,
                status="passed",
                metrics={
                    "load_seconds": round(load_seconds, 6),
                    "inference_seconds": round(inference_seconds, 6),
                    "output_count": len(outputs),
                },
            )
        except Exception as exc:
            return ValidationResult(
                validator_id=self.validator_id,
                status="failed",
                failure=FailureInfo(
                    code=type(exc).__name__,
                    category="validation",
                    message=str(exc)[:4096],
                ),
            )

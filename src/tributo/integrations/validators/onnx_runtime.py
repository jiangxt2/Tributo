"""ONNX Runtime validator — real roundtrip inference check."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from pydantic import BaseModel, ConfigDict, Field

from tributo.exporting.models import (
    ExportSource,
    FailureInfo,
    ResolvedArtifact,
    ValidationResult,
)
from tributo.util.annotations import PublicAPI


class _ONNXRuntimeValidatorOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_samples: int = Field(default=1, ge=1)


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

            import numpy as np

            num_samples = int(getattr(options, "num_samples", 1))
            feed: dict[str, Any] = {}
            for input_info in session.get_inputs():
                shape = [
                    (
                        num_samples
                        if index == 0 and (dim is None or isinstance(dim, str))
                        else 1
                    )
                    if dim is None or isinstance(dim, str)
                    else dim
                    for index, dim in enumerate(input_info.shape)
                ]
                feed[input_info.name] = _dummy_input(
                    np,
                    input_info.type,
                    shape,
                )

            start = time.perf_counter()
            outputs = session.run(None, feed)
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
                    "input_count": len(feed),
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


def _dummy_input(np: Any, onnx_type: str, shape: list[int]) -> Any:
    """Build a deterministic ONNX Runtime input for one tensor signature."""
    prefix = "tensor("
    if not onnx_type.startswith(prefix) or not onnx_type.endswith(")"):
        raise TypeError(f"Unsupported ONNX input type: {onnx_type}")

    element_type = onnx_type[len(prefix) : -1]
    dtype_by_type = {
        "bool": np.bool_,
        "double": np.float64,
        "float": np.float32,
        "float16": np.float16,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
    }
    if element_type == "string":
        return np.full(shape, "", dtype=np.object_)
    if element_type not in dtype_by_type:
        raise TypeError(f"Unsupported ONNX tensor element type: {element_type}")
    return np.zeros(shape, dtype=dtype_by_type[element_type])

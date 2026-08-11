"""ONNX Runtime bundle model flavor.

Loads the artifact entrypoint (``model.onnx``) into an in-memory
``onnxruntime.InferenceSession``.  The session owns its weights after
loading, so prediction keeps working even after the bundle runtime's
temp files are closed — this is the ``close-after-load`` contract that
the serving runtime relies on.

Security mode is ``safe``: ONNX Runtime executes the model graph only —
no pickle payloads, no arbitrary Python code.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from tributo.exceptions import ModelLoadError
from tributo.exporting.models import ResolvedArtifact
from tributo.exporting.runtime import (
    SECURITY_MODE_SAFE,
    BundleModel,
)
from tributo.util.annotations import PublicAPI

__all__ = ["ONNXRuntimeFlavor"]


@PublicAPI(stability="beta")
class ONNXRuntimeFlavor:
    """Loads ``onnx-runtime-v1`` artifacts into an ONNX Runtime session."""

    api_version: ClassVar[int] = 1
    flavor_id: ClassVar[str] = "onnx-runtime-v1"
    supported_formats: ClassVar[tuple[str, ...]] = ("onnx",)
    batch_supported: ClassVar[bool] = True
    serveable: ClassVar[bool] = True
    security_mode: ClassVar[str] = SECURITY_MODE_SAFE
    signature_required: ClassVar[bool] = True
    required_dependencies: ClassVar[tuple[str, ...]] = ("onnxruntime",)

    def load(
        self,
        artifact: ResolvedArtifact,
        *,
        role: str,
        unsafe: bool = False,
        architecture_id: str | None = None,
    ) -> BundleModel:
        """Build an in-memory ONNX model from the artifact entrypoint."""
        del unsafe, architecture_id  # Safe flavor — no rebuilding needed.
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ModelLoadError(
                "onnxruntime is required to serve onnx-runtime-v1 bundles. "
                "Install with: uv sync"
            ) from e

        entrypoint = str(artifact.entrypoint_path)
        try:
            session = ort.InferenceSession(entrypoint)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load ONNX model {entrypoint!r}: {e}"
            ) from e

        logger = __import__("logging").getLogger(__name__)
        logger.info(
            "Loaded onnx-runtime-v1 model for role=%r: inputs=%s outputs=%s",
            role,
            [i.name for i in session.get_inputs()],
            [o.name for o in session.get_outputs()],
        )
        return _ONNXRuntimeModel(session)


#: ONNX Runtime type strings (``inp.type``) → framework-neutral names.
_ONNX_TYPE_TO_CANONICAL: dict[str, str] = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
    "tensor(bool)": "bool",
    "tensor(string)": "string",
    "tensor(float16)": "float16",
    "tensor(uint8)": "uint8",
    "tensor(int8)": "int8",
}


def _onnx_dtype_to_canonical(onnx_type: str) -> str:
    """Map an ONNX Runtime type string to a framework-neutral dtype name."""
    if onnx_type in _ONNX_TYPE_TO_CANONICAL:
        return _ONNX_TYPE_TO_CANONICAL[onnx_type]
    return onnx_type  # Unknown types pass through for diagnostic clarity.


class _ONNXRuntimeModel:
    """In-memory ONNX Runtime session adapted to the ``BundleModel`` protocol."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._input_names = tuple(inp.name for inp in session.get_inputs())
        self._output_names = tuple(
            out.name if out.name else f"output_{i}"
            for i, out in enumerate(session.get_outputs())
        )
        self._input_dtypes = tuple(
            _onnx_dtype_to_canonical(inp.type) for inp in session.get_inputs()
        )
        self._output_dtypes = tuple(
            _onnx_dtype_to_canonical(out.type) for out in session.get_outputs()
        )
        # ONNX dynamic dims arrive as None or strings; both become None.
        self._input_shapes = tuple(
            tuple(dim if isinstance(dim, int) else None for dim in inp.shape)
            for inp in session.get_inputs()
        )
        self._output_shapes = tuple(
            tuple(dim if isinstance(dim, int) else None for dim in out.shape)
            for out in session.get_outputs()
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def input_dtypes(self) -> tuple[str, ...]:
        return self._input_dtypes

    @property
    def output_dtypes(self) -> tuple[str, ...]:
        return self._output_dtypes

    @property
    def input_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        return self._input_shapes

    @property
    def output_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        return self._output_shapes

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = [name for name in self._input_names if name not in inputs]
        if missing:
            raise ModelLoadError(
                f"ONNX model expects inputs {self._input_names!r} but got "
                f"{sorted(inputs)}; missing: {missing}"
            )
        outputs = self._session.run(None, inputs)
        return {
            name: np.asarray(output)
            for name, output in zip(self._output_names, outputs)
        }

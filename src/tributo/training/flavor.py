"""Model flavor abstraction — unified training/export/inference lifecycle.

A *model flavor* wraps a trained model with standard ``save``, ``load``,
and ``predict`` methods, enabling a consistent workflow regardless of the
underlying framework (XGBoost, ONNX, PyTorch, etc.).

.. code-block:: python

    from tributo.training.flavor import ModelFlavor
    from tributo.util.annotations import PublicAPI

    class MyFlavor(ModelFlavor):
        @classmethod
        def save(cls, model, path: str) -> None:
            model.save_model(path)

        @classmethod
        def load(cls, path: str):
            return load_model(path)

        def predict(self, input_data):
            return self._model.predict(input_data)

The flavor abstraction is inspired by MLflow's flavor system but kept
minimal and framework-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class ModelFlavor(ABC):
    """Abstract base class for model flavors.

    Each concrete flavor teaches Tributo how to save, load, and run
    inference for a specific model type (e.g., XGBoost, ONNX).

    Subclasses must implement :meth:`save`, :meth:`load`, and
    :meth:`predict`.
    """

    #: Python packages required for this flavor.  Set on concrete subclasses
    #: so that CLI / runner can emit a helpful install hint when dependencies
    #: are missing.
    _REQUIRED_PACKAGES: ClassVar[list[str]] = []

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    @abstractmethod
    def save(cls, model: Any, path: str) -> None:
        """Persist *model* to *path*."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Any:
        """Load a model from *path* and return a flavor instance."""
        ...

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run inference on *input_data* and return predictions."""
        ...

    @property
    def model(self) -> Any:
        """Return the wrapped model object."""
        return self._model

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={type(self._model).__name__})"


# ── ONNX flavor ─────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ONNXFlavor(ModelFlavor):
    """Model flavor for ONNX Runtime models."""

    _REQUIRED_PACKAGES: ClassVar[list[str]] = ["onnxruntime"]

    @classmethod
    def save(cls, model: Any, path: str) -> None:
        """Save ONNX model to *path*."""
        import onnx

        onnx.save_model(model, path)

    @classmethod
    def load(cls, path: str) -> ONNXFlavor:
        """Load an ONNX model from *path*."""
        import onnxruntime as ort

        session = ort.InferenceSession(path)
        return cls(session)

    def predict(self, input_data: Any) -> Any:
        """Run inference with ONNX Runtime.

        *input_data* is a dict mapping input names to numpy arrays.
        """
        input_feed = {
            inp.name: input_data[inp.name] for inp in self._model.get_inputs()
        }
        return self._model.run(None, input_feed)

"""BasePredictor abstract base class.

Defines a unified interface for batch inference. Subclasses only need to implement _load_model and __call__.
Consistent with Ray Trainer/Predictor pattern: ABC subclassing + explicit Python imports, no registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class BasePredictor(ABC):
    """Base class for batch inference Predictor.

    Subclasses must implement:
    - ``_load_model()``: Load the model (from local path or S3 download).
    - ``__call__(batch)``: Run inference on a single batch, returning a result dict.

    Optional overrides:
    - ``get_feature_names(model_uri, predictor_config)``: Read feature names from model metadata.

    Usage example::

        predictor = MyPredictor(model_uri="s3://bucket/model.onnx", predictor_config={...})
        result = predictor(batch)
    """

    def __init__(
        self,
        model_uri: str,
        predictor_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Predictor and load the model.

        Note: ``__init__`` calls ``_load_model()`` at the end. If a subclass needs to
        set instance attributes (e.g., parse config from predictor_config) before
        ``_load_model()`` is called, it must do so before calling ``super().__init__()``,
        otherwise accessing unset attributes in ``_load_model()`` will raise ``AttributeError``.

        Args:
            model_uri: Model path (local path or s3:// URI).
            predictor_config: Predictor-specific config dict, structure defined by the subclass.
        """
        self.model_uri = model_uri
        self.predictor_config = predictor_config or {}
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load model resources (ONNX session, PyTorch model, etc.)."""

    @abstractmethod
    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on a single batch.

        Args:
            batch: Input batch, keys are column names, values are numpy arrays.

        Returns:
            Result dict containing at least the original columns + prediction column.
        """

    @classmethod
    def get_feature_names(
        cls,
        model_uri: str | None = None,
        predictor_config: dict[str, Any] | None = None,
        *,
        bundle_uri: str | None = None,
        role: str = "inference",
        unsafe: bool = False,
        storage_profile: str | None = None,
    ) -> list[str]:
        """Read feature names from a bundle or model metadata.

        Args:
            model_uri: Model path (legacy entry).
            predictor_config: Predictor-specific config.
            bundle_uri: Published bundle URI (stable entry); feature names
                come from the model's ONNX metadata, falling back to the
                manifest's typed input signature.
            role: Artifact role to serve from the bundle.
            unsafe: Permit bundles without typed signatures or unsafe flavors.
            storage_profile: Storage profile name for S3 bundles.

        Returns:
            List of feature names, default returns empty list.
        """
        return []

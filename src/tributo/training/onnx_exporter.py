"""XGBoost Booster -> ONNX export with onnxruntime inference validation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray
    import xgboost

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
def export_to_onnx(
    booster: "xgboost.Booster",
    n_features: int,
    output_path: str,
    *,
    target_opset: int = 12,
    validate: bool = True,
) -> None:
    """Export an XGBoost Booster to an ONNX file.

    Args:
        booster: Trained XGBoost Booster object.
        n_features: Number of input features, used to define the ONNX input shape.
        output_path: Export path, e.g. ``/tmp/model.onnx``.
        target_opset: ONNX opset version, default 12 (compatible with onnxmltools version constraints inside the container).
        validate: Run a validation inference with onnxruntime after export, enabled by default.

    Raises:
        ImportError: onnxmltools or onnxruntime is not installed.
        RuntimeError: ONNX validation inference failed.
    """
    try:
        import xgboost
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as e:
        raise ImportError(
            "onnxmltools and xgboost are required. "
            "Install with: uv sync --extra training"
        ) from e

    # onnxmltools requires a sklearn wrapper, wrap the Booster with XGBClassifier.
    # Save original feature names to ONNX metadata for automatic column identification on the inference side
    original_names = booster.feature_names
    if original_names is None:
        original_names = [f"f{i}" for i in range(n_features)]
    import json

    feature_names_json = json.dumps(original_names)

    # onnxmltools requires feature_names to be f%d format, temporarily rename
    booster.feature_names = [f"f{i}" for i in range(n_features)]

    wrapper = xgboost.XGBClassifier()
    wrapper._Booster = booster  # noqa: SLF001

    # Inject n_classes_ and classes_ attributes (required by onnxmltools shape calculator)
    import numpy as np

    n_classes = int(booster.attr("num_class") or 0) or 2
    wrapper.__dict__["n_classes_"] = n_classes
    wrapper.__dict__["classes_"] = np.arange(n_classes)

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(
        wrapper,
        initial_types=initial_types,
        target_opset=target_opset,
    )

    # Restore original feature names and write to metadata
    booster.feature_names = original_names
    meta = onnx_model.metadata_props.add()
    meta.key = "feature_names"
    meta.value = feature_names_json

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("ONNX model saved to %s", output_path)

    if validate:
        _validate_onnx(output_path, n_features)


def _validate_onnx(output_path: str, n_features: int) -> None:
    """Run a dummy inference with onnxruntime to verify the exported file is usable."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for validation. "
            "Install with: uv sync --extra training"
        ) from e

    try:
        session = ort.InferenceSession(output_path)
        dummy = np.zeros((1, n_features), dtype=np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: dummy})
        if not outputs:
            raise RuntimeError("ONNX validation failed: empty output from onnxruntime")
        logger.info(
            "ONNX validation passed, output shapes: %s", [o.shape for o in outputs]
        )
    except Exception as exc:
        # Fail-closed: an unverifiable model is a failure, not a warning.
        # Silently swallowing the exception let unusable models through —
        # job scripts relying on validate=True never learned of the broken
        # export.
        logger.error(
            "ONNX validation failed — the exported model is not usable with "
            "the installed onnxruntime. Path: %s",
            output_path,
            exc_info=True,
        )
        raise RuntimeError(f"ONNX validation failed for {output_path}: {exc}") from exc


def export_from_checkpoint(
    checkpoint: "ray.train.Checkpoint",
    output_path: str,
    n_features: int,
    *,
    target_opset: int = 12,
    validate: bool = True,
) -> str:
    """Export an ONNX model from a Ray Train Checkpoint (Driver-side call).

    Args:
        checkpoint: Ray Train Checkpoint object.
        output_path: ONNX file save path.
        n_features: Number of input features.
        target_opset: ONNX opset version, default 12 (compatible with onnxmltools version constraints inside the container).
        validate: Whether to validate the exported model, enabled by default.

    Returns:
        ONNX file path.

    Raises:
        ImportError: ray.train.xgboost is not installed.
        RuntimeError: ONNX validation inference failed.
    """
    try:
        from ray.train.xgboost import XGBoostCheckpoint
    except ImportError as e:
        raise ImportError(
            "ray[train] is required. Install with: uv sync --extra training"
        ) from e

    # Convert generic Checkpoint to XGBoostCheckpoint
    xgb_checkpoint = XGBoostCheckpoint.from_directory(checkpoint.to_directory())
    booster = xgb_checkpoint.get_model()  # type: ignore[attr-defined]

    export_to_onnx(
        booster=booster,
        n_features=n_features,
        output_path=output_path,
        target_opset=target_opset,
        validate=validate,
    )
    return output_path

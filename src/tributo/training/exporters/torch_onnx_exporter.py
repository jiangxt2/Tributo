"""PyTorch -> ONNX model exporter.

Exports PyTorch models to ONNX format, supporting dynamic axes and custom inputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from tributo.util.annotations import PublicAPI

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
@PublicAPI(stability="beta")
def export_pytorch_to_onnx(
    model: Any,
    sample_inputs: dict[str, np.ndarray],
    output_path: str | Path,
    opset_version: int = 12,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model instance.
        sample_inputs: Sample input dictionary, key is feature name, value is numpy array.
        output_path: ONNX file output path.
        opset_version: ONNX opset version.
        input_names: List of input names (optional).
        output_names: List of output names (optional).
        dynamic_axes: Dynamic axes configuration (optional).

    Returns:
        Path to the exported ONNX file.

    Raises:
        ImportError: If PyTorch is not installed.
        RuntimeError: If export fails.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for ONNX export. Install with: pip install torch"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    # Prepare input tensors
    torch_inputs = {}
    for name, value in sample_inputs.items():
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.integer):
                torch_inputs[name] = torch.tensor(value, dtype=torch.long)
            else:
                torch_inputs[name] = torch.tensor(value, dtype=torch.float32)
        else:
            torch_inputs[name] = value

    # If input names not specified, use feature names
    if input_names is None:
        input_names = list(torch_inputs.keys())

    # Default output names
    if output_names is None:
        output_names = ["output"]

    # Default dynamic axes configuration (batch dimension)
    if dynamic_axes is None:
        dynamic_axes = {name: {0: "batch_size"} for name in input_names}
        dynamic_axes["output"] = {0: "batch_size"}

    try:
        # Create wrapper class to convert dict interface to tuple interface
        class ONNXWrapper(torch.nn.Module):
            """ONNX export wrapper that converts dict input to tuple input."""

            def __init__(self, model: torch.nn.Module, input_names: list[str]):
                super().__init__()
                self.model = model
                self.input_names = input_names

            def forward(self, *args: torch.Tensor) -> torch.Tensor:
                """Convert positional args to dict then call original model."""
                inputs = dict(zip(self.input_names, args))
                return self.model(inputs)

        # Wrap model
        wrapper = ONNXWrapper(model, input_names)

        # Prepare input tuple
        input_tuple = tuple(torch_inputs[name] for name in input_names)

        # Export ONNX
        # dynamo=False for compatibility with legacy dynamic_axes API
        torch.onnx.export(
            wrapper,
            input_tuple,
            str(output_path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )

        logger.info("Exported ONNX model to %s", output_path)
        return output_path

    except Exception as e:
        raise RuntimeError(f"Failed to export ONNX model: {e}") from e


def export_model_package(
    model: Any,
    sample_inputs: dict[str, np.ndarray],
    output_dir: str | Path,
    feature_config: dict[str, Any],
    preprocessor_state: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    opset_version: int = 12,
) -> dict[str, Path]:
    """Export a complete model package.

    Includes ONNX model, feature configuration, preprocessor state and training metrics.

    Args:
        model: PyTorch model instance.
        sample_inputs: Sample input dictionary.
        output_dir: Output directory.
        feature_config: Feature column configuration.
        preprocessor_state: Preprocessor state.
        metrics: Training metrics (optional).
        opset_version: ONNX opset version.

    Returns:
        Dictionary containing paths to each file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export ONNX model
    onnx_path = export_pytorch_to_onnx(
        model=model,
        sample_inputs=sample_inputs,
        output_path=output_dir / "model.onnx",
        opset_version=opset_version,
    )

    # Save feature config
    feature_config_path = output_dir / "feature_config.json"
    feature_config_path.write_text(
        json.dumps(feature_config, indent=2, ensure_ascii=False)
    )

    # Save preprocessor state
    preprocessor_path = output_dir / "preprocessor.json"
    preprocessor_path.write_text(
        json.dumps(preprocessor_state, indent=2, ensure_ascii=False)
    )

    # Save training metrics
    metrics_path = output_dir / "metrics.json"
    if metrics is not None:
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str)
        )

    result = {
        "onnx_model": onnx_path,
        "feature_config": feature_config_path,
        "preprocessor": preprocessor_path,
    }
    if metrics is not None:
        result["metrics"] = metrics_path

    logger.info("Exported model package to %s", output_dir)
    return result

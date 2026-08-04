"""Shared assertions for exported ONNX signatures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _normalise_shape(shape: Any) -> tuple[int | None, ...]:
    """Normalise ONNX and Manifest dynamic dimensions for comparison."""
    return tuple(
        None if dim is None or isinstance(dim, str) else int(dim) for dim in shape
    )


def _assert_onnx_signature_matches_manifest(
    onnx_path: Path,
    manifest: Any,
) -> None:
    """Compare the actual ONNX names/shapes with the typed Manifest."""
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(onnx_path))

    manifest_inputs = manifest.input_signature.input_fields
    actual_inputs = session.get_inputs()
    assert len(actual_inputs) == len(manifest_inputs)
    for actual, expected in zip(actual_inputs, manifest_inputs, strict=True):
        assert actual.name == expected.name
        assert _normalise_shape(actual.shape) == _normalise_shape(expected.shape)

    manifest_outputs = manifest.output_signature.output_fields
    actual_outputs = session.get_outputs()
    assert len(actual_outputs) == len(manifest_outputs)
    for actual, expected in zip(actual_outputs, manifest_outputs, strict=True):
        assert actual.name == expected.name
        assert _normalise_shape(actual.shape) == _normalise_shape(expected.shape)

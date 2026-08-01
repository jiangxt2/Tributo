"""Tests for ``ONNXQuantizer.supports()`` dependency probing.

Regression coverage for the dotted ``find_spec`` crash (Phase 0-a hotfix):
probing ``onnxruntime.quantization`` with the parent package missing raises
``ModuleNotFoundError`` instead of returning ``None``, so ``supports()``
crashed instead of reporting ``MISSING_DEPENDENCY``.
"""

from __future__ import annotations

from unittest.mock import patch

from tributo.exporting.models import SupportRequest
from tributo.integrations.exporters.onnx_quantizer import ONNXQuantizer


def test_supports_without_onnxruntime_reports_missing_dependency() -> None:
    """Top-level package missing → MISSING_DEPENDENCY, no exception."""

    def fake_find_spec(name: str) -> object | None:
        # Mirror real find_spec(): probing a dotted name imports the parent
        # first, so a missing onnxruntime raises ModuleNotFoundError instead
        # of returning None. The old implementation probed the dotted name
        # and crashed here, so this side_effect catches that regression.
        if name == "onnxruntime.quantization":
            raise ModuleNotFoundError(
                "No module named 'onnxruntime'", name="onnxruntime"
            )
        if name == "onnxruntime":
            return None
        raise AssertionError(f"unexpected probe: {name}")

    request = SupportRequest(source_kind="dnn_result", upstream_formats=("onnx",))
    with patch(
        "importlib.util.find_spec", side_effect=fake_find_spec
    ) as mock_find_spec:
        result = ONNXQuantizer.supports(request)
    mock_find_spec.assert_called_once_with("onnxruntime")
    assert not result.supported
    assert result.code == "MISSING_DEPENDENCY"
    assert result.missing_dependencies == ("onnxruntime",)

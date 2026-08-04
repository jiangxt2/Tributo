"""Tests for ``ONNXQuantizer.supports()`` dependency probing.

The dotted ``find_spec`` crash (PR #17) is guarded at the unified layer —
``tests/test_dependencies.py`` asserts top-level-only probing and the
ModuleNotFoundError → MISSING mapping. Here we verify the quantizer's
``supports()`` surface: missing onnxruntime reports ``MISSING_DEPENDENCY``
without raising.
"""

from __future__ import annotations

from unittest.mock import patch

from tributo._common.dependencies import ONNXRUNTIME, DependencyState, DependencyStatus
from tributo.exporting.models import SupportRequest
from tributo.integrations.exporters.onnx_quantizer import ONNXQuantizer


def test_supports_without_onnxruntime_reports_missing_dependency() -> None:
    """onnxruntime missing → MISSING_DEPENDENCY, no exception."""
    request = SupportRequest(source_kind="dnn_result", upstream_formats=("onnx",))
    with patch(
        "tributo.integrations.exporters.onnx_quantizer.probe_dependency",
        return_value=DependencyStatus(ONNXRUNTIME, DependencyState.MISSING),
    ):
        result = ONNXQuantizer.supports(request)
    assert not result.supported
    assert result.code == "MISSING_DEPENDENCY"
    assert result.missing_dependencies == ("onnxruntime",)

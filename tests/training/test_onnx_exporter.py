"""ONNX exporter fail-closed validation tests (T3 Core, review P1-10).

``export_from_checkpoint``/``export_to_onnx`` document that ONNX validation
failures raise ``RuntimeError``.  The validator used to swallow inference
errors (warning only), which let unusable models pass silently — job scripts
relying on ``validate=True`` never learned of the broken export.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest


class TestOnnxValidationFailClosed:
    """_validate_onnx 失败必须传播（fail-closed），不得吞异常。"""

    def test_inference_failure_raises_runtime_error(self, monkeypatch):
        """onnxruntime 推理异常 → RuntimeError 传播，而不是 warning 放行。"""
        from tributo.training.onnx_exporter import _validate_onnx

        class BrokenSession:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("inference backend crashed")

        fake_ort = ModuleType("onnxruntime")
        fake_ort.InferenceSession = BrokenSession
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

        with pytest.raises(RuntimeError, match="ONNX validation failed"):
            _validate_onnx("/tmp/model.onnx", n_features=5)

    def test_empty_output_raises_runtime_error(self, monkeypatch):
        """onnxruntime 返回空输出 → RuntimeError（无输出 = 模型不可用）。"""
        from tributo.training.onnx_exporter import _validate_onnx

        class EmptySession:
            def __init__(self, *args, **kwargs):
                pass

            def get_inputs(self):
                return [type("In", (), {"name": "float_input"})()]

            def run(self, *args, **kwargs):
                return []

        fake_ort = ModuleType("onnxruntime")
        fake_ort.InferenceSession = EmptySession
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

        with pytest.raises(RuntimeError, match="empty output"):
            _validate_onnx("/tmp/model.onnx", n_features=5)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))

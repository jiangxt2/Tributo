"""ONNX exporter fail-closed validation tests.

``export_from_checkpoint``/``export_to_onnx`` document that ONNX validation
failures raise ``RuntimeError``.  The validator used to swallow inference
errors (warning only), which let unusable models pass silently — job scripts
relying on ``validate=True`` never learned of the broken export.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

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

    def test_batch_dimension_mismatch_raises(self, monkeypatch):
        """输出 batch 维度与输入不一致 → RuntimeError（模型不可用于打分）。"""
        from tributo.training.onnx_exporter import _validate_onnx

        class WrongBatchSession:
            def __init__(self, *args, **kwargs):
                pass

            def get_inputs(self):
                return [type("In", (), {"name": "float_input"})()]

            def run(self, *args, **kwargs):
                # 输入 1 行，输出 2 行 → batch 维度不匹配
                import numpy as np

                return [np.zeros((2, 3), dtype=np.float32)]

        fake_ort = ModuleType("onnxruntime")
        fake_ort.InferenceSession = WrongBatchSession
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

        with pytest.raises(RuntimeError, match="batch dimension"):
            _validate_onnx("/tmp/model.onnx", n_features=5)


class TestTorchModelPackageMetrics:
    """Training metrics must be valid JSON before model export starts."""

    def test_non_finite_metrics_fail_before_model_export(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        from tributo.training.exporters import torch_onnx_exporter

        export_called = False

        def fake_export(**kwargs: Any) -> Any:
            nonlocal export_called
            export_called = True
            return tmp_path / "model.onnx"

        monkeypatch.setattr(
            torch_onnx_exporter,
            "export_pytorch_to_onnx",
            fake_export,
        )

        with pytest.raises(ValueError, match="Out of range float values"):
            torch_onnx_exporter.export_model_package(
                model=object(),
                sample_inputs={},
                output_dir=tmp_path / "package",
                feature_config={},
                preprocessor_state={},
                metrics={"train_loss": float("nan")},
            )

        assert export_called is False
        assert not (tmp_path / "package").exists()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))

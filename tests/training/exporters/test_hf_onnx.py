"""Focused tests for the HuggingFace ONNX exporter boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tributo.exceptions import JobConfigurationError
from tributo.exporting.models import (
    ExportContext,
    ExportSource,
    ExportTarget,
    PlannedTarget,
)
from tributo.integrations.exporters.hf_onnx import (
    HuggingFaceONNXExporter,
    _export_with_transformers_onnx,
)


def test_transformers_export_receives_source_preprocessor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preprocessor = object()
    captured: dict[str, Any] = {}

    class _FeaturesManager:
        @staticmethod
        def check_supported_model_or_raise(
            model: object,
            *,
            feature: str,
        ) -> tuple[str, Any]:
            del model, feature
            return "fake", lambda config: ("onnx-config", config)

    def fake_export(**kwargs: Any) -> None:
        captured.update(kwargs)

    class _TransformersModule(ModuleType):
        __path__: list[str]

    class _TransformersONNXModule(ModuleType):
        FeaturesManager: Any
        export: Any

    transformers = _TransformersModule("transformers")
    transformers.__path__ = []
    transformers_onnx = _TransformersONNXModule("transformers.onnx")
    transformers_onnx.FeaturesManager = _FeaturesManager
    transformers_onnx.export = fake_export
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.onnx", transformers_onnx)
    monkeypatch.setattr(
        "tributo.integrations.exporters.hf_onnx.require_dependency",
        lambda dependency: None,
    )
    model = SimpleNamespace(config=object())

    output = _export_with_transformers_onnx(
        model,
        preprocessor,
        "default",
        14,
        tmp_path,
    )

    assert output == tmp_path / "model.onnx"
    assert captured["preprocessor"] is preprocessor
    assert captured["model"] is model

    with pytest.raises(JobConfigurationError, match="requires a preprocessor"):
        _export_with_transformers_onnx(model, None, "default", 14, tmp_path)


def test_direct_torch_fallback_does_not_require_preprocessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = ExportTarget(name="model", format="onnx")
    planned = PlannedTarget(
        target=target,
        exporter_id=HuggingFaceONNXExporter.exporter_id,
    )
    context = ExportContext(
        execution_id="execution-1",
        node_id="model",
        artifact_dir=tmp_path,
    )

    output = tmp_path / "model.onnx"
    monkeypatch.setattr(
        "tributo.integrations.exporters.hf_onnx.require_dependency",
        lambda dependency: SimpleNamespace(__version__="test"),
    )
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "transformers.onnx", None)

    def fake_torch_export(model: object, artifact_dir: Path, opset: int) -> Path:
        del model, artifact_dir, opset
        output.write_bytes(b"onnx")
        return output

    monkeypatch.setattr(
        "tributo.integrations.exporters.hf_onnx._export_torch_onnx_fallback",
        fake_torch_export,
    )

    draft = HuggingFaceONNXExporter().export(
        context,
        ExportSource(source_kind="hf_model", model_object=object()),
        {},
        planned,
    )

    assert draft.entrypoint == "model.onnx"

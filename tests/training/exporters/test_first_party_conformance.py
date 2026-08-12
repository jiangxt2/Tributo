"""Unified conformance suites for every first-party exporter and validator."""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tributo.exporting.conftest import (
    ExporterConformanceTest,
    ValidatorConformanceTest,
)
from tributo.exporting.models import (
    ArtifactFile,
    ExportContext,
    ExportSource,
    ExportTarget,
    LogicalArtifact,
    ProducerInfo,
    ResolvedArtifact,
    SupportRequest,
)
from tributo.exporting.validators import StructureValidator
from tributo.integrations.exporters.hf_onnx import HuggingFaceONNXExporter
from tributo.integrations.exporters.onnx_quantizer import ONNXQuantizer
from tributo.integrations.exporters.torch_export import TorchExportExporter
from tributo.integrations.exporters.torch_onnx import TorchONNXExporter
from tributo.integrations.exporters.torch_safetensors import (
    TorchSafetensorsExporter,
)
from tributo.integrations.exporters.xgboost_native import (
    XGBoostJSONExporter,
    XGBoostUBJExporter,
)
from tributo.integrations.exporters.xgboost_onnx import XGBoostONNXExporter
from tributo.integrations.validators.onnx_runtime import ONNXRuntimeValidator

pytestmark = pytest.mark.integration


class _ExporterFixture:
    target_format = "onnx"
    target_options: dict[str, Any] = {}

    def make_target(self) -> ExportTarget:
        return ExportTarget(
            name="model",
            format=self.target_format,
            options=self.target_options,
        )

    def make_context(self, tmp_path: Path) -> ExportContext:
        return ExportContext(
            execution_id="conformance-execution",
            node_id="model",
            artifact_dir=tmp_path,
        )


@cache
def _xgboost_booster() -> Any:
    np = pytest.importorskip("numpy")
    xgboost = pytest.importorskip("xgboost")
    rng = np.random.default_rng(19)
    features = rng.random((40, 4)).astype(np.float32)
    labels = (features[:, 0] > 0.5).astype(np.int64)
    matrix = xgboost.DMatrix(
        features,
        label=labels,
        feature_names=[f"f{i}" for i in range(4)],
    )
    return xgboost.train(
        {"objective": "binary:logistic", "max_depth": 2, "nthread": 1},
        matrix,
        num_boost_round=2,
    )


class _XGBoostFixture(_ExporterFixture):
    def make_source(self) -> ExportSource:
        return ExportSource(
            source_kind="xgboost_result",
            model_object=_xgboost_booster(),
            feature_schema={"feature_names": [f"f{i}" for i in range(4)]},
            metadata={
                "objective": "binary:logistic",
                "has_categorical_features": False,
            },
        )


class TestXGBoostONNXConformance(_XGBoostFixture, ExporterConformanceTest):
    exporter_cls = XGBoostONNXExporter
    target_options = {"opset": 12}


class TestXGBoostUBJConformance(_XGBoostFixture, ExporterConformanceTest):
    exporter_cls = XGBoostUBJExporter
    target_format = "ubj"


class TestXGBoostJSONConformance(_XGBoostFixture, ExporterConformanceTest):
    exporter_cls = XGBoostJSONExporter
    target_format = "xgboost-json"


def _torch_source(*, source_kind: str = "torch_module") -> ExportSource:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    return ExportSource(
        source_kind=source_kind,
        model_object=model,
        model_config_data={"input_dim": 4},
        feature_schema={"feature_names": ["features"]},
        sample_inputs={"features": torch.zeros(2, 4)},
    )


class _TorchFixture(_ExporterFixture):
    def make_source(self) -> ExportSource:
        return _torch_source()


class TestTorchONNXConformance(_TorchFixture, ExporterConformanceTest):
    exporter_cls = TorchONNXExporter
    target_options = {"opset": 18, "dynamo": False}

    def test_dnn_export_requires_preprocessing_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeModule:
            pass

        source = ExportSource(
            source_kind="dnn_result",
            model_object=_FakeModule(),
        )
        target = self.make_target()
        planned = SimpleNamespace(
            target=target,
            typed_options={"opset": 18, "dynamo": False},
        )
        monkeypatch.setattr(
            "tributo.integrations.exporters.torch_onnx.require_dependency",
            lambda _dependency: SimpleNamespace(nn=SimpleNamespace(Module=_FakeModule)),
        )

        with pytest.raises(ValueError, match="requires non-empty preprocessing_state"):
            TorchONNXExporter().export(self.make_context(tmp_path), source, {}, planned)

    def test_dnn_export_requires_model_config_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeModule:
            pass

        source = ExportSource(
            source_kind="dnn_result",
            model_object=_FakeModule(),
            preprocessing_state={
                "features": [{"name": "age"}],
                "label_encoders": {},
                "norm_params": {},
            },
        )
        planned = SimpleNamespace(
            target=self.make_target(),
            typed_options={"opset": 18, "dynamo": False},
        )
        monkeypatch.setattr(
            "tributo.integrations.exporters.torch_onnx.require_dependency",
            lambda _dependency: SimpleNamespace(nn=SimpleNamespace(Module=_FakeModule)),
        )

        with pytest.raises(ValueError, match="requires non-empty model_config_data"):
            TorchONNXExporter().export(self.make_context(tmp_path), source, {}, planned)

    @pytest.mark.parametrize(
        "preprocessing_state, expected_message",
        (
            (
                {"features": [{"name": "age"}], "label_encoders": {}},
                "missing required key",
            ),
            (
                {"features": [], "label_encoders": {}, "norm_params": {}},
                "features.*must not be empty",
            ),
            (
                {
                    "features": [{"dimension": 1}],
                    "label_encoders": {},
                    "norm_params": {},
                },
                "named feature objects",
            ),
        ),
    )
    def test_dnn_export_rejects_malformed_preprocessing_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        preprocessing_state: dict[str, Any],
        expected_message: str,
    ) -> None:
        class _FakeModule:
            pass

        source = ExportSource(
            source_kind="dnn_result",
            model_object=_FakeModule(),
            model_config_data={"features": [{"name": "age"}]},
            preprocessing_state=preprocessing_state,
        )
        planned = SimpleNamespace(
            target=self.make_target(),
            typed_options={"opset": 18, "dynamo": False},
        )
        monkeypatch.setattr(
            "tributo.integrations.exporters.torch_onnx.require_dependency",
            lambda _dependency: SimpleNamespace(nn=SimpleNamespace(Module=_FakeModule)),
        )

        with pytest.raises(ValueError, match=expected_message):
            TorchONNXExporter().export(self.make_context(tmp_path), source, {}, planned)

    @pytest.mark.parametrize(
        ("field", "expected_artifact"),
        (
            ("model_config_data", "model_config.json"),
            ("preprocessing_state", "preprocessor.json"),
        ),
    )
    def test_dnn_export_rejects_non_finite_json_artifacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        field: str,
        expected_artifact: str,
    ) -> None:
        class _FakeModule:
            pass

        values: dict[str, Any] = {
            "model_config_data": {"features": [{"name": "age"}]},
            "preprocessing_state": {
                "features": [{"name": "age"}],
                "label_encoders": {},
                "norm_params": {},
            },
        }
        values[field] = {**values[field], "invalid": float("nan")}
        source = ExportSource(
            source_kind="dnn_result",
            model_object=_FakeModule(),
            model_config_data=values["model_config_data"],
            preprocessing_state=values["preprocessing_state"],
        )
        planned = SimpleNamespace(
            target=self.make_target(),
            typed_options={"opset": 18, "dynamo": False},
        )
        monkeypatch.setattr(
            "tributo.integrations.exporters.torch_onnx.require_dependency",
            lambda _dependency: SimpleNamespace(nn=SimpleNamespace(Module=_FakeModule)),
        )

        with pytest.raises(ValueError, match=f"{expected_artifact}.*finite JSON"):
            TorchONNXExporter().export(self.make_context(tmp_path), source, {}, planned)

    def test_dynamo_export_receives_effective_opset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tributo.integrations.exporters import torch_onnx

        captured: dict[str, Any] = {}

        def fake_export(model: Any, inputs: Any, path: str, **kwargs: Any) -> None:
            del model, inputs
            captured.update(kwargs)
            Path(path).write_bytes(b"onnx")

        fake_torch = SimpleNamespace(
            export=SimpleNamespace(Dim=lambda name: name),
            onnx=SimpleNamespace(export=fake_export),
        )
        monkeypatch.setattr(
            torch_onnx,
            "require_dependency",
            lambda dependency: fake_torch,
        )

        TorchONNXExporter._legacy_export(
            object(),
            (object(),),
            ["features"],
            ["output"],
            18,
            tmp_path,
            False,
            use_dynamo=True,
        )

        assert captured["dynamo"] is True
        assert captured["opset_version"] == 18
        assert captured["dynamic_shapes"] == ({0: "batch_size"},)

    def test_dynamo_export_preserves_dynamic_batch(self, tmp_path: Path) -> None:
        onnx = pytest.importorskip("onnx")
        torch = pytest.importorskip("torch")
        model = torch.nn.Sequential(torch.nn.Linear(4, 2)).eval()

        output_path = TorchONNXExporter._legacy_export(
            model,
            (torch.zeros(2, 4),),
            ["features"],
            ["output"],
            18,
            tmp_path,
            False,
            use_dynamo=True,
        )

        exported = onnx.load(output_path)
        batch_dimension = exported.graph.input[0].type.tensor_type.shape.dim[0]
        assert batch_dimension.dim_param == "batch_size"

    def test_sample_inputs_follow_declared_name_order(self) -> None:
        torch = pytest.importorskip("torch")
        from tributo.integrations.exporters.torch_onnx import _resolve_sample_inputs

        features = torch.tensor([[1.0]])
        category = torch.tensor([[2]])
        source = ExportSource(
            source_kind="torch_module",
            sample_inputs={"category": category, "features": features},
        )

        resolved = _resolve_sample_inputs(source, ["features", "category"])

        assert resolved[0] is features
        assert resolved[1] is category

    def test_sample_inputs_must_match_declared_names(self) -> None:
        torch = pytest.importorskip("torch")
        from tributo.integrations.exporters.torch_onnx import _resolve_sample_inputs

        source = ExportSource(
            source_kind="torch_module",
            sample_inputs={"wrong": torch.tensor([[1.0]])},
        )

        with pytest.raises(ValueError, match="missing=.*features.*unexpected=.*wrong"):
            _resolve_sample_inputs(source, ["features"])


class TestTorchSafetensorsConformance(_TorchFixture, ExporterConformanceTest):
    exporter_cls = TorchSafetensorsExporter
    target_format = "safetensors"


class TestTorchExportConformance(_TorchFixture, ExporterConformanceTest):
    exporter_cls = TorchExportExporter
    target_format = "pt2"


class _TinyHFModel:
    @staticmethod
    def build() -> Any:
        torch = pytest.importorskip("torch")

        class _Model(torch.nn.Module):
            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                return (input_ids.float() * attention_mask.float()).sum(
                    dim=1, keepdim=True
                )

        return _Model()


class TestHuggingFaceONNXConformance(_ExporterFixture, ExporterConformanceTest):
    exporter_cls = HuggingFaceONNXExporter
    target_options = {"opset": 18}

    def make_source(self) -> ExportSource:
        return ExportSource(
            source_kind="hf_model",
            model_object=_TinyHFModel.build(),
        )


def _resolved_onnx(root: Path) -> ResolvedArtifact:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.onnx"
    node = helper.make_node("Identity", inputs=["features"], outputs=["prediction"])
    graph = helper.make_graph(
        [node],
        "tributo-conformance",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 4])],
        [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [None, 4])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.save(model, model_path)

    payload = model_path.read_bytes()
    artifact_file = ArtifactFile(
        relative_path="model.onnx",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        role="model",
    )
    descriptor = LogicalArtifact(
        name="model",
        format="onnx",
        flavor_id="onnx-runtime-v1",
        files=(artifact_file,),
        entrypoint="model.onnx",
        tree_digest=LogicalArtifact.compute_tree_digest((artifact_file,)),
        producer=ProducerInfo(exporter_id="conformance-fixture-v1"),
    )
    return ResolvedArtifact(descriptor=descriptor, root_dir=root)


def _resolved_multi_input_onnx(root: Path) -> ResolvedArtifact:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.onnx"
    nodes = [
        helper.make_node(
            "Cast",
            inputs=["category"],
            outputs=["category_float"],
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Add",
            inputs=["features", "category_float"],
            outputs=["prediction"],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "tributo-multi-input-conformance",
        [
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 1]),
            helper.make_tensor_value_info("category", TensorProto.INT64, [None, 1]),
        ],
        [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [None, 1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.save(model, model_path)

    payload = model_path.read_bytes()
    artifact_file = ArtifactFile(
        relative_path="model.onnx",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        role="model",
    )
    descriptor = LogicalArtifact(
        name="model",
        format="onnx",
        flavor_id="onnx-runtime-v1",
        files=(artifact_file,),
        entrypoint="model.onnx",
        tree_digest=LogicalArtifact.compute_tree_digest((artifact_file,)),
        producer=ProducerInfo(exporter_id="multi-input-conformance-v1"),
    )
    return ResolvedArtifact(descriptor=descriptor, root_dir=root)


class TestONNXQuantizerConformance(_ExporterFixture, ExporterConformanceTest):
    exporter_cls = ONNXQuantizer

    def make_source(self) -> ExportSource:
        return ExportSource(source_kind="dnn_result", model_object=object())

    def make_support_request(self, source: ExportSource) -> SupportRequest:
        return SupportRequest(
            source_kind=source.source_kind,
            upstream_formats=("onnx",),
        )

    def make_upstream(self, tmp_path: Path) -> dict[str, ResolvedArtifact]:
        return {"model": _resolved_onnx(tmp_path / "upstream")}


class TestONNXRuntimeValidatorConformance(ValidatorConformanceTest):
    validator_cls = ONNXRuntimeValidator

    def make_source(self) -> ExportSource:
        return ExportSource(source_kind="dnn_result", model_object=object())

    def make_artifact(self, tmp_path: Path) -> ResolvedArtifact:
        return _resolved_onnx(tmp_path)

    def make_invalid_artifact(self, tmp_path: Path) -> ResolvedArtifact:
        artifact = _resolved_onnx(tmp_path)
        artifact.entrypoint_path.write_bytes(b"not-an-onnx-model")
        return artifact

    def test_multi_input_dtypes_and_dynamic_batch_are_validated(
        self, tmp_path: Path
    ) -> None:
        validator = ONNXRuntimeValidator()
        options = validator.options_model(num_samples=3)
        result = validator.validate(
            self.make_source(),
            _resolved_multi_input_onnx(tmp_path),
            {},
            options,
        )

        assert result.status == "passed"
        assert result.metrics["input_count"] == 2


class TestStructureValidatorConformance(ValidatorConformanceTest):
    validator_cls = StructureValidator

    def make_source(self) -> ExportSource:
        return ExportSource(source_kind="dnn_result", model_object=object())

    def make_artifact(self, tmp_path: Path) -> ResolvedArtifact:
        return _resolved_onnx(tmp_path)

    def make_invalid_artifact(self, tmp_path: Path) -> ResolvedArtifact:
        artifact = _resolved_onnx(tmp_path)
        artifact.entrypoint_path.unlink()
        return artifact


def test_serveable_first_party_onnx_validation_is_required() -> None:
    for exporter_cls in (
        TorchONNXExporter,
        XGBoostONNXExporter,
        ONNXQuantizer,
        HuggingFaceONNXExporter,
    ):
        runtime_binding = next(
            binding
            for binding in exporter_cls.validator_bindings
            if binding.validator_id == "onnx-runtime-v1"
        )
        assert runtime_binding.required is True

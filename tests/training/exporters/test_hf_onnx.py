"""Focused tests for the HuggingFace ONNX exporter boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tributo.exceptions import JobConfigurationError
from tributo.exporting.models import (
    DraftFile,
    ExportContext,
    ExportSource,
    ExportTarget,
    PlannedTarget,
)
from tributo.integrations.exporters.hf_onnx import (
    HuggingFaceONNXExporter,
    _export_with_transformers_onnx,
    _inspect_onnx_artifact,
    _resolve_hf_inputs,
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
        lambda dependency: SimpleNamespace(__version__="4.57.6"),
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
    monkeypatch.setattr(
        "tributo.integrations.exporters.hf_onnx._inspect_onnx_artifact",
        lambda artifact_dir, opset: [DraftFile(relative_path="model.onnx")],
    )

    draft = HuggingFaceONNXExporter().export(
        context,
        ExportSource(source_kind="hf_model", model_object=object()),
        {},
        planned,
    )

    assert draft.entrypoint == "model.onnx"


def test_transformers_without_export_api_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tributo.integrations.exporters.hf_onnx.require_dependency",
        lambda dependency: SimpleNamespace(__version__="5.10.1"),
    )
    with pytest.raises(JobConfigurationError, match="does not provide a supported"):
        _export_with_transformers_onnx(
            SimpleNamespace(config={}), None, "default", 18, tmp_path
        )


def test_explicit_sample_requires_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tributo.integrations.exporters.hf_onnx.require_dependency",
        lambda dependency: SimpleNamespace(is_tensor=lambda value: False),
    )
    with pytest.raises(JobConfigurationError, match="sample_inputs must map"):
        _resolve_hf_inputs(object(), None, "default", {"input_ids": [1, 2]})
    with pytest.raises(JobConfigurationError, match="Encoder-decoder"):
        _resolve_hf_inputs(
            SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=True)),
            None,
            "default",
            {},
        )


def _write_external_model(tmp_path: Path, location: str) -> None:
    import onnx

    graph = onnx.helper.make_graph([], "model", [], [])
    tensor = graph.initializer.add()
    tensor.name = "weights"
    tensor.data_type = onnx.TensorProto.FLOAT
    tensor.dims.append(1)
    tensor.data_location = onnx.TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key, entry.value = "location", location
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 18)]
    )
    (tmp_path / "model.onnx").write_bytes(model.SerializeToString())


def test_external_weights_and_opset_are_checked(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    _write_external_model(tmp_path, "model.onnx.data")
    (tmp_path / "model.onnx.data").write_bytes(b"weight")
    files = _inspect_onnx_artifact(tmp_path, 18)
    assert [(item.relative_path, item.role) for item in files] == [
        ("model.onnx", "model"),
        ("model.onnx.data", "aux"),
    ]
    with pytest.raises(JobConfigurationError, match="Requested ONNX opset"):
        _inspect_onnx_artifact(tmp_path, 14)


@pytest.mark.parametrize(
    "location", ["../weight", "/tmp/weight", "C:/weight", "missing"]
)
def test_external_weights_reject_unsafe_or_missing_locations(
    tmp_path: Path, location: str
) -> None:
    pytest.importorskip("onnx")
    _write_external_model(tmp_path, location)
    with pytest.raises(JobConfigurationError, match="ONNX external data"):
        _inspect_onnx_artifact(tmp_path, 18)


def test_external_weights_reject_symlinks(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    _write_external_model(tmp_path, "link")
    target = tmp_path / "target"
    target.write_bytes(b"weight")
    (tmp_path / "link").symlink_to(target)
    with pytest.raises(JobConfigurationError, match="symlink"):
        _inspect_onnx_artifact(tmp_path, 18)


@pytest.mark.integration
@pytest.mark.filterwarnings(
    r"ignore:`isinstance\(treespec, LeafSpec\)` is deprecated:FutureWarning"
)
@pytest.mark.filterwarnings(r"ignore:# 'dynamic_axes' is not recommended:UserWarning")
@pytest.mark.filterwarnings(
    r"ignore:# The axis name\x3a.*will not be used, since it shares the same shape constraints:UserWarning"
)
@pytest.mark.parametrize(
    ("model_name", "task", "output_names", "use_sample"),
    [
        ("BertForSequenceClassification", "sequence-classification", ["logits"], False),
        (
            "BertForQuestionAnswering",
            "question-answering",
            ["start_logits", "end_logits"],
            True,
        ),
    ],
)
def test_real_hf_export_with_segment_inputs(
    tmp_path: Path,
    model_name: str,
    task: str,
    output_names: list[str],
    use_sample: bool,
) -> None:
    np = pytest.importorskip("numpy")
    ort = pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    torch.set_num_threads(1)
    raw = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"
        )
    )
    raw.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]"
    )
    config = transformers.BertConfig(
        vocab_size=8,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=2,
        attn_implementation="eager",
    )
    model = getattr(transformers, model_name)(config).train()
    model.bert.encoder.layer[0].eval()
    sample = dict(
        tokenizer(
            ["hello world", "hello"],
            ["world", "hello world"],
            padding=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )
    )
    source = ExportSource(
        source_kind="hf_model",
        model_object=model,
        metadata={"preprocessor": tokenizer},
        sample_inputs=sample if use_sample else {},
    )
    planned = PlannedTarget(
        target=ExportTarget(name="model", format="onnx"),
        exporter_id="hf-onnx-v1",
        typed_options={"task": task},
    )
    draft = HuggingFaceONNXExporter().export(
        ExportContext(execution_id="e", node_id="n", artifact_dir=tmp_path),
        source,
        {},
        planned,
    )
    assert model.training and not model.bert.encoder.layer[0].training
    assert draft.producer.effective_options["opset"] == 18
    assert {item.relative_path for item in draft.files} == {
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    options = ort.SessionOptions()
    from tributo.exporting.executor import _materialize_artifact

    assert len(_materialize_artifact(draft, tmp_path, "n").files) == len(draft.files)
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(tmp_path / draft.entrypoint),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    assert [item.name for item in session.get_outputs()] == output_names
    input_names = [item.name for item in session.get_inputs()]
    assert "token_type_ids" in input_names
    model.eval()
    changed = {
        name: torch.cat((value, value[:1]), dim=0)[:, :2]
        for name, value in sample.items()
    }
    for batch in (sample, changed):
        with torch.no_grad():
            expected = model(**batch).to_tuple()
        actual = session.run(None, {name: batch[name].numpy() for name in input_names})
        for left, right in zip(expected, actual, strict=True):
            np.testing.assert_allclose(left.numpy(), right, rtol=1e-3, atol=1e-4)

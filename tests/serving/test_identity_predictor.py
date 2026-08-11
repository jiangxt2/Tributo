"""IdentityPredictor bundle 消费测试（E3）。

覆盖：主 artifact 内辅助文件、独立辅助 artifact（manifest role 定位）、
legacy 裸路径兼容。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tributo.serving.identity_predictor import IdentityPredictor


def _make_single_output_onnx(tmp_path: Path, *, output_name: str) -> str:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    path = tmp_path / f"{output_name}.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", inputs=["score"], outputs=[output_name])],
        "identity-single-output",
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.save(model, path)
    return str(path)


def _make_bundle_with_aux_artifact(tmp_path: Path, onnx_path: str) -> str:
    """构造 bundle：模型 + 独立的 feature config artifact。"""
    from tests.serving.bundle_fixtures import build_test_bundle

    bundle = build_test_bundle(
        tmp_path,
        onnx_path=onnx_path,
        roles={
            "inference": "model",
            "feature_config": "config_art",
        },
        extra_artifacts={
            "config_art": {
                "model_config.json": (
                    "config",
                    json.dumps(
                        {
                            "features": [
                                {
                                    "name": "user_id",
                                    "vocab_size": 1000,
                                    "embedding_dim": 8,
                                },
                            ]
                        }
                    ).encode("utf-8"),
                )
            }
        },
    )
    return str(bundle)


def test_bundle_model_uri_mutually_exclusive(tmp_path: Path):
    """model_path 与 bundle_uri 同时提供应报错。"""
    with pytest.raises(ValueError, match="exactly one"):
        IdentityPredictor(model_path="/x.onnx", bundle_uri="/y")


def test_bundle_loads_aux_artifact_by_role(tmp_path: Path):
    """独立 feature config artifact 按 manifest role 定位并解析 features。"""
    from tests.serving.bundle_fixtures import make_dummy_onnx

    onnx_path = make_dummy_onnx(tmp_path)
    bundle = _make_bundle_with_aux_artifact(tmp_path, onnx_path)

    predictor = IdentityPredictor(bundle_uri=bundle, role="inference")
    assert predictor.model is not None
    assert len(predictor.features) == 1
    assert predictor.features[0].name == "user_id"


def test_auxiliary_file_role_is_used_without_filename_guessing(tmp_path: Path):
    """A config file is loaded by file role even with a nonstandard path."""
    from tests.serving.bundle_fixtures import build_test_bundle, make_dummy_onnx

    onnx_path = make_dummy_onnx(tmp_path)
    bundle = build_test_bundle(
        tmp_path,
        onnx_path=onnx_path,
        roles={"inference": "model", "feature_config": "config_art"},
        extra_artifacts={
            "config_art": {
                "nested/features.json": (
                    "config",
                    json.dumps(
                        {
                            "features": [
                                {
                                    "name": "age",
                                    "vocab_size": 100,
                                    "embedding_dim": 4,
                                }
                            ]
                        }
                    ).encode("utf-8"),
                )
            }
        },
    )

    predictor = IdentityPredictor(bundle_uri=str(bundle), role="inference")

    assert [feature.name for feature in predictor.features] == ["age"]


def test_bundle_without_aux_files_tolerated(tmp_path: Path):
    """无辅助文件的 bundle：transformer/features 为空（与 legacy 一致）。"""
    from tests.serving.bundle_fixtures import build_test_bundle, make_dummy_onnx

    onnx_path = make_dummy_onnx(tmp_path)
    bundle = build_test_bundle(tmp_path, onnx_path=onnx_path)

    predictor = IdentityPredictor(bundle_uri=str(bundle))
    assert predictor.model is not None
    assert predictor.transformer is None
    assert predictor.features == []


def test_legacy_model_path_still_works(tmp_path: Path):
    """legacy 裸模型路径保持兼容。"""
    from tests.serving.bundle_fixtures import make_dummy_onnx

    onnx_path = make_dummy_onnx(tmp_path)
    predictor = IdentityPredictor(model_path=onnx_path)
    assert predictor.model is not None


def test_close_idempotent_and_predict_after_close(tmp_path: Path):
    """close() 幂等；close 后 predict 仍可用（内存模型契约）。"""
    import numpy as np

    from tests.serving.bundle_fixtures import make_dummy_onnx

    onnx_path = make_dummy_onnx(tmp_path)
    bundle = _make_bundle_with_aux_artifact(tmp_path, onnx_path)
    predictor = IdentityPredictor(bundle_uri=bundle, role="inference")

    predictor.close()
    predictor.close()  # 第二次 close 是 no-op

    # The model input is named float_input (skl2onnx fixture) — feed it
    # directly through the adapter; close must not break prediction.
    logits = predictor.model.predict_numpy(
        {"float_input": np.array([[0.5, 0.5]], dtype=np.float32)}
    )
    assert logits.shape == (1,)


def test_close_legacy_path_is_noop(tmp_path: Path):
    """legacy model_path 路径 close() 为 no-op。"""
    from tests.serving.bundle_fixtures import make_dummy_onnx

    onnx_path = make_dummy_onnx(tmp_path)
    predictor = IdentityPredictor(model_path=onnx_path)
    predictor.close()  # 不抛即通过


def test_sklearn_probability_output_is_selected_without_sigmoid(tmp_path: Path):
    """skl2onnx label output must not be treated as a binary logit."""
    import numpy as np
    import onnxruntime as ort

    from tests.serving.bundle_fixtures import make_dummy_onnx

    onnx_path = make_dummy_onnx(tmp_path)
    predictor = IdentityPredictor(model_path=onnx_path)
    inputs = {"float_input": np.array([[0.25, 0.75]], dtype=np.float32)}
    expected = ort.InferenceSession(onnx_path).run(None, inputs)[1][0, 1]

    actual = predictor._predict_probabilities(inputs)

    assert actual.shape == (1,)
    assert actual[0] == pytest.approx(expected)


def test_single_output_logits_predict_and_batch_apply_sigmoid(tmp_path: Path):
    predictor = IdentityPredictor(
        model_path=_make_single_output_onnx(tmp_path, output_name="output")
    )

    single = predictor.predict({"score": 0.0})
    batch = predictor.predict_batch([{"score": -2.0}, {"score": 2.0}])

    assert single == {"probability": pytest.approx(0.5), "prediction": 1}
    assert batch[0]["probability"] == pytest.approx(0.11920292)
    assert batch[0]["prediction"] == 0
    assert batch[1]["probability"] == pytest.approx(0.88079708)
    assert batch[1]["prediction"] == 1


def test_unknown_single_output_is_not_guessed_as_logits(tmp_path: Path):
    predictor = IdentityPredictor(
        model_path=_make_single_output_onnx(tmp_path, output_name="prediction")
    )

    with pytest.raises(ValueError, match="requires a probability or logits output"):
        predictor.predict({"score": 0.0})


def test_aux_load_failure_closes_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """辅助 artifact 解析失败时主 runtime 必须确定性关闭。

    Captures the runtime returned by the loader and asserts ``closed``
    after the constructor raised — only the deterministic exception path
    can have set it (the captured reference keeps ``__del__`` from
    running, so this cannot be a GC-side-effect false positive).
    """
    from tests.serving.bundle_fixtures import build_test_bundle, make_dummy_onnx
    from tributo.exporting.runtime import BundleModelLoader

    onnx_path = make_dummy_onnx(tmp_path)
    bundle = build_test_bundle(
        tmp_path,
        onnx_path=onnx_path,
        roles={
            "inference": "model",
            "feature_config": "config_art",
        },
        extra_artifacts={
            "config_art": {
                "model_config.json": (
                    "config",
                    b"{not-valid-json",  # 触发 JSON 解析失败
                )
            }
        },
    )

    captured: list[Any] = []
    original_open = BundleModelLoader.open

    def capturing_open(self, *args: Any, **kwargs: Any) -> Any:
        runtime = original_open(self, *args, **kwargs)
        captured.append(runtime)
        return runtime

    monkeypatch.setattr(BundleModelLoader, "open", capturing_open)

    with pytest.raises(json.JSONDecodeError):
        IdentityPredictor(bundle_uri=str(bundle), role="inference")

    assert len(captured) == 1
    assert captured[0].closed is True, (
        "auxiliary load failure must close the model runtime deterministically"
    )

"""IdentityPredictor bundle 消费测试（E3）。

覆盖：主 artifact 内辅助文件、独立辅助 artifact（manifest role 定位）、
legacy 裸路径兼容。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tributo.serving.identity_predictor import IdentityPredictor


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

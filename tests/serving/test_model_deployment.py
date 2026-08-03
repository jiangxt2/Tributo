"""serving.model_deployment 单元测试。"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tributo.serving.model_deployment import ONNXModel
from tributo.serving.schema import PredictRequest


def _make_dummy_onnx(tmp_path: Path) -> str:
    """生成一个最小可用的 ONNX 分类模型文件，用于测试。"""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("skl2onnx or sklearn not installed")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    # sklearn 1.6 passes an 'iprint' solver option that scipy >= 1.14
    # warns about; pytest runs with filterwarnings=error, so silence the
    # unrelated solver warning around the fit.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Unknown solver options: iprint", category=Warning
        )
        clf = LogisticRegression().fit(X, y)

    initial_types = [("float_input", FloatTensorType([None, 2]))]
    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_types,
        options={id(clf): {"zipmap": False}},
    )  # plain float probability matrix, like the XGBoost ONNX path

    path = str(tmp_path / "dummy.onnx")
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return path


def test_onnx_model_loads_and_predicts(tmp_path: Path):
    """ONNXModel 能加载模型并返回正确 shape 的预测结果。"""
    model_path = _make_dummy_onnx(tmp_path)
    deployment = ONNXModel(model_path=model_path)

    req = PredictRequest(features=[[0.5, 0.5], [0.1, 0.9]])
    resp = deployment._predict(req)

    assert resp.model_path == model_path
    assert len(resp.predictions) == 2
    assert resp.inference_time_ms >= 0


def test_onnx_model_return_probs_false(tmp_path: Path):
    """return_probs=False 时返回 label 而非概率。"""
    model_path = _make_dummy_onnx(tmp_path)
    deployment = ONNXModel(model_path=model_path)

    req = PredictRequest(features=[[0.5, 0.5]], return_probs=False)
    resp = deployment._predict(req)

    # label 应为 int 类型
    assert isinstance(resp.predictions[0], (int, np.integer))


def test_onnx_model_health():
    """health() 返回预期字段。"""
    with patch("onnxruntime.InferenceSession") as mock_session:
        mock_session.return_value.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.return_value.get_outputs.return_value = []

        deployment = ONNXModel(model_path="/fake/model.onnx")
        health = deployment.health()

    assert health["status"] == "healthy"
    assert health["model_path"] == "/fake/model.onnx"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))


# ── E3 bundle serving path ─────────────────────────────────────────────────────


class TestONNXModelBundle:
    """ONNXModel 经 BundleModelLoader 加载 bundle_uri。"""

    def _make_bundle(self, tmp_path: Path) -> str:
        """构造带 typed signature 的本地 ONNX bundle。"""
        from tests.serving.bundle_fixtures import build_test_bundle, make_dummy_onnx

        onnx_path = make_dummy_onnx(tmp_path)
        bundle = build_test_bundle(tmp_path, onnx_path=onnx_path)
        return str(bundle)

    def test_model_path_and_bundle_uri_mutually_exclusive(self, tmp_path: Path):
        """model_path 与 bundle_uri 同时提供应报错。"""
        with pytest.raises(ValueError, match="exactly one"):
            ONNXModel(model_path="/x.onnx", bundle_uri="/y")

    def test_neither_model_path_nor_bundle_uri_raises(self):
        """两者都不提供应报错。"""
        with pytest.raises(ValueError, match="exactly one"):
            ONNXModel()

    def test_bundle_loads_and_predicts(self, tmp_path: Path):
        """bundle_uri 加载 + versioned inputs 推理。"""
        bundle = self._make_bundle(tmp_path)
        deployment = ONNXModel(bundle_uri=bundle, role="inference")

        req = PredictRequest(
            inputs=[
                {
                    "name": "float_input",
                    "shape": [1, 2],
                    "datatype": "float32",
                    "data": [0.5, 0.5],
                }
            ]
        )
        resp = deployment._predict(req)
        assert resp.model_path == bundle
        assert len(resp.predictions) > 0

    def test_bundle_legacy_features_compat(self, tmp_path: Path):
        """bundle 场景下 legacy features 仍可推理（映射到第一个输入）。"""
        bundle = self._make_bundle(tmp_path)
        deployment = ONNXModel(bundle_uri=bundle)

        req = PredictRequest(features=[[0.5, 0.5]])
        resp = deployment._predict(req)
        assert len(resp.predictions) > 0

    def test_bundle_health(self, tmp_path: Path):
        """bundle 场景 health() 返回 bundle_uri。"""
        bundle = self._make_bundle(tmp_path)
        deployment = ONNXModel(bundle_uri=bundle)
        health = deployment.health()
        assert health["status"] == "healthy"
        assert health["model_path"] == bundle

    def test_unknown_role_fails_fast(self, tmp_path: Path):
        """未知 role 在部署启动时报错（fail-fast）。"""
        bundle = self._make_bundle(tmp_path)
        with pytest.raises(Exception, match="Role"):
            ONNXModel(bundle_uri=bundle, role="serve")

    def test_close_idempotent_and_predict_after_close(self, tmp_path: Path):
        """close() 幂等；close 后 predict 仍可用（内存模型契约）。"""
        bundle = self._make_bundle(tmp_path)
        deployment = ONNXModel(bundle_uri=bundle)

        deployment.close()
        deployment.close()  # 第二次 close 是 no-op

        req = PredictRequest(features=[[0.5, 0.5]])
        resp = deployment._predict(req)
        assert len(resp.predictions) > 0

    def test_close_legacy_path_is_noop(self, tmp_path: Path):
        """legacy model_path 路径 close() 为 no-op。"""
        deployment = ONNXModel(model_path=_make_dummy_onnx(tmp_path))
        deployment.close()  # 不抛即通过


class TestONNXModelLegacyVersionedInputs:
    """E3 fix: 裸模型 compat 路径也支持版本化输入（不再静默 nan）。"""

    def test_legacy_model_accepts_versioned_inputs(self, tmp_path: Path):
        """裸模型路径收到 versioned inputs 应正常推理。"""
        deployment = ONNXModel(model_path=_make_dummy_onnx(tmp_path))

        req = PredictRequest(
            inputs=[
                {
                    "name": "float_input",
                    "shape": [1, 2],
                    "datatype": "float32",
                    "data": [0.5, 0.5],
                }
            ]
        )
        resp = deployment._predict(req)
        assert len(resp.predictions) > 0
        # predictions 不含 nan（此前 np.array(None) 会静默产生 nan）
        import math

        for p in resp.predictions:
            if isinstance(p, list):
                assert all(not math.isnan(float(v)) for v in p)
            else:
                assert not math.isnan(float(p))

    def test_legacy_model_features_still_work(self, tmp_path: Path):
        """裸模型 + legacy features 保持兼容。"""
        deployment = ONNXModel(model_path=_make_dummy_onnx(tmp_path))
        resp = deployment._predict(PredictRequest(features=[[0.5, 0.5]]))
        assert len(resp.predictions) > 0


def _http_request(body: bytes) -> Any:
    """构造一个携带 JSON body 的 starlette Request（绕过 ASGI 服务）。"""
    from starlette.requests import Request

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/predict",
            "raw_path": b"/predict",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8000),
        },
        receive=_receive,
    )


@pytest.mark.asyncio
async def test_http_shape_mismatch_returns_400(tmp_path: Path):
    """版本化输入 shape 与 payload 不匹配 → 400（客户端契约错误）。"""
    deployment = ONNXModel(model_path=_make_dummy_onnx(tmp_path))
    body = json.dumps(
        {
            "inputs": [
                {
                    "name": "float_input",
                    "shape": [2, 2],
                    "datatype": "float32",
                    "data": [1.0],
                }
            ]
        }
    ).encode()

    resp = await deployment(_http_request(body))
    assert resp.status_code == 400
    assert "error" in json.loads(resp.body)


@pytest.mark.asyncio
async def test_http_int_overflow_returns_400(tmp_path: Path):
    """int32 越界整数 → 400（此前冒泡为 500）。"""
    deployment = ONNXModel(model_path=_make_dummy_onnx(tmp_path))
    body = json.dumps(
        {
            "inputs": [
                {
                    "name": "float_input",
                    "shape": [1, 2],
                    "datatype": "int32",
                    "data": [2**40, 1],
                }
            ]
        }
    ).encode()

    resp = await deployment(_http_request(body))
    assert resp.status_code == 400
    assert "out of bounds" in json.loads(resp.body)["error"]


@pytest.mark.asyncio
async def test_http_malformed_json_returns_400(tmp_path: Path):
    """非 JSON body → 400。"""
    deployment = ONNXModel(model_path=_make_dummy_onnx(tmp_path))
    resp = await deployment(_http_request(b"{not-json"))
    assert resp.status_code == 400


def test_versioned_input_unknown_name_rejected(tmp_path: Path):
    """版本化输入名与模型 signature 不匹配 → 客户端错误而非运行时错误。"""
    deployment = ONNXModel(model_path=_make_dummy_onnx(tmp_path))
    req = PredictRequest(
        inputs=[
            {
                "name": "typo_input",
                "shape": [1, 2],
                "datatype": "float32",
                "data": [0.5, 0.5],
            }
        ]
    )
    with pytest.raises(ValueError, match="do not match"):
        deployment._predict(req)


@pytest.mark.asyncio
async def test_http_unknown_input_name_returns_400(tmp_path: Path):
    """输入名 typo 经 HTTP 返回 400，而非 500。"""
    deployment = ONNXModel(model_path=_make_dummy_onnx(tmp_path))
    body = json.dumps(
        {
            "inputs": [
                {
                    "name": "typo_input",
                    "shape": [1, 2],
                    "datatype": "float32",
                    "data": [0.5, 0.5],
                }
            ]
        }
    ).encode()

    resp = await deployment(_http_request(body))
    assert resp.status_code == 400
    assert "do not match" in json.loads(resp.body)["error"]

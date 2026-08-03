"""serving.schema 单元测试。"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tributo.serving.schema import PredictRequest, PredictResponse


def test_predict_request_valid():
    """正常构造 PredictRequest。"""
    req = PredictRequest(features=[[0.1, 0.2], [0.3, 0.4]])
    assert req.features == [[0.1, 0.2], [0.3, 0.4]]
    assert req.return_probs is True


def test_predict_request_return_probs_false():
    """显式设置 return_probs=False。"""
    req = PredictRequest(features=[[1.0]], return_probs=False)
    assert req.return_probs is False


def test_predict_request_empty_features_raises():
    """空 features 应触发 ValidationError。"""
    with pytest.raises(ValidationError):
        PredictRequest(features=[])


def test_predict_response_serialization():
    """PredictResponse 可正确序列化。"""
    resp = PredictResponse(
        predictions=[[0.8, 0.2], [0.3, 0.7]],
        model_path="/workspace/onnx/model.onnx",
        inference_time_ms=12.34,
    )
    dumped = resp.model_dump()
    assert dumped["model_path"] == "/workspace/onnx/model.onnx"
    assert dumped["inference_time_ms"] == 12.34


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))


# ── E3 versioned input protocol ────────────────────────────────────────────────


class TestPredictInput:
    """版本化输入张量 name/shape/datatype/data 契约。"""

    def test_to_numpy_float32(self):
        """float32 datatype 应正确转换并按 shape reshape。"""
        from tributo.serving.schema import PredictInput

        inp = PredictInput(
            name="input", shape=[2, 2], datatype="float32", data=[1, 2, 3, 4]
        )
        arr = inp.to_numpy()
        assert arr.dtype == np.float32
        assert arr.shape == (2, 2)

    def test_to_numpy_int64(self):
        """int64 datatype 应转换为 int64 数组。"""
        from tributo.serving.schema import PredictInput

        inp = PredictInput(name="ids", datatype="int64", data=[1, 2, 3])
        arr = inp.to_numpy()
        assert arr.dtype == np.int64
        assert arr.shape == (3,)

    def test_unsupported_datatype_raises(self):
        """未知 datatype 应 fail-fast。"""
        from tributo.serving.schema import PredictInput

        with pytest.raises(ValueError, match="Unsupported datatype"):
            PredictInput(name="x", datatype="fancy8", data=[1.0]).to_numpy()

    def test_negative_shape_raises(self):
        """负 shape 维度应校验失败。"""
        from tributo.serving.schema import PredictInput

        with pytest.raises(ValidationError):
            PredictInput(name="x", shape=[-1], datatype="float32", data=[1.0])

    def test_dynamic_batch_shape_inferred(self):
        """shape=[0, 2] 时 batch 维由数据长度推断（4 元素 → [2, 2]）。"""
        from tributo.serving.schema import PredictInput

        inp = PredictInput(
            name="x", shape=[0, 2], datatype="float32", data=[1, 2, 3, 4]
        )
        arr = inp.to_numpy()
        assert arr.shape == (2, 2)
        np.testing.assert_allclose(arr, [[1, 2], [3, 4]])

    def test_dynamic_batch_single_element(self):
        """shape=[0, 2] + 2 元素 → batch=1。"""
        from tributo.serving.schema import PredictInput

        inp = PredictInput(name="x", shape=[0, 2], datatype="float32", data=[1, 2])
        assert inp.to_numpy().shape == (1, 2)

    def test_dynamic_dim_only_reshapes_to_flat(self):
        """shape=[0]（纯动态一维）→ 展平数组。"""
        from tributo.serving.schema import PredictInput

        inp = PredictInput(name="x", shape=[0], datatype="float32", data=[1, 2, 3])
        assert inp.to_numpy().shape == (3,)

    def test_dynamic_batch_not_divisible_raises(self):
        """数据量无法被固定维整除时 fail-fast，而非静默错误 reshape。"""
        from tributo.serving.schema import PredictInput

        inp = PredictInput(name="x", shape=[0, 2], datatype="float32", data=[1, 2, 3])
        with pytest.raises(ValueError, match="cannot infer the dynamic"):
            inp.to_numpy()

    def test_multiple_dynamic_dims_raises(self):
        """多个动态维无法唯一推断 → fail-fast。"""
        from tributo.serving.schema import PredictInput

        inp = PredictInput(
            name="x", shape=[0, 0, 2], datatype="float32", data=[1, 2, 3, 4]
        )
        with pytest.raises(ValueError, match="at most one"):
            inp.to_numpy()

    def test_fixed_shape_element_mismatch_raises(self):
        """无动态维时元素数不匹配仍由 reshape 报错（既有行为）。"""
        from tributo.serving.schema import PredictInput

        inp = PredictInput(name="x", shape=[2], datatype="float32", data=[1, 2, 3, 4])
        with pytest.raises(ValueError):
            inp.to_numpy()


class TestPredictRequestVersioned:
    """PredictRequest 的 versioned/legacy 双协议。"""

    def test_versioned_inputs(self):
        """新协议 inputs 列表。"""
        req = PredictRequest(
            inputs=[
                {"name": "a", "datatype": "float32", "data": [1.0, 2.0]},
            ]
        )
        assert req.schema_version == 1
        assert req.inputs is not None and len(req.inputs) == 1

    def test_inputs_and_features_mutually_exclusive(self):
        """inputs 与 features 同时提供应 fail-fast。"""
        with pytest.raises(ValidationError, match="mutually exclusive"):
            PredictRequest(
                inputs=[{"name": "a", "datatype": "float32", "data": [1.0]}],
                features=[[1.0]],
            )

    def test_neither_inputs_nor_features_raises(self):
        """两者都不提供应 fail-fast。"""
        with pytest.raises(ValidationError, match="must be provided"):
            PredictRequest(return_probs=True)

    def test_empty_inputs_rejected(self):
        """空 inputs 列表应在 Pydantic 层拒绝，而非模型层报缺输入。"""
        with pytest.raises(ValidationError, match="must not be empty"):
            PredictRequest(inputs=[])

    def test_legacy_features_still_valid(self):
        """旧 features 协议继续可用（compat）。"""
        req = PredictRequest(features=[[0.1, 0.2]])
        assert req.features == [[0.1, 0.2]]


class TestRequestToInputs:
    """request_to_inputs 归一化为命名 numpy 输入。"""

    def test_versioned_inputs_used_as_is(self):
        """versioned inputs 按名字转换为数组。"""
        from tributo.serving.schema import request_to_inputs

        req = PredictRequest(
            inputs=[{"name": "x", "datatype": "float32", "data": [1.0, 2.0]}]
        )
        result = request_to_inputs(req, input_name="ignored")
        assert set(result) == {"x"}

    def test_legacy_features_map_to_first_input(self):
        """legacy features 映射到第一个输入名。"""
        from tributo.serving.schema import request_to_inputs

        req = PredictRequest(features=[[1.0, 2.0]])
        result = request_to_inputs(req, input_name="float_input")
        np.testing.assert_allclose(result["float_input"], [[1.0, 2.0]])

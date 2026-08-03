"""serving.grpc_deployment 单元测试。"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from tributo.serving.grpc_deployment import _prepare_inputs
from tributo.serving.proto import inference_pb2


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


class _RpcContext:
    """Small Ray/gRPC context double for status and metadata assertions."""

    def __init__(self, metadata: tuple[tuple[str, str], ...] = ()) -> None:
        self.code = None
        self.details = None
        self._metadata = metadata

    def set_code(self, code: object) -> None:
        self.code = code

    def set_details(self, message: str) -> None:
        self.details = message

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata


def test_grpc_deployment_import():
    """gRPCInferenceService 可以正常导入。"""
    from tributo.serving.grpc_deployment import gRPCInferenceService

    assert gRPCInferenceService is not None


def test_grpc_runner_import():
    """gRPC runner 函数可以正常导入。"""
    from tributo.serving.grpc_runner import (
        get_grpc_serving_status,
        start_grpc_serving,
        stop_grpc_serving,
    )

    assert start_grpc_serving is not None
    assert stop_grpc_serving is not None
    assert get_grpc_serving_status is not None


def test_grpc_deployment_class_definition():
    """gRPCInferenceService 是普通类（不由 @serve.decoration 装饰）。"""
    from tributo.serving.grpc_deployment import gRPCInferenceService

    # 不应有 bind 属性（由 deploy_serve_app 统一处理装饰）
    assert not hasattr(gRPCInferenceService, "bind")
    # 应有 __init__ 和推理方法
    assert hasattr(gRPCInferenceService, "__init__")
    assert hasattr(gRPCInferenceService, "Predict")
    assert hasattr(gRPCInferenceService, "StreamPredict")
    assert hasattr(gRPCInferenceService, "BatchPredict")
    assert hasattr(gRPCInferenceService, "health")


def test_inference_pb2_import():
    """inference_pb2 可以正常导入并包含正确的消息类型。"""
    from tributo.serving.proto import inference_pb2

    # 检查消息类型存在
    assert hasattr(inference_pb2, "PredictRequest")
    assert hasattr(inference_pb2, "PredictResponse")


def test_inference_pb2_grpc_import():
    """inference_pb2_grpc 可以正常导入并包含正确的服务类型。"""
    from tributo.serving.proto import inference_pb2_grpc

    # 检查服务类型存在
    assert hasattr(inference_pb2_grpc, "InferenceServiceStub")
    assert hasattr(inference_pb2_grpc, "InferenceServiceServicer")


def test_inference_pb2_module_name_and_pickle():
    """pb2 模块名必须是完整包路径（否则 protobuf 对象无法 pickle）。"""
    import pickle

    from tributo.serving.proto import inference_pb2

    request = inference_pb2.PredictRequest(
        features=[0.5, 0.5],
        model_name="test",
    )

    # 模块名必须与真实 import 路径一致——裸 "inference_pb2" 曾导致
    # PicklingError: Can't pickle <class 'inference_pb2.PredictRequest'>
    assert request.__class__.__module__ == "tributo.serving.proto.inference_pb2"

    restored = pickle.loads(pickle.dumps(request))
    assert list(restored.features) == [0.5, 0.5]
    assert restored.model_name == "test"


def test_predict_request_creation():
    """PredictRequest 可以正确创建。"""
    request = inference_pb2.PredictRequest(
        features=[0.5, 0.5],
        model_name="test",
    )

    assert request.features == [0.5, 0.5]
    assert request.model_name == "test"


def test_predict_response_creation():
    """PredictResponse 可以正确创建。"""
    response = inference_pb2.PredictResponse(
        predictions=[0.8, 0.2],
        confidence=0.8,
    )

    # protobuf float 字段使用 float32，会有精度损失
    assert list(response.predictions) == pytest.approx([0.8, 0.2], abs=1e-6)
    assert response.confidence == pytest.approx(0.8, abs=1e-6)


def test_serve_utils_supports_grpc():
    """serve_utils.py 的 deploy_serve_app 支持 gRPC 相关参数。"""
    import inspect

    from tributo._common.serve_utils import deploy_serve_app

    sig = inspect.signature(deploy_serve_app)
    assert "grpc_port" in sig.parameters
    assert sig.parameters["grpc_port"].default is None
    assert "grpc_servicer_functions" in sig.parameters
    assert sig.parameters["grpc_servicer_functions"].default is None
    assert "enable_http" in sig.parameters
    assert sig.parameters["enable_http"].default is True


def test_cli_grpc_commands():
    """CLI 中包含 grpc 命令组。"""
    from click.testing import CliRunner

    from tributo.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["serve", "grpc", "--help"])

    assert result.exit_code == 0
    assert "gRPC inference service management" in result.output


def test_cli_serve_start_exposes_e3_options():
    """serve start 暴露 --unsafe / --storage-profile（E3 完整透传）。"""
    from click.testing import CliRunner

    from tributo.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["serve", "start", "--help"])

    assert result.exit_code == 0
    assert "--bundle-uri" in result.output
    assert "--unsafe" in result.output
    assert "--storage-profile" in result.output


def test_cli_grpc_start_exposes_e3_options():
    """grpc start 暴露 --unsafe / --storage-profile（E3 完整透传）。"""
    from click.testing import CliRunner

    from tributo.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["serve", "grpc", "start", "--help"])

    assert result.exit_code == 0
    assert "--bundle-uri" in result.output
    assert "--unsafe" in result.output
    assert "--storage-profile" in result.output


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))


# ── E3 versioned input protocol (pure functions, no Ray needed) ────────────────


class TestPrepareInputs:
    """_prepare_inputs 归一化 gRPC 请求输入。"""

    def test_versioned_inputs_to_named_arrays(self):
        """InputTensor 列表转换为命名 numpy 数组。"""
        from tributo.serving.grpc_deployment import _prepare_inputs

        req = inference_pb2.PredictRequest()
        t = req.inputs.add()
        t.schema_version = 1
        t.name = "float_input"
        t.shape.extend([1, 2])
        t.datatype = "float32"
        t.data.extend([0.5, 0.5])

        result = _prepare_inputs(req)
        assert isinstance(result, dict)
        assert set(result) == {"float_input"}
        np.testing.assert_allclose(result["float_input"], [[0.5, 0.5]])

    def test_legacy_features_fallback(self):
        """无 inputs 时回退 legacy 平坦 features。"""
        from tributo.serving.grpc_deployment import _prepare_inputs

        req = inference_pb2.PredictRequest(features=[0.5, 0.5])
        result = _prepare_inputs(req)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2)

    def test_empty_request_returns_none(self):
        """无输入时返回 None（由调用方置 INVALID_ARGUMENT）。"""
        from tributo.serving.grpc_deployment import _prepare_inputs

        req = inference_pb2.PredictRequest()
        assert _prepare_inputs(req) is None

    def test_int64_datatype(self):
        """int64 datatype 经 int64_data 无损转换为 int64 数组。"""
        from tributo.serving.grpc_deployment import _prepare_inputs

        req = inference_pb2.PredictRequest()
        t = req.inputs.add()
        t.schema_version = 1
        t.name = "ids"
        t.datatype = "int64"
        t.int64_data.extend([1, 2, 3])

        result = _prepare_inputs(req)
        assert isinstance(result, dict)
        assert result["ids"].dtype == np.int64

    @pytest.mark.parametrize("schema_version", [0, -1, 99])
    def test_unsupported_schema_version_rejected(self, schema_version: int):
        """Only schema_version 1 is accepted for versioned tensors."""
        from tributo.serving.grpc_deployment import _prepare_inputs

        req = inference_pb2.PredictRequest()
        t = req.inputs.add()
        t.schema_version = schema_version
        t.name = "x"
        t.datatype = "float32"
        t.data.extend([1.0])

        with pytest.raises(ValueError, match="schema_version"):
            _prepare_inputs(req)

    def test_unordered_typed_signature_is_rejected(self):
        """An unordered name set cannot be aligned with typed metadata."""
        from tributo.serving.grpc_deployment import _prepare_inputs

        req = inference_pb2.PredictRequest()
        for name in ("z", "a"):
            tensor = req.inputs.add()
            tensor.schema_version = 1
            tensor.name = name
            tensor.datatype = "float32"
            tensor.data.extend([1.0])

        with pytest.raises(ValueError, match="ordered sequence"):
            _prepare_inputs(
                req,
                expected_names={"z", "a"},
                expected_dtypes=("float32", "int64"),
            )

    def test_int64_without_int64_data_rejected(self):
        """datatype=int64 但未用 int64_data → fail-fast（防 double 精度损失）。"""
        from tributo.serving.grpc_deployment import _prepare_inputs

        req = inference_pb2.PredictRequest()
        t = req.inputs.add()
        t.schema_version = 1
        t.name = "ids"
        t.datatype = "int64"
        t.data.extend([2**60 + 1])  # double 承载会丢精度

        with pytest.raises(ValueError, match="int64_data"):
            _prepare_inputs(req)

    def test_protocol_error_becomes_invalid_argument(self):
        """协议错误经 _prepare_inputs_or_invalid 转为 INVALID_ARGUMENT。"""
        import grpc as grpc_module

        from tributo.serving.grpc_deployment import _prepare_inputs_or_invalid

        class _FakeContext:
            def __init__(self) -> None:
                self.code = None
                self.details = None

            def set_code(self, code: object) -> None:
                self.code = code

            def set_details(self, message: str) -> None:
                self.details = message

        ctx = _FakeContext()
        req = inference_pb2.PredictRequest()
        t = req.inputs.add()
        t.schema_version = 99
        t.name = "x"
        t.datatype = "float32"
        t.data.extend([1.0])

        result = _prepare_inputs_or_invalid(req, ctx)
        assert result is None
        assert ctx.code == grpc_module.StatusCode.INVALID_ARGUMENT
        assert "schema_version" in ctx.details

    def test_int_overflow_becomes_invalid_argument(self):
        """int32 越界整数 → INVALID_ARGUMENT（此前冒泡为 UNKNOWN）。"""
        import grpc as grpc_module

        from tributo.serving.grpc_deployment import _prepare_inputs_or_invalid

        class _FakeContext:
            def __init__(self) -> None:
                self.code = None
                self.details = None

            def set_code(self, code: object) -> None:
                self.code = code

            def set_details(self, message: str) -> None:
                self.details = message

        ctx = _FakeContext()
        req = inference_pb2.PredictRequest()
        t = req.inputs.add()
        t.schema_version = 1
        t.name = "x"
        t.datatype = "int32"
        t.data.extend([2**40, 1])

        result = _prepare_inputs_or_invalid(req, ctx)
        assert result is None
        assert ctx.code == grpc_module.StatusCode.INVALID_ARGUMENT
        assert "out of bounds" in ctx.details

    def test_input_name_mismatch_becomes_invalid_argument(self):
        """输入名与模型 signature 不匹配 → INVALID_ARGUMENT。"""
        import grpc as grpc_module

        from tributo.serving.grpc_deployment import _prepare_inputs_or_invalid

        class _FakeContext:
            def __init__(self) -> None:
                self.code = None
                self.details = None

            def set_code(self, code: object) -> None:
                self.code = code

            def set_details(self, message: str) -> None:
                self.details = message

        ctx = _FakeContext()
        req = inference_pb2.PredictRequest()
        t = req.inputs.add()
        t.schema_version = 1
        t.name = "typo_input"
        t.datatype = "float32"
        t.data.extend([0.5, 0.5])

        result = _prepare_inputs_or_invalid(req, ctx, expected_names={"float_input"})
        assert result is None
        assert ctx.code == grpc_module.StatusCode.INVALID_ARGUMENT
        assert "do not match" in ctx.details


class TestGrpcBundleAndVersionedInputs:
    """E3 fix: gRPC 裸模型/versioned dict 输入与 bundle 真实加载。"""

    def test_legacy_session_accepts_versioned_dict(self, tmp_path: Path):
        """裸模型 session 路径接受 versioned dict（此前 AssertionError）。"""
        from tributo.serving.grpc_deployment import gRPCInferenceService

        service = gRPCInferenceService(model_path=_make_dummy_onnx(tmp_path))
        inputs = {"float_input": np.array([[0.5, 0.5]], dtype=np.float32)}
        outputs = service._run(inputs)
        assert len(outputs) == 2  # label + probabilities
        assert outputs[0].shape == (1,)

    def test_legacy_session_accepts_flat_matrix(self, tmp_path: Path):
        """裸模型 + legacy 平坦矩阵保持兼容。"""
        from tributo.serving.grpc_deployment import gRPCInferenceService

        service = gRPCInferenceService(model_path=_make_dummy_onnx(tmp_path))
        outputs = service._run(np.array([[0.5, 0.5]], dtype=np.float32))
        assert len(outputs) == 2  # label + probabilities

    def test_bundle_loads_and_predicts(self, tmp_path: Path):
        """gRPC 经 bundle_uri 真实加载并推理。"""
        from tests.serving.bundle_fixtures import build_test_bundle, make_dummy_onnx
        from tributo.serving.grpc_deployment import gRPCInferenceService

        onnx_path = make_dummy_onnx(tmp_path)
        bundle = build_test_bundle(tmp_path, onnx_path=onnx_path)
        service = gRPCInferenceService(bundle_uri=str(bundle), role="inference")

        request = inference_pb2.PredictRequest()
        t = request.inputs.add()
        t.schema_version = 1
        t.name = "float_input"
        t.shape.extend([1, 2])
        t.datatype = "float32"
        t.data.extend([0.5, 0.5])

        outputs = service._run(_prepare_inputs(request))
        assert len(outputs) == 2  # label + probabilities
        assert outputs[0].shape[0] == 1

    def test_close_idempotent_and_predict_after_close(self, tmp_path: Path):
        """close() 幂等；close 后 predict 仍可用。"""
        from tests.serving.bundle_fixtures import build_test_bundle, make_dummy_onnx
        from tributo.serving.grpc_deployment import gRPCInferenceService

        onnx_path = make_dummy_onnx(tmp_path)
        bundle = build_test_bundle(tmp_path, onnx_path=onnx_path)
        service = gRPCInferenceService(bundle_uri=str(bundle), role="inference")

        service.close()
        service.close()  # 第二次 close 是 no-op

        request = inference_pb2.PredictRequest()
        t = request.inputs.add()
        t.schema_version = 1
        t.name = "float_input"
        t.shape.extend([1, 2])
        t.datatype = "float32"
        t.data.extend([0.5, 0.5])

        outputs = service._run(_prepare_inputs(request))
        assert len(outputs) == 2

    @pytest.mark.asyncio
    async def test_predict_response_carries_e3_context(self, tmp_path: Path):
        """Unary gRPC responses carry request, trace, and bundle identity."""
        from tributo.serving.grpc_deployment import gRPCInferenceService

        service = gRPCInferenceService(model_path=_make_dummy_onnx(tmp_path))
        request = inference_pb2.PredictRequest(features=[0.5, 0.5])
        traceparent = "00-" + "3" * 32 + "-" + "4" * 16 + "-01"
        context = _RpcContext(
            (
                ("x-request-id", "grpc-request-123"),
                ("traceparent", traceparent),
            )
        )

        response = await service.Predict(request, context)

        assert context.code is None
        assert response.request_id == "grpc-request-123"
        assert response.trace_id == "3" * 32
        assert response.traceparent == traceparent
        assert response.bundle_id == ""
        assert response.model_version == ""

    @pytest.mark.asyncio
    async def test_empty_unary_request_is_invalid_argument(self):
        """Empty unary requests fail with INVALID_ARGUMENT."""
        from tributo.serving.grpc_deployment import gRPCInferenceService

        service = gRPCInferenceService.__new__(gRPCInferenceService)
        service._input_names = ("float_input",)
        context = _RpcContext()

        response = await service.Predict(inference_pb2.PredictRequest(), context)

        import grpc as grpc_module

        assert response.predictions == []
        assert context.code == grpc_module.StatusCode.INVALID_ARGUMENT
        assert "must contain" in context.details

    @pytest.mark.asyncio
    async def test_empty_stream_request_is_invalid_argument(self):
        """Empty server-streaming requests fail with INVALID_ARGUMENT."""
        from tributo.serving.grpc_deployment import gRPCInferenceService

        service = gRPCInferenceService.__new__(gRPCInferenceService)
        service._input_names = ("float_input",)
        context = _RpcContext()

        responses = [
            item
            async for item in service.StreamPredict(
                inference_pb2.PredictRequest(), context
            )
        ]

        import grpc as grpc_module

        assert responses == []
        assert context.code == grpc_module.StatusCode.INVALID_ARGUMENT
        assert "must contain" in context.details

    @pytest.mark.asyncio
    async def test_health_reports_all_bundle_input_names(self):
        """Health metadata exposes every input of a multi-input model."""
        from tributo.serving.grpc_deployment import gRPCInferenceService

        service = gRPCInferenceService.__new__(gRPCInferenceService)
        service._input_name = "first"
        service._input_names = ("first", "second")
        service._runtime = None
        service._session = object()

        health = await service.health()

        assert health["input_names"] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_predict_dtype_mismatch_is_invalid_argument(self, tmp_path: Path):
        """Typed gRPC input mismatches are rejected before ONNX Runtime."""
        import grpc as grpc_module

        from tributo.serving.grpc_deployment import gRPCInferenceService

        service = gRPCInferenceService(model_path=_make_dummy_onnx(tmp_path))
        request = inference_pb2.PredictRequest()
        tensor = request.inputs.add()
        tensor.schema_version = 1
        tensor.name = "float_input"
        tensor.shape.extend([1, 2])
        tensor.datatype = "int64"
        tensor.int64_data.extend([1, 2])
        context = _RpcContext()

        response = await service.Predict(request, context)

        assert response.predictions == []
        assert context.code == grpc_module.StatusCode.INVALID_ARGUMENT
        assert "dtype" in context.details

    def test_prepare_inputs_int64_data(self):
        """int64_data 无损承载整型输入。"""
        from tributo.serving.grpc_deployment import _prepare_inputs

        req = inference_pb2.PredictRequest()
        t = req.inputs.add()
        t.schema_version = 1
        t.name = "ids"
        t.datatype = "int64"
        t.int64_data.extend([2**60, 2**60 + 1])  # 超出 double 精度

        result = _prepare_inputs(req)
        assert isinstance(result, dict)
        assert result["ids"].dtype == np.int64
        assert result["ids"].tolist() == [2**60, 2**60 + 1]

    def test_prepare_inputs_duplicate_name_rejected(self):
        """重复输入名应 fail-fast。"""
        from tributo.serving.grpc_deployment import _prepare_inputs

        req = inference_pb2.PredictRequest()
        for _ in range(2):
            t = req.inputs.add()
            t.schema_version = 1
            t.name = "x"
            t.datatype = "float32"
            t.data.extend([1.0])

        with pytest.raises(ValueError, match="Duplicate input name"):
            _prepare_inputs(req)


class TestBatchPredictInputConsistency:
    """BatchPredict 跨请求输入名一致性契约。"""

    def test_inconsistent_input_names_rejected(self):
        """请求间输入名集合不一致 → INVALID_ARGUMENT fail-fast。"""
        import asyncio

        import grpc as grpc_module

        from tributo.serving.grpc_deployment import gRPCInferenceService

        class _FakeContext:
            def __init__(self) -> None:
                self.code = None
                self.details = None

            def set_code(self, code: object) -> None:
                self.code = code

            def set_details(self, message: str) -> None:
                self.details = message

        async def run() -> _FakeContext:
            # Skip __init__ (no model needed): the signature check
            # fires before any inference.
            service = gRPCInferenceService.__new__(gRPCInferenceService)
            service._input_names = ("a",)
            ctx = _FakeContext()

            async def request_stream():
                for names in (("a",), ("b",)):
                    req = inference_pb2.PredictRequest()
                    for n in names:
                        t = req.inputs.add()
                        t.schema_version = 1
                        t.name = n
                        t.datatype = "float32"
                        t.data.extend([1.0])
                    yield req

            await service.BatchPredict(request_stream(), ctx)
            return ctx

        ctx = asyncio.run(run())
        assert ctx.code == grpc_module.StatusCode.INVALID_ARGUMENT
        # The per-request signature check catches the second request
        # ("b" not in the model inputs) before the cross-request
        # consistency pass would.
        assert "do not match" in ctx.details

    def test_consistent_input_names_pass_fast(self):
        """请求间输入名一致时通过一致性检查（进入推理阶段报模型错）。"""
        import asyncio

        from tributo.serving.grpc_deployment import gRPCInferenceService

        class _FakeContext:
            def __init__(self) -> None:
                self.code = None
                self.details = None

            def set_code(self, code: object) -> None:
                self.code = code

            def set_details(self, message: str) -> None:
                self.details = message

        async def run() -> tuple[_FakeContext, list[dict[str, np.ndarray]]]:
            service = gRPCInferenceService.__new__(gRPCInferenceService)
            service._input_names = ("a",)
            ctx = _FakeContext()

            calls: list[dict[str, np.ndarray]] = []

            def fake_run(inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
                calls.append(inputs)
                return [np.array([1.0])]

            service._run = fake_run

            async def request_stream():
                for _ in range(2):
                    req = inference_pb2.PredictRequest()
                    t = req.inputs.add()
                    t.schema_version = 1
                    t.name = "a"
                    t.datatype = "float32"
                    t.data.extend([1.0])
                    yield req

            await service.BatchPredict(request_stream(), ctx)
            return ctx, calls

        ctx, calls = asyncio.run(run())
        # 一致性通过 → 未设置 INVALID_ARGUMENT，且推理收到拼接后的 batch
        assert ctx.code is None
        assert len(calls) == 1
        assert calls[0]["a"].shape == (2,)

    def test_inconsistent_shape_across_requests_rejected(self):
        """同名输入但非 batch 维 shape 不一致 → INVALID_ARGUMENT。"""
        import asyncio

        import grpc as grpc_module

        from tributo.serving.grpc_deployment import gRPCInferenceService

        class _FakeContext:
            def __init__(self) -> None:
                self.code = None
                self.details = None

            def set_code(self, code: object) -> None:
                self.code = code

            def set_details(self, message: str) -> None:
                self.details = message

        async def run() -> _FakeContext:
            service = gRPCInferenceService.__new__(gRPCInferenceService)
            service._input_names = ("a",)
            ctx = _FakeContext()

            async def request_stream():
                for shape in ((1, 2), (1, 3)):  # 非 batch 维 2 vs 3
                    req = inference_pb2.PredictRequest()
                    t = req.inputs.add()
                    t.schema_version = 1
                    t.name = "a"
                    t.shape.extend(list(shape))
                    t.datatype = "float32"
                    t.data.extend([1.0] * shape[1])
                    yield req

            await service.BatchPredict(request_stream(), ctx)
            return ctx

        ctx = asyncio.run(run())
        assert ctx.code == grpc_module.StatusCode.INVALID_ARGUMENT
        assert "non-batch dimensions" in ctx.details

    def test_mixed_modes_across_requests_rejected(self):
        """versioned 与 legacy 混用 → INVALID_ARGUMENT 而非 AssertionError。"""
        import asyncio

        import grpc as grpc_module

        from tributo.serving.grpc_deployment import gRPCInferenceService

        class _FakeContext:
            def __init__(self) -> None:
                self.code = None
                self.details = None

            def set_code(self, code: object) -> None:
                self.code = code

            def set_details(self, message: str) -> None:
                self.details = message

        async def run() -> _FakeContext:
            service = gRPCInferenceService.__new__(gRPCInferenceService)
            service._input_names = ("a",)
            ctx = _FakeContext()

            async def request_stream():
                # 第一条：versioned inputs
                req1 = inference_pb2.PredictRequest()
                t = req1.inputs.add()
                t.schema_version = 1
                t.name = "a"
                t.datatype = "float32"
                t.data.extend([1.0])
                yield req1
                # 第二条：legacy features
                req2 = inference_pb2.PredictRequest(features=[0.5, 0.5])
                yield req2

            await service.BatchPredict(request_stream(), ctx)
            return ctx

        ctx = asyncio.run(run())
        assert ctx.code == grpc_module.StatusCode.INVALID_ARGUMENT
        assert "Mixed input modes" in ctx.details

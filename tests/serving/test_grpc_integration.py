"""gRPC 推理服务集成测试（基于 Docker Ray 集群）。

运行方式：
    # 在集群容器内运行
    docker exec ray-head python -m pytest /opt/tributo/tests/serving/test_grpc_integration.py -sv -W "ignore::FutureWarning"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import grpc
import pytest
import ray

from tributo.serving.grpc_runner import (
    get_grpc_serving_status,
    start_grpc_serving,
    stop_grpc_serving,
)
from tributo.serving.proto import inference_pb2, inference_pb2_grpc

pytestmark = [pytest.mark.slow, pytest.mark.integration]

# 集群配置
GRPC_ADDRESS = "localhost:8001"
MODEL_PATH = "/workspace/onnx/test_completes.onnx"
APP_NAME = "test-grpc-integration"

# tributo 源码路径（容器内 /opt/tributo/src/tributo）
_TRIBUTO_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
    "tributo",
)


def _check_model_path() -> None:
    """检查模型路径是否存在，不存在则跳过集成测试。"""
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"Model file not found: {MODEL_PATH}")


@pytest.fixture(scope="module")
def ray_cluster():
    """连接 Ray 集群，注入 tributo 源码到 worker 节点。"""
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        runtime_env={"py_modules": [_TRIBUTO_SRC]},
    )
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def grpc_app(ray_cluster):
    """通过 Tributo API 启动 gRPC 服务。"""
    _check_model_path()

    start_grpc_serving(
        MODEL_PATH,
        app_name=APP_NAME,
        grpc_port=8001,
        num_replicas=1,
        enable_http=True,
    )

    yield APP_NAME

    # 清理
    stop_grpc_serving(APP_NAME)


@pytest.fixture(scope="module")
def grpc_channel(grpc_app):
    """gRPC 客户端 channel。"""
    channel = grpc.insecure_channel(GRPC_ADDRESS)

    try:
        grpc.channel_ready_future(channel).result(timeout=30)
    except grpc.FutureTimeoutError:
        channel.close()
        pytest.skip(f"gRPC server at {GRPC_ADDRESS} not ready")

    yield channel
    channel.close()


@pytest.fixture
def grpc_stub(grpc_channel):
    """gRPC stub。"""
    return inference_pb2_grpc.InferenceServiceStub(grpc_channel)


def _grpc_metadata():
    """返回包含 application 名称的 gRPC metadata。

    Ray Serve gRPC 代理通过该 metadata 将请求路由到对应 Serve application。
    """
    return (("application", APP_NAME),)


def test_grpc_unary_predict(grpc_stub):
    """Unary RPC 端到端测试。"""
    request = inference_pb2.PredictRequest(
        features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        model_name="test",
    )

    response = grpc_stub.Predict(
        request,
        metadata=_grpc_metadata(),
        timeout=10,
    )

    assert len(response.predictions) > 0
    assert isinstance(response.confidence, float)
    print(
        f"Unary predict: predictions={response.predictions}, confidence={response.confidence}"
    )


def test_grpc_unary_predict_batch(grpc_stub):
    """Unary RPC 批量特征输入。"""
    request = inference_pb2.PredictRequest(
        features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        model_name="test",
    )

    response = grpc_stub.Predict(
        request,
        metadata=_grpc_metadata(),
        timeout=10,
    )

    assert len(response.predictions) > 0
    print(f"Batch predict: predictions={response.predictions}")


def test_grpc_stream_predict(grpc_stub):
    """Server Streaming 端到端测试。"""
    request = inference_pb2.PredictRequest(
        features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        model_name="test",
    )

    responses = list(
        grpc_stub.StreamPredict(
            request,
            metadata=_grpc_metadata(),
            timeout=10,
        )
    )

    assert len(responses) > 0
    for resp in responses:
        assert len(resp.predictions) == 1
        assert isinstance(resp.confidence, float)
    print(f"Stream predict: {len(responses)} responses")


def test_grpc_batch_predict(grpc_stub):
    """Client Streaming 端到端测试。"""

    def request_generator():
        for i in range(3):
            yield inference_pb2.PredictRequest(
                features=[float(i) * 0.1] * 10,
                model_name="test",
            )

    response = grpc_stub.BatchPredict(
        request_generator(),
        metadata=_grpc_metadata(),
        timeout=10,
    )

    assert len(response.predictions) > 0
    assert isinstance(response.confidence, float)
    print(f"Client streaming predict: predictions={response.predictions}")


def test_grpc_concurrent_requests(grpc_stub):
    """并发请求测试。"""
    import concurrent.futures

    def make_request(i):
        request = inference_pb2.PredictRequest(
            features=[float(i) * 0.1] * 10,
            model_name="test",
        )
        return grpc_stub.Predict(
            request,
            metadata=_grpc_metadata(),
            timeout=10,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(5)]
        results = [f.result() for f in futures]

    assert len(results) == 5
    for r in results:
        assert len(r.predictions) > 0
    print(f"Concurrent requests: {len(results)} succeeded")


def test_grpc_invalid_features(grpc_stub):
    """空特征输入处理。"""
    request = inference_pb2.PredictRequest(
        features=[],
        model_name="test",
    )

    with pytest.raises(grpc.RpcError) as exc_info:
        grpc_stub.Predict(
            request,
            metadata=_grpc_metadata(),
            timeout=10,
        )

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    print(f"Empty features raised INVALID_ARGUMENT: {exc_info.value.details()}")


def test_grpc_service_status(grpc_app):
    """gRPC 服务状态查询。"""
    status = get_grpc_serving_status(APP_NAME)

    assert status["running"] is True
    assert status["app_name"] == APP_NAME
    print(f"Service status: {status['status']}")


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))

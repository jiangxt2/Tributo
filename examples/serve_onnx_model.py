"""End-to-end ONNX model online inference serving example.

This example demonstrates:
1. Starting a Ray Serve deployment with an ONNX model
2. Sending HTTP inference requests
3. Stopping the service

Prerequisites:
    - Ray cluster running (Docker or local)
    - ONNX model file at the specified path

Usage:
    uv run python examples/serve_onnx_model.py /workspace/onnx/test_completes.onnx
"""

from __future__ import annotations

import sys
import time

import requests

from tributo.serving import get_serving_status, start_serving, stop_serving

DEFAULT_DASHBOARD = "http://127.0.0.1:8265"
SERVE_HTTP = "http://127.0.0.1:8000"


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path/to/model.onnx>")
        return 1

    model_path = sys.argv[1]
    app_name = "tributo-onnx-demo"

    # 1. Start the service
    print(f"Starting serve with model: {model_path}")
    start_serving(
        model_path=model_path,
        app_name=app_name,
        route_prefix="/predict",
        num_replicas=1,
    )

    # Wait for service to be ready
    for _ in range(10):
        status = get_serving_status(app_name)
        if status["running"]:
            break
        time.sleep(1)
    else:
        print("Serve app did not start in time")
        return 1

    print(f"Serve running at {SERVE_HTTP}/predict")

    # 2. Send inference request
    dummy_features = [[0.0] * 10, [1.0] * 10]
    resp = requests.post(
        f"{SERVE_HTTP}/predict",
        json={"features": dummy_features, "return_probs": True},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()

    print(f"Predictions: {result['predictions']}")
    print(f"Inference time: {result['inference_time_ms']} ms")
    print(f"Model path: {result['model_path']}")

    # 3. Stop the service
    print("Stopping serve...")
    stop_serving(app_name)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""本地连接到 Ray 集群启动 Serve 服务。"""

from __future__ import annotations

import sys
from pathlib import Path

import ray
from ray import serve

from tributo.serving import ONNXModel


def main() -> int:
    root = Path(__file__).parent.parent.parent
    pkg = root / "src" / "tributo"

    ray.init(
        address="ray://127.0.0.1:10001",
        runtime_env={
            "py_modules": [str(pkg)],
            "env_vars": {
                "PYTHONPATH": "/venv/lib/python3.12/site-packages:/home/ray/anaconda3/lib/python3.12/site-packages",
            },
        },
    )

    serve.start(http_options={"host": "0.0.0.0", "port": 8000})

    deployment = serve.deployment(
        num_replicas=1,
        name="tributo-onnx-test",
    )(ONNXModel)

    serve.run(
        deployment.bind(model_path="/workspace/onnx/test_completes.onnx"),
        name="tributo-onnx-test",
        route_prefix="/predict",
    )

    print("Serve started at http://127.0.0.1:8000/predict")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""XGBoost training job script for execution inside a Ray cluster.

Submitted via Ray Jobs API. Results are written to a temporary directory
(configurable via OUTPUT_DIR environment variable).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import ray
from ray.data import from_pandas
from sklearn.model_selection import train_test_split


def main():
    """主入口，从环境变量读取配置，执行训练，写结果到 JSON。"""
    # Running inside cluster — no address needed
    ray.init()

    # Read configuration from environment variables
    test_name = os.environ.get("TEST_NAME", "default")
    output_dir = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "onnx_output"))
    num_rounds = int(os.environ.get("NUM_ROUNDS", "10"))
    max_depth = int(os.environ.get("MAX_DEPTH", "3"))
    eta = float(os.environ.get("ETA", "0.3"))
    early_stopping = os.environ.get("EARLY_STOPPING_ROUNDS")
    max_rows = os.environ.get("MAX_ROWS_PER_WORKER")
    use_val = os.environ.get("USE_VAL", "true").lower() == "true"

    # Generate synthetic data
    rng = np.random.default_rng(42)
    n, f = 800, 10
    X = rng.normal(size=(n, f)).astype(np.float32)
    y = (X @ rng.normal(size=f) > 0).astype(np.int32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(f)])
    df["label"] = y

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = from_pandas(train_df.reset_index(drop=True))
    val_ds = from_pandas(val_df.reset_index(drop=True)) if use_val else None

    # Build training config
    from tributo.training import build_trainer

    onnx_output = f"{output_dir}/{test_name}.onnx"
    train_config = {
        "label_col": "label",
        "xgb_params": {
            "objective": "binary:logistic",
            "max_depth": max_depth,
            "eta": eta,
        },
        "num_rounds": num_rounds,
        "onnx_output": onnx_output,
    }

    if early_stopping:
        train_config["early_stopping_rounds"] = int(early_stopping)
    if max_rows:
        train_config["max_rows_per_worker"] = int(max_rows)

    trainer = build_trainer(
        ray_dataset=train_ds,
        train_config=train_config,
        val_dataset=val_ds,
        num_workers=2,
    )

    result = trainer.fit()

    # Driver-side ONNX export from checkpoint
    from pathlib import Path

    from tributo.training.onnx_exporter import export_from_checkpoint

    n_features = result.metrics.get("n_features", f)
    Path(onnx_output).parent.mkdir(parents=True, exist_ok=True)
    export_from_checkpoint(
        checkpoint=result.checkpoint,
        output_path=onnx_output,
        n_features=n_features,
        target_opset=12,  # 容器内 onnxmltools 只支持到 opset 15
        validate=True,
    )

    # Write results to JSON
    result_json_path = f"{output_dir}/{test_name}_result.json"
    result_data = {
        "onnx_path": onnx_output,
        "metrics": result.metrics,
        "error": str(result.error) if result.error else None,
    }

    with open(result_json_path, "w") as f:
        json.dump(result_data, f, indent=2)

    # 结果通过 stdout 回传（产物在容器内，宿主机测试侧从 logs 解析，
    # 不依赖任何宿主机挂载路径——任何集群部署均可运行）
    print(f"RESULT: {json.dumps(result_data)}")

    print(f"Training completed. Result written to {result_json_path}")
    print(f"ONNX model: {onnx_output}")

    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

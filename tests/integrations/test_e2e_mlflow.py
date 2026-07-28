"""MLflow end-to-end integration test.

Synthetic data → XGBoost training → MLflow tracking → ONNX export → model registry

Usage (inside Docker container):
    docker exec ray-head python /opt/tributo/tests/integrations/test_e2e_mlflow.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import ray
import requests
from mlflow.tracking import MlflowClient

from tributo.registry.callback import MLflowTrackingCallback
from tributo.registry.model_registry import ModelRegistry
from tributo.training.xgboost_trainer import XGBoostTrainerImpl

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _check_mlflow():
    try:
        return (
            requests.get(
                f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/search",
                params={"max_results": 1},
                timeout=3,
            ).status_code
            == 200
        )
    except Exception:
        return False


def main():
    if not _check_mlflow():
        logger.error("MLflow unreachable: %s", MLFLOW_TRACKING_URI)
        sys.exit(1)

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # ── 1. 生成数据 ──
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    df["label"] = y

    if not ray.is_initialized():
        ray.init(address="auto")
    dataset = ray.data.from_pandas(df)
    logger.info("数据: %d 行, %d 特征", len(df), 5)

    # ── 2. 训练 ──
    tmpdir = tempfile.mkdtemp()
    onnx_path = os.path.join(tmpdir, "xgboost_model.onnx")
    ts = str(int(time.time()))[-6:]

    cb = MLflowTrackingCallback(
        experiment_name=f"e2e-xgboost-training-{ts}",
        tracking_uri=MLFLOW_TRACKING_URI,
        run_name=f"xgboost-binary-classifier-{ts}",
    )

    config = {
        "data": {"type": "csv", "label_col": "label"},
        "model": {"objective": "binary:logistic", "max_depth": 5, "eta": 0.1},
        "training": {"num_rounds": 10, "val_size": 0.2, "seed": 42},
        "ray": {"num_workers": 2},
        "output": {"onnx_path": onnx_path},
    }

    trainer = XGBoostTrainerImpl(
        datasets={"train": dataset},
        config=config,
        callbacks=[cb],
    )
    summary = trainer.run(output_path=onnx_path)
    logger.info("训练完成: %s", summary["status"])

    # ── 3. 验证 ──
    assert summary["status"] == "succeeded"
    assert os.path.exists(onnx_path)
    logger.info("ONNX: %d bytes", os.path.getsize(onnx_path))

    run = client.get_run(cb._run_id)
    assert run.info.status == "FINISHED"
    logger.info("MLflow run: %s", cb._run_id)
    logger.info("  params: %s", dict(run.data.params))
    logger.info("  metrics: %s", dict(run.data.metrics))

    artifacts = client.list_artifacts(cb._run_id)
    artifact_names = [a.path for a in artifacts]
    assert "xgboost_model.onnx" in artifact_names
    logger.info("  artifacts: %s", artifact_names)

    # ── 4. 注册模型 ──
    model_name = f"xgboost-binary-classifier-{ts}"
    reg = ModelRegistry(tracking_uri=MLFLOW_TRACKING_URI)
    mv = reg.register_model(
        model_uri=f"runs:/{cb._run_id}/xgboost_model.onnx",
        name=model_name,
        tags={
            "framework": "xgboost",
            "format": "onnx",
            "task": "binary-classification",
        },
        description="XGBoost binary classifier, 5 features, trained on synthetic data",
    )
    logger.info("模型注册: %s v%d", mv.name, mv.version)

    reg.transition_stage(model_name, mv.version, "Production")
    logger.info("模型阶段: Production")

    logger.info("✅ 端到端测试通过")


if __name__ == "__main__":
    main()

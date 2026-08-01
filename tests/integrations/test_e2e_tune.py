"""Tune end-to-end integration test.

Pipeline:
  Generate data → FIFO hyperparameter search → ASHA early-stopping validation
  → Train with best params (MLflow tracking) → Export ONNX → Register model

Usage (inside Docker container):
    docker cp tests/integrations/test_e2e_tune.py ray-head:/opt/tributo/tests/integrations/
    docker exec ray-head python /opt/tributo/tests/integrations/test_e2e_tune.py

Prerequisites:
    - Docker Ray cluster (ray-head @ 127.0.0.1)
    - MLflow Server (localhost:5000)
    - MinIO (localhost:9000)
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pytest
import ray
import requests

from tributo.registry.callback import MLflowTrackingCallback
from tributo.registry.model_registry import ModelRegistry
from tributo.training.xgboost_trainer import XGBoostTrainerImpl

# mlflow lives in the registry extra - skip collection without it
# (importorskip comes after all imports to avoid E402).
mlflow = pytest.importorskip("mlflow", reason="mlflow not installed")
MlflowClient = mlflow.tracking.MlflowClient

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

    # ── 2. Tune 端到端测试 ──
    tmpdir = tempfile.mkdtemp()
    try:
        _run_tune_pipeline(client, dataset, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_tune_pipeline(client, dataset, tmpdir: str) -> None:
    """Tune 端到端测试主流程（在 tmpdir 内执行，调用方负责清理）。"""
    from ray import tune
    from ray.tune import RunConfig, TuneConfig, Tuner
    from ray.tune.schedulers import ASHAScheduler

    from tributo.training import TuneRunner, TuneSearchConfig, extract_best_params
    from tributo.training.registry import get_trainer

    # ── 2a. FIFO 搜索最优参数（TuneRunner + XGBoostTrainer） ──
    trainer_spec = get_trainer("xgboost")
    search_space = {
        "max_depth": tune.choice([3, 5, 7]),
        "learning_rate": tune.uniform(0.01, 0.3),
    }
    tune_config = TuneSearchConfig(
        metric="train-logloss",
        mode="min",
        num_samples=4,
        search_alg="random",
        scheduler="fifo",
    )

    tune_output = os.path.join(tmpdir, "tune_output")
    os.makedirs(tune_output, exist_ok=True)

    runner = TuneRunner(trainer_spec, tune_config, search_space)
    result_grid = runner.run(
        datasets={"train": dataset},
        output_path=tune_output,
        experiment_name="e2e-tune-fifo",
    )

    best_params = extract_best_params(result_grid, metric="train-logloss", mode="min")
    logger.info("FIFO 最优参数: %s", best_params)
    assert len(result_grid) >= 4, f"至少 4 个 trial，实际 {len(result_grid)}"
    assert "max_depth" in best_params, f"最优参数缺少 max_depth: {best_params}"
    assert "learning_rate" in best_params, f"最优参数缺少 learning_rate: {best_params}"

    # ── 3. ASHA 早停验证（多步 trainable，模拟效果差的 trial 被提前终止） ──
    logger.info("=" * 50)
    logger.info("ASHA 早停验证")
    logger.info("=" * 50)

    def asha_trainable(config: dict) -> None:
        """多步 trainable：loss 随 lr 线性增长，lr 大的 trial 效果差。"""
        lr = config.get("learning_rate", 0.01)
        base_loss = 1.0
        for step in range(20):
            loss = base_loss * (0.9**step) + lr * 10
            tune.report({"loss": loss, "step": step})

    asha_search_space = {
        "learning_rate": tune.loguniform(0.001, 0.1),
    }
    asha_scheduler = ASHAScheduler(
        max_t=20,
        grace_period=5,
        reduction_factor=2,
    )

    asha_tuner = Tuner(
        trainable=asha_trainable,
        param_space=asha_search_space,
        tune_config=TuneConfig(
            metric="loss",
            mode="min",
            num_samples=8,
            scheduler=asha_scheduler,
        ),
        run_config=RunConfig(
            name="e2e-tune-asha",
            storage_path=os.path.join(tmpdir, "asha_output"),
            verbose=1,
        ),
    )

    asha_result = asha_tuner.fit()
    terminated_trials = [r for r in asha_result if not r.error]
    early_stopped = [
        r
        for r in terminated_trials
        if r.metrics and r.metrics.get("training_iteration", 20) < 20
    ]

    logger.info(
        "ASHA 结果: %d trials, %d 早停", len(terminated_trials), len(early_stopped)
    )
    for r in terminated_trials:
        iters = r.metrics.get("training_iteration", "?") if r.metrics else "?"
        loss = r.metrics.get("loss", float("nan")) if r.metrics else float("nan")
        logger.info(
            "  iters=%s, loss=%.4f, lr=%.4f",
            iters,
            loss,
            r.config.get("learning_rate", 0),
        )

    best_asha = asha_result.get_best_result()
    logger.info(
        "ASHA 最优: loss=%.4f, lr=%.4f",
        best_asha.metrics["loss"],
        best_asha.config["learning_rate"],
    )
    assert len(terminated_trials) > 0, "没有成功的 trial"
    assert len(early_stopped) > 0, "ASHA 未触发早停，所有 trial 都跑满了 20 步"

    # ── 4. 用 FIFO 最优参数训练 + MLflow 跟踪 ──
    ts = str(int(time.time()))[-6:]
    onnx_path = os.path.join(tmpdir, "tune_model.onnx")

    cb = MLflowTrackingCallback(
        experiment_name=f"e2e-tune-training-{ts}",
        tracking_uri=MLFLOW_TRACKING_URI,
        run_name=f"tune-best-params-{ts}",
    )

    final_config = {
        "data": {"label_col": "label"},
        "model": {
            "objective": "binary:logistic",
            "max_depth": int(best_params["max_depth"]),
            "learning_rate": float(best_params["learning_rate"]),
        },
        "training": {"num_rounds": 10, "val_size": 0.2, "seed": 42},
        "ray": {"num_workers": 2},
    }

    trainer = XGBoostTrainerImpl(
        datasets={"train": dataset},
        config=final_config,
        callbacks=[cb],
    )
    summary = trainer.run(output_path=onnx_path)
    logger.info("训练完成: %s", summary["status"])

    # ── 5. 验证 ──
    assert summary["status"] == "succeeded", f"训练失败: {summary}"
    assert os.path.exists(onnx_path), f"ONNX 文件不存在: {onnx_path}"
    logger.info("ONNX: %d bytes", os.path.getsize(onnx_path))

    run = client.get_run(cb._run_id)
    assert run.info.status == "FINISHED"
    logger.info("MLflow run: %s", cb._run_id)
    logger.info("  params: %s", dict(run.data.params))
    logger.info("  metrics: %s", dict(run.data.metrics))

    # ── 6. 注册模型 ──
    model_name = f"e2e-tune-model-{ts}"
    reg = ModelRegistry(tracking_uri=MLFLOW_TRACKING_URI)
    mv = reg.register_model(
        model_uri=f"runs:/{cb._run_id}/tune_model.onnx",
        name=model_name,
        tags={
            "framework": "xgboost",
            "format": "onnx",
            "task": "binary-classification",
        },
        description="XGBoost binary classifier with Tune-optimized hyperparameters",
    )
    logger.info("模型注册: %s v%d", mv.name, mv.version)

    reg.transition_stage(model_name, mv.version, "Production")
    logger.info("模型阶段: Production")

    logger.info("✅ Tune 端到端测试全部通过")


if __name__ == "__main__":
    main()

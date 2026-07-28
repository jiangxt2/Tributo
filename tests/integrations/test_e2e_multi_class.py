"""Multi-class XGBoost 端到端集成测试（ClickHouse 数据源）。

3 类合成数据 → multi:softprob 分布式训练 → sklearn macro 指标 → ONNX 导出

运行方式：
    docker exec ray-head python /opt/tributo/tests/integration/test_e2e_multi_class.py

前提：Docker 集群已启动，含 ClickHouse (8123) / MLflow (5000)。
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time

import numpy as np
import ray
import requests

from tributo.registry.callback import MLflowTrackingCallback
from tributo.training.data_loader import load_ray_dataset_from_config
from tributo.training.xgboost_trainer import XGBoostTrainerImpl

CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", 8123))
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "reader")
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "tributo123")
CLICKHOUSE_DB = os.environ.get("CLICKHOUSE_DB", "analytics")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
TABLE = "tributo_e2e_multi_class_test"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _check_clickhouse() -> bool:
    try:
        import clickhouse_connect

        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            database=CLICKHOUSE_DB,
        )
        client.command("SELECT 1")
        return True
    except Exception as e:
        logger.warning("ClickHouse 不可达: %s", e)
        return False


def _check_mlflow() -> bool:
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


def _prepare_clickhouse_table() -> int:
    """Drop & recreate a 3-class dataset in ClickHouse. Returns row count."""
    import clickhouse_connect

    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DB,
    )
    client.command(f"DROP TABLE IF EXISTS {TABLE}")
    client.command(f"""
        CREATE TABLE {TABLE} (
            feature_0 Float64,
            feature_1 Float64,
            feature_2 Float64,
            feature_3 Float64,
            feature_4 Float64,
            label Int32
        ) ENGINE = MergeTree()
        ORDER BY tuple()
    """)

    np.random.seed(42)
    n = 3000
    # 3 classes: 0, 1, 2 based on thresholds of f0 + f1
    f0 = np.random.randn(n)
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    f3 = np.random.randn(n)
    f4 = np.random.randn(n)
    score = f0 + f1
    # Assign class 0, 1, or 2 based on quantiles
    labels = np.digitize(score, bins=[-0.5, 0.5]).astype(np.int32)
    rows = [
        (
            float(f0[i]),
            float(f1[i]),
            float(f2[i]),
            float(f3[i]),
            float(f4[i]),
            int(labels[i]),
        )
        for i in range(n)
    ]
    client.insert(TABLE, rows)
    logger.info(
        "ClickHouse 3-class 表 %s: %d 行已写入 (分布: %s)",
        TABLE,
        n,
        dict(zip(*np.unique(labels, return_counts=True))),
    )
    return n


def main():
    if not _check_clickhouse():
        logger.error("ClickHouse 不可达 (%s:%d)", CLICKHOUSE_HOST, CLICKHOUSE_PORT)
        return 1
    logger.info("ClickHouse 连接正常 ✅")

    if not _check_mlflow():
        logger.error("MLflow 不可达: %s", MLFLOW_TRACKING_URI)
        return 1
    logger.info("MLflow 连接正常 ✅")

    # ── 1. 写入 ClickHouse 3-class 测试数据 ──
    _prepare_clickhouse_table()

    # ── 2. 连接 Ray ──
    if not ray.is_initialized():
        ray.init(address="auto")
    logger.info("Ray 集群就绪: %s", ray.cluster_resources())

    # ── 3. 从 ClickHouse 加载数据 ──
    data_config = {
        "type": "clickhouse",
        "label_col": "label",
        "ch_host": CLICKHOUSE_HOST,
        "ch_port": CLICKHOUSE_PORT,
        "ch_database": CLICKHOUSE_DB,
        "ch_user": CLICKHOUSE_USER,
        "ch_password": CLICKHOUSE_PASSWORD,
        "ch_sql": f"SELECT * FROM {CLICKHOUSE_DB}.{TABLE}",
    }
    ds = load_ray_dataset_from_config(data_config)
    logger.info("Ray Dataset 加载完成: %d 行", ds.count())

    # ── 4. Multi-class 分布式训练 + MLflow ──
    tmpdir = tempfile.mkdtemp()
    onnx_path = os.path.join(tmpdir, "xgboost_multiclass.onnx")
    ts = str(int(time.time()))[-6:]

    cb = MLflowTrackingCallback(
        experiment_name=f"e2e-multiclass-xgboost-{ts}",
        tracking_uri=MLFLOW_TRACKING_URI,
        run_name=f"multiclass-softprob-3-{ts}",
    )

    config = {
        "data": data_config,
        "model": {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 5,
            "eta": 0.1,
        },
        "training": {"num_rounds": 10, "val_size": 0.2, "test_size": 0.1, "seed": 42},
        "ray": {"num_workers": 2},
        "output": {"onnx_path": onnx_path},
    }

    trainer = XGBoostTrainerImpl(
        datasets={"train": ds},
        config=config,
        callbacks=[cb],
    )
    summary = trainer.run(output_path=onnx_path)
    logger.info("训练完成: %s", summary["status"])

    # ── 5. 验证 ──
    assert summary["status"] == "succeeded", f"训练失败: {summary}"
    assert os.path.exists(onnx_path), f"ONNX 模型未生成: {onnx_path}"
    logger.info("ONNX 模型: %d 字节", os.path.getsize(onnx_path))

    # ── 6. 验证多分类指标 ──
    metrics = summary.get("metrics", {})
    logger.info("训练指标 keys: %s", list(metrics.keys())[:20])

    # Verify multi-class specific metrics are present
    eval_f1 = metrics.get("eval_f1_macro")
    assert eval_f1 is not None, (
        "Missing eval_f1_macro — multi-class eval may have been skipped"
    )
    assert eval_f1 > 0.0, f"eval_f1_macro should be positive, got {eval_f1}"
    logger.info("eval_f1_macro = %.4f", eval_f1)

    eval_auc = metrics.get("eval_auc")
    assert eval_auc is not None, "Missing eval_auc (ovr) for multi-class"
    assert eval_auc > 0.0, f"eval_auc (ovr) should be positive, got {eval_auc}"
    logger.info("eval_auc (ovr) = %.4f", eval_auc)

    # Verify no stray binary-only fields
    assert "eval_thr_thresholds" not in metrics, (
        "Binary threshold sweep leaked into multi-class"
    )
    assert "eval_cm_tp" not in metrics, (
        "Binary confusion matrix leaked into multi-class"
    )

    # Verify complete confusion matrix (3x3 → list of lists)
    eval_cm = metrics.get("eval_cm")
    assert eval_cm is not None, "Missing eval_cm (multi-class confusion matrix)"
    logger.info("混淆矩阵: %s", eval_cm)

    logger.info("✅ Multi-class 端到端测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

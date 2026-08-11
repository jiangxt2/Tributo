"""ClickHouse → XGBoost 端到端集成测试。

通过 ClickHouse 数据源加载数据，分布式 XGBoost 训练，MLflow 记录。

运行方式：
    docker exec ray-head python /opt/tributo/tests/integration/test_e2e_clickhouse.py

前提：Docker 集群已启动，含 Ray / Daft / daft-olap-connectors /
ClickHouse (8123) / MinIO / MLflow (5000)。
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

from tributo.data.handle_adapters import adapt_daft_result_to_ray
from tributo.data.ingestion import IngestionRequest, open_ingestion
from tributo.data.source_config import SqlSourceConfig
from tributo.registry.callback import MLflowTrackingCallback
from tributo.training.xgboost_trainer import XGBoostTrainerImpl

CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", 8123))
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "reader")
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "tributo123")
CLICKHOUSE_DB = os.environ.get("CLICKHOUSE_DB", "analytics")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
TABLE = "tributo_e2e_clickhouse_test"

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
    except Exception as exc:
        logger.warning("ClickHouse 不可达: exception_type=%s", type(exc).__name__)
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
    """Drop & recreate test table with 2000 rows of synthetic data. Returns row count."""
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
    n = 2000
    rows = [
        (float(f0), float(f1), float(f2), float(f3), float(f4), int(f0 + f1 > 0))
        for f0, f1, f2, f3, f4 in zip(*(np.random.randn(n) for _ in range(5)))
    ]
    client.insert(TABLE, rows)
    logger.info("ClickHouse 表 %s: %d 行已写入", TABLE, n)
    return n


def main():
    # ── 0. 前置检查 ──
    if not _check_clickhouse():
        logger.error("ClickHouse 不可达 (%s:%d)", CLICKHOUSE_HOST, CLICKHOUSE_PORT)
        return 1
    logger.info("ClickHouse 连接正常 ✅")

    if not _check_mlflow():
        logger.error("MLflow 不可达: %s", MLFLOW_TRACKING_URI)
        return 1
    logger.info("MLflow 连接正常 ✅")

    # ── 1. 写入 ClickHouse 测试数据 ──
    _prepare_clickhouse_table()

    # ── 2. 连接 Ray ──
    if not ray.is_initialized():
        ray.init(address="auto")
    logger.info("Ray 集群就绪: %s", ray.cluster_resources())

    # ── 3. 从 ClickHouse 加载数据 ──
    source = SqlSourceConfig(
        dialect="clickhouse",
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        database=CLICKHOUSE_DB,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        table=TABLE,
    )
    ingestion = open_ingestion(IngestionRequest(source=source, engine="daft"))
    try:
        ds = adapt_daft_result_to_ray(ingestion).handle.dataset
    finally:
        ingestion.close()
    logger.info("Ray Dataset 加载完成: %d 行", ds.count())

    # ── 4. 分布式训练 + MLflow 记录 ──
    tmpdir = tempfile.mkdtemp()
    onnx_path = os.path.join(tmpdir, "xgboost_model.onnx")
    ts = str(int(time.time()))[-6:]

    cb = MLflowTrackingCallback(
        experiment_name=f"e2e-clickhouse-xgboost-{ts}",
        tracking_uri=MLFLOW_TRACKING_URI,
        run_name=f"clickhouse-binary-classifier-{ts}",
    )

    config = {
        "data": {"label_col": "label"},
        "model": {"objective": "binary:logistic", "max_depth": 5, "eta": 0.1},
        "training": {"num_rounds": 10, "val_size": 0.2, "seed": 42},
        "ray": {"num_workers": 2},
        "output": {"onnx_path": onnx_path},
    }

    trainer = XGBoostTrainerImpl(
        datasets={"train": ds},
        config=config,
        callbacks=[cb],
    )
    summary = trainer.run(output_path=onnx_path, legacy_export=True)
    logger.info("训练完成: %s", summary["status"])

    # ── 5. 验证 ──
    assert summary["status"] == "succeeded", f"训练失败: {summary}"
    assert os.path.exists(onnx_path), f"ONNX 模型未生成: {onnx_path}"
    logger.info("ONNX 模型: %d 字节", os.path.getsize(onnx_path))

    logger.info("✅ ClickHouse 端到端测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

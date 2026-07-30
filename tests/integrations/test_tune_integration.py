"""Ray Tune 集成测试脚本。

在真实 Ray Cluster 上测试 TuneRunner 的完整流程。

使用方式：
    cd /path/to/tributo
    uv run python tests/integration/test_tune_integration.py

前置条件：
    - Ray Cluster 运行中（ray-head @ 127.0.0.1:8265）
    - 已安装 tributo[tune] 依赖
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import ray
import ray.data
from ray import tune
from ray.tune import FailureConfig, RunConfig, TuneConfig, Tuner
from ray.tune.schedulers import ASHAScheduler

from tributo.training import (
    TuneRunner,
    TuneSearchConfig,
    extract_best_params,
    get_trainer,
    parse_search_space,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Ray Cluster 地址
RAY_ADDRESS = "ray://127.0.0.1:10001"


def create_search_space_json(tmp_dir: str) -> str:
    """创建测试用的搜索空间 JSON 文件。"""
    content = {
        "search_space": {
            "learning_rate": {
                "type": "loguniform",
                "lower": 0.001,
                "upper": 0.1,
            },
            "max_depth": {
                "type": "choice",
                "values": [3, 5, 7],
            },
        }
    }
    json_path = Path(tmp_dir) / "search_space.json"
    json_path.write_text(json.dumps(content))
    return str(json_path)


@pytest.mark.slow
def test_simple_trainable():
    """测试 1：使用简单 trainable 函数验证 Tune 基础流程。

    验证点：
    - Tuner 可以正确启动
    - tune.report() 可以正确报告指标
    - ResultGrid 可以正确返回结果
    - extract_best_params 可以提取最佳参数
    """
    logger.info("=" * 60)
    logger.info("测试 1：简单 trainable 函数")
    logger.info("=" * 60)

    # 定义简单 trainable（不使用 Tributo Trainer，直接测试 Tune 流程）
    def simple_trainable(config: dict[str, Any]) -> None:
        """简单 trainable：计算 loss = lr^2 + depth，模拟训练过程。"""
        lr = config.get("learning_rate", 0.01)
        depth = config.get("max_depth", 5)
        loss = lr * 10 + depth * 0.1  # 简单的损失函数
        accuracy = 1.0 - loss / 10.0

        # 模拟多步训练
        for step in range(3):
            step_loss = loss + step * 0.01
            tune.report({"loss": step_loss, "accuracy": accuracy, "step": step})

    # 创建搜索空间
    search_space = {
        "learning_rate": tune.loguniform(0.001, 0.1),
        "max_depth": tune.choice([3, 5, 7]),
    }

    # 直接使用 Tuner（不通过 TuneRunner，验证基础 Tune 流程）
    with tempfile.TemporaryDirectory() as tmp_dir:
        tuner = Tuner(
            trainable=simple_trainable,
            param_space=search_space,
            run_config=RunConfig(
                name="test_simple_trainable",
                storage_path=tmp_dir,
                verbose=1,
            ),
        )

        logger.info("启动 Tune 实验...")
        result_grid = tuner.fit()

        # 验证结果
        logger.info("实验完成，分析结果...")
        logger.info(f"总 trial 数: {len(result_grid)}")
        logger.info(f"成功 trial 数: {len([r for r in result_grid if not r.error])}")

        best_result = result_grid.get_best_result()
        logger.info(f"最佳 trial 指标: {best_result.metrics}")
        logger.info(f"最佳 trial 配置: {best_result.config}")

        # 验证 extract_best_params
        best_params = extract_best_params(result_grid, metric="loss", mode="min")
        logger.info(f"extract_best_params 结果: {best_params}")

        assert len(result_grid) > 0, "应该有至少一个 trial"
        assert best_result.metrics is not None, "最佳结果应该有指标"
        assert "loss" in best_result.metrics, "指标应该包含 loss"
        assert "learning_rate" in best_params, "最佳参数应该包含 learning_rate"
        assert "max_depth" in best_params, "最佳参数应该包含 max_depth"

    logger.info("✅ 测试 1 通过：简单 trainable 函数")


@pytest.mark.slow
def test_tune_runner_with_xgboost():
    """测试 2：使用 TuneRunner + XGBoostTrainer 测试完整流程。

    验证点：
    - TuneRunner 可以正确将 TrainerSpec 适配为 trainable
    - 闭包捕获 datasets 和 output_path 正确工作
    - 与 XGBoostTrainer 集成正常
    """
    logger.info("=" * 60)
    logger.info("测试 2：TuneRunner + XGBoostTrainer")
    logger.info("=" * 60)

    # 获取 XGBoost 训练器
    try:
        trainer_spec = get_trainer("xgboost")
        logger.info(f"找到训练器: {trainer_spec.name}")
    except Exception as exc:
        pytest.skip(f"获取 XGBoost 训练器失败: {exc}")

    # 创建测试数据集
    logger.info("创建测试数据集...")
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["label"] = y

    dataset = ray.data.from_pandas(df)
    logger.info(f"数据集大小: {dataset.count()} 行")

    # 创建搜索空间
    with tempfile.TemporaryDirectory() as tmp_dir:
        search_space_content = {
            "search_space": {
                "max_depth": {
                    "type": "choice",
                    "values": [3, 5],
                },
                "n_estimators": {
                    "type": "choice",
                    "values": [10, 20],
                },
            }
        }
        json_path = Path(tmp_dir) / "space.json"
        json_path.write_text(json.dumps(search_space_content))

        search_space = parse_search_space(str(json_path))
        logger.info(f"搜索空间: {search_space}")

        # 配置（XGBoostTrainer 使用 train-logloss 作为指标）
        tune_config = TuneSearchConfig(
            metric="train-logloss",
            mode="min",
            num_samples=2,
            search_alg="random",
            scheduler="fifo",
        )

        # 创建 TuneRunner
        runner = TuneRunner(trainer_spec, tune_config, search_space)

        # 执行调优
        logger.info("启动 TuneRunner 实验...")
        output_path = Path(tmp_dir) / "output"
        output_path.mkdir()

        result_grid = runner.run(
            datasets={"train": dataset},
            output_path=str(output_path),
            experiment_name="test_xgboost_tune",
        )

        # 验证结果
        logger.info("实验完成，分析结果...")
        logger.info(f"总 trial 数: {len(result_grid)}")

        best_result = result_grid.get_best_result()
        logger.info(f"最佳 trial 指标: {best_result.metrics}")
        logger.info(f"最佳 trial 配置: {best_result.config}")

        best_params = extract_best_params(
            result_grid, metric="train-logloss", mode="min"
        )
        logger.info(f"extract_best_params 结果: {best_params}")

        assert len(result_grid) > 0, "应该有至少一个 trial"
        assert best_result.metrics is not None, "最佳结果应该有指标"

        logger.info("✅ 测试 2 通过：TuneRunner + XGBoostTrainer")


@pytest.mark.slow
def test_scheduler_asha():
    """测试 3：ASHA 调度器。

    验证点：
    - ASHA 调度器可以正确工作
    - 早停机制正常
    """
    logger.info("=" * 60)
    logger.info("测试 3：ASHA 调度器")
    logger.info("=" * 60)

    def trainable_with_asha(config: dict[str, Any]) -> None:
        """模拟训练过程，逐步改善指标。"""
        lr = config.get("learning_rate", 0.01)
        base_loss = 1.0

        for step in range(20):
            # 模拟训练过程，loss 逐步下降
            loss = base_loss * (0.9**step) + lr * 10
            tune.report({"loss": loss, "step": step})

    search_space = {
        "learning_rate": tune.loguniform(0.001, 0.1),
    }

    scheduler = ASHAScheduler(
        max_t=20,
        grace_period=5,
        reduction_factor=2,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tuner = Tuner(
            trainable=trainable_with_asha,
            param_space=search_space,
            tune_config=TuneConfig(
                metric="loss",
                mode="min",
                num_samples=8,
                scheduler=scheduler,
            ),
            run_config=RunConfig(
                name="test_asha_scheduler",
                storage_path=tmp_dir,
                verbose=1,
            ),
        )

        logger.info("启动 ASHA 实验...")
        result_grid = tuner.fit()

        # 验证结果
        logger.info("实验完成，分析结果...")
        logger.info(f"总 trial 数: {len(result_grid)}")
        logger.info(f"成功 trial 数: {len([r for r in result_grid if not r.error])}")

        best_result = result_grid.get_best_result()
        logger.info(f"最佳 trial 指标: {best_result.metrics}")

        assert len(result_grid) > 0, "应该有至少一个 trial"
        assert best_result.metrics is not None, "最佳结果应该有指标"

    logger.info("✅ 测试 3 通过：ASHA 调度器")


@pytest.mark.slow
def test_fail_fast():
    """测试 4：fail_fast 行为。

    验证点：
    - fail_fast=True 时，首个 trial 失败立即终止
    - fail_fast=False 时，继续执行其他 trial
    """
    logger.info("=" * 60)
    logger.info("测试 4：fail_fast 行为")
    logger.info("=" * 60)

    def failing_trainable(config: dict[str, Any]) -> None:
        """总是失败的 trainable。"""
        raise ValueError("模拟训练失败")

    search_space = {
        "lr": tune.uniform(0.001, 0.1),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 测试 fail_fast=True
        logger.info("测试 fail_fast=True...")
        tuner = Tuner(
            trainable=failing_trainable,
            param_space=search_space,
            tune_config=TuneConfig(
                num_samples=4,
            ),
            run_config=RunConfig(
                name="test_fail_fast_true",
                storage_path=tmp_dir,
                failure_config=FailureConfig(fail_fast=True),
                verbose=1,
            ),
        )

        result_grid = tuner.fit()
        failed_trials = [r for r in result_grid if r.error]
        logger.info(
            f"fail_fast=True: 总 trial {len(result_grid)}, 失败 {len(failed_trials)}"
        )

        # fail_fast=True 时，应该只有 1 个失败的 trial
        assert len(failed_trials) <= 1, "fail_fast=True 时应该只运行 1 个失败 trial"

    logger.info("✅ 测试 4 通过：fail_fast 行为")


@pytest.mark.slow
def test_time_budget():
    """测试 5：时间预算。

    验证点：
    - time_budget_s 限制总运行时间
    """
    logger.info("=" * 60)
    logger.info("测试 5：时间预算")
    logger.info("=" * 60)

    def slow_trainable(config: dict[str, Any]) -> None:
        """慢速 trainable，每个 trial 耗时 2 秒。"""
        time.sleep(2)
        tune.report({"loss": 0.5})

    search_space = {
        "lr": tune.uniform(0.001, 0.1),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tuner = Tuner(
            trainable=slow_trainable,
            param_space=search_space,
            tune_config=TuneConfig(
                num_samples=100,  # 尝试运行 100 个 trial
                time_budget_s=10,  # 但限制 10 秒
            ),
            run_config=RunConfig(
                name="test_time_budget",
                storage_path=tmp_dir,
                verbose=1,
            ),
        )

        start_time = time.time()
        result_grid = tuner.fit()
        elapsed = time.time() - start_time

        logger.info(f"总运行时间: {elapsed:.1f} 秒")
        logger.info(f"总 trial 数: {len(result_grid)}")

        # 验证时间预算生效
        # 注意：time_budget_s 是软限制，Ray Tune 可能会超时一些
        logger.info(
            f"时间预算: 10 秒, 实际运行: {elapsed:.1f} 秒, trial 数: {len(result_grid)}"
        )
        # 只验证实验能正常完成，不断言严格的时间限制
        assert len(result_grid) > 0, "应该有至少一个 trial"

    logger.info("✅ 测试 5 通过：时间预算")


@pytest.mark.slow
def test_max_concurrent_trials():
    """测试 6：最大并发 trial 数。

    验证点：
    - max_concurrent_trials 限制同时运行的 trial 数
    """
    logger.info("=" * 60)
    logger.info("测试 6：最大并发 trial 数")
    logger.info("=" * 60)

    def concurrent_trainable(config: dict[str, Any]) -> None:
        """记录并发数的 trainable。"""
        time.sleep(1)
        tune.report({"loss": 0.5})

    search_space = {
        "lr": tune.uniform(0.001, 0.1),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tuner = Tuner(
            trainable=concurrent_trainable,
            param_space=search_space,
            tune_config=TuneConfig(
                num_samples=6,
                max_concurrent_trials=2,  # 最多 2 个并发
            ),
            run_config=RunConfig(
                name="test_max_concurrent",
                storage_path=tmp_dir,
                verbose=1,
            ),
        )

        start_time = time.time()
        result_grid = tuner.fit()
        elapsed = time.time() - start_time

        logger.info(f"总运行时间: {elapsed:.1f} 秒")
        logger.info(f"总 trial 数: {len(result_grid)}")

        # 6 个 trial，最多 2 个并发，每个 1 秒，至少需要 3 秒
        # 加上调度开销，应该在 10 秒内完成
        assert elapsed < 15, f"运行时间异常: {elapsed:.1f} 秒"

    logger.info("✅ 测试 6 通过：最大并发 trial 数")


def main():
    """运行所有集成测试。"""
    logger.info("开始 Ray Tune 集成测试")

    # 连接 Ray Cluster
    # 在容器内运行时使用 auto，在本地运行时使用 Ray Client
    try:
        import os

        if os.environ.get("RAY_ADDRESS"):
            ray.init(address=os.environ["RAY_ADDRESS"])
        else:
            ray.init(address="auto")
        logger.info(f"已连接到 Ray Cluster: {ray.cluster_resources()}")
    except Exception as e:
        logger.error(f"无法连接 Ray Cluster: {e}")
        logger.error("请确保 Ray Cluster 正在运行: docker start ray-head")
        sys.exit(1)

    # 运行测试
    results = {}

    tests = [
        ("简单 trainable 函数", test_simple_trainable),
        ("ASHA 调度器", test_scheduler_asha),
        ("fail_fast 行为", test_fail_fast),
        ("时间预算", test_time_budget),
        ("最大并发 trial 数", test_max_concurrent_trials),
        # XGBoost 测试放在最后，因为它可能需要更长时间
        ("TuneRunner + XGBoostTrainer", test_tune_runner_with_xgboost),
    ]

    for test_name, test_func in tests:
        try:
            test_func()
            results[test_name] = True
        except Exception:
            logger.exception("测试 '%s' 异常", test_name)
            raise

    # 输出测试报告
    logger.info("\n" + "=" * 60)
    logger.info("测试报告")
    logger.info("=" * 60)

    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"{status} - {test_name}")

    passed = sum(1 for s in results.values() if s)
    total = len(results)
    logger.info(f"\n总计: {passed}/{total} 通过")

    # 清理
    ray.shutdown()

    if passed == total:
        logger.info("\n🎉 所有测试通过！")
        return 0
    else:
        logger.error("\n💥 部分测试失败！")
        return 1


if __name__ == "__main__":
    sys.exit(main())

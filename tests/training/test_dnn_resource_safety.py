"""DNN 单 worker 资源安全测试。

覆盖 DNNTrainingConfig 的 resource 预算默认值，以及 worker 内 train/val
共享预算的超限 fail-fast。worker 级测试通过 mock
``ray.train`` 上下文运行，不依赖真实 Ray 集群。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


class TestDNNResourceConfig:
    """DNNTrainingConfig resource 配置。"""

    def test_default_resource_budget_is_active(self):
        """预算默认启用。"""
        from tributo.training.dnn_trainer import DNNTrainingConfig
        from tributo.training.resource import MIB

        cfg = DNNTrainingConfig()
        assert cfg.resource.max_batch_bytes == 64 * MIB
        assert cfg.resource.max_worker_materialization_bytes == 1024 * MIB
        assert cfg.resource.max_input_rows_per_worker is None

    def test_custom_resource_budget(self):
        from tributo.training.dnn_trainer import DNNTrainingConfig

        cfg = DNNTrainingConfig(resource={"max_batch_bytes": 1024})
        assert cfg.resource.max_batch_bytes == 1024

    def test_resource_passed_through_train_loop_config(self):
        """resource 随 train_loop_config 传给 worker。"""
        from tributo.training.dnn_trainer import DNNTrainerImpl

        trainer = DNNTrainerImpl(
            datasets={},
            config={
                "features": [],
                "resource": {"max_worker_materialization_bytes": 2048},
            },
        )
        assert trainer._train_config.resource.max_worker_materialization_bytes == 2048


class TestDNNWorkerBudget:
    """worker 内预算校验（mock ray.train，无真实集群）。"""

    def _patch_ray(self, monkeypatch, schema=None, batch=None):
        """Mock ray.train context + shards; returns the fake dataset factory."""
        from types import SimpleNamespace

        import ray.train

        monkeypatch.setattr(
            ray.train,
            "get_context",
            lambda: SimpleNamespace(get_world_rank=lambda: 0),
        )

        class FakeShard:
            def schema(self):
                return schema

            def iter_batches(self, **kwargs):
                if batch is not None:
                    yield batch

        monkeypatch.setattr(ray.train, "get_dataset_shard", lambda key: FakeShard())

    def test_worker_budget_exceeded_fails_before_concat(self, monkeypatch):
        """train 收集超预算在 concat 前失败，算法上下文完整。"""
        pytest.importorskip("torch")

        from tributo.exceptions import ResourceBudgetExceededError
        from tributo.training.dnn_trainer import dnn_train_loop_per_worker

        self._patch_ray(
            monkeypatch, schema=None, batch=pd.DataFrame({"a": list(range(10))})
        )

        with pytest.raises(ResourceBudgetExceededError) as excinfo:
            dnn_train_loop_per_worker(
                {
                    "features": [],
                    "label_col": "label",
                    "model": {},
                    "loss": {},
                    "pu_learning": {},
                    "training": {},
                    "resource": {"max_worker_materialization_bytes": 10},
                }
            )
        assert excinfo.value.algorithm == "dnn"
        assert excinfo.value.split == "train"
        assert excinfo.value.worker_rank == 0

    def test_worker_preflight_rejects_absurd_schema(self, monkeypatch):
        """预检：单行估算即超单批预算 → 在读取任何 batch 前拒绝。"""
        pytest.importorskip("torch")

        import pyarrow as pa

        from tributo.exceptions import ResourceBudgetExceededError
        from tributo.training.dnn_trainer import dnn_train_loop_per_worker

        # string 列估算 32B/行 > max_batch_bytes=8 → 直接拒绝
        schema = pa.schema([("name", pa.string())])
        self._patch_ray(monkeypatch, schema=schema)

        with pytest.raises(ResourceBudgetExceededError) as excinfo:
            dnn_train_loop_per_worker(
                {
                    "features": [],
                    "label_col": "label",
                    "model": {},
                    "loss": {},
                    "pu_learning": {},
                    "training": {},
                    "resource": {
                        "max_batch_bytes": 8,
                        "max_worker_materialization_bytes": 10**9,
                    },
                }
            )
        assert "single row" in str(excinfo.value)

    def test_worker_row_guard_fails_fast_no_truncation(self, monkeypatch):
        """P2-7: max_input_rows_per_worker 超限 → fail-fast，不截断 DataFrame。"""
        pytest.importorskip("torch")

        from tributo.exceptions import ResourceBudgetExceededError
        from tributo.training.dnn_trainer import dnn_train_loop_per_worker

        self._patch_ray(
            monkeypatch,
            schema=None,
            batch=pd.DataFrame({"a": list(range(40))}),
        )
        with pytest.raises(ResourceBudgetExceededError) as excinfo:
            dnn_train_loop_per_worker(
                {
                    "features": [],
                    "label_col": "label",
                    "model": {},
                    "loss": {},
                    "pu_learning": {},
                    "training": {},
                    "resource": {
                        "max_input_rows_per_worker": 10,
                        "max_batch_bytes": 10**9,
                        "max_worker_materialization_bytes": 10**9,
                    },
                }
            )
        assert excinfo.value.observed_rows == 40  # 不截断为 10
        assert excinfo.value.max_rows == 10

    def test_worker_within_budget_trains_successfully(self, monkeypatch):
        """预算内正常路径：默认预算下小数据完整跑通 worker 训练。

        覆盖默认预算与共享预算的 happy path——train/val 共享预算、收集通过、
        训练循环执行、metrics 上报。mock ray.train，不依赖真实集群。
        """
        pytest.importorskip("torch")
        from types import SimpleNamespace

        import numpy as np
        import pyarrow as pa
        import ray.train

        from tributo.training.dnn_trainer import dnn_train_loop_per_worker

        n = 32
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "f0": rng.normal(size=n).astype(np.float32),
                "f1": rng.normal(size=n).astype(np.float32),
                "label": (rng.random(n) > 0.5).astype(np.float32),
            }
        )
        schema = pa.schema(
            [
                ("f0", pa.float32()),
                ("f1", pa.float32()),
                ("label", pa.float32()),
            ]
        )

        reported: dict[str, Any] = {}

        def fake_report(metrics, checkpoint=None):
            reported.update(metrics)

        monkeypatch.setattr(ray.train, "report", fake_report)
        monkeypatch.setattr(
            ray.train,
            "get_context",
            lambda: SimpleNamespace(get_world_rank=lambda: 0),
        )

        class FakeShard:
            def schema(self):
                return schema

            def iter_batches(self, **kwargs):
                yield df

        monkeypatch.setattr(ray.train, "get_dataset_shard", lambda key: FakeShard())

        dnn_train_loop_per_worker(
            {
                "features": [
                    {"name": "f0", "type": "dense"},
                    {"name": "f1", "type": "dense"},
                ],
                "label_col": "label",
                "model": {"dnn_hidden_units": [8]},
                "loss": {"type": "bce"},
                "pu_learning": {},
                "training": {"epochs": 1, "batch_size": 8},
                "resource": {},  # 默认预算
            }
        )
        assert reported["epoch"] == 1  # 训练完成且 metrics 已上报

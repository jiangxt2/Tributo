"""DNN distributed-worker contract and resource safety tests.

覆盖 DNNTrainingConfig 的 resource 预算默认值，以及 worker 内 train/val
共享预算的超限 fail-fast。worker 级测试通过 mock
``ray.train`` 上下文运行，不依赖真实 Ray 集群。
"""

from __future__ import annotations

from pathlib import Path
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
        assert cfg.ray.num_workers == 1

    def test_distributed_resume_requires_same_world_size_and_spec(self) -> None:
        from types import SimpleNamespace

        from tributo.exceptions import JobConfigurationError
        from tributo.training.dnn_trainer import (
            _validate_distributed_resume_metadata,
        )

        valid = SimpleNamespace(
            payload_metadata={
                "world_size": 2,
                "distribution_spec_digest": "a" * 64,
            }
        )
        _validate_distributed_resume_metadata(
            valid,
            expected_world_size=2,
            expected_distribution_digest="a" * 64,
        )

        with pytest.raises(JobConfigurationError, match="world_size"):
            _validate_distributed_resume_metadata(
                valid,
                expected_world_size=3,
                expected_distribution_digest="a" * 64,
            )
        with pytest.raises(JobConfigurationError, match="DistributionSpec"):
            _validate_distributed_resume_metadata(
                valid,
                expected_world_size=2,
                expected_distribution_digest="b" * 64,
            )

        legacy = SimpleNamespace(payload_metadata={})
        _validate_distributed_resume_metadata(
            legacy,
            expected_world_size=1,
            expected_distribution_digest=None,
        )
        with pytest.raises(JobConfigurationError, match="Legacy.*world_size"):
            _validate_distributed_resume_metadata(
                legacy,
                expected_world_size=2,
                expected_distribution_digest=None,
            )

    def test_bundle_destination_is_distinct_from_legacy_onnx_path(self):
        from pydantic import ValidationError

        from tributo.training.dnn_trainer import DNNTrainingConfig

        with pytest.raises(ValidationError, match="cannot be combined"):
            DNNTrainingConfig(
                output={
                    "bundle_uri": "s3://bucket/bundles",
                    "onnx_path": "model.onnx",
                }
            )

        cfg = DNNTrainingConfig(output={"bundle_uri": "s3://bucket/bundles"})
        assert cfg.output.bundle_uri == "s3://bucket/bundles"

    def test_num_workers_gt_one_is_supported_by_config(self) -> None:
        from tributo.training.dnn_trainer import DNNTrainingConfig

        config = DNNTrainingConfig.model_validate({"ray": {"num_workers": 2}})
        assert config.ray.num_workers == 2

    def test_non_collective_fallback_never_consumes_an_undeclared_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch = pytest.importorskip("torch")
        import tributo.training.distributed_torch as distributed_torch

        monkeypatch.setattr(distributed_torch, "collective_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        model, device = distributed_torch.prepare_model(torch.nn.Linear(2, 1))

        assert device.type == "cpu"
        assert next(model.parameters()).device.type == "cpu"

    def test_nnpu_requires_explicit_prior(self) -> None:
        from pydantic import ValidationError

        from tributo.training.dnn_trainer import DNNTrainingConfig

        with pytest.raises(ValidationError, match="class_prior"):
            DNNTrainingConfig.model_validate(
                {
                    "loss": {"type": "nnpu"},
                    "pu_learning": {"enabled": True},
                }
            )

    def test_nnpu_public_api_routes_to_canonical_pu_trainer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tributo.training.pu_trainer as pu_module
        from tributo.training.dnn_trainer import run_dnn_training_with_config

        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            pu_module,
            "run_pu_training_with_config",
            lambda config: captured.update(config) or {"status": "succeeded"},
        )

        result = run_dnn_training_with_config(
            {
                "data": {"type": "parquet", "path": "/tmp/not-opened.parquet"},
                "loss": {"type": "nnpu"},
                "pu_learning": {"enabled": True, "class_prior": 0.2},
                "ray": {"num_workers": 2},
            }
        )

        assert result == {"status": "succeeded"}
        assert captured["pu"]["loss_type"] == "nnpu"
        assert captured["ray"]["num_workers"] == 2

    def test_nnpu_cannot_construct_a_second_dnn_trainer_path(self) -> None:
        from tributo.exceptions import JobConfigurationError
        from tributo.training.dnn_trainer import DNNTrainerImpl

        with pytest.raises(JobConfigurationError, match="canonical 'pu'"):
            DNNTrainerImpl(
                datasets={},
                config={
                    "loss": {"type": "nnpu"},
                    "pu_learning": {"enabled": True, "class_prior": 0.2},
                },
            )

    @pytest.mark.parametrize(
        "pu_learning",
        (
            {"enabled": True, "class_prior": 0.2, "beta": -0.1},
            {"enabled": True, "class_prior": 0.2, "gamma": -0.1},
            {"enabled": True, "class_prior": 0.2, "gamma": 1.1},
        ),
    )
    def test_nnpu_rejects_invalid_correction_parameters(
        self,
        pu_learning: dict[str, float | bool],
    ) -> None:
        from pydantic import ValidationError

        from tributo.training.dnn_trainer import DNNTrainingConfig

        with pytest.raises(ValidationError):
            DNNTrainingConfig.model_validate(
                {
                    "loss": {"type": "nnpu"},
                    "pu_learning": pu_learning,
                }
            )

    def test_nnpu_rejects_batch_size_one(self) -> None:
        from pydantic import ValidationError

        from tributo.training.dnn_trainer import DNNTrainingConfig

        with pytest.raises(ValidationError, match="batch_size"):
            DNNTrainingConfig.model_validate(
                {
                    "loss": {"type": "nnpu"},
                    "pu_learning": {"enabled": True, "class_prior": 0.2},
                    "training": {"batch_size": 1},
                }
            )

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

    def test_worker_rejects_invalid_prior_before_reading_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("torch")
        from types import SimpleNamespace

        import ray.train

        from tributo.exceptions import JobConfigurationError
        from tributo.training.pu_trainer import pu_train_loop_per_worker

        monkeypatch.setattr(
            ray.train,
            "get_context",
            lambda: SimpleNamespace(
                get_world_size=lambda: 1,
                get_world_rank=lambda: 0,
            ),
        )
        monkeypatch.setattr(
            ray.train,
            "get_dataset_shard",
            lambda key: pytest.fail("dataset must not be opened"),
        )

        with pytest.raises(JobConfigurationError, match="range \\(0, 1\\)"):
            pu_train_loop_per_worker(
                {
                    "features": [],
                    "pu": {"loss_type": "nnpu", "class_prior": 2.0},
                    "training": {},
                    "resource": {},
                }
            )

    def _patch_ray(self, monkeypatch, schema=None, batch=None):
        """Mock ray.train context + shards; returns the fake dataset factory."""
        from types import SimpleNamespace

        import ray.train

        monkeypatch.setattr(
            ray.train,
            "get_context",
            lambda: SimpleNamespace(
                get_world_size=lambda: 1,
                get_world_rank=lambda: 0,
            ),
        )
        monkeypatch.setattr(
            ray,
            "get_runtime_context",
            lambda: SimpleNamespace(
                get_assigned_resources=lambda: {"CPU": 1.0},
                get_worker_id=lambda: "test-worker",
                get_node_id=lambda: "test-node",
            ),
        )

        class FakeShard:
            def schema(self):
                return schema

            def iter_batches(self, **kwargs):
                if batch is not None:
                    yield batch

        def get_dataset_shard(key):
            if key == "val":
                raise KeyError(key)
            return FakeShard()

        monkeypatch.setattr(ray.train, "get_dataset_shard", get_dataset_shard)

    @pytest.mark.parametrize("loss_type", ("bce", "focal"))
    def test_worker_requires_val_shard_when_validation_is_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        loss_type: str,
    ) -> None:
        """Supervised losses must not silently disable configured validation."""
        pytest.importorskip("torch")

        from tributo.exceptions import JobConfigurationError
        from tributo.training.dnn_trainer import dnn_train_loop_per_worker

        self._patch_ray(monkeypatch)

        with pytest.raises(
            JobConfigurationError,
            match="validation is enabled.*'val' dataset shard",
        ) as exc_info:
            dnn_train_loop_per_worker(
                {
                    "features": [],
                    "loss": {"type": loss_type},
                    "pu_learning": {},
                    "training": {"val_size": 0.2},
                    "resource": {},
                }
            )
        assert isinstance(exc_info.value.__cause__, KeyError)

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
                    "training": {"val_size": 0},
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
                    "training": {"val_size": 0},
                    "resource": {
                        "max_batch_bytes": 8,
                        "max_worker_materialization_bytes": 10**9,
                    },
                }
            )
        assert "single row" in str(excinfo.value)

    def test_worker_row_guard_fails_fast_no_truncation(self, monkeypatch):
        """max_input_rows_per_worker 超限 → fail-fast，不截断 DataFrame。"""
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
                    "training": {"val_size": 0},
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
        torch = pytest.importorskip("torch")
        from types import SimpleNamespace

        import numpy as np
        import pyarrow as pa
        import ray.train

        from tributo.training.dnn_trainer import dnn_train_loop_per_worker
        from tributo.training.pu_trainer import pu_train_loop_per_worker

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
        checkpoint_dirs: list[Path] = []

        def fake_report(metrics, checkpoint=None):
            reported.update(metrics)

        monkeypatch.setattr(ray.train, "report", fake_report)
        monkeypatch.setattr(
            ray.train.Checkpoint,
            "from_directory",
            staticmethod(lambda path: checkpoint_dirs.append(Path(path)) or object()),
        )
        monkeypatch.setattr(
            ray.train,
            "get_context",
            lambda: SimpleNamespace(
                get_world_size=lambda: 1,
                get_world_rank=lambda: 0,
            ),
        )
        monkeypatch.setattr(
            ray,
            "get_runtime_context",
            lambda: SimpleNamespace(
                get_assigned_resources=lambda: {"CPU": 1.0},
                get_worker_id=lambda: "test-worker",
                get_node_id=lambda: "test-node",
            ),
        )

        class FakeShard:
            def __init__(self, frame: pd.DataFrame) -> None:
                self.frame = frame

            def schema(self):
                return schema

            def iter_batches(self, **kwargs):
                yield self.frame

        def get_dataset_shard(key):
            if key == "val":
                raise KeyError(key)
            if key == "positive":
                return FakeShard(df[df["label"] == 1].reset_index(drop=True))
            if key == "unlabeled":
                return FakeShard(df[df["label"] == 0].reset_index(drop=True))
            return FakeShard(df)

        monkeypatch.setattr(ray.train, "get_dataset_shard", get_dataset_shard)

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
                "training": {"epochs": 1, "batch_size": 8, "val_size": 0},
                "resource": {},  # 默认预算
            }
        )
        assert reported["epoch"] == 1  # 训练完成且 metrics 已上报
        assert checkpoint_dirs and all(not path.exists() for path in checkpoint_dirs)

        reported.clear()
        pu_train_loop_per_worker(
            {
                "features": [
                    {"name": "f0", "type": "dense"},
                    {"name": "f1", "type": "dense"},
                ],
                "label_col": "label",
                "model": {"dnn_hidden_units": [8]},
                "pu": {"loss_type": "nnpu", "class_prior": 0.2},
                "training": {"epochs": 1, "batch_size": 8},
                "resource": {},
            }
        )
        assert reported["epoch"] == 1
        assert "train_loss" in reported
        assert "train_optimization_objective" in reported
        assert all(not path.exists() for path in checkpoint_dirs)
        assert "train_observed_label_accuracy" in reported
        assert reported["train_acc"] == reported["train_observed_label_accuracy"]

        import tempfile

        created_dirs: list[Path] = []
        original_mkdtemp = tempfile.mkdtemp

        def tracked_mkdtemp(*args: Any, **kwargs: Any) -> str:
            path = Path(original_mkdtemp(*args, **kwargs))
            created_dirs.append(path)
            return str(path)

        def fail_checkpoint_write(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise OSError("checkpoint write failed")

        monkeypatch.setattr(tempfile, "mkdtemp", tracked_mkdtemp)
        monkeypatch.setattr(torch, "save", fail_checkpoint_write)

        with pytest.raises(OSError, match="checkpoint write failed"):
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
                    "training": {"epochs": 1, "batch_size": 8, "val_size": 0},
                    "resource": {},
                }
            )

        assert created_dirs and all(not path.exists() for path in created_dirs)


class TestDNNTrainerRunConfig:
    """DNN forwards only the existing Ray Train identity overrides."""

    @pytest.mark.parametrize(
        ("run_config", "expected_storage", "expected_name"),
        [
            (None, "/configured-storage", "tributo-dnn"),
            (
                {"name": None, "storage_path": None},
                "/configured-storage",
                "tributo-dnn",
            ),
            (
                {"name": "tune-dnn-trial-001", "storage_path": "/trial-storage"},
                "/trial-storage",
                "tune-dnn-trial-001",
            ),
        ],
    )
    def test_training_loop_applies_narrow_run_config_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        run_config: dict[str, Any] | None,
        expected_storage: str,
        expected_name: str,
    ) -> None:
        import sys
        from types import SimpleNamespace

        import ray

        import tributo.training.checkpoint as checkpoint_module
        from tributo.training.dnn_trainer import DNNTrainerImpl

        captured: dict[str, Any] = {}
        resume_checkpoint = object()

        class FakeTorchTrainer:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

            def fit(self) -> Any:
                return SimpleNamespace(metrics={})

        monkeypatch.setattr(ray, "is_initialized", lambda: True)
        monkeypatch.setitem(
            sys.modules,
            "ray.train.torch",
            SimpleNamespace(TorchTrainer=FakeTorchTrainer),
        )
        monkeypatch.setattr(
            checkpoint_module,
            "load_initial_checkpoint",
            lambda path: resume_checkpoint if path == "/persisted-checkpoint" else None,
        )
        trainer = DNNTrainerImpl(
            datasets={},
            config={
                "features": [],
                "ray": {
                    "storage_path": "/configured-storage",
                    "resume": {"checkpoint_path": "/persisted-checkpoint"},
                },
            },
            run_config=run_config,
        )

        trainer.training_loop()

        inner_run_config = captured["run_config"]
        assert inner_run_config.storage_path == expected_storage
        assert inner_run_config.name == expected_name
        assert captured["resume_from_checkpoint"] is resume_checkpoint
        assert captured["train_loop_config"]["resume"]["checkpoint_path"] == (
            "/persisted-checkpoint"
        )

    @pytest.mark.parametrize(
        "run_config",
        (
            {"name": ""},
            {"name": 1},
            {"storage_path": ""},
            {"storage_path": 1},
        ),
    )
    def test_training_loop_rejects_invalid_run_config_override(
        self,
        run_config: dict[str, Any],
    ) -> None:
        from tributo.exceptions import JobConfigurationError
        from tributo.training.dnn_trainer import DNNTrainerImpl

        trainer = DNNTrainerImpl(
            datasets={},
            config={"features": []},
            run_config=run_config,
        )

        with pytest.raises(JobConfigurationError, match="run_config"):
            trainer.training_loop()

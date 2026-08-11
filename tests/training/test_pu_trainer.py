"""PUTrainer 单元测试和端到端测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


class TestPUTrainingConfig:
    """PUTrainingConfig 配置验证测试。"""

    def test_default_config(self):
        """默认配置应合法。"""
        from tributo.training.pu_trainer import PUTrainingConfig

        cfg = PUTrainingConfig(pu={"class_prior": 0.2})
        assert cfg.pu.loss_type == "nnpu"
        assert cfg.pu.class_prior == 0.2
        assert cfg.pu.class_prior_method == "label_frequency"
        assert cfg.pu.beta == 0.0
        assert cfg.pu.gamma == 1.0

    def test_bundle_destination_is_supported(self):
        from tributo.training.pu_trainer import PUTrainingConfig

        cfg = PUTrainingConfig(
            pu={"class_prior": 0.2},
            output={"bundle_uri": "s3://bucket/pu-bundles"},
        )
        assert cfg.output.bundle_uri == "s3://bucket/pu-bundles"

    def test_custom_config(self):
        """自定义配置应合法。"""
        from tributo.training.pu_trainer import PUTrainingConfig

        cfg = PUTrainingConfig(
            pu={
                "loss_type": "upu",
                "class_prior": 0.15,
                "class_prior_method": "em",
                "beta": 0.1,
                "gamma": 0.5,
            },
            training={"epochs": 20, "batch_size": 64},
        )
        assert cfg.pu.loss_type == "upu"
        assert cfg.pu.class_prior == 0.15
        assert cfg.pu.class_prior_method == "em"
        assert cfg.training.epochs == 20

    def test_invalid_loss_type(self):
        """无效损失类型应抛异常。"""
        from pydantic import ValidationError

        from tributo.training.pu_trainer import PUTrainingConfig

        with pytest.raises(ValidationError):
            PUTrainingConfig(pu={"loss_type": "invalid", "class_prior": 0.2})

    def test_prior_method_is_compatibility_metadata(self):
        """Legacy prior-method values are preserved but do not drive training."""
        from tributo.training.pu_trainer import PUTrainingConfig

        cfg = PUTrainingConfig(
            pu={"class_prior": 0.2, "class_prior_method": "external_estimator"}
        )

        assert cfg.pu.class_prior_method == "external_estimator"

    def test_missing_class_prior(self) -> None:
        """PU training must not silently infer a class prior from observed labels."""
        from pydantic import ValidationError

        from tributo.training.pu_trainer import PUTrainingConfig

        with pytest.raises(ValidationError, match="class_prior"):
            PUTrainingConfig()

    @pytest.mark.parametrize(
        "pu",
        (
            {"class_prior": 0.2, "beta": -0.1},
            {"class_prior": 0.2, "gamma": -0.1},
            {"class_prior": 0.2, "gamma": 1.1},
        ),
    )
    def test_invalid_correction_parameters(self, pu: dict[str, float]) -> None:
        from pydantic import ValidationError

        from tributo.training.pu_trainer import PUTrainingConfig

        with pytest.raises(ValidationError):
            PUTrainingConfig(pu=pu)

    def test_batch_size_one_is_rejected(self) -> None:
        from pydantic import ValidationError

        from tributo.training.pu_trainer import PUTrainingConfig

        with pytest.raises(ValidationError, match="batch_size"):
            PUTrainingConfig(
                pu={"class_prior": 0.2},
                training={"batch_size": 1},
            )


class TestPUTrainerRegistration:
    """PUTrainer 注册表测试。"""

    def test_registered(self):
        """PUTrainer 应已注册。"""
        from tributo.training.registry import get_trainer

        spec = get_trainer("pu")
        assert spec is not None
        assert spec.name == "pu"


class TestPUBatchContract:
    """PU training and validation preserve independent P/U samples."""

    def test_paired_loader_keeps_both_groups_in_sparse_batches(self) -> None:
        torch = pytest.importorskip("torch")

        from tributo.training.dnn_trainer import build_pu_train_loader

        labels = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        dataset = [
            {"feature": torch.tensor(float(index)), "label": torch.tensor(label)}
            for index, label in enumerate(labels)
        ]

        loader = build_pu_train_loader(
            dataset,
            labels,
            batch_size=4,
            seed=7,
        )

        for batch in loader:
            assert set(batch["label"].tolist()) == {0.0, 1.0}

    def test_paired_loader_uses_absolute_epoch_for_resume(self) -> None:
        torch = pytest.importorskip("torch")

        from tributo.training.dnn_trainer import build_pu_train_loader

        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        dataset = [
            {"row": torch.tensor(index), "label": torch.tensor(label)}
            for index, label in enumerate(labels)
        ]
        loader = build_pu_train_loader(dataset, labels, batch_size=4, seed=7)

        loader.batch_sampler.set_epoch(3)
        first = [batch["row"].tolist() for batch in loader]
        repeated = [batch["row"].tolist() for batch in loader]
        loader.batch_sampler.set_epoch(3)
        resumed = [batch["row"].tolist() for batch in loader]

        assert repeated == first
        assert resumed == first

    def test_stratified_split_preserves_both_groups(self) -> None:
        from tributo.training.dnn_trainer import split_pu_indices

        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        train_indices, val_indices = split_pu_indices(
            labels,
            val_size=0.25,
            seed=42,
        )

        assert set(labels[train_indices]) == {0.0, 1.0}
        assert set(labels[val_indices]) == {0.0, 1.0}

    def test_validation_split_rejects_insufficient_positive_rows(self) -> None:
        from tributo.exceptions import JobConfigurationError
        from tributo.training.dnn_trainer import split_pu_indices

        labels = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        with pytest.raises(JobConfigurationError, match="val_size=0"):
            split_pu_indices(labels, val_size=0.25, seed=42)

    @pytest.mark.parametrize("invalid_label", (None, "not-a-number", np.nan))
    def test_invalid_labels_raise_configuration_error(
        self, invalid_label: object
    ) -> None:
        from tributo.exceptions import JobConfigurationError
        from tributo.training.dnn_trainer import validate_pu_labels

        labels = np.array([1.0, invalid_label, 0.0], dtype=object)

        with pytest.raises(JobConfigurationError, match="row 1"):
            validate_pu_labels(labels, split="train")

    def test_split_risk_error_is_classified_as_configuration_error(self) -> None:
        torch = pytest.importorskip("torch")

        from tributo.exceptions import JobConfigurationError
        from tributo.training.dnn_trainer import evaluate_pu_split
        from tributo.training.losses.pu_loss import PULoss

        class ConstantModel:
            def eval(self) -> None:
                pass

            def __call__(self, inputs: dict[str, Any]) -> Any:
                return torch.zeros(2)

        dataloader = [{"label": torch.ones(2)}]

        with pytest.raises(JobConfigurationError, match="Invalid PU evaluation split"):
            evaluate_pu_split(
                ConstantModel(),
                dataloader,
                PULoss(class_prior=0.2),
                torch.device("cpu"),
            )

    @pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
    def test_non_finite_training_metrics_fail_closed(self, value: float) -> None:
        from tributo.exceptions import JobExecutionError
        from tributo.training.dnn_trainer import validate_finite_training_metrics

        with pytest.raises(JobExecutionError, match="train_loss"):
            validate_finite_training_metrics(
                {"train_loss": value},
                algorithm="PU",
            )

    def test_non_default_prior_method_logs_compatibility_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        from tributo.training.dnn_trainer import warn_if_ignored_class_prior_method

        warn_if_ignored_class_prior_method(
            "em",
            default_method="label_frequency",
            config_path="pu.class_prior_method",
            target_logger=logging.getLogger("tributo.test"),
        )

        assert "does not trigger class-prior estimation" in caplog.text


class TestPUE2E:
    """PU Learning 端到端测试（本地单机，不依赖 Ray）。"""

    def _generate_pu_data(self, n_samples: int = 500) -> pd.DataFrame:
        """生成 PU 测试数据。"""
        rng = np.random.RandomState(42)

        department = rng.randint(0, 10, n_samples)
        job_level = rng.randint(0, 5, n_samples)
        years_of_service = rng.exponential(5, n_samples).astype(np.float32)
        salary_range = rng.uniform(0, 1, n_samples).astype(np.float32)
        use_enterprise_email = rng.binomial(1, 0.3, n_samples)

        # 约 15% 正例
        positive_prob = (
            0.05
            + 0.15 * (years_of_service > 3).astype(float)
            + 0.1 * (salary_range > 0.6).astype(float)
            + 0.2 * use_enterprise_email.astype(float)
            + 0.05 * (job_level >= 3).astype(float)
        )
        label = (rng.random(n_samples) < positive_prob).astype(int)

        return pd.DataFrame(
            {
                "department": department,
                "job_level": job_level,
                "years_of_service": years_of_service,
                "salary_range": salary_range,
                "use_enterprise_email": use_enterprise_email,
                "label": label,
            }
        )

    def test_pu_training_bce(self):
        """PU 训练（nnPU）→ ONNX 推理。"""
        import torch

        from tributo.training.dnn_trainer import build_pu_train_loader
        from tributo.training.features.column_types import (
            DenseFeat,
            NormMethod,
            SparseFeat,
        )
        from tributo.training.features.dataset import IdentityDataset
        from tributo.training.features.transformer import FeatureTransformer
        from tributo.training.losses.pu_loss import PULoss, compute_class_prior
        from tributo.training.models.dnn import DNNModel

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 1. 生成数据
            df = self._generate_pu_data(500)

            # 2. 定义特征
            features = [
                SparseFeat(name="department", vocab_size=10, embedding_dim=4),
                SparseFeat(name="job_level", vocab_size=5, embedding_dim=2),
                DenseFeat(name="years_of_service", norm=NormMethod.LOG),
                DenseFeat(name="salary_range", norm=NormMethod.MINMAX),
                DenseFeat(name="use_enterprise_email"),
            ]

            # 3. 预处理
            feature_names = [f.name for f in features]
            data_dict = {name: df[name].values for name in feature_names}
            labels = df["label"].values.astype(np.float32)

            transformer = FeatureTransformer(features)
            processed = transformer.fit_transform(data_dict)

            # 4. 创建 Dataset
            dataset = IdentityDataset(processed, labels, features)
            torch_dataset = dataset.to_torch_dataset()
            dataloader = build_pu_train_loader(
                torch_dataset, labels, batch_size=32, seed=42
            )

            # 5. 创建模型
            model = DNNModel(features, dnn_hidden_units=[32, 16])

            # 6. PU 损失
            class_prior = compute_class_prior(
                positive_count=int(labels.sum()),
                total_count=len(labels),
            )
            criterion = PULoss(
                class_prior=class_prior, beta=0.0, gamma=1.0, loss_type="nnpu"
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # 7. 训练
            model.train()
            for _epoch in range(3):
                for batch in dataloader:
                    inputs = {k: v for k, v in batch.items() if k != "label"}
                    labels_batch = batch["label"]
                    logits = model(inputs)
                    loss = criterion(logits, labels_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # 8. 导出 ONNX
            from tributo.training.exporters.torch_onnx_exporter import (
                export_model_package,
            )

            sample_inputs = {
                name: (
                    np.array([0, 1], dtype=np.int64)
                    if isinstance(f, SparseFeat)
                    else np.array([0.0, 1.0], dtype=np.float32)
                )
                for name, f in zip(feature_names, features)
            }

            preprocessor_path = tmpdir / "preprocessor.json"
            transformer.save(preprocessor_path)
            preprocessor_state = json.loads(preprocessor_path.read_text())

            output_dir = tmpdir / "model_output"
            result_paths = export_model_package(
                model=model,
                sample_inputs=sample_inputs,
                output_dir=output_dir,
                feature_config=[f.__dict__ for f in features],
                preprocessor_state=preprocessor_state,
                metrics={"class_prior": class_prior, "final_loss": loss.item()},
            )

            # 9. 验证 ONNX 推理
            import onnxruntime as ort

            session = ort.InferenceSession(str(result_paths["onnx_model"]))
            test_inputs = {
                "department": np.array([1, 2], dtype=np.int64),
                "job_level": np.array([0, 1], dtype=np.int64),
                "years_of_service": np.array([1.0, 5.0], dtype=np.float32),
                "salary_range": np.array([0.3, 0.8], dtype=np.float32),
                "use_enterprise_email": np.array([0.0, 1.0], dtype=np.float32),
            }
            outputs = session.run(None, test_inputs)
            probs = 1.0 / (1.0 + np.exp(-outputs[0]))
            assert all(0 <= p <= 1 for p in probs)

    def test_pu_training_upu(self):
        """PU 训练（uPU）→ ONNX 推理。"""
        import torch

        from tributo.training.dnn_trainer import build_pu_train_loader
        from tributo.training.features.column_types import (
            DenseFeat,
            NormMethod,
            SparseFeat,
        )
        from tributo.training.features.dataset import IdentityDataset
        from tributo.training.features.transformer import FeatureTransformer
        from tributo.training.losses.pu_loss import PULoss, compute_class_prior
        from tributo.training.models.dnn import DNNModel

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            df = self._generate_pu_data(500)
            features = [
                SparseFeat(name="department", vocab_size=10, embedding_dim=4),
                SparseFeat(name="job_level", vocab_size=5, embedding_dim=2),
                DenseFeat(name="years_of_service", norm=NormMethod.LOG),
                DenseFeat(name="salary_range", norm=NormMethod.MINMAX),
                DenseFeat(name="use_enterprise_email"),
            ]
            feature_names = [f.name for f in features]
            data_dict = {name: df[name].values for name in feature_names}
            labels = df["label"].values.astype(np.float32)

            transformer = FeatureTransformer(features)
            processed = transformer.fit_transform(data_dict)

            dataset = IdentityDataset(processed, labels, features)
            dataloader = build_pu_train_loader(
                dataset.to_torch_dataset(), labels, batch_size=32, seed=42
            )

            model = DNNModel(features, dnn_hidden_units=[32, 16])
            class_prior = compute_class_prior(int(labels.sum()), len(labels))
            criterion = PULoss(class_prior=class_prior, loss_type="upu")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            model.train()
            for _epoch in range(3):
                for batch in dataloader:
                    inputs = {k: v for k, v in batch.items() if k != "label"}
                    logits = model(inputs)
                    loss = criterion(logits, batch["label"])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # ONNX 导出验证
            from tributo.training.exporters.torch_onnx_exporter import (
                export_model_package,
            )

            sample_inputs = {
                name: (
                    np.array([0, 1], dtype=np.int64)
                    if isinstance(f, SparseFeat)
                    else np.array([0.0, 1.0], dtype=np.float32)
                )
                for name, f in zip(feature_names, features)
            }
            preprocessor_path = tmpdir / "preprocessor.json"
            transformer.save(preprocessor_path)
            preprocessor_state = json.loads(preprocessor_path.read_text())

            result_paths = export_model_package(
                model=model,
                sample_inputs=sample_inputs,
                output_dir=tmpdir / "model_output",
                feature_config=[f.__dict__ for f in features],
                preprocessor_state=preprocessor_state,
                metrics={"class_prior": class_prior},
            )
            assert result_paths["onnx_model"].exists()

    def test_pu_metrics_computation(self):
        """PU 指标计算。"""
        from tributo.training.priors import label_frequency_prior
        from tributo.training.pu_metrics import compute_pu_metrics

        # 模拟预测结果
        rng = np.random.RandomState(42)
        y_true = np.concatenate([np.ones(100), np.zeros(400)])
        y_scores = np.concatenate(
            [
                rng.uniform(0.6, 1.0, 100),
                rng.uniform(0.0, 1.0, 400),
            ]
        )
        class_prior = label_frequency_prior(100, 500)

        metrics = compute_pu_metrics(y_true, y_scores, class_prior)
        assert "pu_precision" in metrics
        assert "pu_f1" in metrics
        assert "pu_auc" in metrics
        assert 0.0 <= metrics["pu_auc"] <= 1.0


class TestPUTrainerResourceSafety:
    """PU 单 worker 资源安全。"""

    def test_default_resource_budget_is_active(self):
        """预算默认启用。"""
        from tributo.training.pu_trainer import PUTrainingConfig
        from tributo.training.resource import MIB

        cfg = PUTrainingConfig(pu={"class_prior": 0.2})
        assert cfg.resource.max_batch_bytes == 64 * MIB
        assert cfg.resource.max_worker_materialization_bytes == 1024 * MIB
        assert cfg.resource.max_input_rows_per_worker is None

    def test_custom_resource_budget(self):
        from tributo.training.pu_trainer import PUTrainingConfig

        cfg = PUTrainingConfig(
            pu={"class_prior": 0.2}, resource={"max_batch_bytes": 1024}
        )
        assert cfg.resource.max_batch_bytes == 1024

    def test_num_workers_gt_1_rejected_at_construction(self):
        """构造期拒绝 num_workers > 1（早于任何训练）。"""
        from pydantic import ValidationError

        from tributo.training.pu_trainer import PUTrainingConfig

        with pytest.raises(ValidationError, match="num_workers"):
            PUTrainingConfig(pu={"class_prior": 0.2}, ray={"num_workers": 2})

    def test_num_workers_1_constructs(self):
        from tributo.training.pu_trainer import PUTrainerImpl

        trainer = PUTrainerImpl(
            datasets={},
            config={"pu": {"class_prior": 0.2}, "ray": {"num_workers": 1}},
        )
        assert trainer._pu_config.ray.num_workers == 1

    def test_worker_loop_rejects_missing_prior_before_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing class prior fails before opening the configured source."""
        pytest.importorskip("torch")
        from types import SimpleNamespace

        import ray.train

        import tributo.training.data_loader as data_loader_mod
        from tributo.exceptions import JobConfigurationError
        from tributo.training.pu_trainer import pu_train_loop_per_worker

        monkeypatch.setattr(
            ray.train,
            "get_context",
            lambda: SimpleNamespace(get_world_size=lambda: 1, get_world_rank=lambda: 0),
        )
        source_opened = False

        def fail_if_source_opens(_source: object) -> object:
            nonlocal source_opened
            source_opened = True
            raise AssertionError("source must not be opened")

        monkeypatch.setattr(
            data_loader_mod,
            "load_ray_dataset_from_source",
            fail_if_source_opens,
        )
        with pytest.raises(JobConfigurationError, match="class_prior"):
            pu_train_loop_per_worker(
                {
                    "data": {"source": {"type": "parquet", "path": "x"}},
                    "features": [],
                    "label_col": "label",
                    "model": {},
                    "pu": {},
                    "training": {},
                    "resource": {},
                }
            )
        assert not source_opened

    def test_worker_loop_rejects_world_size_gt_1(self, monkeypatch):
        """worker 入口二次拒绝，早于数据加载。"""
        pytest.importorskip("torch")
        from types import SimpleNamespace

        import ray.train

        from tributo.exceptions import JobConfigurationError
        from tributo.training.pu_trainer import pu_train_loop_per_worker

        monkeypatch.setattr(
            ray.train,
            "get_context",
            lambda: SimpleNamespace(get_world_size=lambda: 2, get_world_rank=lambda: 0),
        )
        with pytest.raises(JobConfigurationError, match="world_size=2"):
            pu_train_loop_per_worker(
                {
                    "data": {"source": {"type": "parquet", "path": "x"}},
                    "features": [],
                    "label_col": "label",
                    "model": {},
                    "pu": {"class_prior": 0.2},
                    "training": {},
                    "resource": {},
                }
            )

    def test_worker_budget_exceeded_fails_before_concat(self, monkeypatch):
        """worker 加载超预算在 concat 前失败，不返回部分数据。"""
        pytest.importorskip("torch")
        from types import SimpleNamespace

        import pandas as pd
        import ray.train

        import tributo.training.data_loader as data_loader_mod
        from tributo.exceptions import ResourceBudgetExceededError
        from tributo.training.pu_trainer import pu_train_loop_per_worker

        monkeypatch.setattr(
            ray.train,
            "get_context",
            lambda: SimpleNamespace(get_world_size=lambda: 1, get_world_rank=lambda: 0),
        )

        class FakeDataset:
            def schema(self):
                return None

            def iter_batches(self, **kwargs):
                yield pd.DataFrame({"a": list(range(10))})

        monkeypatch.setattr(
            data_loader_mod,
            "load_ray_dataset_from_source",
            lambda source: FakeDataset(),
        )
        with pytest.raises(ResourceBudgetExceededError) as excinfo:
            pu_train_loop_per_worker(
                {
                    "data": {"source": {"type": "parquet", "path": "x"}},
                    "features": [],
                    "label_col": "label",
                    "model": {},
                    "pu": {"class_prior": 0.2},
                    "training": {},
                    "resource": {"max_worker_materialization_bytes": 10},
                }
            )
        assert excinfo.value.algorithm == "pu"
        assert excinfo.value.split == "train"

    def test_worker_row_guard_fails_fast_no_truncation(self, monkeypatch):
        """max_input_rows_per_worker 超限 → fail-fast，不截断。"""
        pytest.importorskip("torch")
        from types import SimpleNamespace

        import pandas as pd
        import ray.train

        import tributo.training.data_loader as data_loader_mod
        from tributo.exceptions import ResourceBudgetExceededError
        from tributo.training.pu_trainer import pu_train_loop_per_worker

        monkeypatch.setattr(
            ray.train,
            "get_context",
            lambda: SimpleNamespace(get_world_size=lambda: 1, get_world_rank=lambda: 0),
        )

        class FakeDataset:
            def schema(self):
                return None

            def iter_batches(self, **kwargs):
                yield pd.DataFrame({"a": list(range(40))})

        monkeypatch.setattr(
            data_loader_mod,
            "load_ray_dataset_from_source",
            lambda source: FakeDataset(),
        )
        with pytest.raises(ResourceBudgetExceededError) as excinfo:
            pu_train_loop_per_worker(
                {
                    "data": {"source": {"type": "parquet", "path": "x"}},
                    "features": [],
                    "label_col": "label",
                    "model": {},
                    "pu": {"class_prior": 0.2},
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

        覆盖默认预算的 happy path——收集通过、训练循环执行、metrics 上报。
        mock ray.train 与数据源，不依赖真实集群。
        """
        torch = pytest.importorskip("torch")
        from types import SimpleNamespace

        import numpy as np
        import pyarrow as pa
        import ray.train

        import tributo.training.data_loader as data_loader_mod
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
            lambda: SimpleNamespace(get_world_size=lambda: 1, get_world_rank=lambda: 0),
        )

        class FakeDataset:
            def schema(self):
                return schema

            def iter_batches(self, **kwargs):
                yield df

        monkeypatch.setattr(
            data_loader_mod,
            "load_ray_dataset_from_source",
            lambda source: FakeDataset(),
        )

        pu_train_loop_per_worker(
            {
                "data": {"source": {"type": "parquet", "path": "x"}},
                "features": [
                    {"name": "f0", "type": "dense"},
                    {"name": "f1", "type": "dense"},
                ],
                "label_col": "label",
                "model": {"dnn_hidden_units": [8]},
                "pu": {"loss_type": "nnpu", "class_prior": 0.2},
                "training": {"epochs": 1, "batch_size": 8},
                "resource": {},  # 默认预算
            }
        )
        assert reported["epoch"] == 1  # 训练完成且 metrics 已上报
        assert "train_loss" in reported
        assert "train_optimization_objective" in reported
        assert "train_observed_label_accuracy" in reported
        assert reported["train_acc"] == reported["train_observed_label_accuracy"]
        assert checkpoint_dirs and all(not path.exists() for path in checkpoint_dirs)

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
            pu_train_loop_per_worker(
                {
                    "data": {"source": {"type": "parquet", "path": "x"}},
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

        assert created_dirs and all(not path.exists() for path in created_dirs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

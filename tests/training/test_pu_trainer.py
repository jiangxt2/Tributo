"""PUTrainer 单元测试和端到端测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestPUTrainingConfig:
    """PUTrainingConfig 配置验证测试。"""

    def test_default_config(self):
        """默认配置应合法。"""
        from tributo.training.pu_trainer import PUTrainingConfig

        cfg = PUTrainingConfig()
        assert cfg.pu.loss_type == "nnpu"
        assert cfg.pu.class_prior_method == "label_frequency"
        assert cfg.pu.beta == 0.0
        assert cfg.pu.gamma == 1.0

    def test_custom_config(self):
        """自定义配置应合法。"""
        from tributo.training.pu_trainer import PUTrainingConfig

        cfg = PUTrainingConfig(
            pu={
                "loss_type": "upu",
                "class_prior": 0.15,
                "class_prior_method": "em",
                "beta": 0.1,
                "gamma": 2.0,
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
            PUTrainingConfig(pu={"loss_type": "invalid"})

    def test_invalid_prior_method(self):
        """无效 class prior 方法应抛异常。"""
        from pydantic import ValidationError

        from tributo.training.pu_trainer import PUTrainingConfig

        with pytest.raises(ValidationError):
            PUTrainingConfig(pu={"class_prior_method": "invalid"})


class TestPUTrainerRegistration:
    """PUTrainer 注册表测试。"""

    def test_registered(self):
        """PUTrainer 应已注册。"""
        from tributo.training.registry import get_trainer

        spec = get_trainer("pu")
        assert spec is not None
        assert spec.name == "pu"


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
        from torch.utils.data import DataLoader

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
            dataloader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

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
                name: np.array([0, 1], dtype=np.int64)
                if isinstance(f, SparseFeat)
                else np.array([0.0, 1.0], dtype=np.float32)
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
        from torch.utils.data import DataLoader

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
            dataloader = DataLoader(
                dataset.to_torch_dataset(), batch_size=32, shuffle=True
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
                name: np.array([0, 1], dtype=np.int64)
                if isinstance(f, SparseFeat)
                else np.array([0.0, 1.0], dtype=np.float32)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

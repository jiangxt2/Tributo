"""身份挖掘端到端测试。

测试完整流程：数据读取 → 预处理 → 训练 → ONNX 导出 → ONNX 推理。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def generate_identity_dataset(
    n_samples: int = 500,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """生成身份挖掘测试数据集。

    模拟场景：企业员工识别
    - 正例（企业员工）：约 15%
    - 特征：部门、职级、工龄、薪资范围、是否使用企业邮箱

    Args:
        n_samples: 样本数量。
        output_path: 保存路径（可选）。

    Returns:
        pandas DataFrame。
    """
    np.random.seed(42)

    # 生成特征
    department = np.random.randint(0, 10, n_samples)  # 10 个部门
    job_level = np.random.randint(0, 5, n_samples)  # 5 个职级
    years_of_service = np.random.exponential(5, n_samples)  # 工龄
    salary_range = np.random.uniform(0, 1, n_samples)  # 薪资范围
    use_enterprise_email = np.random.binomial(1, 0.3, n_samples)  # 是否使用企业邮箱

    # 生成标签（与特征相关）
    # 工龄长、薪资高、使用企业邮箱的更可能是企业员工
    positive_prob = (
        0.05
        + 0.15 * (years_of_service > 3).astype(float)
        + 0.1 * (salary_range > 0.6).astype(float)
        + 0.2 * use_enterprise_email.astype(float)
        + 0.05 * (job_level >= 3).astype(float)
    )
    label = (np.random.random(n_samples) < positive_prob).astype(int)

    df = pd.DataFrame(
        {
            "department": department,
            "job_level": job_level,
            "years_of_service": years_of_service.astype(np.float32),
            "salary_range": salary_range.astype(np.float32),
            "use_enterprise_email": use_enterprise_email,
            "label": label,
        }
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    return df


class TestIdentityE2E:
    """身份挖掘端到端测试。"""

    def test_parquet_to_onnx_bce(self):
        """测试 Parquet → 训练(BCE) → ONNX 完整流程。"""
        import torch
        from torch.utils.data import DataLoader

        from tributo.training.features.column_types import (
            DenseFeat,
            NormMethod,
            SparseFeat,
        )
        from tributo.training.features.dataset import IdentityDataset
        from tributo.training.features.transformer import FeatureTransformer
        from tributo.training.models.dnn import DNNModel

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 1. 生成测试数据并保存为 Parquet
            data_path = tmpdir / "train.parquet"
            generate_identity_dataset(500, data_path)
            assert data_path.exists()

            # 2. 读取 Parquet 数据
            df_loaded = pd.read_parquet(data_path)
            assert len(df_loaded) == 500
            assert "label" in df_loaded.columns

            # 3. 定义特征列
            features = [
                SparseFeat(name="department", vocab_size=10, embedding_dim=4),
                SparseFeat(name="job_level", vocab_size=5, embedding_dim=2),
                DenseFeat(name="years_of_service", norm=NormMethod.LOG),
                DenseFeat(name="salary_range", norm=NormMethod.MINMAX),
                DenseFeat(name="use_enterprise_email"),
            ]

            # 4. 预处理
            feature_names = [f.name for f in features]
            data_dict = {name: df_loaded[name].values for name in feature_names}
            labels = df_loaded["label"].values.astype(np.float32)

            transformer = FeatureTransformer(features)
            processed = transformer.fit_transform(data_dict)

            # 5. 创建 Dataset 和 DataLoader
            dataset = IdentityDataset(processed, labels, features)
            torch_dataset = dataset.to_torch_dataset()
            dataloader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

            # 6. 训练模型
            model = DNNModel(features, dnn_hidden_units=[32, 16])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.BCEWithLogitsLoss()

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

            # 7. 导出 ONNX
            from tributo.training.exporters.torch_onnx_exporter import (
                export_model_package,
            )

            sample_inputs = {
                name: np.array([0, 1], dtype=np.int64)
                if isinstance(f, SparseFeat)
                else np.array([0.0, 1.0], dtype=np.float32)
                for name, f in zip(feature_names, features)
            }

            # 保存预处理器状态
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
                metrics={"final_loss": loss.item()},
            )

            # 8. 验证输出文件
            assert result_paths["onnx_model"].exists()
            assert result_paths["feature_config"].exists()
            assert result_paths["preprocessor"].exists()

            # 9. 验证 ONNX 模型可以推理
            import onnxruntime as ort

            session = ort.InferenceSession(str(result_paths["onnx_model"]))

            # 准备测试输入
            test_inputs = {
                "department": np.array([1, 2], dtype=np.int64),
                "job_level": np.array([0, 1], dtype=np.int64),
                "years_of_service": np.array([1.0, 5.0], dtype=np.float32),
                "salary_range": np.array([0.3, 0.8], dtype=np.float32),
                "use_enterprise_email": np.array([0.0, 1.0], dtype=np.float32),
            }

            outputs = session.run(None, test_inputs)
            assert len(outputs) >= 1
            assert outputs[0].shape[0] == 2  # batch_size=2

    def test_parquet_to_onnx_pu_learning(self):
        """测试 Parquet → 训练(nnPU) → ONNX 完整流程。"""
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

            # 1. 生成测试数据
            data_path = tmpdir / "train.parquet"
            generate_identity_dataset(500, data_path)

            # 2. 读取数据
            df_loaded = pd.read_parquet(data_path)

            # 3. 定义特征列
            features = [
                SparseFeat(name="department", vocab_size=10, embedding_dim=4),
                SparseFeat(name="job_level", vocab_size=5, embedding_dim=2),
                DenseFeat(name="years_of_service", norm=NormMethod.LOG),
                DenseFeat(name="salary_range", norm=NormMethod.MINMAX),
                DenseFeat(name="use_enterprise_email"),
            ]

            # 4. 预处理
            feature_names = [f.name for f in features]
            data_dict = {name: df_loaded[name].values for name in feature_names}
            labels = df_loaded["label"].values.astype(np.float32)

            transformer = FeatureTransformer(features)
            processed = transformer.fit_transform(data_dict)

            # 5. 创建 Dataset
            dataset = IdentityDataset(processed, labels, features)
            torch_dataset = dataset.to_torch_dataset()
            dataloader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

            # 6. 创建模型
            model = DNNModel(features, dnn_hidden_units=[32, 16])

            # 7. 配置 PU Learning 损失
            class_prior = compute_class_prior(
                positive_count=int(labels.sum()),
                total_count=len(labels),
            )
            criterion = PULoss(class_prior=class_prior, beta=0.0, gamma=1.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # 8. 训练
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

            # 9. 导出 ONNX
            from tributo.training.exporters.torch_onnx_exporter import (
                export_model_package,
            )

            sample_inputs = {
                name: np.array([0, 1], dtype=np.int64)
                if isinstance(f, SparseFeat)
                else np.array([0.0, 1.0], dtype=np.float32)
                for name, f in zip(feature_names, features)
            }

            # 保存预处理器状态
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

            # 10. 验证输出
            assert result_paths["onnx_model"].exists()

            # 11. 验证 ONNX 推理
            import onnxruntime as ort

            session = ort.InferenceSession(str(result_paths["onnx_model"]))
            test_inputs = {
                "department": np.array([1], dtype=np.int64),
                "job_level": np.array([0], dtype=np.int64),
                "years_of_service": np.array([2.0], dtype=np.float32),
                "salary_range": np.array([0.5], dtype=np.float32),
                "use_enterprise_email": np.array([0.0], dtype=np.float32),
            }

            outputs = session.run(None, test_inputs)
            logits = outputs[0]

            # 应用 sigmoid 得到概率
            prob = 1.0 / (1.0 + np.exp(-logits))
            assert 0 <= prob[0] <= 1

    def test_parquet_to_onnx_focal_loss(self):
        """测试 Parquet → 训练(Focal Loss) → ONNX 完整流程。"""
        import torch
        from torch.utils.data import DataLoader

        from tributo.training.features.column_types import (
            DenseFeat,
            NormMethod,
            SparseFeat,
        )
        from tributo.training.features.dataset import IdentityDataset
        from tributo.training.features.transformer import FeatureTransformer
        from tributo.training.losses.focal_loss import FocalLoss
        from tributo.training.models.dnn import DNNModel

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 1. 生成测试数据
            data_path = tmpdir / "train.parquet"
            generate_identity_dataset(500, data_path)

            # 2. 读取数据
            df_loaded = pd.read_parquet(data_path)

            # 3. 定义特征列
            features = [
                SparseFeat(name="department", vocab_size=10, embedding_dim=4),
                SparseFeat(name="job_level", vocab_size=5, embedding_dim=2),
                DenseFeat(name="years_of_service", norm=NormMethod.LOG),
                DenseFeat(name="salary_range", norm=NormMethod.MINMAX),
                DenseFeat(name="use_enterprise_email"),
            ]

            # 4. 预处理
            feature_names = [f.name for f in features]
            data_dict = {name: df_loaded[name].values for name in feature_names}
            labels = df_loaded["label"].values.astype(np.float32)

            transformer = FeatureTransformer(features)
            processed = transformer.fit_transform(data_dict)

            # 5. 创建 Dataset
            dataset = IdentityDataset(processed, labels, features)
            torch_dataset = dataset.to_torch_dataset()
            dataloader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

            # 6. 创建模型
            model = DNNModel(features, dnn_hidden_units=[32, 16])

            # 7. 配置 Focal Loss
            # 根据正例比例设置 alpha
            positive_ratio = labels.mean()
            criterion = FocalLoss(alpha=1 - positive_ratio, gamma=2.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # 8. 训练
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

            # 9. 导出 ONNX
            from tributo.training.exporters.torch_onnx_exporter import (
                export_model_package,
            )

            sample_inputs = {
                name: np.array([0, 1], dtype=np.int64)
                if isinstance(f, SparseFeat)
                else np.array([0.0, 1.0], dtype=np.float32)
                for name, f in zip(feature_names, features)
            }

            # 保存预处理器状态
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
                metrics={"final_loss": loss.item()},
            )

            # 10. 验证输出
            assert result_paths["onnx_model"].exists()

            # 11. 验证 ONNX 推理
            import onnxruntime as ort

            session = ort.InferenceSession(str(result_paths["onnx_model"]))
            test_inputs = {
                "department": np.array([1], dtype=np.int64),
                "job_level": np.array([0], dtype=np.int64),
                "years_of_service": np.array([2.0], dtype=np.float32),
                "salary_range": np.array([0.5], dtype=np.float32),
                "use_enterprise_email": np.array([0.0], dtype=np.float32),
            }

            outputs = session.run(None, test_inputs)
            logits = outputs[0]
            prob = 1.0 / (1.0 + np.exp(-logits))
            assert 0 <= prob[0] <= 1

    def test_preprocessing_consistency(self):
        """测试预处理三端一致性（训练/批量/在线）。"""

        from tributo.training.features.column_types import (
            DenseFeat,
            NormMethod,
            SparseFeat,
        )
        from tributo.training.features.transformer import FeatureTransformer

        # 生成数据
        df = generate_identity_dataset(100)

        # 定义特征
        features = [
            SparseFeat(name="department", vocab_size=10),
            SparseFeat(name="job_level", vocab_size=5),
            DenseFeat(name="years_of_service", norm=NormMethod.LOG),
            DenseFeat(name="salary_range", norm=NormMethod.MINMAX),
            DenseFeat(name="use_enterprise_email"),
        ]

        feature_names = [f.name for f in features]
        data_dict = {name: df[name].values for name in feature_names}

        # 训练端：拟合并转换
        transformer = FeatureTransformer(features)
        train_processed = transformer.fit_transform(data_dict)

        # 保存预处理器
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor_path = Path(tmpdir) / "preprocessor.json"
            transformer.save(preprocessor_path)

            # 在线端：加载预处理器并转换
            loaded_transformer = FeatureTransformer.load(preprocessor_path)
            online_processed = loaded_transformer.transform(data_dict)

            # 验证一致性
            for name in feature_names:
                np.testing.assert_array_equal(
                    train_processed[name],
                    online_processed[name],
                    err_msg=f"Feature '{name}' preprocessing mismatch",
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

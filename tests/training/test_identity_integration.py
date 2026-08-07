"""身份挖掘集成测试。

测试完整的训练→ONNX→推理流程。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tributo.training.features.column_types import (
    DenseFeat,
    NormMethod,
    SparseFeat,
)
from tributo.training.features.transformer import FeatureTransformer


def generate_test_data(n_samples: int = 100) -> dict:
    """生成测试数据。"""
    np.random.seed(42)

    features = {
        "category": np.random.randint(0, 5, n_samples),
        "value": np.random.randn(n_samples).astype(np.float32),
    }
    labels = (np.random.random(n_samples) > 0.7).astype(np.float32)

    return {"features": features, "labels": labels}


class TestTransformerIntegration:
    """预处理器集成测试。"""

    def test_fit_transform_save_load(self):
        """测试完整的拟合→转换→保存→加载流程。"""
        data = generate_test_data(100)
        feature_list = [
            SparseFeat(name="category", vocab_size=5),
            DenseFeat(name="value", norm=NormMethod.STANDARD),
        ]

        # 拟合和转换
        transformer = FeatureTransformer(feature_list)
        processed = transformer.fit_transform(data["features"])

        assert "category" in processed
        assert "value" in processed
        assert len(processed["category"]) == 100

        # 保存和加载
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "transformer.json"
            transformer.save(save_path)

            loaded = FeatureTransformer.load(save_path)
            loaded_processed = loaded.transform(data["features"])

            # 验证结果一致
            np.testing.assert_array_equal(
                processed["category"], loaded_processed["category"]
            )
            np.testing.assert_array_almost_equal(
                processed["value"], loaded_processed["value"]
            )


try:
    import torch
    from torch.utils.data import DataLoader

    from tributo.training.dnn_trainer import build_pu_train_loader
    from tributo.training.features.dataset import IdentityDataset
    from tributo.training.losses.pu_loss import PULoss
    from tributo.training.models.dnn import DNNModel

    class TestDNNE2E:
        """DNN 端到端测试（需要 PyTorch）。"""

        def test_train_predict_export(self):
            """测试训练→预测→导出流程。"""
            data = generate_test_data(100)
            feature_list = [
                SparseFeat(name="category", vocab_size=5, embedding_dim=4),
                DenseFeat(name="value", norm=NormMethod.MINMAX),
            ]

            # 预处理
            transformer = FeatureTransformer(feature_list)
            processed = transformer.fit_transform(data["features"])

            # 创建 Dataset 和 DataLoader
            dataset = IdentityDataset(processed, data["labels"], feature_list)
            torch_dataset = dataset.to_torch_dataset()
            dataloader = DataLoader(torch_dataset, batch_size=16, shuffle=True)

            # 创建模型
            model = DNNModel(feature_list, dnn_hidden_units=[32, 16])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # 训练 2 个 epoch
            model.train()
            for _epoch in range(2):
                for batch in dataloader:
                    inputs = {k: v for k, v in batch.items() if k != "label"}
                    labels = batch["label"]

                    logits = model(inputs)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # 预测
            model.eval()
            test_input = {
                "category": torch.tensor([0, 1, 2], dtype=torch.long),
                "value": torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32),
            }
            probs = model.predict_proba(test_input)

            assert probs.shape == (3,)
            assert torch.all(probs >= 0)
            assert torch.all(probs <= 1)

            # 导出 ONNX
            with tempfile.TemporaryDirectory() as tmpdir:
                from tributo.training.exporters.torch_onnx_exporter import (
                    export_pytorch_to_onnx,
                )

                sample_inputs = {
                    "category": np.array([0, 1], dtype=np.int64),
                    "value": np.array([0.1, 0.5], dtype=np.float32),
                }
                onnx_path = export_pytorch_to_onnx(
                    model=model,
                    sample_inputs=sample_inputs,
                    output_path=Path(tmpdir) / "model.onnx",
                )

                assert onnx_path.exists()

        def test_pu_learning_training(self):
            """测试 PU Learning 训练。"""
            data = generate_test_data(100)
            feature_list = [
                SparseFeat(name="category", vocab_size=5, embedding_dim=4),
                DenseFeat(name="value"),
            ]

            # 预处理
            transformer = FeatureTransformer(feature_list)
            processed = transformer.fit_transform(data["features"])

            # 创建 Dataset
            dataset = IdentityDataset(processed, data["labels"], feature_list)
            torch_dataset = dataset.to_torch_dataset()
            dataloader = build_pu_train_loader(
                torch_dataset, data["labels"], batch_size=16, seed=42
            )

            # 创建模型
            model = DNNModel(feature_list, dnn_hidden_units=[32, 16])

            # PU Learning 损失
            class_prior = data["labels"].mean()
            criterion = PULoss(class_prior=class_prior)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # 训练
            model.train()
            for _epoch in range(2):
                for batch in dataloader:
                    inputs = {k: v for k, v in batch.items() if k != "label"}
                    labels = batch["label"]

                    logits = model(inputs)
                    loss = criterion(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # 验证模型可以正常预测
            model.eval()
            test_input = {
                "category": torch.tensor([0, 1], dtype=torch.long),
                "value": torch.tensor([0.1, 0.5], dtype=torch.float32),
            }
            probs = model.predict_proba(test_input)
            assert probs.shape == (2,)

    class TestIdentityPredictorIntegration:
        """IdentityPredictor 集成测试。"""

        def test_predictor_with_model_package(self):
            """测试 IdentityPredictor 加载模型包。"""
            from tributo.serving.identity_predictor import IdentityPredictor
            from tributo.training.exporters.torch_onnx_exporter import (
                export_model_package,
            )

            data = generate_test_data(100)
            feature_list = [
                SparseFeat(name="category", vocab_size=5, embedding_dim=4),
                DenseFeat(name="value", norm=NormMethod.MINMAX),
            ]

            # 预处理
            transformer = FeatureTransformer(feature_list)
            processed = transformer.fit_transform(data["features"])

            # 创建并训练简单模型
            model = DNNModel(feature_list, dnn_hidden_units=[32, 16])
            dataset = IdentityDataset(processed, data["labels"], feature_list)
            torch_dataset = dataset.to_torch_dataset()
            dataloader = DataLoader(torch_dataset, batch_size=16, shuffle=True)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model.train()
            for batch in dataloader:
                inputs = {k: v for k, v in batch.items() if k != "label"}
                labels = batch["label"]
                logits = model(inputs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 导出模型包
            with tempfile.TemporaryDirectory() as tmpdir:
                sample_inputs = {
                    "category": np.array([0, 1], dtype=np.int64),
                    "value": np.array([0.1, 0.5], dtype=np.float32),
                }

                # 保存预处理器状态
                preprocessor_path = Path(tmpdir) / "preprocessor.json"
                transformer.save(preprocessor_path)
                preprocessor_state = json.loads(preprocessor_path.read_text())

                export_model_package(
                    model=model,
                    sample_inputs=sample_inputs,
                    output_dir=Path(tmpdir) / "model_output",
                    feature_config=[f.__dict__ for f in feature_list],
                    preprocessor_state=preprocessor_state,
                )

                # 加载 IdentityPredictor
                predictor = IdentityPredictor(model_path=Path(tmpdir) / "model_output")

                # 预测
                result = predictor.predict(
                    {"category": 1, "value": 0.5},
                    threshold=0.5,
                )

                assert "probability" in result
                assert "prediction" in result
                assert 0 <= result["probability"] <= 1
                assert result["prediction"] in (0, 1)

except ImportError:
    pass  # PyTorch 未安装时跳过


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

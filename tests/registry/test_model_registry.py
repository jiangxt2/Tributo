"""ModelRegistry 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from tributo.registry.model_registry import ModelRegistry
from tributo.registry.schema import ModelVersion

# mlflow lives in the registry extra - skip collection without it
# (importorskip comes after all imports to avoid E402).
mlflow = pytest.importorskip("mlflow", reason="mlflow not installed")
MlflowException = mlflow.exceptions.MlflowException


def _make_model_version(
    name: str = "test-model",
    version: int = 1,
    stage: str = "None",
    run_id: str = "run-001",
) -> MagicMock:
    """创建模拟的 MLflow ModelVersion 对象。"""
    mv = MagicMock()
    mv.name = name
    mv.version = str(version)
    mv.current_stage = stage
    mv.run_id = run_id
    mv.source = f"s3://bucket/{name}/{version}"
    mv.creation_timestamp = 1700000000
    mv.description = None
    mv.tags = {}
    return mv


class TestModelRegistryInit:
    """ModelRegistry 初始化测试。"""

    def test_import_error_raises(self):
        with patch("tributo.registry.model_registry.MlflowClient", None):
            with pytest.raises(ImportError, match="mlflow is required"):
                ModelRegistry()

    def test_creates_client_with_tracking_uri(self):
        with patch("tributo.registry.model_registry.MlflowClient") as MockClient:
            ModelRegistry(tracking_uri="http://mlflow:5000")
            MockClient.assert_called_once_with(tracking_uri="http://mlflow:5000")


class TestRegisterModel:
    """register_model 方法测试。"""

    def test_registers_and_returns_model_version(self):
        mock_client = MagicMock()
        mock_mv = _make_model_version()
        mock_client.create_model_version.return_value = mock_mv

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            result = reg.register_model(
                model_uri="runs:/run-001/model",
                name="test-model",
                tags={"team": "ml"},
                description="A test model",
            )

        assert isinstance(result, ModelVersion)
        assert result.name == "test-model"
        assert result.version == 1
        mock_client.create_model_version.assert_called_once_with(
            name="test-model", source="runs:/run-001/model"
        )
        mock_client.set_model_version_tag.assert_called_once_with(
            "test-model", "1", "team", "ml"
        )
        mock_client.update_model_version.assert_called_once()

    def test_creates_registered_model_when_not_found(self):
        mock_client = MagicMock()
        mock_mv = _make_model_version()
        mock_client.create_model_version.return_value = mock_mv

        not_found = MlflowException("not found")
        not_found.error_code = "RESOURCE_DOES_NOT_EXIST"
        mock_client.get_registered_model.side_effect = not_found

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            reg.register_model(model_uri="runs:/run-001/model", name="test-model")

        mock_client.create_registered_model.assert_called_once_with("test-model")

    def test_reraises_non_not_found_mlflow_exception(self):
        mock_client = MagicMock()
        forbidden = MlflowException("forbidden")
        forbidden.error_code = "PERMISSION_DENIED"
        mock_client.get_registered_model.side_effect = forbidden

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            with pytest.raises(MlflowException, match="forbidden"):
                reg.register_model(model_uri="runs:/run-001/model", name="test-model")

        mock_client.create_registered_model.assert_not_called()

    def test_registers_without_optional_fields(self):
        mock_client = MagicMock()
        mock_mv = _make_model_version()
        mock_client.create_model_version.return_value = mock_mv

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            result = reg.register_model(
                model_uri="runs:/run-001/model", name="test-model"
            )

        assert result.name == "test-model"
        mock_client.set_model_version_tag.assert_not_called()
        mock_client.update_model_version.assert_not_called()

    def test_cleans_up_created_version_on_tag_failure(self):
        """版本创建成功但 tag 失败时，应删除已创建的版本。"""
        mock_client = MagicMock()
        mock_mv = _make_model_version()
        mock_client.create_model_version.return_value = mock_mv
        mock_client.set_model_version_tag.side_effect = RuntimeError("tag failed")

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            with pytest.raises(RuntimeError, match="tag failed"):
                reg.register_model(
                    model_uri="runs:/run-001/model",
                    name="test-model",
                    tags={"team": "ml"},
                )

        mock_client.delete_model_version.assert_called_once_with(
            name="test-model", version="1"
        )
        mock_client.delete_registered_model.assert_not_called()

    def test_cleans_up_created_model_on_version_failure(self):
        """模型是本次新建但版本创建失败时，应删除空模型。"""
        mock_client = MagicMock()
        not_found = MlflowException("not found")
        not_found.error_code = "RESOURCE_DOES_NOT_EXIST"
        mock_client.get_registered_model.side_effect = not_found
        mock_client.create_model_version.side_effect = RuntimeError("version failed")

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            with pytest.raises(RuntimeError, match="version failed"):
                reg.register_model(model_uri="runs:/run-001/model", name="test-model")

        mock_client.create_registered_model.assert_called_once_with("test-model")
        mock_client.delete_registered_model.assert_called_once_with(name="test-model")
        mock_client.delete_model_version.assert_not_called()


class TestGetModel:
    """get_model 方法测试。"""

    def test_get_by_version(self):
        mock_client = MagicMock()
        mock_mv = _make_model_version(version=3)
        mock_client.get_model_version.return_value = mock_mv

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            result = reg.get_model(name="test-model", version=3)

        assert result.version == 3
        mock_client.get_model_version.assert_called_once_with(
            name="test-model", version="3"
        )

    def test_get_by_stage(self):
        mock_client = MagicMock()
        mv1 = _make_model_version(version=1, stage="Staging")
        mv2 = _make_model_version(version=2, stage="Staging")
        mv3 = _make_model_version(version=3, stage="None")
        mock_client.search_model_versions.return_value = [mv1, mv2, mv3]

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            result = reg.get_model(name="test-model", stage="Staging")

        # 应返回最新版本（version=2）
        assert result.version == 2

    def test_get_by_stage_not_found(self):
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = []

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            with pytest.raises(ValueError, match="No model"):
                reg.get_model(name="test-model", stage="Production")

    def test_get_by_stage_invalid_name_raises(self):
        """模型名包含非法字符时应直接拒绝，避免过滤字符串注入。"""
        mock_client = MagicMock()

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            with pytest.raises(ValueError, match="Invalid model name"):
                reg.get_model(name="test' or '1'='1", stage="Production")

        mock_client.search_model_versions.assert_not_called()

    def test_get_without_version_or_stage_raises(self):
        mock_client = MagicMock()
        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            with pytest.raises(ValueError, match="Either version or stage"):
                reg.get_model(name="test-model")


class TestListModels:
    """list_models 方法测试。"""

    def test_returns_model_names_from_registered_models(self):
        mock_client = MagicMock()
        rm_a = MagicMock()
        rm_a.name = "model-a"
        rm_b = MagicMock()
        rm_b.name = "model-b"
        mock_client.search_registered_models.return_value = [rm_a, rm_b]

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            result = reg.list_models()

        assert result == ["model-a", "model-b"]
        mock_client.search_model_versions.assert_not_called()

    def test_falls_back_to_search_model_versions(self):
        mock_client = MagicMock()
        mock_client.search_registered_models.side_effect = MlflowException(
            "unsupported"
        )
        mv_a = MagicMock()
        mv_a.name = "model-a"
        mv_b = MagicMock()
        mv_b.name = "model-b"
        mv_a2 = MagicMock()
        mv_a2.name = "model-a"  # 重复名称
        mock_client.search_model_versions.return_value = [mv_a, mv_b, mv_a2]

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            result = reg.list_models()

        assert result == ["model-a", "model-b"]


class TestTransitionStage:
    """transition_stage 方法测试。"""

    def test_transitions_stage(self):
        mock_client = MagicMock()
        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            reg.transition_stage("test-model", 1, "Production")

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="test-model", version=1, stage="Production"
        )


class TestCompareModels:
    """compare_models 方法测试。"""

    def test_batch_query(self):
        mock_client = MagicMock()
        mv1 = _make_model_version(version=1, run_id="run-001")
        mv2 = _make_model_version(version=2, run_id="run-002")
        mock_client.get_model_version.side_effect = [mv1, mv2]

        run1 = MagicMock()
        run1.info.run_id = "run-001"
        run1.data.metrics = {"loss": 0.5}
        run2 = MagicMock()
        run2.info.run_id = "run-002"
        run2.data.metrics = {"loss": 0.3}
        mock_client.get_run.side_effect = [run1, run2]

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            result = reg.compare_models("test-model", [1, 2], metric="loss")

        assert result == {1: 0.5, 2: 0.3}
        assert mock_client.get_model_version.call_args_list == [
            call(name="test-model", version="1"),
            call(name="test-model", version="2"),
        ]

    def test_per_version_query(self):
        mock_client = MagicMock()
        mv1 = _make_model_version(version=1, run_id="run-001")
        mock_client.get_model_version.return_value = mv1

        run1 = MagicMock()
        run1.data.metrics = {"loss": 0.5}
        mock_client.get_run.return_value = run1

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            result = reg.compare_models("test-model", [1], metric="loss")

        assert result == {1: 0.5}

    def test_empty_versions(self):
        mock_client = MagicMock()
        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            result = reg.compare_models("test-model", [], metric="loss")

        assert result == {}

    def test_extracts_run_id_from_source_uri(self):
        """run_id 为空时从 runs:/<run_id>/... URI 中提取。"""
        mock_client = MagicMock()
        mv1 = _make_model_version(version=1, run_id="")
        mv1.source = "runs:/run-abc/model"
        mock_client.get_model_version.return_value = mv1

        run1 = MagicMock()
        run1.data.metrics = {"loss": 0.5}
        mock_client.get_run.return_value = run1

        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            result = reg.compare_models("test-model", [1], metric="loss")

        mock_client.get_run.assert_called_once_with("run-abc")
        assert result == {1: 0.5}


class TestDeleteModel:
    """delete_model / delete_model_version 方法测试。"""

    def test_delete_model(self):
        mock_client = MagicMock()
        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            reg.delete_model("test-model")

        mock_client.delete_registered_model.assert_called_once_with(name="test-model")

    def test_delete_model_version(self):
        mock_client = MagicMock()
        with patch(
            "tributo.registry.model_registry.MlflowClient",
            return_value=mock_client,
        ):
            reg = ModelRegistry()
            reg.delete_model_version("test-model", 2)

        mock_client.delete_model_version.assert_called_once_with(
            name="test-model", version="2"
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))

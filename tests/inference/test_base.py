"""BasePredictor 抽象基类测试。"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tributo.inference.base import BasePredictor


class TestBasePredictorABC:
    """BasePredictor 抽象方法测试。"""

    def test_incomplete_subclass_raises_type_error(self):
        """未实现抽象方法的子类实例化时应抛 TypeError。"""

        class IncompletePredictor(BasePredictor):
            pass

        with pytest.raises(TypeError, match="_load_model"):
            IncompletePredictor(model_uri="dummy")

    def test_partial_subclass_raises_type_error(self):
        """只实现 _load_model 未实现 __call__ 时应抛 TypeError。"""

        class PartialPredictor(BasePredictor):
            def _load_model(self) -> None:
                pass

        with pytest.raises(TypeError, match="__call__"):
            PartialPredictor(model_uri="dummy")

    def test_complete_subclass_works(self):
        """完整实现的子类应可正常实例化和调用。"""

        class MockPredictor(BasePredictor):
            def _load_model(self) -> None:
                self.loaded = True

            def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                batch["pred"] = np.ones(len(next(iter(batch.values()))))
                return batch

        predictor = MockPredictor(model_uri="dummy")
        assert predictor.loaded is True
        assert predictor.model_uri == "dummy"
        assert predictor.predictor_config == {}

        result = predictor({"a": np.array([1, 2, 3])})
        assert "pred" in result
        assert "a" in result
        np.testing.assert_array_equal(result["pred"], [1, 1, 1])

    def test_predictor_config_passed_through(self):
        """predictor_config 应正确传递到子类。"""

        class ConfigPredictor(BasePredictor):
            def _load_model(self) -> None:
                self.return_probs = self.predictor_config.get("return_probs", False)

            def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                return batch

        predictor = ConfigPredictor(
            model_uri="dummy",
            predictor_config={"return_probs": True, "extra": 42},
        )
        assert predictor.return_probs is True
        assert predictor.predictor_config["extra"] == 42

    def test_get_feature_names_default_returns_empty(self):
        """基类 get_feature_names 默认返回空列表。"""
        assert BasePredictor.get_feature_names("dummy") == []
        assert BasePredictor.get_feature_names("dummy", {"key": "val"}) == []

    def test_get_feature_names_override(self):
        """子类可覆盖 get_feature_names。"""

        class FeaturePredictor(BasePredictor):
            def _load_model(self) -> None:
                pass

            def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                return batch

            @classmethod
            def get_feature_names(
                cls,
                model_uri: str,
                predictor_config: dict | None = None,
            ) -> list[str]:
                return ["feat_a", "feat_b"]

        assert FeaturePredictor.get_feature_names("dummy") == ["feat_a", "feat_b"]


class TestXGBoostONNXPredictorInheritance:
    """XGBoostONNXPredictor 继承关系测试。"""

    def test_is_subclass_of_base_predictor(self):
        from tributo.inference.batch_predictor import XGBoostONNXPredictor

        assert issubclass(XGBoostONNXPredictor, BasePredictor)

    def test_get_feature_names_is_overridden(self):
        """XGBoostONNXPredictor 应覆盖 get_feature_names。"""
        from tributo.inference.batch_predictor import XGBoostONNXPredictor

        # 覆盖后的方法应存在且不是基类的默认实现
        assert "get_feature_names" in XGBoostONNXPredictor.__dict__


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))

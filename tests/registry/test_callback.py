"""MLflowTrackingCallback 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tributo.registry.callback import MLflowTrackingCallback


def _make_trainer(config: dict | None = None) -> MagicMock:
    """创建模拟的 BaseTrainer 实例。"""
    trainer = MagicMock()
    trainer.config = config or {"lr": 0.01, "epochs": 10}
    return trainer


class TestMLflowTrackingCallbackInit:
    """MLflowTrackingCallback 初始化测试。"""

    def test_default_params(self):
        cb = MLflowTrackingCallback(experiment_name="exp1")
        assert cb._experiment_name == "exp1"
        assert cb._tracking_uri is None
        assert cb._run_name is None
        assert cb._tags is None
        assert cb._raise_on_error is False
        assert cb.failure_policy == "best_effort"
        assert cb._util is None
        assert cb._run_id is None

    def test_custom_params(self):
        cb = MLflowTrackingCallback(
            experiment_name="exp2",
            tracking_uri="http://mlflow:5000",
            run_name="run1",
            tags={"team": "ml"},
            raise_on_error=True,
        )
        assert cb._tracking_uri == "http://mlflow:5000"
        assert cb._run_name == "run1"
        assert cb._tags == {"team": "ml"}
        assert cb._raise_on_error is True
        assert cb.failure_policy == "required"


class TestOnSetupStart:
    """on_setup_start 回调测试。"""

    @patch("tributo.registry.callback._MLflowTrackerUtil")
    def test_success(self, MockUtil):
        mock_util = MagicMock()
        mock_util.start_run.return_value = "run-123"
        MockUtil.return_value = mock_util

        cb = MLflowTrackingCallback(experiment_name="exp1")
        trainer = _make_trainer()
        cb.on_setup_start(trainer)

        MockUtil.assert_called_once_with(None, raise_on_error=False)
        mock_util.setup_experiment.assert_called_once_with("exp1")
        mock_util.start_run.assert_called_once_with(run_name=None, tags=None)
        mock_util.log_params.assert_called_once_with(trainer.config)
        assert cb._run_id == "run-123"

    @patch("tributo.registry.callback._MLflowTrackerUtil")
    def test_failure_silent(self, MockUtil):
        MockUtil.side_effect = ConnectionError("refused")
        cb = MLflowTrackingCallback(experiment_name="exp1", raise_on_error=False)
        trainer = _make_trainer()
        # 不应抛异常
        cb.on_setup_start(trainer)
        assert cb._util is None

    @patch("tributo.registry.callback._MLflowTrackerUtil")
    def test_failure_raises(self, MockUtil):
        MockUtil.side_effect = ConnectionError("refused")
        cb = MLflowTrackingCallback(experiment_name="exp1", raise_on_error=True)
        trainer = _make_trainer()
        with pytest.raises(ConnectionError):
            cb.on_setup_start(trainer)


class TestOnTrainingEnd:
    """on_training_end 回调测试。"""

    def test_skips_when_util_is_none(self):
        cb = MLflowTrackingCallback(experiment_name="exp1")
        # 不应抛异常
        cb.on_training_end(MagicMock(), MagicMock())

    @patch("tributo.registry.callback._MLflowTrackerUtil")
    def test_logs_metrics(self, MockUtil):
        mock_util = MagicMock()
        MockUtil.return_value = mock_util

        cb = MLflowTrackingCallback(experiment_name="exp1")
        cb.on_setup_start(_make_trainer())

        result = MagicMock()
        result.metrics = {"loss": 0.5, "acc": 0.9}
        cb.on_training_end(MagicMock(), result)
        mock_util.log_metrics.assert_called_with({"loss": 0.5, "acc": 0.9})

    @patch("tributo.registry.callback._MLflowTrackerUtil")
    def test_skips_when_no_metrics_attr(self, MockUtil):
        mock_util = MagicMock()
        MockUtil.return_value = mock_util

        cb = MLflowTrackingCallback(experiment_name="exp1")
        cb.on_setup_start(_make_trainer())

        result = "plain_string"
        cb.on_training_end(MagicMock(), result)
        mock_util.log_metrics.assert_not_called()


class TestOnExportEnd:
    """on_export_end 回调测试。"""

    def test_skips_when_util_is_none(self):
        cb = MLflowTrackingCallback(experiment_name="exp1")
        cb.on_export_end(MagicMock(), "/tmp/model.onnx")

    @patch("tributo.registry.callback._MLflowTrackerUtil")
    def test_logs_artifact(self, MockUtil):
        mock_util = MagicMock()
        MockUtil.return_value = mock_util

        cb = MLflowTrackingCallback(experiment_name="exp1")
        cb.on_setup_start(_make_trainer())

        cb.on_export_end(MagicMock(), "/tmp/model.onnx")
        mock_util.log_artifact.assert_called_once_with("/tmp/model.onnx")


class TestOnRunComplete:
    """on_run_complete 回调测试。"""

    def test_skips_when_util_is_none(self):
        cb = MLflowTrackingCallback(experiment_name="exp1")
        cb.on_run_complete(MagicMock(), {"status": "succeeded"})

    @patch("tributo.registry.callback._MLflowTrackerUtil")
    def test_logs_summary_and_ends_run(self, MockUtil):
        mock_util = MagicMock()
        MockUtil.return_value = mock_util

        cb = MLflowTrackingCallback(experiment_name="exp1")
        cb.on_setup_start(_make_trainer())

        summary = {"status": "succeeded", "loss": 0.1}
        cb.on_run_complete(MagicMock(), summary)
        mock_util.log_metrics.assert_called_with(summary)
        mock_util.end_run.assert_called_once_with(status="FINISHED")


class TestOnRunError:
    """on_run_error 回调测试。"""

    def test_skips_when_util_is_none(self):
        cb = MLflowTrackingCallback(experiment_name="exp1")
        cb.on_run_error(MagicMock(), RuntimeError("boom"))

    @patch("tributo.registry.callback._MLflowTrackerUtil")
    def test_ends_run_failed(self, MockUtil):
        mock_util = MagicMock()
        MockUtil.return_value = mock_util

        cb = MLflowTrackingCallback(experiment_name="exp1")
        cb.on_setup_start(_make_trainer())

        cb.on_run_error(MagicMock(), RuntimeError("boom"))
        mock_util.end_run.assert_called_once_with(status="FAILED")


class TestGracefulDegradation:
    """graceful degradation 单元测试（无需真实 MLflow server）。"""

    @patch("tributo.registry.callback._MLflowTrackerUtil")
    def test_setup_failure_does_not_block_training(self, MockUtil):
        """_MLflowTrackerUtil 初始化失败 → 后续回调不抛异常。"""
        MockUtil.side_effect = ConnectionError("refused")

        cb = MLflowTrackingCallback(
            experiment_name="exp1",
            tracking_uri="http://127.0.0.1:19999",
            raise_on_error=False,
        )
        trainer = _make_trainer()

        cb.on_setup_start(trainer)

        assert cb._util is None
        cb.on_run_complete(trainer, {"status": "succeeded"})
        cb.on_run_error(trainer, RuntimeError("test"))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))

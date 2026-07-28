"""_MLflowTrackerUtil 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tributo.registry.mlflow_util import _MLflowTrackerUtil


class TestFlattenDict:
    """_flatten_dict 静态方法测试。"""

    def test_flat_dict(self):
        d = {"a": 1, "b": 2}
        assert _MLflowTrackerUtil._flatten_dict(d) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        d = {"a": {"b": 1, "c": 2}}
        result = _MLflowTrackerUtil._flatten_dict(d)
        assert result == {"a.b": 1, "a.c": 2}

    def test_deeply_nested(self):
        d = {"a": {"b": {"c": 3}}}
        result = _MLflowTrackerUtil._flatten_dict(d)
        assert result == {"a.b.c": 3}

    def test_custom_separator(self):
        d = {"a": {"b": 1}}
        result = _MLflowTrackerUtil._flatten_dict(d, sep="__")
        assert result == {"a__b": 1}

    def test_empty_dict(self):
        assert _MLflowTrackerUtil._flatten_dict({}) == {}


class TestToSafeParamValue:
    """_to_safe_param_value 静态方法测试。"""

    def test_string(self):
        assert _MLflowTrackerUtil._to_safe_param_value("hello") == "hello"

    def test_int(self):
        assert _MLflowTrackerUtil._to_safe_param_value(42) == "42"

    def test_float(self):
        assert _MLflowTrackerUtil._to_safe_param_value(3.14) == "3.14"

    def test_bool(self):
        assert _MLflowTrackerUtil._to_safe_param_value(True) == "True"

    def test_dict_to_json(self):
        result = _MLflowTrackerUtil._to_safe_param_value({"k": "v"})
        assert result == '{"k": "v"}'

    def test_list_to_json(self):
        result = _MLflowTrackerUtil._to_safe_param_value([1, 2, 3])
        assert result == "[1, 2, 3]"


class TestMLflowTrackerUtilInit:
    """_MLflowTrackerUtil 初始化测试。"""

    @patch("tributo.registry.mlflow_util.logger")
    def test_import_error_raises(self, _mock_logger):
        with patch.dict("sys.modules", {"mlflow": None}):
            with pytest.raises(ImportError, match="mlflow is required"):
                _MLflowTrackerUtil()

    @patch("tributo.registry.mlflow_util.logger")
    def test_tracking_uri_set(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil(tracking_uri="http://localhost:5000")
            mock_mlflow.set_tracking_uri.assert_called_once_with(
                "http://localhost:5000"
            )
            assert util._available is True

    @patch("tributo.registry.mlflow_util.logger")
    def test_tracking_uri_failure_graceful(self, mock_logger):
        mock_mlflow = MagicMock()
        mock_mlflow.set_tracking_uri.side_effect = ConnectionError("refused")
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil(
                tracking_uri="http://bad:5000", raise_on_error=False
            )
            assert util._available is False
            mock_logger.warning.assert_called()

    @patch("tributo.registry.mlflow_util.logger")
    def test_tracking_uri_failure_raises(self, _mock_logger):
        mock_mlflow = MagicMock()
        mock_mlflow.set_tracking_uri.side_effect = ConnectionError("refused")
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            with pytest.raises(ConnectionError):
                _MLflowTrackerUtil(tracking_uri="http://bad:5000", raise_on_error=True)


class TestSafeCall:
    """_safe_call 方法测试。"""

    @patch("tributo.registry.mlflow_util.logger")
    def test_unavailable_returns_none(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            util._available = False
            result = util._safe_call(lambda: "should not run")
            assert result is None

    @patch("tributo.registry.mlflow_util.logger")
    def test_success_returns_value(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            result = util._safe_call(lambda: 42)
            assert result == 42

    @patch("tributo.registry.mlflow_util.logger")
    def test_failure_silent(self, mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil(raise_on_error=False)
            result = util._safe_call(lambda: 1 / 0)
            assert result is None
            mock_logger.warning.assert_called()

    @patch("tributo.registry.mlflow_util.logger")
    def test_failure_raises(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil(raise_on_error=True)
            with pytest.raises(ZeroDivisionError):
                util._safe_call(lambda: 1 / 0)


class TestSetupExperiment:
    """setup_experiment 方法测试。"""

    @patch("tributo.registry.mlflow_util.logger")
    def test_creates_new_experiment(self, _mock_logger):
        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp-123"
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            result = util.setup_experiment("test-exp")
            assert result == "exp-123"
            mock_mlflow.create_experiment.assert_called_once_with(
                "test-exp", artifact_location=None
            )

    @patch("tributo.registry.mlflow_util.logger")
    def test_returns_existing_experiment(self, _mock_logger):
        mock_mlflow = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp-existing"
        mock_mlflow.get_experiment_by_name.return_value = mock_exp
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            result = util.setup_experiment("test-exp")
            assert result == "exp-existing"
            mock_mlflow.create_experiment.assert_not_called()


class TestStartRun:
    """start_run 方法测试。"""

    @patch("tributo.registry.mlflow_util.logger")
    def test_starts_run(self, _mock_logger):
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "run-abc"
        mock_mlflow.start_run.return_value = mock_run
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            result = util.start_run(run_name="my-run", tags={"k": "v"})
            assert result == "run-abc"
            mock_mlflow.start_run.assert_called_once_with(
                run_name="my-run", tags={"k": "v"}
            )


class TestLogParams:
    """log_params 方法测试。"""

    @patch("tributo.registry.mlflow_util.logger")
    def test_logs_flat_params(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            util.log_params({"lr": 0.01, "epochs": 10})
            assert mock_mlflow.log_param.call_count == 2

    @patch("tributo.registry.mlflow_util.logger")
    def test_logs_nested_params(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            util.log_params({"model": {"hidden_size": 128}})
            mock_mlflow.log_param.assert_called_once_with("model.hidden_size", "128")

    @patch("tributo.registry.mlflow_util.logger")
    def test_skips_long_keys(self, mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            long_key = "x" * 300
            util.log_params({long_key: 1, "ok": 2})
            mock_mlflow.log_param.assert_called_once_with("ok", "2")
            mock_logger.warning.assert_called()

    @patch("tributo.registry.mlflow_util.logger")
    def test_truncates_long_values(self, mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            long_value = "x" * 600
            util.log_params({"key": long_value})
            call_args = mock_mlflow.log_param.call_args
            assert call_args[0][0] == "key"
            assert len(call_args[0][1]) == 500
            mock_logger.warning.assert_called()


class TestLogMetrics:
    """log_metrics 方法测试。"""

    @patch("tributo.registry.mlflow_util.logger")
    def test_logs_numeric_metrics(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            util.log_metrics({"loss": 0.5, "acc": 0.9, "name": "skip"})
            assert mock_mlflow.log_metric.call_count == 2

    @patch("tributo.registry.mlflow_util.logger")
    def test_logs_with_step(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            util.log_metrics({"loss": 0.3}, step=5)
            mock_mlflow.log_metric.assert_called_once_with("loss", 0.3, step=5)

    @patch("tributo.registry.mlflow_util.logger")
    def test_skips_nan_and_inf(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            util.log_metrics({"loss": float("nan"), "acc": float("inf")})
            mock_mlflow.log_metric.assert_not_called()

    @patch("tributo.registry.mlflow_util.logger")
    def test_bool_metric_logged_as_float(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            util.log_metrics({"flag": True})
            mock_mlflow.log_metric.assert_called_once_with("flag", 1.0, step=None)


class TestEndRun:
    """end_run 方法测试。"""

    @patch("tributo.registry.mlflow_util.logger")
    def test_end_run_finished(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            util.end_run(status="FINISHED")
            mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    @patch("tributo.registry.mlflow_util.logger")
    def test_end_run_failed(self, _mock_logger):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            util = _MLflowTrackerUtil()
            util.end_run(status="FAILED")
            mock_mlflow.end_run.assert_called_once_with(status="FAILED")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))

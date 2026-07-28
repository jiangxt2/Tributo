"""MLflow tracking utility class (internal use).

Referencing the Ray _MLflowLoggerUtil design, wraps the mlflow module
reference to provide a unified logging interface. Supports graceful
degradation: when the MLflow Server is unreachable, log calls fail
silently without blocking the training flow.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class _MLflowTrackerUtil:
    """MLflow tracking utility class (internal use).

    Wraps the mlflow module reference, providing experiment management,
    param/metric logging, and artifact logging capabilities.
    Supports graceful degradation: silently degrades when the MLflow
    Server is unreachable.
    """

    # MLflow param limits (compatible with 2.18.0 and earlier server): key <=250, value <=500.
    # The 2.22.x client raised the value limit to 8000, but conservative handling
    # avoids errors from older servers.
    _MAX_PARAM_KEY_LEN = 250
    _MAX_PARAM_VALUE_LEN = 500

    def __init__(
        self,
        tracking_uri: str | None = None,
        raise_on_error: bool = False,
    ):
        """Initialize the MLflow tracking utility.

        Args:
            tracking_uri: MLflow tracking server URI, None uses the default.
            raise_on_error: Whether to raise on MLflow operation failure.
                False (default) silently logs warnings without blocking training.
        """
        try:
            import mlflow

            self._mlflow = mlflow
        except ImportError as err:
            raise ImportError(
                "mlflow is required for registry module. "
                "Install with: pip install tributo[registry]"
            ) from err

        self._raise_on_error = raise_on_error
        self._available = True

        if tracking_uri:
            try:
                self._mlflow.set_tracking_uri(tracking_uri)
            except Exception as e:
                logger.warning("Failed to set MLflow tracking URI: %s", e)
                if raise_on_error:
                    raise
                self._available = False

    def _safe_call(self, fn, *args, **kwargs) -> Any:
        """Safely call the MLflow API, supporting graceful degradation.

        Args:
            fn: MLflow API function.
            *args, **kwargs: Function arguments.

        Returns:
            Function return value, or None on failure.
        """
        if not self._available:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning("MLflow API call failed: %s", e)
            if self._raise_on_error:
                raise
            return None

    def setup_experiment(
        self,
        experiment_name: str,
        artifact_location: str | None = None,
    ) -> str | None:
        """Set up or create an experiment.

        Args:
            experiment_name: Experiment name.
            artifact_location: Artifact storage location.

        Returns:
            Experiment ID, or None on failure.
        """

        def _setup() -> str:
            experiment = self._mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self._mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location,
                )
            else:
                experiment_id = experiment.experiment_id
            # Set as the active experiment to ensure start_run uses the correct experiment
            self._mlflow.set_experiment(experiment_name)
            return experiment_id

        return self._safe_call(_setup)

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """Start a new Run.

        Args:
            run_name: Run name.
            tags: Run tags.

        Returns:
            Run ID, or None on failure.
        """

        def _start() -> str:
            run = self._mlflow.start_run(run_name=run_name, tags=tags)
            return run.info.run_id

        return self._safe_call(_start)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params (automatically flattens nested dicts).

        Non-serializable values (e.g., custom objects) are converted to strings.
        Keys/values exceeding MLflow length limits log a warning and are skipped
        or truncated to avoid server rejection of the entire batch.

        Args:
            params: Params dict, supports nesting.
        """
        flat_params = self._flatten_dict(params)
        for key, value in flat_params.items():
            if len(key) > self._MAX_PARAM_KEY_LEN:
                logger.warning(
                    "MLflow param key too long (%d > %d), skipping: %s...",
                    len(key),
                    self._MAX_PARAM_KEY_LEN,
                    key[:50],
                )
                continue
            safe_value = self._to_safe_param_value(value)
            if len(safe_value) > self._MAX_PARAM_VALUE_LEN:
                logger.warning(
                    "MLflow param value too long (%d > %d), truncating key '%s'",
                    len(safe_value),
                    self._MAX_PARAM_VALUE_LEN,
                    key,
                )
                safe_value = (
                    safe_value[: self._MAX_PARAM_VALUE_LEN - 11] + "[truncated]"
                )
            self._safe_call(self._mlflow.log_param, key, safe_value)

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
    ) -> None:
        """Log metrics (automatically converts to float).

        Only numeric metrics convertible to float are logged; NaN/Inf and
        non-numeric types are silently skipped.

        Args:
            metrics: Metrics dict.
            step: Step number.
        """
        for key, value in metrics.items():
            try:
                float_value = float(value)
            except (ValueError, TypeError):
                logger.debug(
                    "Cannot log metric '%s' with value '%s' as float, skipping.",
                    key,
                    value,
                )
                continue
            if float_value != float_value:  # NaN
                logger.debug("Skipping NaN metric '%s'.", key)
                continue
            if float_value in (float("inf"), float("-inf")):
                logger.debug("Skipping Inf metric '%s'.", key)
                continue
            self._safe_call(self._mlflow.log_metric, key, float_value, step=step)

    def log_artifact(self, local_path: str) -> None:
        """Log an artifact file.

        Args:
            local_path: Local file path.
        """
        self._safe_call(self._mlflow.log_artifact, local_path)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current Run.

        Args:
            status: Run status (FINISHED / FAILED).
        """
        self._safe_call(self._mlflow.end_run, status=status)

    @staticmethod
    def _flatten_dict(
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        """Flatten a nested dict.

        Args:
            d: The dict to flatten.
            parent_key: Parent key prefix.
            sep: Separator.

        Returns:
            The flattened dict.
        """
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_MLflowTrackerUtil._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def _to_safe_param_value(value: Any) -> str:
        """Convert param value to an MLflow-acceptable string type.

        Args:
            value: Raw param value.

        Returns:
            String representation of the param value.
        """
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)

"""MLflow training tracking callback.

Integrates with BaseTrainer's Template Method lifecycle,
automatically logging training params, metrics, and model artifacts.

All callback methods have exception protection to ensure MLflow failures
do not block the training flow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tributo.registry.mlflow_util import _MLflowTrackerUtil

if TYPE_CHECKING:
    from tributo.training.base import BaseTrainer

from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class MLflowTrackingCallback:
    """MLflow training tracking callback.

    Integrates with the BaseTrainer lifecycle, automatically logging
    training params, metrics, and model artifacts.

    All callback methods have exception protection to ensure MLflow failures
    do not block the training flow.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        raise_on_error: bool = False,
    ):
        """Initialize the MLflow tracking callback.

        Args:
            experiment_name: MLflow experiment name.
            tracking_uri: MLflow tracking server URI.
            run_name: Run name.
            tags: Run tags.
            raise_on_error: Whether to raise on MLflow operation failure.
        """
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._run_name = run_name
        self._tags = tags
        self._raise_on_error = raise_on_error
        self._util: _MLflowTrackerUtil | None = None
        self._run_id: str | None = None

    def on_setup_start(self, trainer: BaseTrainer) -> None:
        """Before training starts: create MLflow Run, log params.

        Args:
            trainer: BaseTrainer instance.
        """
        try:
            self._util = _MLflowTrackerUtil(
                self._tracking_uri,
                raise_on_error=self._raise_on_error,
            )
            self._util.setup_experiment(self._experiment_name)
            self._run_id = self._util.start_run(
                run_name=self._run_name,
                tags=self._tags,
            )
            self._util.log_params(trainer.config)
        except Exception as e:
            logger.warning("MLflow on_setup_start failed: %s", e)
            if self._raise_on_error:
                raise

    def on_training_end(self, trainer: BaseTrainer, result: Any) -> None:
        """After training ends: log metrics.

        Args:
            trainer: BaseTrainer instance.
            result: Return value from training_loop.
        """
        if self._util is None:
            return
        try:
            metrics = getattr(result, "metrics", None)
            if metrics is not None:
                self._util.log_metrics(metrics)
        except Exception as e:
            logger.warning("MLflow on_training_end failed: %s", e)
            if self._raise_on_error:
                raise

    def on_export_end(self, trainer: BaseTrainer, output_path: str) -> None:
        """After model export: log artifact.

        Args:
            trainer: BaseTrainer instance.
            output_path: Model export path.
        """
        if self._util is None:
            return
        try:
            self._util.log_artifact(output_path)
        except Exception as e:
            logger.warning("MLflow on_export_end failed: %s", e)
            if self._raise_on_error:
                raise

    def on_run_complete(self, trainer: BaseTrainer, summary: dict[str, Any]) -> None:
        """Training complete: log summary, end Run.

        Args:
            trainer: BaseTrainer instance.
            summary: Training result summary.
        """
        if self._util is None:
            return
        try:
            self._util.log_metrics(summary)
            self._util.end_run(status="FINISHED")
            logger.info("MLflow Run %s completed.", self._run_id)
        except Exception as e:
            logger.warning("MLflow on_run_complete failed: %s", e)
            if self._raise_on_error:
                raise

    def on_run_error(self, trainer: BaseTrainer, error: Exception) -> None:
        """Training failed: end Run (FAILED status).

        Args:
            trainer: BaseTrainer instance.
            error: The exception that caused training to fail.
        """
        if self._util is None:
            return
        try:
            self._util.end_run(status="FAILED")
            logger.error("MLflow Run %s failed: %s", self._run_id, error)
        except Exception as e:
            logger.warning("MLflow on_run_error failed: %s", e)
            if self._raise_on_error:
                raise

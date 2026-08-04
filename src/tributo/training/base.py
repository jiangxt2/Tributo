"""Base trainer and registration spec.

``BaseTrainer`` uses the Template Method pattern — subclasses only need to
implement ``setup``, ``training_loop``, and ``export_model`` (or the newer
``export_artifacts`` for non-model outputs).  An optional callback mechanism
supports integration with experiment trackers such as MLflow.

.. deprecated:: 0.5.0
    Override ``export_artifacts()`` instead of ``export_model()``.
    ``export_model()`` is kept as a backward-compatible alias with a
    no-op default implementation.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

from tributo.training.algorithm_spec import AlgorithmSpec
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backward-compatible alias (Phase 1: pure alias, no warning)
# ---------------------------------------------------------------------------
# v1.x — TrainerSpec = AlgorithmSpec (zero-friction migration)
# v2.0 — add @deprecated + DeprecationWarning on constructor use
# v3.0 — remove TrainerSpec name
TrainerSpec = AlgorithmSpec


class TrainerCallback(Protocol):
    """Trainer callback protocol.

    Callback objects should implement the corresponding lifecycle methods
    on ``BaseTrainer``.
    """

    def on_setup_start(self, trainer: BaseTrainer) -> None: ...
    def on_training_end(self, trainer: BaseTrainer, result: Any) -> None: ...
    def on_export_end(self, trainer: BaseTrainer, output_path: str) -> None: ...
    def on_artifacts_exported(self, trainer: BaseTrainer, output_path: str) -> None: ...
    def on_run_complete(
        self, trainer: BaseTrainer, summary: dict[str, Any]
    ) -> None: ...
    def on_run_error(self, trainer: BaseTrainer, error: Exception) -> None: ...


@PublicAPI(stability="beta")
class BaseTrainer(ABC):
    """Base class for trainers — ``setup → training_loop → export_artifacts``.

    Subclasses must implement two abstract methods:
    - ``setup()``: Initialise model, optimizer, data preprocessing, etc.
    - ``training_loop()``: Run training, return a checkpoint or model object.

    For artifact export, subclasses should override ``export_artifacts()``
    (the new extension point).  ``export_model()`` is kept for backward
    compatibility — its default no-op implementation is called by
    ``export_artifacts()`` when the subclass only overrides the legacy
    ``export_model()`` method.

    The base class provides ``run()`` as a Template Method that orchestrates
    the three steps in order.

    An optional callback mechanism supports integration with experiment
    trackers such as MLflow.  Callback methods are guarded with exception
    handling by default so that callback failures do not block the training
    pipeline.  Note: ``on_setup_start`` exceptions are handled by the
    callback itself (based on its ``raise_on_error`` attribute); the base
    class does NOT catch them in ``run()`` so that callbacks can abort
    training early when needed.

    Args:
        datasets: Dataset dictionary.
        config: Training configuration.
        run_config: Optional Ray run configuration.
        **kwargs: Extension parameters for backward-compatible API evolution.
            Currently supported keys:
            - ``callbacks``: ``list[TrainerCallback]`` — callback objects
              implementing the lifecycle methods (``on_setup_start``,
              ``on_training_end``, ``on_export_end``,
              ``on_artifacts_exported``, ``on_run_complete``,
              ``on_run_error``).
    """

    def __init__(
        self,
        datasets: dict[str, ray.data.Dataset],
        config: dict[str, Any],
        run_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the trainer.

        Args:
            datasets: Dataset dict, keys are dataset names (e.g. ``"train"``,
                ``"val"``).
            config: Training configuration dict.
            run_config: Optional Ray run configuration.
            **kwargs: Extension parameters.  Currently supports ``callbacks``
                (a list of ``TrainerCallback``).
        """
        self.datasets = datasets
        self.config = config
        self.run_config = run_config or {}
        # Extract callbacks from kwargs to avoid changing subclass signatures.
        # Shallow-copy so external list mutations don't leak into the trainer.
        self._callbacks: list[TrainerCallback] = list(kwargs.get("callbacks", []))
        # Run summary, populated by ``run()``.  Declared here so subclasses
        # writing export results into ``self._summary`` (the contract
        # documented on ``run()``) do not trigger type-check errors.
        self._summary: dict[str, Any] = {}

    @abstractmethod
    def setup(self) -> None:
        """Initialise model, optimizer, and other resources."""

    @abstractmethod
    def training_loop(self) -> Any:
        """Run training and return a checkpoint or model object."""

    # -- artifact export (Phase 4: algorithm extensibility) --------------------

    def export_artifacts(self, checkpoint: Any, output_path: str) -> None:
        """Export training artifacts to *output_path*.

        The default implementation delegates to ``export_model()`` for
        backward compatibility.  Subclasses that produce non-model
        artifacts (reports, diagnostics, graph snapshots) should override
        this method directly.

        Args:
            checkpoint: The checkpoint or model object returned by
                ``training_loop``.
            output_path: Export destination (local path or S3 URI).
        """
        # Detect the no-op path: if neither export_artifacts nor export_model
        # was overridden by the subclass, warn the user.
        if type(self).export_model is BaseTrainer.export_model:
            logger.warning(
                "%s does not override export_artifacts() or export_model(); "
                "no artifact will be exported.",
                type(self).__name__,
            )
        self.export_model(checkpoint, output_path)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Warn at class-definition time when a subclass still uses the
        # legacy export_model path (instead of export_artifacts).  The
        # export_artifacts default delegates to export_model, so existing
        # subclasses keep working — this is a gentle nudge, not a break.
        if "export_model" in cls.__dict__ and "export_artifacts" not in cls.__dict__:
            warnings.warn(
                f"{cls.__name__} overrides export_model() — "
                "export_artifacts() is the preferred hook. "
                "Override export_artifacts() instead for artifact-kit-aware export.",
                DeprecationWarning,
                stacklevel=2,
            )

    def _export_artifacts_default(self, checkpoint: Any, output_path: str) -> None:
        """Default artifact-export dispatch."""
        if type(self).export_artifacts is not BaseTrainer.export_artifacts:
            self.export_artifacts(checkpoint, output_path)
        elif type(self).export_model is not BaseTrainer.export_model:
            self.export_model(checkpoint, output_path)
        else:
            logger.warning(
                "%s does not override export_artifacts() or export_model(); "
                "no artifact will be exported.",
                type(self).__name__,
            )

    def export_model(self, checkpoint: Any, output_path: str) -> None:  # noqa: B027
        """Export the model to the given path.

        .. deprecated:: 0.5.0
            Override ``export_artifacts()`` instead.  This method is kept
            as a backward-compatible alias — the default implementation
            is a no-op.

        Args:
            checkpoint: The checkpoint or model object returned by
                ``training_loop``.
            output_path: Export destination (local path or S3 URI).
        """
        pass

    # -- Template Method -------------------------------------------------------

    def run(
        self,
        output_path: str = "",
        *,
        bundle_config: Any | None = None,
    ) -> dict[str, Any]:
        """Template Method: ``setup → training_loop → export_artifacts``.

        Subclasses should write their actual results into ``self._summary``
        inside ``export_artifacts`` (or the legacy ``export_model``); this
        method returns that dict.

        When *bundle_config* is set (a ``BundleOutputConfig`` with non-empty
        ``targets``), the export path is routed through
        ``BundleExportService`` instead of the legacy ``export_artifacts()``
        method.  This is the **bundle mode** entry point — it uses the
        new export framework (planner → executor → publisher).

        Args:
            output_path: Artifact export path (legacy mode only; bundle
                mode passes an empty string by default).
            bundle_config: Optional ``BundleOutputConfig`` for bundle-mode
                export.  When provided and has non-empty targets, the
                export is routed through the new bundle pipeline.

        Returns:
            A summary dict, containing at minimum ``{"status":
            "succeeded" | "partial"}`` — a partial bundle is still a
            successful run, but the summary marks it ``partial``.
        """
        from tributo.training.callbacks import CallbackDispatcher
        from tributo.training.lifecycle import TrainingLifecycle

        lifecycle = TrainingLifecycle(self, CallbackDispatcher(self._callbacks))
        return lifecycle.run(output_path, bundle_config=bundle_config)

    def _export_bundle(self, checkpoint: Any, bundle_config: Any) -> None:
        """Legacy bundle-export hook (backward compatible).

        Kept so subclasses that override the protected hook keep working:
        ``TrainingLifecycle.run`` dispatches to the subclass override when
        present.  The default implementation forwards to the current bundle
        route, so ``super()._export_bundle(...)`` inside an override still
        runs the real pipeline.  The default path never calls this method.
        Overrides bypass the default Ray Jobs identity binding and must handle
        ``TRIBUTO_RUN_ID`` and ``TRIBUTO_ATTEMPT_ID`` themselves when needed.
        """
        from tributo.training.callbacks import CallbackDispatcher
        from tributo.training.lifecycle import TrainingLifecycle

        TrainingLifecycle(self, CallbackDispatcher(self._callbacks))._export_bundle(
            checkpoint, bundle_config, self._summary
        )

    @staticmethod
    def _get_trainer_type() -> str:
        """Return the trainer_type string for this trainer class.

        Every Bundle-capable trainer must override this method.  A default
        trainer identity would silently route a new trainer through the
        wrong source provider, so missing declarations fail fast.
        """
        raise NotImplementedError(
            "BaseTrainer subclasses must declare _get_trainer_type() "
            "before using Bundle export"
        )

    @staticmethod
    def _get_tributo_version() -> str:
        """Return the current tributo version string."""
        try:
            from importlib.metadata import version

            return version("tributo")
        except Exception:
            return "0.0.0"

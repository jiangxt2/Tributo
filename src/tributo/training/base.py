"""Base trainer and registration spec.

``BaseTrainer`` uses the Template Method pattern — subclasses only need to
implement ``setup``, ``training_loop``, and ``export_model``.  An optional
callback mechanism supports integration with experiment trackers such as
MLflow.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

from tributo.training.algorithm_spec import AlgorithmSpec
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data

logger = logging.getLogger(__name__)

# ── Plugin helpers ────────────────────────────────────────────────────────────────

_provider_plugins_loaded = False


def _load_provider_plugins(registry: Any) -> None:
    """Load source-provider plugins into *registry* (cached)."""
    global _provider_plugins_loaded
    if not _provider_plugins_loaded:
        from tributo.plugin import discover_source_provider_plugins

        for cls in discover_source_provider_plugins():
            registry.register(cls)
        _provider_plugins_loaded = True


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
    def on_run_complete(
        self, trainer: BaseTrainer, summary: dict[str, Any]
    ) -> None: ...
    def on_run_error(self, trainer: BaseTrainer, error: Exception) -> None: ...


@PublicAPI(stability="beta")
class BaseTrainer(ABC):
    """Base class for trainers — ``setup → training_loop → export_model``.

    Subclasses must implement three abstract methods:
    - ``setup()``: Initialise model, optimizer, data preprocessing, etc.
    - ``training_loop()``: Run training, return a checkpoint or model object.
    - ``export_model()``: Export the model to the given path.

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
              ``on_training_end``, ``on_export_end``, ``on_run_complete``,
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

    @abstractmethod
    def setup(self) -> None:
        """Initialise model, optimizer, and other resources."""

    @abstractmethod
    def training_loop(self) -> Any:
        """Run training and return a checkpoint or model object."""

    @abstractmethod
    def export_model(self, checkpoint: Any, output_path: str) -> None:
        """Export the model to the given path.

        Args:
            checkpoint: The checkpoint or model object returned by
                ``training_loop``.
            output_path: Export destination (local path or S3 URI).
        """

    def run(
        self,
        output_path: str,
        *,
        bundle_config: Any | None = None,
    ) -> dict[str, Any]:
        """Template Method: ``setup → training_loop → export_model``.

        Subclasses should write their actual results into ``self._summary``
        inside ``export_model``; this method returns that dict.

        When *bundle_config* is set (a ``BundleOutputConfig`` with non-empty
        ``targets``), the export path is routed through
        ``BundleExportService`` instead of the legacy ``export_model()``
        method.  This is the **bundle mode** entry point — it uses the
        new export framework (planner → executor → publisher).

        Args:
            output_path: Model export path (legacy mode).
            bundle_config: Optional ``BundleOutputConfig`` for bundle-mode
                export.  When provided and has non-empty targets, the
                export is routed through the new bundle pipeline.

        Returns:
            A summary dict, containing at minimum ``{"status": "succeeded"}``.
        """
        self._summary: dict[str, Any] = {"status": "succeeded"}

        # Let the callback decide whether to raise (via raise_on_error).
        # The base class does not catch here so callbacks can abort early.
        for cb in self._callbacks:
            cb.on_setup_start(self)

        try:
            logger.info("Starting %s training...", type(self).__name__)
            self.setup()
            checkpoint = self.training_loop()

            # Fire on_training_end
            for cb in self._callbacks:
                try:
                    cb.on_training_end(self, checkpoint)
                except Exception as e:
                    logger.warning("Callback on_training_end failed: %s", e)

            # ── Bundle mode routing ──
            if (
                bundle_config is not None
                and hasattr(bundle_config, "targets")
                and bundle_config.targets is not None
                and len(bundle_config.targets) > 0
            ):
                self._export_bundle(checkpoint, bundle_config)
            else:
                self.export_model(checkpoint, output_path)

            # Fire on_export_end
            for cb in self._callbacks:
                try:
                    cb.on_export_end(self, output_path)
                except Exception as e:
                    logger.warning("Callback on_export_end failed: %s", e)

            logger.info("%s training completed.", type(self).__name__)

            # Fire on_run_complete
            for cb in self._callbacks:
                try:
                    cb.on_run_complete(self, self._summary)
                except Exception as e:
                    logger.warning("Callback on_run_complete failed: %s", e)

        except Exception as e:
            # Fire on_run_error; use exception chaining if a callback also
            # raises so the original training error is preserved for debugging.
            callback_error: Exception | None = None
            for cb in self._callbacks:
                try:
                    cb.on_run_error(self, e)
                except Exception as cb_err:
                    logger.warning("Callback on_run_error failed: %s", cb_err)
                    if callback_error is None:
                        callback_error = cb_err
            if callback_error is not None:
                raise callback_error from e
            raise

        return self._summary

    def _export_bundle(
        self,
        checkpoint: Any,
        bundle_config: Any,
    ) -> None:
        """Route export through the new bundle pipeline.

        Resolves a ``SourceProvider`` from the training checkpoint, creates
        an ``ExportSource``, and delegates to ``BundleExportService``.
        Results are written to ``self._summary``.

        Args:
            checkpoint: The training result (Ray Result or raw model).
            bundle_config: A ``BundleOutputConfig`` with non-empty targets.
        """
        from tributo.exporting.service import BundleExportService
        from tributo.exporting.registries import SourceProviderRegistry

        # Resolve the source provider for this trainer type.
        provider_registry = SourceProviderRegistry()
        _load_provider_plugins(provider_registry)

        trainer_type = self._get_trainer_type()
        provider_cls = provider_registry.resolve(trainer_type)
        provider = provider_cls()

        # Open the source and run the bundle pipeline.
        with provider.open_source(checkpoint) as source:
            service = BundleExportService()
            result = service.export_bundle(
                source=source,
                config=bundle_config,
                provider=provider,
                tributo_version=self._get_tributo_version(),
            )

        # Populate summary from bundle result.
        self._summary.update({
            "bundle_id": result.bundle_id,
            "canonical_uri": result.canonical_uri,
            "manifest_sha256": result.manifest_sha256,
            "artifacts": [
                {"name": a.name, "format": a.format, "tree_digest": a.tree_digest}
                for a in result.artifacts
            ],
            "node_results": [
                {"node_id": nr.node_id, "status": nr.status, "target_name": nr.target_name}
                for nr in result.node_results
            ],
        })

    @staticmethod
    def _get_trainer_type() -> str:
        """Return the trainer_type string for this trainer class.

        Subclasses override this to declare their trainer type identity.
        Default returns ``"xgboost"`` — the most common case.
        """
        return "xgboost"

    @staticmethod
    def _get_tributo_version() -> str:
        """Return the current tributo version string."""
        try:
            from importlib.metadata import version
            return version("tributo")
        except Exception:
            return "0.0.0"

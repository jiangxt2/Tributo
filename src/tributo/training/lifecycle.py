"""Training lifecycle orchestration.

``TrainingLifecycle`` owns the template flow that used to be inlined in
``BaseTrainer.run``: event dispatch, export routing (bundle mode vs legacy
artifact export) and summary population.  ``BaseTrainer`` keeps the
algorithm abstraction (``setup``/``training_loop``/``export_artifacts``)
and delegates ``run`` to this class.
"""

from __future__ import annotations

import logging
from typing import Any

from tributo.training.base import BaseTrainer
from tributo.training.callbacks import CallbackDispatcher

logger = logging.getLogger(__name__)

# ── Source-provider plugin discovery (bundle export path) ────────────────

_provider_plugins_cache: list[Any] | None = None


def _load_provider_plugins(registry: Any) -> None:
    """Load source-provider plugins into *registry*.

    Discovery runs once (cached), but classes are re-registered into every
    fresh registry so that repeated ``.run()`` calls in the same process
    see the full provider set.
    """
    global _provider_plugins_cache
    if _provider_plugins_cache is None:
        from tributo.plugin import discover_source_provider_plugins

        _provider_plugins_cache = discover_source_provider_plugins()

    for cls in _provider_plugins_cache:
        if cls.provider_id not in {p.provider_id for p in registry._by_id.values()}:
            registry.register(cls)


class TrainingLifecycle:
    """Orchestrate a single trainer run.

    Args:
        trainer: The ``BaseTrainer`` whose ``setup``/``training_loop``
            define the algorithm.
        dispatcher: Callback dispatcher receiving lifecycle events.
    """

    def __init__(self, trainer: BaseTrainer, dispatcher: CallbackDispatcher) -> None:
        self._trainer = trainer
        self._dispatcher = dispatcher

    def run(
        self,
        output_path: str = "",
        *,
        bundle_config: Any | None = None,
    ) -> dict[str, Any]:
        """Run the ``setup → training_loop → export`` flow.

        When *bundle_config* has non-empty ``targets``, export is routed
        through ``BundleExportService`` (bundle mode); otherwise the legacy
        ``export_artifacts``/``export_model`` hooks run.  Returns the
        summary dict, containing at minimum ``{"status": "succeeded" |
        "partial"}`` — a partial bundle is still a successful run.
        """
        trainer = self._trainer
        summary: dict[str, Any] = {"status": "succeeded"}
        # Subclasses write export results into ``trainer._summary`` (the
        # contract documented on ``BaseTrainer.run``); keep the same dict
        # in both places so that contract keeps working.
        trainer._summary = summary

        # Let the callback decide whether to raise (via raise_on_error).
        # Not caught here so callbacks can abort training early.
        self._dispatcher.on_setup_start(trainer)

        try:
            logger.info("Starting %s training...", type(trainer).__name__)
            trainer.setup()
            checkpoint = trainer.training_loop()

            self._dispatcher.on_training_end(trainer, checkpoint)

            if (
                bundle_config is not None
                and hasattr(bundle_config, "targets")
                and bundle_config.targets is not None
                and len(bundle_config.targets) > 0
            ):
                # Bundle mode: post-publish actions run through the
                # PublicationRunner hooks — legacy artifact-export
                # callbacks are not fired (backward-compat contract).
                # Subclass overrides of the historical
                # ``BaseTrainer._export_bundle`` hook keep working: dispatch
                # to them when present (MRO lookup, so indirect overrides
                # via an intermediate base class are found too), otherwise
                # use the default route.
                if type(trainer)._export_bundle is not BaseTrainer._export_bundle:
                    trainer._export_bundle(checkpoint, bundle_config)
                else:
                    self._export_bundle(checkpoint, bundle_config, summary)
            else:
                trainer._export_artifacts_default(checkpoint, output_path)
                self._dispatcher.on_artifacts_exported(trainer, output_path)

            logger.info("%s training completed.", type(trainer).__name__)
            self._dispatcher.on_run_complete(trainer, summary)

        except Exception as e:
            # Fire on_run_error; use exception chaining if a callback also
            # raises so the original training error is preserved.
            callback_error = self._dispatcher.on_run_error(trainer, e)
            if callback_error is not None:
                raise callback_error from e
            raise

        return summary

    def _export_bundle(
        self,
        checkpoint: Any,
        bundle_config: Any,
        summary: dict[str, Any],
    ) -> None:
        """Route export through the new bundle pipeline.

        Resolves a ``SourceProvider`` from the training checkpoint, creates
        an ``ExportSource``, and delegates to ``BundleExportService``.
        Results are written into *summary*.
        """
        from tributo.exporting.registries import SourceProviderRegistry
        from tributo.exporting.service import BundleExportService

        trainer = self._trainer

        provider_registry = SourceProviderRegistry()
        _load_provider_plugins(provider_registry)

        trainer_type = trainer._get_trainer_type()
        provider_cls = provider_registry.resolve(trainer_type)
        provider = provider_cls()

        with provider.open_source(checkpoint) as source:
            service = BundleExportService()
            result = service.export_bundle(
                source=source,
                config=bundle_config,
                provider=provider,
                tributo_version=trainer._get_tributo_version(),
            )

        summary.update(
            {
                "status": result.status,
                "bundle_id": result.bundle_id,
                "canonical_uri": result.canonical_uri,
                "manifest_sha256": result.manifest_sha256,
                "artifacts": [
                    {"name": a.name, "format": a.format, "tree_digest": a.tree_digest}
                    for a in result.artifacts
                ],
                "node_results": [
                    {
                        "node_id": nr.node_id,
                        "status": nr.status,
                        "target_name": nr.target_name,
                    }
                    for nr in result.node_results
                ],
            }
        )

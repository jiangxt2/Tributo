"""Training lifecycle orchestration.

``TrainingLifecycle`` owns the template flow that used to be inlined in
``BaseTrainer.run``: event dispatch, export routing (bundle mode vs legacy
artifact export) and summary population.  ``BaseTrainer`` keeps the
algorithm abstraction (``setup``/``training_loop``/``export_artifacts``)
and delegates ``run`` to this class.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import warnings
from typing import Any

from ray.exceptions import RayTaskError, TaskCancelledError

from tributo.training.base import BaseTrainer
from tributo.training.callbacks import CallbackDispatcher
from tributo.training.results import (
    BundleStatus,
    TrainingHookStatus,
    TrainingResult,
    TrainingStatus,
    aggregate_hook_status,
)

logger = logging.getLogger(__name__)

# ── Source-provider plugin discovery (bundle export path) ────────────────

_provider_plugins_cache: list[Any] | None = None


def _is_cancellation(error: BaseException) -> bool:
    """Recognize real local or Ray cancellation exceptions, including wrappers."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    cancellation_types = (
        asyncio.CancelledError,
        concurrent.futures.CancelledError,
        TaskCancelledError,
    )
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, cancellation_types):
            return True
        # Only explicit exception chaining represents wrapper intent. Implicit
        # ``__context__`` merely records that a different exception was raised
        # while handling another one and must not turn that new failure into a
        # cancellation.
        candidates: list[object] = [current.__cause__]
        if isinstance(current, RayTaskError):
            candidates.append(current.cause)
        pending.extend(
            candidate
            for candidate in candidates
            if isinstance(candidate, BaseException)
        )
    return False


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
        if cls.provider_id not in registry.list_all():
            registry.register(cls)


def _bind_job_identity(bundle_config: Any) -> tuple[Any, str | None]:
    """Bind job attempt identity to a bundle config when running in Ray Jobs."""
    run_id = os.environ.get("TRIBUTO_RUN_ID")
    attempt_id = os.environ.get("TRIBUTO_ATTEMPT_ID")
    if not run_id:
        return bundle_config, attempt_id

    from tributo.exporting.models import BundleOutputConfig

    if not isinstance(bundle_config, BundleOutputConfig):
        return bundle_config, attempt_id
    if bundle_config.run_id not in (None, run_id):
        raise ValueError("Bundle run_id conflicts with TRIBUTO_RUN_ID")
    if bundle_config.request_id not in (None, run_id):
        raise ValueError("Bundle request_id conflicts with TRIBUTO_RUN_ID")

    config_data = bundle_config.model_dump()
    config_data["run_id"] = run_id
    return BundleOutputConfig.model_validate(config_data), attempt_id


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
        legacy_export: bool = False,
    ) -> dict[str, Any]:
        """Run the ``setup → training_loop → export`` flow.

        First-party trainers require an explicit Bundle destination and use
        their standard targets when none are supplied.  Their legacy
        ``export_artifacts``/``export_model`` hooks run only when callers opt
        in with ``legacy_export=True``.  Compatible third-party trainers that
        do not declare Bundle defaults retain their existing lifecycle.
        """
        trainer = self._trainer

        summary: dict[str, Any] = {"status": "succeeded"}
        # Subclasses write export results into ``trainer._summary`` (the
        # contract documented on ``BaseTrainer.run``); keep the same dict
        # in both places so that contract keeps working.
        trainer._summary = summary

        training_completed = False
        bundle_attempted = False
        committed_bundle: Any | None = None
        terminal_result: TrainingResult | None = None

        try:
            bundle_config = self._prepare_bundle_config(
                output_path,
                bundle_config,
                legacy_export=legacy_export,
            )
            # Required setup callbacks may abort before trainer setup.  They
            # remain inside the lifecycle error boundary so on_run_error can
            # close any external run that was created before the failure.
            self._dispatcher.on_setup_start(trainer)
            logger.info("Starting %s training...", type(trainer).__name__)
            trainer.setup()
            checkpoint = trainer.training_loop()
            training_completed = True

            self._dispatcher.on_training_end(trainer, checkpoint)

            if (
                bundle_config is not None
                and hasattr(bundle_config, "targets")
                and bundle_config.targets is not None
                and len(bundle_config.targets) > 0
            ):
                bundle_attempted = True
                # Bundle mode: post-publish actions run through the
                # publication Hook dispatcher — legacy artifact-export
                # callbacks are not fired (backward-compat contract).
                # Subclass overrides of the historical
                # ``BaseTrainer._export_bundle`` hook keep working: dispatch
                # to them when present (MRO lookup, so indirect overrides
                # via an intermediate base class are found too), otherwise
                # use the default route.
                if type(trainer)._export_bundle is not BaseTrainer._export_bundle:
                    trainer._export_bundle(checkpoint, bundle_config)
                else:
                    committed_bundle = self._export_bundle(
                        checkpoint, bundle_config, summary
                    )
            else:
                trainer._export_artifacts_default(checkpoint, output_path)
                self._dispatcher.on_artifacts_exported(trainer, output_path)

            terminal_result = self._build_success_result(
                summary,
                output_path=output_path,
                bundle_attempted=bundle_attempted,
                bundle_result=committed_bundle,
            )
            self._merge_result(summary, terminal_result)
            logger.info("%s training completed.", type(trainer).__name__)
            self._dispatcher.on_run_complete(trainer, summary)

        except BaseException as e:
            if not isinstance(e, Exception) and not _is_cancellation(e):
                raise
            training_result = self._build_error_result(
                e,
                summary,
                training_completed=training_completed,
                bundle_attempted=bundle_attempted,
                committed_result=terminal_result,
            )
            self._merge_result(summary, training_result)
            try:
                vars(e)["training_result"] = training_result
            except Exception:
                e.add_note(
                    "TrainingResult could not be attached to this exception type; "
                    f"terminal state was {training_result.model_dump(mode='json')}"
                )
            # Error callbacks may fail too, but the training exception remains
            # the primary error and is always re-raised unchanged.
            callback_error = self._dispatcher.on_run_error(trainer, e)
            if callback_error is not None:
                e.add_note(
                    "on_run_error callback also failed: "
                    f"{type(callback_error).__name__}: {callback_error}"
                )
            raise

        return summary

    def _prepare_bundle_config(
        self,
        output_path: str,
        bundle_config: Any | None,
        *,
        legacy_export: bool,
    ) -> Any | None:
        """Apply first-party Bundle defaults before any training side effect."""
        default_targets = self._trainer._default_bundle_targets()
        if default_targets is None:
            return bundle_config
        if legacy_export:
            warnings.warn(
                "legacy_export=True is deprecated for first-party trainers; "
                "configure a Bundle URI instead",
                DeprecationWarning,
                stacklevel=3,
            )
            return None

        from tributo.exporting.models import BundleOutputConfig

        if bundle_config is None:
            if not output_path:
                raise ValueError(
                    "First-party trainers require an explicit Bundle URI via "
                    "bundle_config.bundle_uri or output_path"
                )
            return BundleOutputConfig(
                bundle_uri=output_path,
                targets=list(default_targets),
                roles=self._trainer._default_bundle_roles(),
            )
        if not isinstance(bundle_config, BundleOutputConfig):
            raise TypeError("bundle_config must be a BundleOutputConfig")
        if bundle_config.targets is not None:
            return bundle_config

        data = bundle_config.model_dump()
        if data.get("bundle_uri") is None and output_path:
            data["bundle_uri"] = output_path
        data["targets"] = [target.model_dump() for target in default_targets]
        if not data.get("roles"):
            data["roles"] = self._trainer._default_bundle_roles()
        return BundleOutputConfig.model_validate(data)

    @staticmethod
    def _merge_result(summary: dict[str, Any], result: TrainingResult) -> None:
        summary.update(result.model_dump(mode="json"))
        summary["status"] = (
            "partial"
            if result.bundle_status == BundleStatus.PARTIAL
            else result.training_status.value
        )

    @staticmethod
    def _summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
        metrics = summary.get("metrics")
        return dict(metrics) if isinstance(metrics, dict) else {}

    def _build_success_result(
        self,
        summary: dict[str, Any],
        *,
        output_path: str,
        bundle_attempted: bool,
        bundle_result: Any | None,
    ) -> TrainingResult:
        if bundle_attempted:
            bundle_status = BundleStatus(
                getattr(bundle_result, "status", summary.get("status", "succeeded"))
            )
            bundle_uri = getattr(
                bundle_result, "canonical_uri", summary.get("canonical_uri")
            )
            execution_id = getattr(
                bundle_result, "execution_id", summary.get("execution_id")
            )
            if bundle_uri is None or execution_id is None:
                raise RuntimeError(
                    "Bundle publication completed without the required canonical URI "
                    "and execution ID"
                )
            receipts = getattr(bundle_result, "hook_receipts", ())
            hook_status = aggregate_hook_status(receipts)
            return TrainingResult(
                model_uri=bundle_uri,
                bundle_uri=bundle_uri,
                metrics=self._summary_metrics(summary),
                training_status=TrainingStatus.SUCCEEDED,
                bundle_status=bundle_status,
                hook_status=hook_status,
                execution_id=execution_id,
            )

        legacy_uri = summary.get("onnx_path") or output_path or None
        return TrainingResult(
            model_uri=legacy_uri,
            metrics=self._summary_metrics(summary),
            legacy_artifact_uri=legacy_uri,
            training_status=TrainingStatus.SUCCEEDED,
            bundle_status=BundleStatus.NOT_STARTED,
            hook_status=TrainingHookStatus.NOT_CONFIGURED,
        )

    def _build_error_result(
        self,
        error: BaseException,
        summary: dict[str, Any],
        *,
        training_completed: bool,
        bundle_attempted: bool,
        committed_result: TrainingResult | None,
    ) -> TrainingResult:
        if committed_result is not None:
            error.add_note(
                "Training and Bundle publication had already reached a terminal "
                "state before this callback failed"
            )
            return committed_result

        bundle_result = getattr(error, "bundle_result", None)
        if bundle_result is not None:
            bundle_uri = getattr(bundle_result, "canonical_uri", None)
            execution_id = getattr(bundle_result, "execution_id", None)
            if bundle_uri is not None and execution_id is not None:
                return TrainingResult(
                    model_uri=bundle_uri,
                    bundle_uri=bundle_uri,
                    metrics=self._summary_metrics(summary),
                    training_status=TrainingStatus.SUCCEEDED,
                    bundle_status=BundleStatus(bundle_result.status),
                    hook_status=aggregate_hook_status(
                        getattr(error, "receipts", ())
                        or getattr(bundle_result, "hook_receipts", ())
                    ),
                    execution_id=execution_id,
                )
            error.add_note(
                "Bundle error metadata omitted canonical_uri or execution_id; "
                "publication is reported as failed instead of replacing the "
                "original exception with a result-validation error"
            )

        if training_completed and bundle_attempted:
            execution = getattr(error, "execution_result", None)
            return TrainingResult(
                metrics=self._summary_metrics(summary),
                training_status=TrainingStatus.SUCCEEDED,
                bundle_status=BundleStatus.FAILED,
                hook_status=TrainingHookStatus.NOT_CONFIGURED,
                execution_id=getattr(execution, "execution_id", None),
            )

        status = (
            TrainingStatus.CANCELLED
            if _is_cancellation(error)
            else TrainingStatus.FAILED
        )
        return TrainingResult(
            metrics=self._summary_metrics(summary),
            training_status=status,
            bundle_status=BundleStatus.NOT_STARTED,
            hook_status=TrainingHookStatus.NOT_CONFIGURED,
        )

    def _export_bundle(
        self,
        checkpoint: Any,
        bundle_config: Any,
        summary: dict[str, Any],
    ) -> Any:
        """Route export through the new bundle pipeline.

        Resolves an ``ExportSourceProvider`` from the training checkpoint, creates
        an ``ExportSource``, and delegates to ``BundleExportService``.
        Results are written into *summary*.
        """
        from tributo.exporting.registries import SourceProviderRegistry
        from tributo.exporting.service import BundleExportService

        trainer = self._trainer
        bound_config, attempt_id = _bind_job_identity(bundle_config)

        provider_registry = SourceProviderRegistry()
        _load_provider_plugins(provider_registry)

        trainer_type = trainer._get_trainer_type()
        provider_cls = provider_registry.resolve(trainer_type)
        provider = provider_cls()

        with provider.open_source(checkpoint) as source:
            service = BundleExportService()
            result = service.export_bundle(
                source=source,
                config=bound_config,
                tributo_version=trainer._get_tributo_version(),
                attempt_id=attempt_id,
            )

        summary.update(
            {
                "status": result.status,
                "bundle_id": result.bundle_id,
                "execution_id": result.execution_id,
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
        return result

"""Unit tests for training/lifecycle.py — TrainingLifecycle."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from ray.exceptions import RayTaskError, TaskCancelledError

from tributo.exceptions import (
    BundleExportError,
    JobConfigurationError,
    PostPublishCallbackError,
)
from tributo.exporting.models import (
    BundleOutputConfig,
    ExportTarget,
    HookStatus,
)
from tributo.training.algorithm_spec import AlgorithmSpec, DataLoadingMode
from tributo.training.base import BaseTrainer, TrainerCallback
from tributo.training.callbacks import CallbackDispatcher
from tributo.training.lifecycle import TrainingLifecycle
from tributo.training.local_runner import run_local_trial
from tributo.training.results import (
    BundleStatus,
    TrainingHookStatus,
    TrainingResult,
    TrainingStatus,
    aggregate_hook_status,
)


class _FakeTrainer(BaseTrainer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(datasets={}, config={}, **kwargs)
        self.events: list[str] = []

    def setup(self) -> None:
        self.events.append("setup")

    def training_loop(self) -> Any:
        self.events.append("training_loop")
        return "checkpoint"

    @staticmethod
    def _get_trainer_type() -> str:
        return "xgboost"


class _FailingSetupTrainer(_FakeTrainer):
    def setup(self) -> None:
        self.events.append("setup")
        raise RuntimeError("setup boom")


class _ExportingTrainer(_FakeTrainer):
    def export_artifacts(self, checkpoint: Any, output_path: str) -> None:
        self.events.append(f"export_artifacts:{output_path}")


class _SummaryWritingTrainer(_FakeTrainer):
    """Legacy contract: export hooks write results into ``self._summary``."""

    def export_model(self, checkpoint: Any, output_path: str) -> None:
        self._summary["metrics"] = {"accuracy": 0.9}


class _MetricsCheckpointTrainer(_FakeTrainer):
    def training_loop(self) -> Any:
        self.events.append("training_loop")
        return SimpleNamespace(
            metrics={"eval-logloss_history": [0.8, 0.4], "eval-logloss": 0.4}
        )


class _EntryTrainer(BaseTrainer):
    """Production-shaped trainer: accepts ``datasets``/``config`` like
    ``run_local_trial`` constructs them (``trainer_cls(datasets=...,
    config=...)``) — ``_FakeTrainer`` pins those to fixed values instead."""

    def __init__(
        self,
        datasets: dict[str, Any],
        config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(datasets=datasets, config=config, **kwargs)
        self.events: list[str] = []

    def setup(self) -> None:
        self.events.append("setup")

    def training_loop(self) -> Any:
        self.events.append("training_loop")
        return "checkpoint"


class _FirstPartyTrainer(_FakeTrainer):
    @staticmethod
    def _default_bundle_targets() -> tuple[Any, ...]:
        return (
            ExportTarget(name="onnx-model", format="onnx"),
            ExportTarget(name="native", format="ubj"),
        )

    @staticmethod
    def _default_bundle_roles() -> dict[str, str]:
        return {"inference": "onnx-model"}


class _RecordingCallback:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, *args))

    def on_setup_start(self, trainer: BaseTrainer) -> None:
        self._record("setup_start")

    def on_training_end(self, trainer: BaseTrainer, result: Any) -> None:
        self._record("training_end", result)

    def on_export_end(self, trainer: BaseTrainer, output_path: str) -> None:
        self._record("export_end", output_path)

    def on_artifacts_exported(self, trainer: BaseTrainer, output_path: str) -> None:
        self._record("artifacts_exported", output_path)

    def on_run_complete(self, trainer: BaseTrainer, summary: dict[str, Any]) -> None:
        self._record("run_complete", summary)

    def on_run_error(self, trainer: BaseTrainer, error: Exception) -> None:
        self._record("run_error", error)


def _lifecycle(
    trainer: BaseTrainer, callbacks: list[TrainerCallback] | None = None
) -> TrainingLifecycle:
    return TrainingLifecycle(trainer, CallbackDispatcher(callbacks or []))


class TestLegacyFlow:
    def test_local_runner_rejects_portable_registration(self) -> None:
        spec = AlgorithmSpec(
            name="portable",
            trainer_cls=None,
            data_loading=DataLoadingMode.CANONICAL_TRAINER,
            operations=("fit",),
        )

        with pytest.raises(JobConfigurationError, match="portable execution path"):
            run_local_trial(
                spec,
                "/tmp/out",
                effective_config={"data": {"source": {}}},
            )

    def test_run_orders_events_and_returns_summary(self) -> None:
        trainer = _ExportingTrainer()
        cb = _RecordingCallback()

        summary = _lifecycle(trainer, [cb]).run("/tmp/out")

        # setup → training_loop → export → callbacks in order
        assert trainer.events == [
            "setup",
            "training_loop",
            "export_artifacts:/tmp/out",
        ]
        assert [c[0] for c in cb.calls] == [
            "setup_start",
            "training_end",
            "artifacts_exported",
            "run_complete",
        ]
        assert summary["status"] == "succeeded"
        assert summary["training_status"] == "succeeded"
        assert summary["bundle_status"] == "not_started"

    def test_base_trainer_run_delegates_to_lifecycle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``BaseTrainer.run()`` must delegate to ``TrainingLifecycle``, not
        inline the orchestration.

        The historical version of this test only asserted the run contract
        and stayed green when the delegation was broken by the E1 inline
        regression; the spy below pins the actual delegation chain (trainer
        + callback dispatcher construction, argument forwarding, and the
        return value passing straight through).
        """
        constructed: list[tuple[BaseTrainer, CallbackDispatcher]] = []
        invoked: dict[str, Any] = {}

        class _SpyLifecycle:
            def __init__(
                self, trainer: BaseTrainer, dispatcher: CallbackDispatcher
            ) -> None:
                constructed.append((trainer, dispatcher))

            # Mirror the real TrainingLifecycle.run signature (keyword-only
            # bundle_config) so the spy guards the invocation shape too.
            def run(
                self,
                output_path: str = "",
                *,
                bundle_config: Any = None,
                legacy_export: bool = False,
            ) -> dict[str, Any]:
                invoked["output_path"] = output_path
                invoked["bundle_config"] = bundle_config
                invoked["legacy_export"] = legacy_export
                return {"status": "succeeded", "delegated": True}

        # base.py imports TrainingLifecycle inside run(), so the spy must
        # replace the attribute on the lifecycle module itself.
        monkeypatch.setattr(
            "tributo.training.lifecycle.TrainingLifecycle", _SpyLifecycle
        )

        trainer = _FakeTrainer()
        bundle_config = SimpleNamespace(targets=[1])
        summary = trainer.run("/tmp/out", bundle_config=bundle_config)

        # The return value comes straight from the lifecycle, not from
        # inline logic.
        assert summary == {"status": "succeeded", "delegated": True}
        # Exactly one lifecycle was constructed, with this trainer and a
        # CallbackDispatcher wrapping its callbacks.
        assert len(constructed) == 1
        delegate, dispatcher = constructed[0]
        assert delegate is trainer
        assert isinstance(dispatcher, CallbackDispatcher)
        assert invoked == {
            "output_path": "/tmp/out",
            "bundle_config": bundle_config,
            "legacy_export": False,
        }

    def test_run_local_trial_entry_reaches_lifecycle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The production entry ``run_local_trial`` reaches
        ``TrainingLifecycle`` through ``BaseTrainer.run()``'s thin
        delegation — the exit gate.

        The loader must never inline the orchestration again: this test
        fails if the delegation chain from the entry point to the
        lifecycle is broken.
        """
        delegated: list[tuple[str, Any]] = []

        class _SpyLifecycle:
            def __init__(self, trainer: BaseTrainer, dispatcher: Any) -> None:
                pass

            # Mirror the real TrainingLifecycle.run signature so the spy
            # guards the invocation shape through the production entry too.
            def run(
                self,
                output_path: str = "",
                *,
                bundle_config: Any = None,
                legacy_export: bool = False,
            ) -> dict[str, Any]:
                assert legacy_export is False
                delegated.append((output_path, bundle_config))
                return {"status": "succeeded"}

        monkeypatch.setattr(
            "tributo.training.lifecycle.TrainingLifecycle", _SpyLifecycle
        )

        spec = AlgorithmSpec(
            name="t1r-probe",
            trainer_cls=_EntryTrainer,
            data_loading=DataLoadingMode.CANONICAL_TRAINER,
        )
        # CANONICAL_TRAINER always requires data.source (PU rule), though
        # the trainer itself loads data — supply the minimal shape.
        summary = run_local_trial(
            spec, "/tmp/out", effective_config={"data": {"source": {}}}
        )

        assert summary["status"] == "succeeded"
        assert delegated == [("/tmp/out", None)]

    def test_local_trial_routes_validation_and_test_sources_through_gateway(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loaded: list[tuple[dict[str, Any], Path | None]] = []

        def fake_load(
            source: dict[str, Any],
            *,
            project_root_path: Path | None = None,
        ) -> object:
            loaded.append((source, project_root_path))
            return object()

        monkeypatch.setattr(
            "tributo.training.data_loader.load_ray_dataset_from_source",
            fake_load,
        )
        spec = AlgorithmSpec(
            name="gateway-data-probe",
            trainer_cls=_EntryTrainer,
            data_loading=DataLoadingMode.CANONICAL_DRIVER,
        )

        summary = run_local_trial(
            spec,
            "/tmp/out",
            effective_config={
                "data": {
                    "source": {"type": "parquet", "path": "train.parquet"},
                    "val_path": "validation.parquet",
                    "test_path": "test.parquet",
                }
            },
        )

        assert summary["status"] == "succeeded"
        assert loaded == [
            (
                {
                    "type": "parquet",
                    "path": "train.parquet",
                    "columns": None,
                    "s3": None,
                },
                None,
            ),
            ({"type": "parquet", "path": "validation.parquet"}, Path.cwd()),
            ({"type": "parquet", "path": "test.parquet"}, Path.cwd()),
        ]

    def test_no_export_override_warns_but_succeeds(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        trainer = _FakeTrainer()
        with caplog.at_level(logging.WARNING):
            summary = _lifecycle(trainer).run("/tmp/out")

        assert summary["status"] == "succeeded"
        assert summary["bundle_status"] == "not_started"
        assert "does not override export_artifacts" in caplog.text

    def test_export_results_written_to_trainer_summary_are_returned(self) -> None:
        """Subclasses write into ``self._summary``; run() must return it."""
        trainer = _SummaryWritingTrainer()
        summary = _lifecycle(trainer).run("/tmp/out")

        assert summary["metrics"] == {"accuracy": 0.9}
        assert trainer._summary is summary

    def test_ray_checkpoint_metrics_are_preserved_for_replay(self) -> None:
        trainer = _MetricsCheckpointTrainer()
        summary = _lifecycle(trainer).run("/tmp/out")

        assert summary["metrics"] == {
            "eval-logloss_history": [0.8, 0.4],
            "eval-logloss": 0.4,
        }


class TestBundleMode:
    def test_first_party_defaults_to_bundle_before_setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = _FirstPartyTrainer()
        lifecycle = _lifecycle(trainer)
        captured: list[BundleOutputConfig] = []

        def fake_export(checkpoint: Any, config: Any, summary: dict[str, Any]):
            del checkpoint
            captured.append(config)
            summary["canonical_uri"] = f"{config.bundle_uri}/bundle-1"
            return SimpleNamespace(
                status="succeeded",
                bundle_id="bundle-1",
                execution_id="exec-1",
                canonical_uri=summary["canonical_uri"],
                manifest_sha256="a" * 64,
                artifacts=(),
                node_results=(),
                hook_receipts=(),
            )

        monkeypatch.setattr(lifecycle, "_export_bundle", fake_export)

        summary = lifecycle.run(str(tmp_path / "bundles"))

        assert [target.format for target in captured[0].targets or []] == [
            "onnx",
            "ubj",
        ]
        assert captured[0].roles == {"inference": "onnx-model"}
        assert summary["bundle_status"] == "succeeded"
        assert summary["bundle_uri"].endswith("/bundle-1")

    def test_first_party_targets_none_uses_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = _FirstPartyTrainer()
        lifecycle = _lifecycle(trainer)
        captured: list[BundleOutputConfig] = []

        def fake_export(checkpoint: Any, config: Any, summary: dict[str, Any]):
            del checkpoint, summary
            captured.append(config)
            return SimpleNamespace(
                status="succeeded",
                bundle_id="bundle-1",
                execution_id="exec-1",
                canonical_uri=f"{config.bundle_uri}/bundle-1",
                manifest_sha256="a" * 64,
                artifacts=(),
                node_results=(),
                hook_receipts=(),
            )

        monkeypatch.setattr(lifecycle, "_export_bundle", fake_export)
        lifecycle.run(
            bundle_config=BundleOutputConfig(bundle_uri=str(tmp_path / "bundles"))
        )

        assert [target.name for target in captured[0].targets or []] == [
            "onnx-model",
            "native",
        ]

    def test_first_party_requires_explicit_bundle_uri_before_setup(self) -> None:
        trainer = _FirstPartyTrainer()

        with pytest.raises(ValueError, match="explicit Bundle URI") as exc_info:
            _lifecycle(trainer).run()

        assert trainer.events == []
        result = exc_info.value.training_result
        assert result.training_status == TrainingStatus.FAILED
        assert result.bundle_status == BundleStatus.NOT_STARTED

    @pytest.mark.parametrize(
        "error",
        [
            TaskCancelledError(error_message="task cancelled"),
            RayTaskError(
                "train",
                "remote traceback",
                TaskCancelledError(error_message="nested cancellation"),
            ),
        ],
    )
    def test_ray_cancellation_status_uses_real_exception_types(
        self, error: Exception
    ) -> None:
        class _ErrorTrainer(_FakeTrainer):
            def setup(self) -> None:
                raise error

        with pytest.raises(type(error)) as exc_info:
            _lifecycle(_ErrorTrainer()).run("/tmp/out")

        assert exc_info.value.training_result.training_status == (
            TrainingStatus.CANCELLED
        )

    def test_asyncio_cancellation_is_captured_and_re_raised(self) -> None:
        callback = _RecordingCallback()

        class _ErrorTrainer(_FakeTrainer):
            def setup(self) -> None:
                raise asyncio.CancelledError("cancelled")

        with pytest.raises(asyncio.CancelledError) as exc_info:
            _lifecycle(_ErrorTrainer(), [callback]).run("/tmp/out")

        assert exc_info.value.training_result.training_status == (
            TrainingStatus.CANCELLED
        )
        assert callback.calls[-1][0] == "run_error"

    def test_explicit_cancellation_cause_is_preserved(self) -> None:
        class _ErrorTrainer(_FakeTrainer):
            def setup(self) -> None:
                cancellation = TaskCancelledError(error_message="cancelled")
                raise RuntimeError("wrapped cancellation") from cancellation

        with pytest.raises(RuntimeError) as exc_info:
            _lifecycle(_ErrorTrainer()).run("/tmp/out")

        assert exc_info.value.training_result.training_status == (
            TrainingStatus.CANCELLED
        )

    def test_implicit_cancellation_context_does_not_mask_new_failure(self) -> None:
        def raise_replacement_failure() -> None:
            raise ValueError("replacement failure")

        class _ErrorTrainer(_FakeTrainer):
            def setup(self) -> None:
                try:
                    raise TaskCancelledError(error_message="cancelled")
                except TaskCancelledError:
                    raise_replacement_failure()

        with pytest.raises(ValueError) as exc_info:
            _lifecycle(_ErrorTrainer()).run("/tmp/out")

        assert exc_info.value.__context__ is not None
        assert exc_info.value.training_result.training_status == TrainingStatus.FAILED

    def test_ray_wrapper_with_non_cancellation_cause_is_failed(self) -> None:
        error = RayTaskError("train", "remote traceback", ValueError("bad input"))

        class _ErrorTrainer(_FakeTrainer):
            def setup(self) -> None:
                raise error

        with pytest.raises(RayTaskError) as exc_info:
            _lifecycle(_ErrorTrainer()).run("/tmp/out")

        assert exc_info.value.training_result.training_status == TrainingStatus.FAILED

    def test_exception_name_does_not_define_cancellation_semantics(self) -> None:
        error_type = type("TaskCancelledError", (RuntimeError,), {})

        class _ErrorTrainer(_FakeTrainer):
            def setup(self) -> None:
                raise error_type("setup stopped")

        with pytest.raises(error_type) as exc_info:
            _lifecycle(_ErrorTrainer()).run("/tmp/out")

        assert exc_info.value.training_result.training_status == TrainingStatus.FAILED

    def test_non_cancellation_base_exception_bypasses_error_mapping(self) -> None:
        callback = _RecordingCallback()

        class _InterruptTrainer(_FakeTrainer):
            def setup(self) -> None:
                raise KeyboardInterrupt("stop")

        with pytest.raises(KeyboardInterrupt) as exc_info:
            _lifecycle(_InterruptTrainer(), [callback]).run("/tmp/out")

        assert not hasattr(exc_info.value, "training_result")
        assert callback.calls == [("setup_start",)]

    def test_legacy_export_is_explicit_and_deprecated(self) -> None:
        trainer = _FirstPartyTrainer()

        with pytest.warns(DeprecationWarning, match="legacy_export"):
            summary = _lifecycle(trainer).run("/tmp/out", legacy_export=True)

        assert summary["legacy_artifact_uri"] == "/tmp/out"
        assert summary["bundle_status"] == "not_started"

    def test_bundle_route_skips_artifact_callbacks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = _FakeTrainer()
        cb = _RecordingCallback()
        lifecycle = _lifecycle(trainer, [cb])

        def fake_export_bundle(
            checkpoint: Any, config: Any, summary: dict[str, Any]
        ) -> None:
            summary.update(
                {
                    "status": "succeeded",
                    "bundle_id": "b-1",
                    "canonical_uri": "s3://bucket/bundle-b-1",
                    "execution_id": "exec-1",
                }
            )

        monkeypatch.setattr(lifecycle, "_export_bundle", fake_export_bundle)
        bundle_config = SimpleNamespace(targets=["model"])

        summary = lifecycle.run(bundle_config=bundle_config)

        assert summary["bundle_id"] == "b-1"
        assert [c[0] for c in cb.calls] == [
            "setup_start",
            "training_end",
            "run_complete",
        ]
        # Legacy artifact-export callbacks must not fire in bundle mode.
        assert "artifacts_exported" not in [c[0] for c in cb.calls]
        assert "export_end" not in [c[0] for c in cb.calls]

    def test_empty_targets_treats_as_legacy(self) -> None:
        trainer = _ExportingTrainer()
        summary = _lifecycle(trainer).run(
            "/tmp/out", bundle_config=SimpleNamespace(targets=[])
        )

        assert summary["status"] == "succeeded"
        assert summary["bundle_status"] == "not_started"
        assert "export_artifacts:/tmp/out" in trainer.events

    def test_subclass_export_bundle_override_is_dispatched(self) -> None:
        """Subclass overrides of the historical hook still win."""

        class _BundleOverrideTrainer(_FakeTrainer):
            def _export_bundle(self, checkpoint: Any, bundle_config: Any) -> None:
                self._summary.update(
                    {
                        "override_hit": True,
                        "canonical_uri": "s3://bucket/bundle-override",
                        "execution_id": "exec-override",
                    }
                )

        trainer = _BundleOverrideTrainer()
        summary = _lifecycle(trainer).run(
            bundle_config=SimpleNamespace(targets=["model"])
        )

        assert summary["override_hit"] is True

    def test_indirect_override_is_dispatched(self) -> None:
        """Overrides inherited through an intermediate base class work."""

        class _MiddleTrainer(_FakeTrainer):
            def _export_bundle(self, checkpoint: Any, bundle_config: Any) -> None:
                self._summary.update(
                    {
                        "middle_hit": True,
                        "canonical_uri": "s3://bucket/bundle-middle",
                        "execution_id": "exec-middle",
                    }
                )

        class _ChildTrainer(_MiddleTrainer):
            pass

        trainer = _ChildTrainer()
        summary = _lifecycle(trainer).run(
            bundle_config=SimpleNamespace(targets=["model"])
        )

        assert summary["middle_hit"] is True

    def test_super_call_in_override_uses_default_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """super()._export_bundle(...) inside an override still runs the
        real pipeline instead of raising."""

        class _SuperCallingTrainer(_FakeTrainer):
            def _export_bundle(self, checkpoint: Any, bundle_config: Any) -> None:
                self._summary["wrapped"] = True
                super()._export_bundle(checkpoint, bundle_config)

        trainer = _SuperCallingTrainer()
        # Default route needs the exporting stack; stub the heavy parts and
        # assert the pipeline runs end to end.
        from unittest.mock import MagicMock

        import tributo.exporting.registries
        import tributo.exporting.service
        from tributo.training import lifecycle as lifecycle_module

        class _SourceContext:
            def __enter__(self) -> str:
                return "source"

            def __exit__(self, *exc: Any) -> None:
                return None

        class _FakeProvider:
            def open_source(self, checkpoint: Any) -> _SourceContext:
                return _SourceContext()

        fake_service = MagicMock()
        fake_service.export_bundle.return_value = SimpleNamespace(
            status="succeeded",
            bundle_id="b-1",
            execution_id="exec-1",
            canonical_uri="s3://bucket/b-1",
            manifest_sha256="abc",
            hook_receipts=(),
            artifacts=[SimpleNamespace(name="model", format="onnx", tree_digest="d1")],
            node_results=[
                SimpleNamespace(node_id="n1", status="succeeded", target_name="t1")
            ],
        )
        fake_service_cls = MagicMock(return_value=fake_service)
        fake_registry = MagicMock()
        fake_registry.resolve.return_value = _FakeProvider

        monkeypatch.setattr(lifecycle_module, "_load_provider_plugins", lambda r: None)
        monkeypatch.setattr(
            tributo.exporting.registries,
            "SourceProviderRegistry",
            lambda: fake_registry,
        )
        monkeypatch.setattr(
            tributo.exporting.service, "BundleExportService", fake_service_cls
        )

        summary = _lifecycle(trainer).run(
            bundle_config=SimpleNamespace(targets=["model"])
        )

        fake_service.export_bundle.assert_called_once()
        assert summary["wrapped"] is True
        assert summary["bundle_id"] == "b-1"

    def test_first_party_source_providers_do_not_require_entry_points(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tributo.exporting.registries import SourceProviderRegistry
        from tributo.training import lifecycle as lifecycle_module

        monkeypatch.setattr(
            "tributo.plugin.discover_source_provider_plugins", lambda: []
        )
        monkeypatch.setattr(lifecycle_module, "_provider_plugins_cache", None)
        registry = SourceProviderRegistry()

        lifecycle_module._load_provider_plugins(registry)

        assert registry.list_all() == [
            "ray-dnn-v1",
            "ray-pu-v1",
            "ray-xgboost-v1",
        ]

    def test_bundle_real_routing_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provider resolution → BundleExportService → summary mapping."""
        from unittest.mock import MagicMock

        import tributo.exporting.registries
        import tributo.exporting.service
        from tributo.exporting.models import BundleOutputConfig, ExportTarget
        from tributo.training import lifecycle as lifecycle_module

        class _SourceContext:
            def __enter__(self) -> str:
                return "source"

            def __exit__(self, *exc: Any) -> None:
                return None

        class _FakeProvider:
            def open_source(self, checkpoint: Any) -> _SourceContext:
                return _SourceContext()

        fake_result = SimpleNamespace(
            status="succeeded",
            bundle_id="b-1",
            execution_id="exec-1",
            canonical_uri="s3://bucket/b-1",
            manifest_sha256="abc",
            hook_receipts=(),
            artifacts=[SimpleNamespace(name="model", format="onnx", tree_digest="d1")],
            node_results=[
                SimpleNamespace(node_id="n1", status="succeeded", target_name="t1")
            ],
        )
        fake_service = MagicMock()
        fake_service.export_bundle.return_value = fake_result
        fake_service_cls = MagicMock(return_value=fake_service)
        fake_registry = MagicMock()
        fake_registry.resolve.return_value = _FakeProvider

        monkeypatch.setattr(lifecycle_module, "_load_provider_plugins", lambda r: None)
        monkeypatch.setattr(
            tributo.exporting.registries,
            "SourceProviderRegistry",
            lambda: fake_registry,
        )
        monkeypatch.setattr(
            tributo.exporting.service, "BundleExportService", fake_service_cls
        )

        trainer = _FakeTrainer()
        cb = _RecordingCallback()
        monkeypatch.setenv("TRIBUTO_RUN_ID", "run-42")
        monkeypatch.setenv("TRIBUTO_ATTEMPT_ID", "attempt-2")
        summary = _lifecycle(trainer, [cb]).run(
            bundle_config=BundleOutputConfig(
                bundle_uri="s3://bucket/bundle",
                targets=[ExportTarget(name="model", format="onnx")],
            )
        )

        fake_registry.resolve.assert_called_with("xgboost")
        fake_service.export_bundle.assert_called_once()
        assert summary["bundle_id"] == "b-1"
        call = fake_service.export_bundle.call_args.kwargs
        assert call["config"].run_id == "run-42"
        assert call["attempt_id"] == "attempt-2"
        assert summary["artifacts"][0]["name"] == "model"
        assert summary["node_results"][0]["node_id"] == "n1"
        assert summary["training_status"] == "succeeded"
        assert summary["bundle_status"] == "succeeded"
        assert summary["hook_status"] == "not_configured"
        assert summary["execution_id"] == "exec-1"
        # Artifact-export callbacks stay silent in bundle mode.
        assert [c[0] for c in cb.calls] == [
            "setup_start",
            "training_end",
            "run_complete",
        ]

    def test_explicit_algorithm_run_identity_overrides_parent_ray_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tributo.exporting.models import BundleOutputConfig
        from tributo.training import lifecycle as lifecycle_module

        monkeypatch.setenv("TRIBUTO_RUN_ID", "parent-job")
        bound, _ = lifecycle_module._bind_job_identity(
            BundleOutputConfig(bundle_uri="/tmp/algorithm-bundle"),
            run_id="algorithm-run",
        )

        assert bound.request_id == "algorithm-run"
        assert bound.run_id == "algorithm-run"


class TestFailurePaths:
    def test_setup_callback_failure_fires_on_run_error(self) -> None:
        class _RequiredSetupCallback(_RecordingCallback):
            failure_policy = "required"

            def on_setup_start(self, trainer: BaseTrainer) -> None:
                self._record("setup_start")
                raise RuntimeError("callback setup boom")

        callback = _RequiredSetupCallback()
        with pytest.raises(RuntimeError, match="callback setup boom"):
            _lifecycle(_FakeTrainer(), [callback]).run()
        assert [call[0] for call in callback.calls] == ["setup_start", "run_error"]

    def test_setup_failure_fires_on_run_error_and_preserves_error(self) -> None:
        trainer = _FailingSetupTrainer()
        cb = _RecordingCallback()

        with pytest.raises(RuntimeError, match="setup boom"):
            _lifecycle(trainer, [cb]).run()

        assert [c[0] for c in cb.calls] == ["setup_start", "run_error"]
        assert isinstance(cb.calls[1][1], RuntimeError)

    def test_callback_error_never_replaces_original(self) -> None:
        class _RaisingErrorCallback(_RecordingCallback):
            def on_run_error(self, trainer: BaseTrainer, error: Exception) -> None:
                raise ValueError("callback blew up")

        with pytest.raises(RuntimeError, match="setup boom") as exc_info:
            _lifecycle(_FailingSetupTrainer(), [_RaisingErrorCallback()]).run()

        assert any("callback blew up" in note for note in exc_info.value.__notes__)

    def test_required_bundle_failure_carries_training_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = _FirstPartyTrainer()
        lifecycle = _lifecycle(trainer)
        execution = SimpleNamespace(execution_id="exec-failed")

        def fail_export(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise BundleExportError("required artifact failed", execution)

        monkeypatch.setattr(lifecycle, "_export_bundle", fail_export)
        with pytest.raises(BundleExportError) as exc_info:
            lifecycle.run(str(tmp_path / "bundles"))

        result = exc_info.value.training_result
        assert result.training_status == TrainingStatus.SUCCEEDED
        assert result.bundle_status == BundleStatus.FAILED
        assert result.hook_status == TrainingHookStatus.NOT_CONFIGURED
        assert result.execution_id == "exec-failed"

    def test_required_hook_failure_keeps_committed_bundle_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = _FirstPartyTrainer()
        lifecycle = _lifecycle(trainer)
        receipt = SimpleNamespace(status=HookStatus.TERMINAL_FAILED)
        bundle = SimpleNamespace(
            status="succeeded",
            canonical_uri=f"{tmp_path}/bundles/bundle-1",
            execution_id="exec-1",
            hook_receipts=(receipt,),
        )

        def fail_hook(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise PostPublishCallbackError(
                "required hook failed", bundle_result=bundle, receipts=(receipt,)
            )

        monkeypatch.setattr(lifecycle, "_export_bundle", fail_hook)
        with pytest.raises(PostPublishCallbackError) as exc_info:
            lifecycle.run(str(tmp_path / "bundles"))

        result = exc_info.value.training_result
        assert result.training_status == TrainingStatus.SUCCEEDED
        assert result.bundle_status == BundleStatus.SUCCEEDED
        assert result.hook_status == TrainingHookStatus.FAILED
        assert result.bundle_uri == bundle.canonical_uri

    def test_run_complete_failure_keeps_already_committed_bundle_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _RequiredCompletionCallback(_RecordingCallback):
            failure_policy = "required"

            def on_run_complete(
                self, trainer: BaseTrainer, summary: dict[str, Any]
            ) -> None:
                del trainer, summary
                raise RuntimeError("completion callback failed")

        bundle = SimpleNamespace(
            status="succeeded",
            canonical_uri=f"{tmp_path}/bundles/bundle-1",
            execution_id="exec-1",
            hook_receipts=(),
        )
        lifecycle = _lifecycle(
            _FirstPartyTrainer(),
            [_RequiredCompletionCallback()],
        )
        monkeypatch.setattr(lifecycle, "_export_bundle", lambda *args: bundle)

        with pytest.raises(RuntimeError, match="completion callback failed") as exc:
            lifecycle.run(str(tmp_path / "bundles"))

        result = exc.value.training_result
        assert result.bundle_status == BundleStatus.SUCCEEDED
        assert result.bundle_uri == bundle.canonical_uri
        assert result.execution_id == "exec-1"

    def test_incomplete_bundle_error_metadata_preserves_original_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lifecycle = _lifecycle(_FirstPartyTrainer())
        incomplete_bundle = SimpleNamespace(
            status="succeeded",
            canonical_uri=f"{tmp_path}/bundles/bundle-1",
            execution_id=None,
            hook_receipts=(),
        )

        def fail_after_commit(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise PostPublishCallbackError(
                "callback failed",
                bundle_result=incomplete_bundle,
            )

        monkeypatch.setattr(lifecycle, "_export_bundle", fail_after_commit)

        with pytest.raises(PostPublishCallbackError, match="callback failed") as exc:
            lifecycle.run(str(tmp_path / "bundles"))

        assert exc.value.training_result.bundle_status == BundleStatus.FAILED
        assert any(
            "omitted canonical_uri or execution_id" in note
            for note in exc.value.__notes__
        )


class TestTrainingResultContract:
    def test_hook_aggregation_is_closed_and_deterministic(self) -> None:
        def receipt(status: HookStatus) -> SimpleNamespace:
            return SimpleNamespace(status=status)

        assert aggregate_hook_status([]) == TrainingHookStatus.NOT_CONFIGURED
        assert aggregate_hook_status([receipt(HookStatus.ACCEPTED)]) == "pending"
        assert aggregate_hook_status([receipt(HookStatus.SKIPPED)]) == "skipped"
        assert (
            aggregate_hook_status(
                [receipt(HookStatus.SUCCEEDED), receipt(HookStatus.SKIPPED)]
            )
            == "succeeded"
        )
        assert (
            aggregate_hook_status(
                [receipt(HookStatus.SUCCEEDED), receipt(HookStatus.TERMINAL_FAILED)]
            )
            == "partial"
        )
        assert (
            aggregate_hook_status(
                [
                    receipt(HookStatus.RETRYABLE_FAILED),
                    receipt(HookStatus.TERMINAL_FAILED),
                ]
            )
            == "failed"
        )

    def test_illegal_state_combinations_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="failed or cancelled"):
            TrainingResult(
                training_status=TrainingStatus.FAILED,
                bundle_status=BundleStatus.SUCCEEDED,
                hook_status=TrainingHookStatus.NOT_CONFIGURED,
                bundle_uri="s3://bucket/bundle-1",
                execution_id="exec-1",
            )
        with pytest.raises(ValidationError, match="hooks cannot run"):
            TrainingResult(
                training_status=TrainingStatus.SUCCEEDED,
                bundle_status=BundleStatus.FAILED,
                hook_status=TrainingHookStatus.FAILED,
            )


class TestTrainerConstructorContract:
    """PU/DNN must accept callbacks like XGBoost (unified contract, §6.5)."""

    def test_pu_trainer_accepts_callbacks(self) -> None:
        from tributo.training.pu_trainer import PUTrainerImpl

        cb = _RecordingCallback()
        trainer = PUTrainerImpl(
            datasets={},
            config={"features": [], "pu": {"class_prior": 0.2}},
            callbacks=[cb],
        )
        assert trainer._callbacks == [cb]

    def test_dnn_trainer_accepts_callbacks(self) -> None:
        from tributo.training.dnn_trainer import DNNTrainerImpl

        cb = _RecordingCallback()
        trainer = DNNTrainerImpl(datasets={}, config={"features": []}, callbacks=[cb])
        assert trainer._callbacks == [cb]

    def test_xgboost_trainer_accepts_callbacks(self) -> None:
        from tributo.training.xgboost_trainer import XGBoostTrainerImpl

        cb = _RecordingCallback()
        trainer = XGBoostTrainerImpl(datasets={}, config={}, callbacks=[cb])
        assert trainer._callbacks == [cb]

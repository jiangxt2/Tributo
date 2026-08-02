"""Unit tests for training/lifecycle.py — TrainingLifecycle."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from tributo.training.base import BaseTrainer, TrainerCallback
from tributo.training.callbacks import CallbackDispatcher
from tributo.training.lifecycle import TrainingLifecycle


class _FakeTrainer(BaseTrainer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(datasets={}, config={}, **kwargs)
        self.events: list[str] = []

    def setup(self) -> None:
        self.events.append("setup")

    def training_loop(self) -> Any:
        self.events.append("training_loop")
        return "checkpoint"


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
        assert summary == {"status": "succeeded"}

    def test_base_trainer_run_delegates_to_lifecycle(self) -> None:
        trainer = _FakeTrainer()
        summary = trainer.run()

        assert summary == {"status": "succeeded"}
        assert trainer.events == ["setup", "training_loop"]

    def test_no_export_override_warns_but_succeeds(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        trainer = _FakeTrainer()
        with caplog.at_level(logging.WARNING):
            summary = _lifecycle(trainer).run("/tmp/out")

        assert summary == {"status": "succeeded"}
        assert "does not override export_artifacts" in caplog.text

    def test_export_results_written_to_trainer_summary_are_returned(self) -> None:
        """Subclasses write into ``self._summary``; run() must return it."""
        trainer = _SummaryWritingTrainer()
        summary = _lifecycle(trainer).run("/tmp/out")

        assert summary["metrics"] == {"accuracy": 0.9}
        assert trainer._summary is summary


class TestBundleMode:
    def test_bundle_route_skips_artifact_callbacks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = _FakeTrainer()
        cb = _RecordingCallback()
        lifecycle = _lifecycle(trainer, [cb])

        def fake_export_bundle(
            checkpoint: Any, config: Any, summary: dict[str, Any]
        ) -> None:
            summary.update({"status": "succeeded", "bundle_id": "b-1"})

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

        assert summary == {"status": "succeeded"}
        assert "export_artifacts:/tmp/out" in trainer.events

    def test_subclass_export_bundle_override_is_dispatched(self) -> None:
        """Subclass overrides of the historical hook still win."""

        class _BundleOverrideTrainer(_FakeTrainer):
            def _export_bundle(self, checkpoint: Any, bundle_config: Any) -> None:
                self._summary["override_hit"] = True

        trainer = _BundleOverrideTrainer()
        summary = _lifecycle(trainer).run(
            bundle_config=SimpleNamespace(targets=["model"])
        )

        assert summary["override_hit"] is True

    def test_indirect_override_is_dispatched(self) -> None:
        """Overrides inherited through an intermediate base class work."""

        class _MiddleTrainer(_FakeTrainer):
            def _export_bundle(self, checkpoint: Any, bundle_config: Any) -> None:
                self._summary["middle_hit"] = True

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
            canonical_uri="s3://bucket/b-1",
            manifest_sha256="abc",
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

    def test_bundle_real_routing_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provider resolution → BundleExportService → summary mapping."""
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

        fake_result = SimpleNamespace(
            status="succeeded",
            bundle_id="b-1",
            canonical_uri="s3://bucket/b-1",
            manifest_sha256="abc",
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
        summary = _lifecycle(trainer, [cb]).run(
            bundle_config=SimpleNamespace(targets=["model"])
        )

        fake_registry.resolve.assert_called_with("xgboost")
        fake_service.export_bundle.assert_called_once()
        assert summary["bundle_id"] == "b-1"
        assert summary["artifacts"][0]["name"] == "model"
        assert summary["node_results"][0]["node_id"] == "n1"
        # Artifact-export callbacks stay silent in bundle mode.
        assert [c[0] for c in cb.calls] == [
            "setup_start",
            "training_end",
            "run_complete",
        ]


class TestFailurePaths:
    def test_setup_failure_fires_on_run_error_and_preserves_error(self) -> None:
        trainer = _FailingSetupTrainer()
        cb = _RecordingCallback()

        with pytest.raises(RuntimeError, match="setup boom"):
            _lifecycle(trainer, [cb]).run()

        assert [c[0] for c in cb.calls] == ["setup_start", "run_error"]
        assert isinstance(cb.calls[1][1], RuntimeError)

    def test_callback_error_chained_from_original(self) -> None:
        class _RaisingErrorCallback(_RecordingCallback):
            def on_run_error(self, trainer: BaseTrainer, error: Exception) -> None:
                raise ValueError("callback blew up")

        with pytest.raises(ValueError, match="callback blew up") as exc_info:
            _lifecycle(_FailingSetupTrainer(), [_RaisingErrorCallback()]).run()

        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestTrainerConstructorContract:
    """PU/DNN must accept callbacks like XGBoost (unified contract, §6.5)."""

    def test_pu_trainer_accepts_callbacks(self) -> None:
        from tributo.training.pu_trainer import PUTrainerImpl

        cb = _RecordingCallback()
        trainer = PUTrainerImpl(datasets={}, config={"features": []}, callbacks=[cb])
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

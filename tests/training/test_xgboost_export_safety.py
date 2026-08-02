"""Export-safety tests for XGBoostTrainerImpl (ADR-001 required artifacts).

Contract: once an ONNX output path is configured, the model is a required
artifact — a failed export must fail the run instead of silently
completing without the model.  When no ONNX output path is configured,
export is skipped and the run succeeds.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tributo.training.xgboost_trainer import XGBoostTrainerImpl


def _make_result() -> MagicMock:
    result = MagicMock()
    result.metrics = {"n_features": 2}
    result.checkpoint = MagicMock()
    return result


def _make_trainer(
    monkeypatch: pytest.MonkeyPatch, onnx_path: str | None = "model.onnx"
) -> tuple[XGBoostTrainerImpl, MagicMock]:
    """Build a trainer whose setup/training_loop are stubbed out.

    ``training_loop`` returns the shared fake result so tests can mutate
    it (e.g. drop the checkpoint) before calling ``run()``.
    """
    ds = MagicMock()
    ds.schema.return_value.names = ["a", "b", "label"]
    trainer = XGBoostTrainerImpl(
        datasets={"train": ds},
        config={
            "data": {"type": "csv", "path": "data/train.csv", "label_col": "label"},
            "output": {"onnx_path": onnx_path},
            "training": {"val_size": 0, "test_size": 0, "seed": 42},
        },
    )
    result = _make_result()
    monkeypatch.setattr(trainer, "setup", lambda: None)
    monkeypatch.setattr(trainer, "training_loop", lambda: result)
    # Skip real metrics computation — not part of the export-safety contract.
    monkeypatch.setattr(
        "tributo.training.xgboost_evaluator.compute_metrics_summary",
        lambda metrics: metrics,
    )
    return trainer, result


class TestRequiredOnnxFailure:
    """ADR-001: configured ONNX is required — export failure fails the run."""

    def test_export_error_fails_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated ONNX export failure")

        trainer, _ = _make_trainer(monkeypatch)
        monkeypatch.setattr("tributo.training.xgboost_exporter.export_onnx", _boom)

        with pytest.raises(RuntimeError, match="simulated ONNX export failure"):
            trainer.run(output_path="model.onnx")

    def test_missing_checkpoint_fails_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer, result = _make_trainer(monkeypatch)
        result.checkpoint = None

        with pytest.raises(RuntimeError, match="no checkpoint"):
            trainer.run(output_path="model.onnx")

    def test_no_onnx_path_is_not_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer, _ = _make_trainer(monkeypatch, onnx_path=None)

        summary = trainer.run()

        assert summary["status"] == "succeeded"
        assert summary["onnx_path"] is None

"""Worker-loaded fixtures for the bounded legacy Trainer adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from tributo.algorithms.input.fake import FakeTabularPayload
from tributo.algorithms.input.tabular import InMemoryTabularInputView
from tributo.algorithms.spi import PreparedInput, WorkerInputPayload
from tributo.training.base import BaseTrainer


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: float = 1.0
    fail: bool = False
    nonfinite: bool = False
    require_ray_dataset: bool = False
    resume: dict[str, int] | None = None


class ProbeLegacyTrainer(BaseTrainer):
    """Return portable metrics while proving setup precedes training."""

    def __init__(
        self,
        datasets: dict[str, Any],
        config: dict[str, Any],
        run_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(datasets, config, run_config, **kwargs)
        self._config = _Config.model_validate(config)
        self._setup_complete = False

    def setup(self) -> None:
        if "train" not in self.datasets:
            raise ValueError("missing train input")
        if self._config.require_ray_dataset:
            import ray.data

            dataset = self.datasets["train"]
            if not isinstance(dataset, ray.data.Dataset) or dataset.count() != 8:
                raise ValueError(
                    "legacy adapter did not provide the native Ray Dataset"
                )
        self._setup_complete = True

    def training_loop(self) -> object:
        if not self._setup_complete:
            raise RuntimeError("setup did not run")
        if self._config.fail:
            raise RuntimeError("probe failure secret=do-not-leak")

        metrics: dict[str, Any] = {
            "loss": float("inf") if self._config.nonfinite else self._config.metric,
            "nested": {"epochs": 1},
        }
        if self._config.resume is not None:
            metrics["resume_attempt"] = self._config.resume["attempt"]

        class Result:
            checkpoint = object()

            def __init__(self, result_metrics: dict[str, Any]) -> None:
                self.metrics = result_metrics

        return Result(metrics)


def prepare_native_input(payload: WorkerInputPayload) -> PreparedInput:
    """Pass the fixture value through without framework conversion."""
    return PreparedInput({payload.input_name: payload.value})


def prepare_materialized_input(payload: WorkerInputPayload) -> PreparedInput:
    """Expose the fake job payload through the production materialized view contract."""
    if not isinstance(payload.value, FakeTabularPayload):
        raise TypeError("legacy materialized fixture requires FakeTabularPayload")
    return PreparedInput(
        {
            payload.input_name: InMemoryTabularInputView(
                _columns=payload.value.columns_by_name,
                feature_names=payload.binding.feature_names,
                label_name=payload.binding.label_name,
            )
        }
    )


__all__ = [
    "ProbeLegacyTrainer",
    "prepare_materialized_input",
    "prepare_native_input",
]

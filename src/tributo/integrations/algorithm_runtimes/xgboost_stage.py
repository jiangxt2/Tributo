"""Reusable bounded Ray Train XGBoost stage execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import ray

from tributo.algorithms.api import AlgorithmExecutionError, WorkerResources
from tributo.integrations.sources.ray_xgboost import RayXGBoostSourceProvider
from tributo.training.xgboost_trainer import _build_trainer


class XGBoostEvidenceCollectorState:
    """Collect one bounded evidence record per XGBoost worker rank."""

    def __init__(self) -> None:
        self._records: dict[int, dict[str, object]] = {}

    def record(self, value: dict[str, object]) -> None:
        rank_value = value["rank"]
        if not isinstance(rank_value, int) or isinstance(rank_value, bool):
            raise TypeError("XGBoost evidence rank must be an integer")
        rank = rank_value
        if rank in self._records:
            raise ValueError(f"duplicate XGBoost worker evidence rank: {rank}")
        self._records[rank] = dict(value)

    def snapshot(self) -> list[dict[str, object]]:
        return [self._records[rank] for rank in sorted(self._records)]


@dataclass(frozen=True)
class FittedXGBoostStage:
    """Bounded result of one native distributed XGBoost fit."""

    name: str
    result: object
    checkpoint: object
    booster_raw: bytes
    evidence: dict[str, object]


class XGBoostStageRunner:
    """Run named XGBoost stages through Tributo's existing Ray Train kernel."""

    def __init__(
        self,
        *,
        worker_count: int,
        resources_per_worker: WorkerResources,
        storage_path: str | None,
        run_identity: str,
        input_binding_digest: str,
    ) -> None:
        self.worker_count = worker_count
        self.resources_per_worker = resources_per_worker
        self.storage_path = storage_path
        self.run_identity = run_identity
        self.input_binding_digest = input_binding_digest

    def fit(
        self,
        name: str,
        dataset: Any,
        *,
        label_col: str,
        xgb_params: dict[str, Any],
        num_rounds: int,
        validation: Any | None = None,
        expected_training_rows: int | None = None,
    ) -> FittedXGBoostStage:
        """Fit one stage and return its real worker/state evidence."""
        prepared_dataset = dataset.materialize()
        prepared_validation = (
            validation.materialize() if validation is not None else None
        )
        rows = (
            int(prepared_dataset.count())
            if expected_training_rows is None
            else expected_training_rows
        )
        if rows < self.worker_count:
            raise AlgorithmExecutionError(
                f"XGBoost stage {name!r} has {rows} rows for "
                f"{self.worker_count} workers"
            )
        collector_type = ray.remote(XGBoostEvidenceCollectorState).options(num_cpus=0)
        collector: Any = collector_type.remote()
        binding_digest = hashlib.sha256(
            f"{self.input_binding_digest}:{name}".encode("utf-8")
        ).hexdigest()
        try:
            trainer = _build_trainer(
                ray_dataset=prepared_dataset,
                val_dataset=prepared_validation,
                train_config={
                    "label_col": label_col,
                    "xgb_params": dict(xgb_params),
                    "num_rounds": num_rounds,
                    "resume": {},
                    "_tributo_evidence_actor": collector,
                    "_tributo_input_binding_digest": binding_digest,
                },
                num_workers=self.worker_count,
                use_gpu=self.resources_per_worker.num_gpus > 0,
                resources_per_worker={
                    "CPU": self.resources_per_worker.num_cpus,
                    "GPU": self.resources_per_worker.num_gpus,
                    **dict(self.resources_per_worker.custom),
                },
                storage_path=self.storage_path,
                max_failures=0,
                run_name=f"tributo-x-learner-{self.run_identity}-{name}",
                dataset_config=self._exact_coverage_data_config(),
            )
            result = trainer.fit()
            checkpoint = getattr(result, "checkpoint", None)
            if checkpoint is None:
                raise AlgorithmExecutionError(
                    f"XGBoost stage {name!r} has no checkpoint"
                )
            workers = ray.get(collector.snapshot.remote())
            if len(workers) != self.worker_count:
                raise AlgorithmExecutionError(
                    f"XGBoost stage {name!r} did not report every worker"
                )
            digests = {str(item["model_state_digest"]) for item in workers}
            if len(digests) != 1:
                raise AlgorithmExecutionError(
                    f"XGBoost stage {name!r} workers produced divergent models"
                )
            with RayXGBoostSourceProvider().open_source(result) as source:
                booster_raw = bytes(source.model_object.save_raw(raw_format="ubj"))
            evidence: dict[str, object] = {
                "workers": workers,
                "state": {
                    "coordination": "framework_native",
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": next(iter(digests)),
                    "details": {"framework": "xgboost", "stage": name},
                },
                "input_complete": True,
                "expected_training_rows": rows,
            }
            return FittedXGBoostStage(
                name=name,
                result=result,
                checkpoint=checkpoint,
                booster_raw=booster_raw,
                evidence=evidence,
            )
        finally:
            ray.kill(collector, no_restart=True)

    @staticmethod
    def _exact_coverage_data_config() -> object:
        """Reuse the audited Ray DataConfig that retains uneven final rows."""
        from tributo.integrations.algorithm_runtimes.ray_data_config import (
            ExactCoverageDataConfig,
        )

        return ExactCoverageDataConfig()

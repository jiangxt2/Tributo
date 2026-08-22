"""Ray Train collective Runtime Adapter for iterative distributed algorithms."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, cast

import ray

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    CollectivePolicy,
    DistributionStrategy,
    ResolvedAlgorithmPlan,
    ResultPolicy,
    WorkerExecutionEvidence,
    WorkerExecutionResult,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.algorithms.core.worker import (
    _actual_environment_versions,
    _load_reference,
    _validate_module_digest,
)
from tributo.algorithms.spi import (
    CollectiveAlgorithm,
    PreparedInput,
    RuntimeExecutionEnvelope,
)
from tributo.integrations.algorithm_runtimes.portable_metrics import (
    portable_fit_only_metrics,
)
from tributo.util.annotations import DeveloperAPI

RAY_TRAIN_COLLECTIVE_RUNTIME_ID = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
    DistributionStrategy.RAY_TRAIN_COLLECTIVE
].runtime_id


def _ray_run_name(algorithm: str, run_id: str) -> str:
    """Bind Ray storage identity to the existing Tributo logical run."""
    suffix = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    return f"tributo-{algorithm}-{suffix}"


def _collective_execution_result(
    *,
    result: object,
    metrics: Mapping[str, Any],
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    """Apply the declared result policy after collective semantics succeed."""
    distribution = plan.distribution_spec
    if distribution is not None and distribution.result_policy is ResultPolicy.FIT_ONLY:
        # FIT_ONLY deliberately discards the trained model/checkpoint; receipt
        # metadata carries the execution evidence for this non-public result.
        return AlgorithmExecutionResult(
            status="succeeded",
            metrics=portable_fit_only_metrics(metrics),
        )
    exporter_ref = plan.implementation.exporter_ref
    if exporter_ref is None:
        raise AlgorithmConfigurationError(
            "collective bundle_required fit requires an explicit exporter"
        )
    exporter = _load_reference(exporter_ref)
    if not callable(exporter):
        raise AlgorithmConfigurationError(
            "collective exporter reference is not callable"
        )
    execution = exporter(result=result, plan=plan, run_id=run_id)
    if not isinstance(execution, AlgorithmExecutionResult):
        raise AlgorithmExecutionError(
            "collective exporter must return AlgorithmExecutionResult"
        )
    if not execution.outputs.get("bundle_uri"):
        raise AlgorithmExecutionError(
            "collective fit completed without the required Bundle publication"
        )
    return execution


def _load_algorithm(envelope: RuntimeExecutionEnvelope) -> CollectiveAlgorithm:
    plan = envelope.plan
    spec = plan.distribution_spec
    if spec is None or spec.strategy is not DistributionStrategy.RAY_TRAIN_COLLECTIVE:
        raise AlgorithmConfigurationError(
            "collective runtime requires a ray_train_collective DistributionSpec"
        )
    _validate_module_digest(
        plan.implementation.implementation_ref,
        plan.implementation.code_digest,
    )
    implementation = _load_reference(plan.implementation.implementation_ref)
    factory = _load_reference(plan.implementation.executable_factory_ref)
    if not callable(factory):
        raise AlgorithmConfigurationError(
            "collective executable factory reference is not callable"
        )
    algorithm = factory(plan=plan, implementation=implementation, artifacts=())
    if not isinstance(algorithm, CollectiveAlgorithm):
        raise AlgorithmConfigurationError(
            "collective executable factory must return CollectiveAlgorithm"
        )
    return algorithm


def _prepare_datasets(envelope: RuntimeExecutionEnvelope) -> PreparedInput:
    adapter = _load_reference(envelope.plan.runtime.worker_input_adapter_ref)
    if not callable(adapter):
        raise AlgorithmConfigurationError(
            "collective input adapter reference is not callable"
        )
    prepared = adapter(envelope.input_payloads[0])
    if not isinstance(prepared, PreparedInput):
        raise AlgorithmConfigurationError(
            "collective input adapter must return PreparedInput"
        )
    if not prepared.views:
        prepared.close()
        raise AlgorithmConfigurationError(
            "collective input adapter did not expose any Ray Dataset"
        )
    return prepared


def _worker_evidence(
    metrics: Mapping[str, Any],
    *,
    worker_count: int,
    num_cpus: float,
    num_gpus: float,
    custom_resources: Mapping[str, float],
    expected_input_rows: Mapping[str, int],
) -> tuple[list[dict[str, object]], str]:
    raw_workers = metrics.get("execution_workers")
    global_digest = metrics.get("model_state_digest")
    if not isinstance(raw_workers, (list, tuple)) or len(raw_workers) != worker_count:
        raise AlgorithmExecutionError(
            "collective training did not report every requested worker"
        )
    if not isinstance(global_digest, str) or len(global_digest) != 64:
        raise AlgorithmExecutionError(
            "collective training did not report a consolidated model digest"
        )
    workers: list[dict[str, object]] = []
    ranks: set[int] = set()
    shards: set[str] = set()
    digests: set[str] = set()
    input_rows_by_worker: list[dict[str, int]] = []
    for value in raw_workers:
        if not isinstance(value, Mapping):
            raise AlgorithmExecutionError(
                "collective worker evidence must contain mappings"
            )
        try:
            evidence = WorkerExecutionEvidence.from_dict(value)
            rank = evidence.rank
            worker = evidence.to_dict()
        except (AlgorithmConfigurationError, KeyError, TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                "collective worker evidence is malformed"
            ) from exc
        if worker["world_size"] != worker_count:
            raise AlgorithmExecutionError(
                "collective worker evidence has an inconsistent world size"
            )
        if (
            not evidence.input_rows
            or sum(evidence.input_rows.values()) <= 0
            or not evidence.rows_processed
            or not evidence.batch_count
            or not evidence.collective_steps
        ):
            raise AlgorithmExecutionError(
                "collective worker evidence does not prove consumed input batches"
            )
        resources = worker["resources"]
        if not isinstance(resources, Mapping) or (
            float(resources.get("num_cpus", 0.0)) < num_cpus
            or float(resources.get("num_gpus", 0.0)) < num_gpus
        ):
            raise AlgorithmExecutionError(
                "collective worker evidence does not satisfy requested resources"
            )
        reported_custom = resources.get("custom", {})
        if not isinstance(reported_custom, Mapping) or any(
            float(reported_custom.get(name, 0.0)) < amount
            for name, amount in custom_resources.items()
        ):
            raise AlgorithmExecutionError(
                "collective worker evidence does not satisfy requested custom resources"
            )
        ranks.add(rank)
        shards.add(str(worker["shard_id"]))
        digests.add(str(worker["model_state_digest"]))
        workers.append(worker)
        input_rows_by_worker.append(dict(evidence.input_rows))
    if ranks != set(range(worker_count)) or len(shards) != worker_count:
        raise AlgorithmExecutionError(
            "collective training did not prove unique ranks and input shards"
        )
    if digests != {global_digest}:
        raise AlgorithmExecutionError(
            "collective workers did not report one synchronized model state"
        )
    observed_input_rows = {
        name: sum(input_rows.get(name, 0) for input_rows in input_rows_by_worker)
        for name in expected_input_rows
    }
    if observed_input_rows != dict(expected_input_rows) or any(
        set(input_rows) != set(expected_input_rows)
        for input_rows in input_rows_by_worker
    ):
        raise AlgorithmExecutionError(
            "collective worker evidence does not prove complete input coverage"
        )
    return sorted(
        workers,
        key=lambda item: cast(int, item["rank"]),
    ), global_digest


@DeveloperAPI
class RayTrainCollectiveRuntime:
    """Execute one constrained user or first-party collective algorithm."""

    @property
    def runtime_id(self) -> str:
        """Return the formal runtime identity."""
        return RAY_TRAIN_COLLECTIVE_RUNTIME_ID

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        """Build a Ray Train worker group and require synchronization evidence."""
        if not ray.is_initialized():
            raise AlgorithmExecutionError(
                "Ray must be initialized before collective algorithm execution"
            )
        if envelope.cancelled:
            raise AlgorithmExecutionError("collective execution was cancelled")
        algorithm = _load_algorithm(envelope)
        prepared = _prepare_datasets(envelope)
        try:
            from ray.train import FailureConfig, RunConfig, ScalingConfig
            from ray.train.torch import TorchConfig, TorchTrainer

            from tributo.integrations.algorithm_runtimes.ray_data_config import (
                ExactCoverageDataConfig,
            )
            from tributo.training.checkpoint import (
                ResumeConfig,
                checkpoint_config,
            )

            plan = envelope.plan
            if plan.distribution_spec is None or not isinstance(
                plan.distribution_spec.policy, CollectivePolicy
            ):
                raise AlgorithmConfigurationError(
                    "collective plan lost its CollectivePolicy"
                )
            policy = plan.distribution_spec.policy
            resource_map = {"CPU": plan.runtime.num_cpus}
            if plan.runtime.num_gpus:
                resource_map["GPU"] = plan.runtime.num_gpus
            resource_map.update(plan.runtime.custom_resources)
            ray_config = plan.algorithm_config.get("ray", {})
            if not isinstance(ray_config, Mapping):
                raise AlgorithmConfigurationError("ray config must be a mapping")
            storage_path = ray_config.get("storage_path")
            max_failures = ray_config.get("max_failures", 0)
            if (
                not isinstance(max_failures, int)
                or isinstance(max_failures, bool)
                or max_failures < -1
            ):
                raise AlgorithmConfigurationError(
                    "ray.max_failures must be -1 or a non-negative integer"
                )
            resume_config = ResumeConfig.model_validate(ray_config.get("resume", {}))
            datasets = algorithm.bind_datasets(prepared.views)
            if not isinstance(datasets, Mapping) or not datasets:
                raise AlgorithmConfigurationError(
                    "collective bind_datasets must return named Ray Datasets"
                )
            expected_input_rows: dict[str, int] = {}
            for name, dataset in datasets.items():
                count = getattr(dataset, "count", None)
                if not isinstance(name, str) or not name or not callable(count):
                    raise AlgorithmConfigurationError(
                        "collective datasets must be named Ray Datasets with count()"
                    )
                expected_input_rows[name] = int(count())
            if any(count < 1 for count in expected_input_rows.values()):
                raise AlgorithmConfigurationError(
                    "collective datasets must be non-empty before worker sharding"
                )
            train_loop_config = dict(plan.algorithm_config)
            train_loop_config["_tributo_expected_input_rows"] = expected_input_rows
            train_loop_config["_tributo_metric_reducers"] = {
                name: reducer.value for name, reducer in policy.metric_reducers.items()
            }
            trainer = TorchTrainer(
                train_loop_per_worker=algorithm.train_loop_per_worker,
                train_loop_config=train_loop_config,
                scaling_config=ScalingConfig(
                    num_workers=plan.runtime.worker_count,
                    use_gpu=plan.runtime.num_gpus > 0,
                    resources_per_worker=resource_map,
                    placement_strategy="SPREAD",
                ),
                datasets=dict(cast(Mapping[str, Any], datasets)),
                run_config=RunConfig(
                    name=_ray_run_name(plan.resolution.algorithm, envelope.run_id),
                    storage_path=storage_path,
                    failure_config=FailureConfig(max_failures=max_failures),
                    checkpoint_config=checkpoint_config(resume_config),
                ),
                torch_config=TorchConfig(
                    backend=None if policy.backend == "auto" else policy.backend
                ),
                dataset_config=ExactCoverageDataConfig(),
            )
            result = trainer.fit()
            metrics = result.metrics or {}
            workers, global_digest = _worker_evidence(
                metrics,
                worker_count=plan.runtime.worker_count,
                num_cpus=plan.runtime.num_cpus,
                num_gpus=plan.runtime.num_gpus,
                custom_resources=plan.runtime.custom_resources,
                expected_input_rows=expected_input_rows,
            )
            observed_backend = metrics.get("collective_backend")
            if (
                plan.runtime.worker_count >= 2
                and policy.backend != "auto"
                and observed_backend != policy.backend
            ):
                raise AlgorithmExecutionError(
                    "collective training did not use the declared synchronization "
                    "backend"
                )
            if metrics.get("checkpoint_owner_rank") != policy.checkpoint_owner_rank:
                raise AlgorithmExecutionError(
                    "collective training did not report the declared checkpoint owner"
                )
            expected_reducers = {
                name: reducer.value for name, reducer in policy.metric_reducers.items()
            }
            if metrics.get("metric_reducers") != expected_reducers:
                raise AlgorithmExecutionError(
                    "collective training did not report the declared metric reducers"
                )
            if not any(name in metrics for name in expected_reducers):
                raise AlgorithmExecutionError(
                    "collective training did not report any declared global metric"
                )
            execution = _collective_execution_result(
                result=result,
                metrics=metrics,
                plan=plan,
                run_id=envelope.run_id,
            )
            versions = _actual_environment_versions(
                plan.environment.python,
                plan.environment.dependencies,
            )
            return WorkerExecutionResult(
                execution=execution,
                actual_versions=versions,
                worker_metadata={
                    "topology": "ray_train_collective",
                    "workers": workers,
                    "state": {
                        "coordination": "all_reduce",
                        "synchronized": True,
                        "bounded": True,
                        "global_model_digest": global_digest,
                        "details": {
                            "backend": getattr(policy, "backend", "auto"),
                            "checkpoint_owner_rank": policy.checkpoint_owner_rank,
                        },
                    },
                    "input_complete": True,
                    "driver_materialized_training_rows": 0,
                },
            )
        except ray.exceptions.RayError as exc:
            raise AlgorithmExecutionError(
                f"Ray collective execution failed: {type(exc).__name__}"
            ) from exc
        finally:
            prepared.close()


__all__ = ["RayTrainCollectiveRuntime"]

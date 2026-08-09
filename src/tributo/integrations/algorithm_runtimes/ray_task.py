"""Ray task Runtime Adapter for portable bounded execution."""

from __future__ import annotations

from typing import Any

import ray

from tributo.algorithms.api import (
    AlgorithmExecutionError,
    ResolvedAlgorithmPlan,
    RuntimeTopology,
    WorkerExecutionResult,
)
from tributo.algorithms.core.worker import reduce_worker_group, worker_bootstrap
from tributo.algorithms.spi import ExecutionEnvelope, RuntimeExecutionEnvelope
from tributo.util.annotations import DeveloperAPI

RAY_TASK_RUNTIME_ID = "tributo.ray_task"


@ray.remote
def _execute_on_worker(
    envelope: ExecutionEnvelope,
    world_rank: int,
    world_size: int,
    framework_parallelism: int,
) -> WorkerExecutionResult:
    """Collect Ray-only metadata and enter the framework-neutral bootstrap."""
    runtime_context = ray.get_runtime_context()
    metadata: dict[str, Any] = {
        "node_id": str(runtime_context.get_node_id()),
        "worker_id": str(runtime_context.get_worker_id()),
        "job_id": str(runtime_context.get_job_id()),
        "world_rank": world_rank,
        "world_size": world_size,
        "framework_parallelism": framework_parallelism,
        "topology": envelope.plan.runtime.topology.value,
    }
    return worker_bootstrap(envelope, metadata)


@ray.remote(num_cpus=0, max_retries=0)
def _reduce_on_worker(
    plan: ResolvedAlgorithmPlan,
    results: tuple[WorkerExecutionResult, ...],
) -> WorkerExecutionResult:
    """Load and execute a user reducer outside the Driver control plane."""
    runtime_context = ray.get_runtime_context()
    metadata: dict[str, Any] = {
        "node_id": str(runtime_context.get_node_id()),
        "worker_id": str(runtime_context.get_worker_id()),
        "job_id": str(runtime_context.get_job_id()),
    }
    return reduce_worker_group(plan, results, metadata)


@DeveloperAPI
class RayTaskRuntime:
    """Submit one bounded execution to a Ray Worker task."""

    @property
    def runtime_id(self) -> str:
        """Return the registration identity for this Runtime Adapter."""
        return RAY_TASK_RUNTIME_ID

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        """Execute the generic Worker bootstrap with declared task resources."""
        if not ray.is_initialized():
            raise AlgorithmExecutionError(
                "Ray must be initialized before portable algorithm execution"
            )
        try:
            topology = envelope.plan.runtime.topology
            world_size = (
                envelope.plan.runtime.worker_count
                if topology is RuntimeTopology.DATA_PARALLEL
                else 1
            )
            references = [
                _execute_on_worker.options(
                    num_cpus=envelope.plan.runtime.num_cpus,
                    num_gpus=envelope.plan.runtime.num_gpus,
                    max_retries=envelope.plan.runtime.max_retries,
                    retry_exceptions=False,
                ).remote(
                    ExecutionEnvelope(
                        plan=envelope.plan,
                        input_payload=payload,
                        artifacts=envelope.artifacts,
                        cancelled=envelope.cancelled,
                    ),
                    rank,
                    world_size,
                    envelope.plan.runtime.framework_parallelism,
                )
                for rank, payload in enumerate(envelope.input_payloads)
            ]
            results = tuple(ray.get(references))
            if any(not isinstance(item, WorkerExecutionResult) for item in results):
                raise AlgorithmExecutionError(
                    "Ray Worker group returned an invalid execution result"
                )
            if topology is RuntimeTopology.DATA_PARALLEL:
                result = ray.get(_reduce_on_worker.remote(envelope.plan, results))
            else:
                result = results[0]
        except ray.exceptions.RayError as exc:
            raise AlgorithmExecutionError(
                f"Ray Worker execution failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(result, WorkerExecutionResult):
            raise AlgorithmExecutionError(
                "Ray Worker returned an invalid execution result"
            )
        return result


__all__ = ["RayTaskRuntime"]

"""Default composition root for formal and compatibility algorithm execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    from tributo.algorithms.core import AlgorithmDispatcher, RayRuntimeManager


@PublicAPI(stability="alpha")
def build_algorithm_dispatcher(
    *,
    ingestion_gateway: Any | None = None,
    runtime_manager: RayRuntimeManager | None = None,
) -> AlgorithmDispatcher:
    """Build one Dispatcher over the sole Registry and canonical input bridge."""
    from tributo.algorithms.core import (
        AlgorithmDispatcher,
        AlgorithmPlanner,
        AlgorithmRunCoordinator,
    )
    from tributo.integrations.algorithm_inputs import (
        IngestionInputResolver,
        IngestionInputRuntimeAdapter,
    )
    from tributo.integrations.algorithm_runtimes.collective import (
        RayTrainCollectiveRuntime,
        RayTrainRecipeV2Runtime,
    )
    from tributo.integrations.algorithm_runtimes.framework_native import (
        FrameworkNativeRuntime,
    )
    from tributo.integrations.algorithm_runtimes.iterative_optimization import (
        RayIterativeOptimizationRuntime,
    )
    from tributo.integrations.algorithm_runtimes.joblib_estimator import (
        RayJoblibEstimatorRuntime,
    )
    from tributo.integrations.algorithm_runtimes.map_reduce import (
        RayMapReduceRuntime,
    )
    from tributo.integrations.algorithm_runtimes.parallel_ensemble import (
        RayParallelUnitRuntime,
    )
    from tributo.integrations.algorithm_runtimes.ray_task import RayTaskRuntime
    from tributo.training.registry import get_execution_registry

    resolver = IngestionInputResolver(
        ingestion_gateway,
        accepted_handle_kinds=("ray_data",),
    )
    input_adapter = IngestionInputRuntimeAdapter()
    runtimes = (
        RayTaskRuntime(),
        RayTrainCollectiveRuntime(),
        RayTrainRecipeV2Runtime(),
        RayMapReduceRuntime(),
        FrameworkNativeRuntime(),
        RayJoblibEstimatorRuntime(),
        RayParallelUnitRuntime(),
        RayIterativeOptimizationRuntime(),
    )
    return AlgorithmDispatcher(
        AlgorithmPlanner(
            get_execution_registry(),
            {resolver.resolver_id: resolver},
        ),
        AlgorithmRunCoordinator(
            resolvers={resolver.resolver_id: resolver},
            input_adapters={resolver.resolver_id: input_adapter},
            runtimes={runtime.runtime_id: runtime for runtime in runtimes},
        ),
        runtime_manager=runtime_manager,
    )


__all__ = ["build_algorithm_dispatcher"]

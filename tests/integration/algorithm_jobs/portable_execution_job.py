"""Run one portable algorithm channel inside a real Ray Job."""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Any, cast

import ray

from tributo.algorithms.api import (
    AlgorithmOperation,
    AlgorithmRegistration,
    AlgorithmRequest,
    BackendInputCompatibility,
    EnvironmentSpec,
    ExecutionMode,
    InputBinding,
    QualifiedReference,
    RuntimeBinding,
    RuntimeTopology,
)
from tributo.algorithms.core import (
    AlgorithmBuilder,
    AlgorithmDispatcher,
    AlgorithmPlanner,
    AlgorithmRegistrationRegistry,
    AlgorithmRunCoordinator,
)
from tributo.algorithms.input import (
    FAKE_RESOLVER_ID,
    FakeInputInvocation,
    FakeInputResolver,
    FakeInputRuntimeAdapter,
    FakeTabularPayload,
)
from tributo.algorithms.spi import (
    InputExecutionContext,
    InputResolutionContext,
    InputResolverPort,
    InputRuntimeAdapter,
)
from tributo.data import (
    DistributionVersionEvidence,
    IngestionGateway,
    IngestionRequest,
    ParquetSourceConfig,
    RayDataHandle,
)
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingDescriptor,
    BindingKey,
    EngineBinding,
    EngineBindings,
)
from tributo.data.scan_plan import ScanKind
from tributo.integrations.algorithm_inputs import (
    INGESTION_RESOLVER_ID,
    IngestionInputInvocation,
    IngestionInputResolver,
    IngestionInputRuntimeAdapter,
)
from tributo.integrations.algorithm_runtimes.ray_task import RayTaskRuntime
from tributo.training.algorithm_spec import AlgorithmSpec, ProblemType

_USER_MODULE = "tests.support.portable_algorithms"


def _fake_input_compatibility(
    *topologies: RuntimeTopology,
) -> BackendInputCompatibility:
    return BackendInputCompatibility(
        accepted_input_views=("materialized_tabular",),
        accepted_ingestion_engines=("tributo.fake_tabular",),
        required_input_capabilities=("materializable",),
        supported_explicit_adapters=(
            QualifiedReference.parse("tributo.algorithms.input.fake:prepare_input"),
        ),
        distribution_policy=topologies,
    )


def _runtime_binding(
    *,
    production_input: bool = False,
    topology: RuntimeTopology = RuntimeTopology.SINGLE_WORKER,
    worker_count: int = 1,
    framework_parallelism: int = 1,
) -> RuntimeBinding:
    worker_input_ref = (
        "tributo.integrations.algorithm_inputs.ingestion:prepare_ingestion_input"
        if production_input
        else "tributo.algorithms.input.fake:prepare_input"
    )
    return RuntimeBinding(
        runtime_id="tributo.ray_task",
        worker_input_adapter_ref=QualifiedReference.parse(worker_input_ref),
        topology=topology,
        worker_count=worker_count,
        framework_parallelism=framework_parallelism,
        result_reducer_ref=(
            QualifiedReference.parse(f"{_USER_MODULE}:reduce_custom_training_results")
            if topology is RuntimeTopology.DATA_PARALLEL
            else None
        ),
        num_cpus=0,
        max_retries=0,
    )


def _spec(name: str, operations: tuple[str, ...], mode: ExecutionMode) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        trainer_cls=None,
        version="1.0.0",
        supported_tasks=operations,
        operations=operations,
        problem_types=(ProblemType.BINARY_CLASSIFICATION,),
        learning_paradigm="supervised",
        model_family="external_test",
        data_modalities=("tabular",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=(mode.value,),
        config_contract_ref=f"tests.{name}.config.v1",
        input_contract_ref="tests.tabular.binary.v1",
        output_contract_ref="tests.binary.output.v1",
    )


def _request(
    algorithm: str,
    operation: AlgorithmOperation,
    config: dict[str, object] | None = None,
) -> AlgorithmRequest:
    return AlgorithmRequest(
        algorithm=algorithm,
        operation=operation,
        input_binding=InputBinding(
            name="train",
            resolver_id=FAKE_RESOLVER_ID,
            reference="portable-job-data",
            feature_names=("x0", "x1"),
            label_name="label",
        ),
        algorithm_config=config or {},
    )


def _ingestion_request(
    algorithm: str,
    operation: AlgorithmOperation,
    config: dict[str, object] | None = None,
) -> AlgorithmRequest:
    return AlgorithmRequest(
        algorithm=algorithm,
        operation=operation,
        input_binding=InputBinding(
            name="train",
            resolver_id=INGESTION_RESOLVER_ID,
            reference="jobs.production-input",
            feature_names=("x0", "x1"),
            label_name="label",
        ),
        algorithm_config=config or {},
    )


def _context(close_calls: list[str]) -> InputExecutionContext:
    payload = FakeTabularPayload(
        {
            "x0": (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0),
            "x1": (-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0),
            "label": (0, 0, 0, 0, 1, 1, 1, 1),
        }
    )
    return InputExecutionContext(
        {
            "portable-job-data": FakeInputInvocation(
                payload,
                close_callback=lambda: close_calls.append("closed"),
                cancel_callback=lambda: close_calls.append("cancelled"),
            )
        }
    )


class _JobBindings:
    """Compile the Gateway fixture to a real Ray Dataset without file I/O."""

    def __init__(self, lifecycle_events: list[str]) -> None:
        self._lifecycle_events = lifecycle_events

    @staticmethod
    def _descriptor() -> BindingDescriptor:
        return BindingDescriptor(
            key=BindingKey(
                "tributo.ray_data",
                ScanKind.FILE,
                "parquet",
                "jobs.ray.parquet",
            ),
            factory=lambda: cast(EngineBinding, object()),
            capabilities=frozenset(),
            distribution_name="tributo",
            distribution_version="1.0.0",
            engine_version_spec=f"=={ray.__version__}",
        )

    def describe(
        self,
        **kwargs: object,
    ) -> tuple[BindingDescriptor, frozenset[object]]:
        del kwargs
        return self._descriptor(), frozenset()

    def compile(
        self,
        **kwargs: object,
    ) -> tuple[
        BindingCompilation,
        BindingDescriptor,
        tuple[DistributionVersionEvidence, ...],
    ]:
        del kwargs
        dataset = ray.data.from_items(
            [
                {"x0": -2.0, "x1": -1.0, "label": 0},
                {"x0": -1.5, "x1": -0.5, "label": 0},
                {"x0": -1.0, "x1": -0.2, "label": 0},
                {"x0": -0.5, "x1": -0.1, "label": 0},
                {"x0": 0.5, "x1": 0.1, "label": 1},
                {"x0": 1.0, "x1": 0.2, "label": 1},
                {"x0": 1.5, "x1": 0.5, "label": 1},
                {"x0": 2.0, "x1": 1.0, "label": 1},
            ]
        )
        return (
            BindingCompilation(
                handle=RayDataHandle(dataset),
                engine_version=ray.__version__,
                reader_api="ray.data.from_items",
                transport_id="ray.object_store",
                close_callback=lambda: self._lifecycle_events.append("closed"),
                cancel_callback=lambda: self._lifecycle_events.append("cancelled"),
            ),
            self._descriptor(),
            (
                DistributionVersionEvidence(
                    distribution_name="ray",
                    driver_version=ray.__version__,
                ),
                DistributionVersionEvidence(
                    distribution_name="tributo",
                    driver_version="1.0.0",
                ),
            ),
        )


def _ingestion_contexts() -> tuple[InputResolutionContext, InputExecutionContext]:
    invocation = IngestionInputInvocation(
        IngestionRequest(
            source=ParquetSourceConfig(path="/not-read/algorithm-input.parquet"),
            engine="ray",
            trace_context={"trace_id": "portable-ingestion-job"},
        )
    )
    values = {"jobs.production-input": invocation}
    return InputResolutionContext(values=values), InputExecutionContext(values)


def _dispatcher(
    registration: AlgorithmRegistration,
    *,
    production_input: bool = False,
    lifecycle_events: list[str] | None = None,
) -> AlgorithmDispatcher:
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver: InputResolverPort
    input_adapter: InputRuntimeAdapter
    if production_input:
        if lifecycle_events is None:
            raise ValueError("production input requires lifecycle events")
        gateway = IngestionGateway(cast(EngineBindings, _JobBindings(lifecycle_events)))
        resolver = IngestionInputResolver(
            gateway,
            accepted_handle_kinds=("ray_data",),
        )
        input_adapter = IngestionInputRuntimeAdapter()
    else:
        resolver = FakeInputResolver()
        input_adapter = FakeInputRuntimeAdapter()
    runtime = RayTaskRuntime()
    return AlgorithmDispatcher(
        AlgorithmPlanner(registry, {resolver.resolver_id: resolver}),
        AlgorithmRunCoordinator(
            resolvers={resolver.resolver_id: resolver},
            input_adapters={resolver.resolver_id: input_adapter},
            runtimes={runtime.runtime_id: runtime},
        ),
    )


def _run_sklearn(*, production_input: bool = False) -> dict[str, Any]:
    registration = AlgorithmBuilder.from_sklearn(
        spec=_spec(
            "jobs.sklearn",
            ("fit", "evaluate", "predict"),
            ExecutionMode.MANAGED_ESTIMATOR,
        ),
        implementation_id="jobs.sklearn_logistic",
        implementation_version="1.0.0",
        estimator_factory=f"{_USER_MODULE}:logistic_pipeline_factory",
        environment=EnvironmentSpec(
            environment_id="jobs.sklearn",
            dependencies=("ray==2.55.1", "scikit-learn>=1.4,<2"),
        ),
        runtime=None if production_input else _runtime_binding(),
        input_compatibility=(
            None
            if production_input
            else _fake_input_compatibility(
                RuntimeTopology.SINGLE_WORKER,
                RuntimeTopology.FRAMEWORK_MANAGED,
            )
        ),
        num_cpus=0 if production_input else None,
        allowed_config_keys=("model__C",),
        trusted_pickle=True,
        is_default=True,
    )
    lifecycle_events: list[str] = []
    dispatcher = _dispatcher(
        registration,
        production_input=production_input,
        lifecycle_events=lifecycle_events,
    )
    close_calls: list[str] = []
    resolution_context, execution_context = (
        _ingestion_contexts() if production_input else (None, _context(close_calls))
    )
    request_factory = _ingestion_request if production_input else _request
    fit = dispatcher.execute(
        request_factory(
            "jobs.sklearn",
            AlgorithmOperation.FIT,
            {"model__C": 0.5},
        ),
        execution_context,
        resolution_context=resolution_context,
    )
    if fit.execution.status != "succeeded":
        raise AssertionError(f"sklearn fit failed: {fit.execution}")
    model = fit.execution.artifacts[0]
    evaluate = dispatcher.execute(
        request_factory("jobs.sklearn", AlgorithmOperation.EVALUATE),
        execution_context,
        artifacts=(model,),
        resolution_context=resolution_context,
    )
    predict = dispatcher.execute(
        request_factory("jobs.sklearn", AlgorithmOperation.PREDICT),
        execution_context,
        artifacts=(model,),
        resolution_context=resolution_context,
    )
    if evaluate.execution.status != "succeeded":
        raise AssertionError(f"sklearn evaluate failed: {evaluate.execution}")
    if predict.execution.status != "succeeded":
        raise AssertionError(f"sklearn predict failed: {predict.execution}")
    return {
        "channel": "sklearn",
        "fit_accuracy": fit.execution.metrics["accuracy"],
        "evaluate_accuracy": evaluate.execution.metrics["accuracy"],
        "prediction_count": len(predict.execution.outputs["predictions"]),
        "artifact_sha256": model.sha256,
        "actual_sklearn": fit.actual_versions["scikit-learn"],
        "actual_ray": fit.actual_versions["ray"],
        "close_calls": close_calls,
        "ingestion_lifecycle": lifecycle_events,
        "input_resolver": fit.input_provenance["resolver_id"],
        "input_engine": fit.input_provenance.get("engine_id"),
        "request_body_in_provenance": "source" in fit.input_provenance,
    }


def _run_distributed_sklearn() -> dict[str, Any]:
    registration = AlgorithmBuilder.from_sklearn(
        spec=_spec(
            "jobs.sklearn_distributed",
            ("fit",),
            ExecutionMode.MANAGED_ESTIMATOR,
        ),
        implementation_id="jobs.sklearn_ray_joblib",
        implementation_version="1.0.0",
        estimator_factory=f"{_USER_MODULE}:ray_joblib_probe_factory",
        environment=EnvironmentSpec(
            environment_id="jobs.sklearn_ray_joblib",
            dependencies=("ray==2.55.1", "scikit-learn>=1.4,<2"),
        ),
        runtime=_runtime_binding(
            topology=RuntimeTopology.FRAMEWORK_MANAGED,
            framework_parallelism=2,
        ),
        input_compatibility=_fake_input_compatibility(
            RuntimeTopology.SINGLE_WORKER,
            RuntimeTopology.FRAMEWORK_MANAGED,
        ),
        allowed_config_keys=("n_jobs", "task_count"),
        trusted_pickle=True,
        is_default=True,
    )
    dispatcher = _dispatcher(registration)
    result = dispatcher.execute(
        _request(
            "jobs.sklearn_distributed",
            AlgorithmOperation.FIT,
            {"n_jobs": 2, "task_count": 6},
        ),
        _context([]),
    )
    if result.execution.status != "succeeded":
        raise AssertionError(f"distributed sklearn fit failed: {result.execution}")
    driver_imported_user_module = _USER_MODULE in sys.modules
    driver_imported_sklearn = "sklearn" in sys.modules
    estimator = pickle.loads(result.execution.artifacts[0].payload)
    evidence = cast(list[dict[str, object]], estimator.worker_evidence_)
    return {
        "channel": "distributed_sklearn",
        "task_count": len(evidence),
        "worker_ids": sorted({str(item["worker_id"]) for item in evidence}),
        "node_ids": sorted({str(item["node_id"]) for item in evidence}),
        "outer_worker_id": result.worker_metadata["worker_id"],
        "framework_parallelism": result.worker_metadata["framework_parallelism"],
        "actual_sklearn": result.actual_versions["scikit-learn"],
        "actual_ray": result.actual_versions["ray"],
        "driver_imported_user_module": driver_imported_user_module,
        "driver_imported_sklearn": driver_imported_sklearn,
    }


def _run_function(
    *,
    fail: bool = False,
    cancelled: bool = False,
    production_input: bool = False,
    data_parallel: bool = False,
) -> dict[str, Any]:
    explicit_runtime = (
        None
        if production_input
        else _runtime_binding(
            topology=(
                RuntimeTopology.DATA_PARALLEL
                if data_parallel
                else RuntimeTopology.SINGLE_WORKER
            ),
            worker_count=2 if data_parallel else 1,
        )
    )
    registration = AlgorithmBuilder.from_ray_function(
        spec=_spec(
            "jobs.user_function",
            ("fit",),
            ExecutionMode.CUSTOM_RAY_FUNCTION,
        ),
        implementation_id="jobs.user_function",
        implementation_version="1.0.0",
        function=(
            f"{_USER_MODULE}:sensitive_failure_fragment"
            if fail
            else (
                f"{_USER_MODULE}:cancellation_aware_fragment"
                if cancelled
                else f"{_USER_MODULE}:custom_training_fragment"
            )
        ),
        environment=EnvironmentSpec(
            environment_id="jobs.user_function",
            dependencies=("ray==2.55.1",),
        ),
        runtime=explicit_runtime,
        input_compatibility=(
            None
            if production_input
            else _fake_input_compatibility(
                RuntimeTopology.SINGLE_WORKER,
                RuntimeTopology.DATA_PARALLEL,
            )
        ),
        worker_count=2 if data_parallel and explicit_runtime is None else 1,
        result_reducer=(
            f"{_USER_MODULE}:reduce_custom_training_results"
            if data_parallel and explicit_runtime is None
            else None
        ),
        allowed_config_keys=("threshold",),
        is_default=True,
    )
    lifecycle_events: list[str] = []
    dispatcher = _dispatcher(
        registration,
        production_input=production_input,
        lifecycle_events=lifecycle_events,
    )
    close_calls: list[str] = []
    resolution_context, execution_context = (
        _ingestion_contexts() if production_input else (None, _context(close_calls))
    )
    request_factory = _ingestion_request if production_input else _request
    result = dispatcher.execute(
        request_factory(
            "jobs.user_function",
            AlgorithmOperation.FIT,
            {"threshold": 0.8},
        ),
        execution_context,
        resolution_context=resolution_context,
        cancelled=cancelled,
    )
    if fail:
        return {
            "channel": "function_failure",
            "status": result.execution.status,
            "failure_category": result.execution.failure_category,
            "error_type": result.execution.error_type,
            "error_message": result.execution.error_message,
            "actual_ray": result.actual_versions["ray"],
            "close_calls": close_calls,
            "ingestion_lifecycle": lifecycle_events,
            "input_resolver": result.input_provenance["resolver_id"],
        }
    if cancelled:
        return {
            "channel": "function_cancelled",
            "cancelled": result.execution.outputs["cancelled"],
            "actual_ray": result.actual_versions["ray"],
            "close_calls": close_calls,
            "ingestion_lifecycle": lifecycle_events,
            "input_resolver": result.input_provenance["resolver_id"],
        }
    if result.execution.status != "succeeded":
        raise AssertionError(f"custom function failed: {result.execution}")
    if data_parallel:
        return {
            "channel": "distributed_function",
            "row_count": result.execution.metrics["row_count"],
            "positive_rate": result.execution.metrics["positive_rate"],
            "ranks": result.execution.outputs["ranks"],
            "worker_ids": result.execution.outputs["worker_ids"],
            "shard_values": result.execution.outputs["shard_values"],
            "artifact_kinds": [
                artifact.kind for artifact in result.execution.artifacts
            ],
            "runtime_workers": result.worker_metadata["workers"],
            "reducer_worker": result.worker_metadata["reducer_worker"],
            "actual_ray": result.actual_versions["ray"],
            "ingestion_lifecycle": lifecycle_events,
            "input_resolver": result.input_provenance["resolver_id"],
            "input_engine": result.input_provenance.get("engine_id"),
        }
    return {
        "channel": "function",
        "positive_rate": result.execution.metrics["positive_rate"],
        "threshold": result.execution.outputs["threshold"],
        "worker_id": result.execution.outputs["worker_id"],
        "artifact_kinds": [artifact.kind for artifact in result.execution.artifacts],
        "actual_ray": result.actual_versions["ray"],
        "close_calls": close_calls,
        "ingestion_lifecycle": lifecycle_events,
        "input_resolver": result.input_provenance["resolver_id"],
        "input_engine": result.input_provenance.get("engine_id"),
        "request_body_in_provenance": "source" in result.input_provenance,
    }


def _run_ingestion_function_matrix() -> dict[str, Any]:
    success = _run_function(production_input=True)
    failure = _run_function(fail=True, production_input=True)
    cancellation = _run_function(cancelled=True, production_input=True)
    return {
        "channel": "ingestion_function",
        "success": success,
        "failure": failure,
        "cancellation": cancellation,
    }


def main() -> int:
    """Execute the requested channel and emit one machine-readable result."""
    channel = os.environ["PORTABLE_CHANNEL"]
    sys.modules.pop(_USER_MODULE, None)
    ray.init()
    try:
        if channel == "sklearn":
            result = _run_sklearn()
        elif channel == "function":
            result = _run_function()
        elif channel == "function_failure":
            result = _run_function(fail=True)
        elif channel == "distributed_sklearn":
            result = _run_distributed_sklearn()
        elif channel == "distributed_function":
            result = _run_function(production_input=True, data_parallel=True)
        elif channel == "ingestion_sklearn":
            result = _run_sklearn(production_input=True)
            result["channel"] = "ingestion_sklearn"
        elif channel == "ingestion_function":
            result = _run_ingestion_function_matrix()
        else:
            raise ValueError(f"unsupported portable channel: {channel!r}")
        result.setdefault("driver_imported_user_module", _USER_MODULE in sys.modules)
        result.setdefault("driver_imported_sklearn", "sklearn" in sys.modules)
        print(f"RESULT: {json.dumps(result, sort_keys=True)}")
    finally:
        ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

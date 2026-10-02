"""Unit tests for portable descriptors, plans, and registration invariants."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Literal, cast

import pytest

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionResult,
    AlgorithmOperation,
    AlgorithmRegistration,
    AlgorithmRequest,
    ArtifactDraft,
    BackendInputCompatibility,
    CollectivePolicy,
    DistributedAlgorithmDescriptor,
    DistributionSpec,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    ExecutionRequest,
    FrameworkNativePolicy,
    ImplementationDescriptor,
    InputBinding,
    InputDistribution,
    IterativeOptimizationPolicy,
    JoblibEstimatorPolicy,
    MapReducePolicy,
    MetricReduction,
    ParallelEnsemblePolicy,
    QualifiedReference,
    ResolvedInputDescriptor,
    ResultPolicy,
    RuntimeBinding,
    RuntimeTopology,
    StateCoordination,
    StateField,
    WorkerExecutionResult,
    WorkerRange,
    WorkerResources,
    canonical_digest,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.algorithms.core import AlgorithmPlanner, AlgorithmRegistrationRegistry

from .conftest import fake_runtime_binding, make_spec


def _formal_policy(
    strategy: DistributionStrategy,
) -> (
    CollectivePolicy
    | MapReducePolicy
    | FrameworkNativePolicy
    | JoblibEstimatorPolicy
    | ParallelEnsemblePolicy
    | IterativeOptimizationPolicy
):
    if strategy in {
        DistributionStrategy.RAY_TRAIN_COLLECTIVE,
    }:
        return CollectivePolicy(
            backend="gloo",
            metric_reducers={"loss": MetricReduction.WEIGHTED_MEAN},
        )
    if strategy is DistributionStrategy.RAY_MAP_REDUCE:
        return MapReducePolicy(
            state_schema=(StateField("sum", "float64", (None,)),),
            max_partial_state_bytes=4096,
            reducer_ref="tests.support.portable_algorithms:merge_map_reduce_states",
            finalizer_ref="tests.support.portable_algorithms:finalize_map_reduce_model",
        )
    if strategy is DistributionStrategy.RAY_JOBLIB_ESTIMATOR:
        return JoblibEstimatorPolicy()
    if strategy is DistributionStrategy.RAY_PARALLEL_ENSEMBLE:
        return ParallelEnsemblePolicy(max_units=8)
    if strategy is DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION:
        return IterativeOptimizationPolicy(max_rounds=4)
    return FrameworkNativePolicy(
        framework="example",
        evidence_collector_ref=(
            "tests.support.portable_algorithms:collect_framework_evidence"
        ),
    )


def _formal_descriptor(
    strategy: DistributionStrategy,
    *,
    operations: tuple[str, ...] = ("fit",),
    result_policy: ResultPolicy = ResultPolicy.FIT_ONLY,
    input_compatibility: BackendInputCompatibility | None = None,
    exporter: str | None = None,
    flavor_id: str | None = None,
) -> DistributedAlgorithmDescriptor:
    expected_mode = {
        DistributionStrategy.RAY_TRAIN_COLLECTIVE: ExecutionMode.COLLECTIVE,
        DistributionStrategy.RAY_MAP_REDUCE: ExecutionMode.MAP_REDUCE,
        DistributionStrategy.FRAMEWORK_NATIVE: ExecutionMode.FRAMEWORK_NATIVE,
        DistributionStrategy.RAY_JOBLIB_ESTIMATOR: ExecutionMode.JOBLIB_ESTIMATOR,
        DistributionStrategy.RAY_PARALLEL_ENSEMBLE: ExecutionMode.PARALLEL_ENSEMBLE,
        DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION: (
            ExecutionMode.ITERATIVE_OPTIMIZATION
        ),
    }[strategy]
    return AlgorithmBuilder.from_distributed_algorithm(
        spec=make_spec(
            f"builder_{strategy.value}",
            operations=operations,
            mode=expected_mode,
        ),
        implementation_id=f"tests.builder.{strategy.value}",
        implementation_version="1.0.0",
        implementation="tests.support.portable_algorithms:ExampleMapReduceAlgorithm",
        executable_factory="tests.support.portable_algorithms:map_reduce_factory",
        distribution="example-distributed-algorithm",
        framework="example",
        environment=EnvironmentSpec(
            environment_id=f"tests.builder.{strategy.value}",
            dependencies=("example-distributed-algorithm==1.0.0",),
        ),
        allowed_config_keys=(),
        strategy=strategy,
        supported_worker_range=WorkerRange(1, 8),
        supported_execution_profiles=(
            ExecutionProfile.LOCAL,
            ExecutionProfile.CLUSTER,
        ),
        resources_per_worker=WorkerResources(),
        policy=_formal_policy(strategy),
        package_name="example-distributed-algorithm",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
        result_policy=result_policy,
        input_compatibility=input_compatibility,
        exporter=exporter,
        flavor_id=flavor_id,
        tested=True,
        supported=True,
        validated_execution_profiles=(ExecutionProfile.LOCAL,),
        is_default=True,
    )


def test_qualified_reference_is_validated_without_import() -> None:
    reference = QualifiedReference.parse("package.integration:factory.create")
    assert str(reference) == "package.integration:factory.create"

    for invalid in ("missing_separator", "pkg:<locals>", "pkg:lambda value"):
        with pytest.raises(AlgorithmConfigurationError):
            QualifiedReference.parse(invalid)

    with pytest.raises(AlgorithmConfigurationError, match="module-qualified strings"):
        QualifiedReference.parse(cast(str, lambda: None))


def test_request_rejects_callable_configuration() -> None:
    with pytest.raises(
        AlgorithmConfigurationError,
        match="portable JSON values",
    ):
        AlgorithmRequest(
            algorithm="external",
            operation=AlgorithmOperation.FIT,
            input_binding=InputBinding(
                name="train",
                resolver_id="tributo.fake_tabular",
                reference="fixture",
                feature_names=("x",),
                label_name="label",
            ),
            algorithm_config={"factory": lambda: None},
        )

    with pytest.raises(AlgorithmConfigurationError, match="canonical JSON"):
        AlgorithmRequest(
            algorithm="external",
            operation=AlgorithmOperation.FIT,
            input_binding=InputBinding(
                name="train",
                resolver_id="tributo.fake_tabular",
                reference="fixture",
                feature_names=("x",),
            ),
            algorithm_config={"threshold": float("nan")},
        )

    with pytest.raises(AlgorithmConfigurationError, match="canonical JSON"):
        AlgorithmExecutionResult(
            status="succeeded",
            metrics={"loss": float("nan")},
        )


def test_execution_results_reject_invalid_nested_contract_values() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="ArtifactDraft"):
        AlgorithmExecutionResult(
            status="succeeded",
            artifacts=cast(tuple[ArtifactDraft, ...], (object(),)),
        )
    with pytest.raises(AlgorithmConfigurationError, match="AlgorithmExecutionResult"):
        WorkerExecutionResult(
            execution=cast(AlgorithmExecutionResult, object()),
            actual_versions={},
        )


def test_request_rejects_nested_credentials_but_accepts_secret_references() -> None:
    binding = InputBinding(
        name="train",
        resolver_id="tributo.fake_tabular",
        reference="fixture",
        feature_names=("x",),
        label_name="label",
    )
    with pytest.raises(AlgorithmConfigurationError, match="sensitive field"):
        AlgorithmRequest(
            algorithm="external",
            operation=AlgorithmOperation.FIT,
            input_binding=binding,
            algorithm_config={"service": {"api-key": "plaintext"}},
        )

    request = AlgorithmRequest(
        algorithm="external",
        operation=AlgorithmOperation.FIT,
        input_binding=binding,
        algorithm_config={"service": {"secret_ref": "vault://algorithm/key"}},
    )
    assert request.algorithm_config["service"]["secret_ref"] == (
        "vault://algorithm/key"
    )

    with pytest.raises(AlgorithmConfigurationError, match="must not contain"):
        replace(
            binding,
            reference="https://alice:private@example.test/data?token=plaintext",
        )

    for credential_fragment in (
        "https://example.test/data#token=plaintext",
        "https://example.test/data#/view?api-key=plaintext",
    ):
        with pytest.raises(AlgorithmConfigurationError, match="must not contain"):
            replace(binding, reference=credential_fragment)

    safe_fragment = replace(
        binding,
        reference="https://example.test/data#partition=2026-08-09",
    )
    assert safe_fragment.reference.endswith("#partition=2026-08-09")


def test_portable_registration_rejects_legacy_trainer() -> None:
    spec = make_spec(
        "legacy",
        operations=("fit",),
        mode=ExecutionMode.MANAGED_ESTIMATOR,
    )
    spec = replace(spec, trainer_cls=type("LegacyTrainer", (), {}))

    with pytest.raises(
        AlgorithmConfigurationError,
        match="must not store trainer_cls",
    ):
        AlgorithmBuilder.from_sklearn(
            spec=spec,
            implementation_id="tests.legacy",
            implementation_version="1.0.0",
            estimator_factory=(
                "tests.support.portable_algorithms:logistic_regression_factory"
            ),
            environment=EnvironmentSpec(environment_id="tests.legacy"),
            runtime=fake_runtime_binding(),
            allowed_config_keys=(),
        )


def test_environment_normalizes_dependency_order() -> None:
    environment = EnvironmentSpec(
        environment_id="tests.environment",
        dependencies=("ray==2.55.1", "scikit-learn>=1.4,<2"),
    )
    assert environment.dependencies == (
        "ray==2.55.1",
        "scikit-learn<2,>=1.4",
    )
    assert json.dumps(environment.dependencies)

    with pytest.raises(AlgorithmConfigurationError, match="more than once"):
        EnvironmentSpec(
            environment_id="tests.duplicates",
            dependencies=("Scikit-Learn>=1.4", "scikit-learn<2"),
        )
    with pytest.raises(AlgorithmConfigurationError, match="not URLs"):
        EnvironmentSpec(
            environment_id="tests.url",
            dependencies=("example @ https://example.test/example.whl",),
        )


@pytest.mark.parametrize(
    (
        "strategy",
        "expected_mode",
        "expected_runtime",
        "expected_topology",
        "expected_input_distribution",
        "expected_coordination",
        "expected_adapter",
    ),
    [
        (
            DistributionStrategy.RAY_TRAIN_COLLECTIVE,
            ExecutionMode.COLLECTIVE,
            "tributo.ray_train_collective",
            RuntimeTopology.RAY_TRAIN_COLLECTIVE,
            InputDistribution.SHARDED,
            StateCoordination.ALL_REDUCE,
            "prepare_ray_train_input",
        ),
        (
            DistributionStrategy.RAY_MAP_REDUCE,
            ExecutionMode.MAP_REDUCE,
            "tributo.ray_map_reduce",
            RuntimeTopology.RAY_MAP_REDUCE,
            InputDistribution.SHARDED,
            StateCoordination.ASSOCIATIVE_REDUCE,
            "prepare_ray_batch_input",
        ),
        (
            DistributionStrategy.FRAMEWORK_NATIVE,
            ExecutionMode.FRAMEWORK_NATIVE,
            "tributo.framework_native",
            RuntimeTopology.FRAMEWORK_NATIVE,
            InputDistribution.FRAMEWORK_OWNED,
            StateCoordination.FRAMEWORK_NATIVE,
            "prepare_ray_train_input",
        ),
    ],
)
def test_distributed_builder_lowers_each_strategy_deterministically(
    strategy: DistributionStrategy,
    expected_mode: ExecutionMode,
    expected_runtime: str,
    expected_topology: RuntimeTopology,
    expected_input_distribution: InputDistribution,
    expected_coordination: StateCoordination,
    expected_adapter: str,
) -> None:
    first = _formal_descriptor(strategy)
    second = _formal_descriptor(strategy)

    assert first == second
    assert first.registration == second.registration
    registration = first.registration
    distribution = registration.distribution_spec
    second_distribution = second.registration.distribution_spec
    assert distribution is not None
    assert second_distribution is not None
    assert distribution.digest == second_distribution.digest
    assert registration.implementation.execution_mode is expected_mode
    assert registration.implementation.runtime_id == expected_runtime
    assert distribution.input_distribution is expected_input_distribution
    assert distribution.state_coordination is expected_coordination
    compatibility = registration.implementation.input_compatibility
    worker_adapter = registration.implementation.worker_input_adapter_ref
    assert worker_adapter is not None
    assert compatibility.distribution_policy == (expected_topology,)
    assert str(worker_adapter).endswith(f":{expected_adapter}")
    assert compatibility.supported_explicit_adapters == (worker_adapter,)
    assert registration.implementation.exporter_ref is None
    assert registration.implementation.flavor_id is None


def test_distributed_builder_contract_is_shared_with_registration_fields() -> None:
    for strategy, contract in FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS.items():
        if strategy is DistributionStrategy.RAY_TRAIN_TORCH:
            continue
        registration = _formal_descriptor(strategy).registration
        distribution = registration.distribution_spec
        assert distribution is not None
        assert registration.implementation.execution_mode is contract.execution_mode
        assert registration.implementation.runtime_id == contract.runtime_id
        assert distribution.input_distribution is contract.input_distribution
        assert distribution.state_coordination is contract.state_coordination
        assert registration.implementation.worker_input_adapter_ref is not None
        assert (
            str(registration.implementation.worker_input_adapter_ref)
            == contract.worker_input_adapter_ref
        )
        assert registration.implementation.input_compatibility.distribution_policy == (
            contract.topology,
        )


def test_distributed_builder_rejects_non_fit_operations() -> None:
    with pytest.raises(
        AlgorithmConfigurationError,
        match="only operation='fit'",
    ):
        _formal_descriptor(
            DistributionStrategy.RAY_MAP_REDUCE,
            operations=("fit", "predict"),
        )


def test_distributed_builder_rejects_result_and_input_contract_conflicts() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="bundle_required"):
        _formal_descriptor(
            DistributionStrategy.RAY_MAP_REDUCE,
            result_policy=ResultPolicy.BUNDLE_REQUIRED,
        )
    with pytest.raises(AlgorithmConfigurationError, match="declared together"):
        _formal_descriptor(
            DistributionStrategy.RAY_MAP_REDUCE,
            exporter="tests.support.portable_algorithms:export_map_reduce_model",
        )

    incompatible = BackendInputCompatibility(
        accepted_input_views=("ray_data",),
        accepted_ingestion_engines=("tributo.ray_data",),
        required_input_capabilities=("shardable",),
        supported_explicit_adapters=(
            QualifiedReference.parse(
                "tributo.integrations.algorithm_inputs.ingestion:prepare_ray_batch_input"
            ),
        ),
        distribution_policy=(RuntimeTopology.RAY_TRAIN_COLLECTIVE,),
    )
    with pytest.raises(AlgorithmConfigurationError, match="strategy topology"):
        _formal_descriptor(
            DistributionStrategy.RAY_MAP_REDUCE,
            input_compatibility=incompatible,
        )


def test_bundle_required_distributed_builder_keeps_publication_contract() -> None:
    descriptor = _formal_descriptor(
        DistributionStrategy.RAY_MAP_REDUCE,
        result_policy=ResultPolicy.BUNDLE_REQUIRED,
        exporter="tests.support.portable_algorithms:export_map_reduce_model",
        flavor_id="tests.map_reduce.bundle",
    )

    assert descriptor.registration.distribution_spec is not None
    assert (
        descriptor.registration.distribution_spec.result_policy
        is ResultPolicy.BUNDLE_REQUIRED
    )
    assert descriptor.registration.implementation.exporter_ref is not None
    assert descriptor.registration.implementation.flavor_id == "tests.map_reduce.bundle"


def test_distributed_builder_registration_produces_deterministic_plans() -> None:
    descriptor = _formal_descriptor(DistributionStrategy.RAY_MAP_REDUCE)
    registry = AlgorithmRegistrationRegistry()
    registry.register(descriptor.registration)

    class RayDataResolver:
        resolver_id = "tests.ray_data"

        def describe(
            self, binding: InputBinding, _context: object
        ) -> ResolvedInputDescriptor:
            return ResolvedInputDescriptor(
                resolver_id=self.resolver_id,
                reference=binding.reference,
                descriptor_version=1,
                binding_digest=canonical_digest(binding.descriptor_payload()),
                engine_id="tributo.ray_data",
                view_kind="ray_data",
                input_capabilities=("materializable", "shardable"),
                compatible_worker_input_adapter_refs=(
                    "tributo.integrations.algorithm_inputs.ingestion:"
                    "prepare_ray_batch_input",
                ),
            )

    resolver = RayDataResolver()
    planner = AlgorithmPlanner(registry, {resolver.resolver_id: resolver})
    request = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm="builder_ray_map_reduce",
            operation=AlgorithmOperation.FIT,
            input_binding=InputBinding(
                name="train",
                resolver_id=resolver.resolver_id,
                reference="fixture",
                feature_names=("x",),
                label_name="label",
            ),
            algorithm_config={},
        ),
        profile=ExecutionProfile.LOCAL,
        worker_count=2,
    )

    first = planner.plan(request, available_resources={"CPU": 2.0})
    second = planner.plan(request, available_resources={"CPU": 2.0})

    assert first == second
    assert first.plan_id == second.plan_id
    assert first.config_digest == second.config_digest
    assert first.input_descriptor.binding_digest == (
        second.input_descriptor.binding_digest
    )
    assert first.distribution_spec is not None
    assert second.distribution_spec is not None
    assert first.distribution_spec.digest == second.distribution_spec.digest
    assert first.to_dict() == second.to_dict()


def test_distributed_builder_matches_manual_reference_contracts() -> None:
    built = _formal_descriptor(DistributionStrategy.RAY_MAP_REDUCE)
    built_registration = built.registration
    manual_registration = AlgorithmRegistration(
        spec=built_registration.spec,
        implementation=ImplementationDescriptor(
            implementation_id="tests.builder.ray_map_reduce",
            version="1.0.0",
            execution_mode=ExecutionMode.MAP_REDUCE,
            implementation_ref=QualifiedReference.parse(
                "tests.support.portable_algorithms:ExampleMapReduceAlgorithm"
            ),
            executable_factory_ref=QualifiedReference.parse(
                "tests.support.portable_algorithms:map_reduce_factory"
            ),
            operations=(AlgorithmOperation.FIT,),
            input_compatibility=BackendInputCompatibility(
                accepted_input_views=("ray_data",),
                accepted_ingestion_engines=("tributo.ray_data",),
                required_input_capabilities=("shardable",),
                supported_explicit_adapters=(
                    QualifiedReference.parse(
                        "tributo.integrations.algorithm_inputs.ingestion:"
                        "prepare_ray_batch_input"
                    ),
                ),
                distribution_policy=(RuntimeTopology.RAY_MAP_REDUCE,),
            ),
            distribution="example-distributed-algorithm",
            framework="example",
            allowed_config_keys=(),
            runtime_id="tributo.ray_map_reduce",
            worker_input_adapter_ref=QualifiedReference.parse(
                "tributo.integrations.algorithm_inputs.ingestion:"
                "prepare_ray_batch_input"
            ),
        ),
        environment=EnvironmentSpec(
            environment_id="tests.builder.ray_map_reduce",
            dependencies=("example-distributed-algorithm==1.0.0",),
        ),
        distribution_spec=DistributionSpec(
            strategy=DistributionStrategy.RAY_MAP_REDUCE,
            supported_worker_range=WorkerRange(1, 8),
            supported_execution_profiles=(
                ExecutionProfile.LOCAL,
                ExecutionProfile.CLUSTER,
            ),
            resources_per_worker=WorkerResources(),
            input_distribution=InputDistribution.SHARDED,
            state_coordination=StateCoordination.ASSOCIATIVE_REDUCE,
            policy=_formal_policy(DistributionStrategy.RAY_MAP_REDUCE),
            result_policy=ResultPolicy.FIT_ONLY,
        ),
        is_default=True,
    )
    manual = DistributedAlgorithmDescriptor(
        registration=manual_registration,
        package_name="example-distributed-algorithm",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
        tested=True,
        supported=True,
        validated_execution_profiles=(ExecutionProfile.LOCAL,),
    )

    assert built.registration == manual_registration
    assert built == manual


def test_runtime_resources_must_be_finite() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="finite"):
        RuntimeBinding(
            runtime_id="tests.runtime",
            worker_input_adapter_ref=QualifiedReference.parse("package.input:prepare"),
            num_cpus=float("nan"),
        )


@pytest.mark.parametrize(
    "runtime_kwargs",
    [
        {"worker_count": 2},
        {
            "topology": RuntimeTopology.FRAMEWORK_MANAGED,
            "framework_parallelism": 1,
        },
        {
            "topology": RuntimeTopology.FRAMEWORK_MANAGED,
            "worker_count": 2,
            "framework_parallelism": 2,
        },
        {
            "topology": RuntimeTopology.DATA_PARALLEL,
            "worker_count": 2,
        },
        {
            "topology": RuntimeTopology.DATA_PARALLEL,
            "worker_count": 2,
            "framework_parallelism": 2,
            "result_reducer_ref": QualifiedReference.parse(
                "package.module:reduce_results"
            ),
        },
        {
            "topology": RuntimeTopology.DATA_PARALLEL,
            "worker_count": 2,
            "result_reducer_ref": QualifiedReference.parse(
                "package.module:reduce_results"
            ),
            "max_retries": 1,
        },
    ],
)
def test_runtime_topology_rejects_inconsistent_resource_shapes(
    runtime_kwargs: dict[str, Any],
) -> None:
    with pytest.raises(AlgorithmConfigurationError):
        RuntimeBinding(
            runtime_id="tests.runtime",
            worker_input_adapter_ref=QualifiedReference.parse("package.input:prepare"),
            **runtime_kwargs,
        )


def test_implementation_rejects_unknown_mode_operation_and_artifact_format() -> None:
    def build(
        *,
        execution_mode: ExecutionMode,
        operations: tuple[AlgorithmOperation, ...] = (AlgorithmOperation.FIT,),
        artifact_format: Literal["none", "trusted_pickle"] = "none",
    ) -> ImplementationDescriptor:
        return ImplementationDescriptor(
            implementation_id="tests.invalid",
            version="1.0.0",
            execution_mode=execution_mode,
            implementation_ref=QualifiedReference.parse("package.module:factory"),
            executable_factory_ref=QualifiedReference.parse(
                "package.runtime:create_executable"
            ),
            operations=operations,
            input_compatibility=BackendInputCompatibility(
                accepted_input_views=("materialized_tabular",),
                accepted_ingestion_engines=("tests.engine",),
                required_input_capabilities=(),
                supported_explicit_adapters=(
                    QualifiedReference.parse("package.input:prepare"),
                ),
                distribution_policy=(RuntimeTopology.SINGLE_WORKER,),
            ),
            artifact_format=artifact_format,
        )

    with pytest.raises(AlgorithmConfigurationError, match="mode or operation"):
        build(execution_mode=cast(ExecutionMode, "unknown"))
    with pytest.raises(AlgorithmConfigurationError, match="mode or operation"):
        build(
            execution_mode=ExecutionMode.MANAGED_ESTIMATOR,
            operations=cast(tuple[AlgorithmOperation, ...], ("unknown",)),
        )
    with pytest.raises(AlgorithmConfigurationError, match="artifact_format"):
        build(
            execution_mode=ExecutionMode.MANAGED_ESTIMATOR,
            artifact_format=cast(Literal["none", "trusted_pickle"], "pickle"),
        )


def test_builders_reject_unsupported_or_unsafe_registration_shapes() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="does not support"):
        AlgorithmBuilder.from_sklearn(
            spec=make_spec(
                "transformer",
                operations=("transform",),
                mode=ExecutionMode.MANAGED_ESTIMATOR,
            ),
            implementation_id="tests.transformer",
            implementation_version="1.0.0",
            estimator_factory="package.module:factory",
            environment=EnvironmentSpec(
                environment_id="tests.transformer",
                dependencies=("scikit-learn>=1.4,<2",),
            ),
            runtime=fake_runtime_binding(),
            allowed_config_keys=(),
            trusted_pickle=True,
        )

    with pytest.raises(AlgorithmConfigurationError, match="persistence mode"):
        AlgorithmBuilder.from_sklearn(
            spec=make_spec(
                "predictor",
                operations=("fit", "predict"),
                mode=ExecutionMode.MANAGED_ESTIMATOR,
            ),
            implementation_id="tests.predictor",
            implementation_version="1.0.0",
            estimator_factory="package.module:factory",
            environment=EnvironmentSpec(
                environment_id="tests.predictor",
                dependencies=("scikit-learn>=1.4,<2",),
            ),
            runtime=fake_runtime_binding(),
            allowed_config_keys=(),
        )

    with pytest.raises(AlgorithmConfigurationError, match="module-qualified strings"):
        AlgorithmBuilder.from_ray_function(
            spec=make_spec(
                "callable_object",
                operations=("fit",),
                mode=ExecutionMode.CUSTOM_RAY_FUNCTION,
            ),
            implementation_id="tests.callable_object",
            implementation_version="1.0.0",
            function=cast(str, lambda: None),
            environment=EnvironmentSpec(environment_id="tests.callable_object"),
            runtime=fake_runtime_binding(),
        )


def test_builders_offer_production_defaults_without_runtime_objects() -> None:
    sklearn_registration = AlgorithmBuilder.from_sklearn(
        spec=make_spec(
            "default_sklearn",
            operations=("fit",),
            mode=ExecutionMode.MANAGED_ESTIMATOR,
        ),
        implementation_id="tests.default_sklearn",
        implementation_version="1.0.0",
        estimator_factory="package.sklearn:factory",
        environment=EnvironmentSpec(
            environment_id="tests.default_sklearn",
            dependencies=("scikit-learn>=1.4,<2",),
        ),
        allowed_config_keys=(),
        framework_parallelism=2,
    )
    function_registration = AlgorithmBuilder.from_ray_function(
        spec=make_spec(
            "default_function",
            operations=("fit",),
            mode=ExecutionMode.CUSTOM_RAY_FUNCTION,
        ),
        implementation_id="tests.default_function",
        implementation_version="1.0.0",
        function="package.function:train",
        environment=EnvironmentSpec(environment_id="tests.default_function"),
        worker_count=2,
        result_reducer="package.function:reduce",
    )

    assert sklearn_registration.runtime.topology is RuntimeTopology.FRAMEWORK_MANAGED
    assert sklearn_registration.runtime.framework_parallelism == 2
    assert sklearn_registration.runtime.num_cpus == 0
    assert function_registration.runtime.topology is RuntimeTopology.DATA_PARALLEL
    assert function_registration.runtime.worker_count == 2
    assert function_registration.runtime.num_cpus == 1
    assert str(function_registration.runtime.worker_input_adapter_ref).endswith(
        ":prepare_ingestion_input"
    )
    for registration in (sklearn_registration, function_registration):
        compatibility = registration.implementation.input_compatibility
        assert compatibility.accepted_input_views == (
            "daft_dataframe",
            "ray_data",
        )
        assert compatibility.accepted_ingestion_engines == (
            "tributo.daft",
            "tributo.ray_data",
        )
        assert all(
            "tributo.algorithms.input.fake" not in str(reference)
            for reference in compatibility.supported_explicit_adapters
        )


def test_builders_reject_multiple_runtime_topology_fact_sources() -> None:
    sklearn_spec = make_spec(
        "conflicting_sklearn_runtime",
        operations=("fit",),
        mode=ExecutionMode.MANAGED_ESTIMATOR,
    )
    function_spec = make_spec(
        "conflicting_function_runtime",
        operations=("fit",),
        mode=ExecutionMode.CUSTOM_RAY_FUNCTION,
    )

    with pytest.raises(AlgorithmConfigurationError, match="either runtime"):
        AlgorithmBuilder.from_sklearn(
            spec=sklearn_spec,
            implementation_id="tests.conflicting_sklearn_runtime",
            implementation_version="1.0.0",
            estimator_factory="package.sklearn:factory",
            environment=EnvironmentSpec(
                environment_id="tests.conflicting_sklearn_runtime",
                dependencies=("scikit-learn>=1.4,<2",),
            ),
            allowed_config_keys=(),
            runtime=fake_runtime_binding(),
            framework_parallelism=2,
            num_cpus=0,
        )

    with pytest.raises(AlgorithmConfigurationError, match="either runtime"):
        AlgorithmBuilder.from_ray_function(
            spec=function_spec,
            implementation_id="tests.conflicting_function_runtime",
            implementation_version="1.0.0",
            function="package.function:train",
            environment=EnvironmentSpec(
                environment_id="tests.conflicting_function_runtime"
            ),
            runtime=fake_runtime_binding(),
            worker_count=2,
            result_reducer="package.function:reduce",
            num_gpus=1,
        )


def test_data_parallel_builder_rejects_reducer_from_another_module() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="same module"):
        AlgorithmBuilder.from_ray_function(
            spec=make_spec(
                "cross_module_reducer",
                operations=("fit",),
                mode=ExecutionMode.CUSTOM_RAY_FUNCTION,
            ),
            implementation_id="tests.cross_module_reducer",
            implementation_version="1.0.0",
            function="package.function:train",
            environment=EnvironmentSpec(environment_id="tests.cross_module_reducer"),
            worker_count=2,
            result_reducer="package.reducer:reduce",
        )


def test_target_spec_fields_project_to_legacy_read_views() -> None:
    spec = make_spec(
        "portable",
        operations=("fit", "predict"),
        mode=ExecutionMode.MANAGED_ESTIMATOR,
    )

    assert spec.trainer_cls is None
    assert spec.operations == ("fit", "predict")
    assert spec.supported_tasks == spec.operations
    assert spec.data_modalities == ("tabular",)
    assert spec.data_modality == spec.data_modalities


def test_target_and_legacy_spec_fields_cannot_conflict() -> None:
    with pytest.raises(ValueError, match="data_modalities and data_modality conflict"):
        replace(
            make_spec(
                "portable",
                operations=("fit",),
                mode=ExecutionMode.MANAGED_ESTIMATOR,
            ),
            data_modality=("graph",),
        )

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
    AlgorithmRequest,
    ArtifactDraft,
    BackendInputCompatibility,
    EnvironmentSpec,
    ExecutionMode,
    ImplementationDescriptor,
    InputBinding,
    QualifiedReference,
    RuntimeBinding,
    RuntimeTopology,
    WorkerExecutionResult,
)

from .conftest import fake_runtime_binding, make_spec


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

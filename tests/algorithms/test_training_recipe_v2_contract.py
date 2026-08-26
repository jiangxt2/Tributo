"""Contract tests for low-code TrainingRecipeV2."""

from __future__ import annotations

from tributo.algorithms import AlgorithmBuilder, TrainingRecipeV2
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MetricReduction,
    QualifiedReference,
    RuntimeTopology,
    WorkerRange,
    WorkerResources,
)
from tributo.training.algorithm_spec import AlgorithmSpec

from .conftest import make_spec


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tests.support.portable_contracts:{validator}"
        ),
    )


def _contracts(spec: AlgorithmSpec) -> ContractBindingSet:
    assert spec.config_contract_ref is not None
    assert spec.input_contract_ref is not None
    assert spec.output_contract_ref is not None
    return ContractBindingSet(
        config=_binding(spec.config_contract_ref, "a", "ConfigValidator"),
        input=_binding(spec.input_contract_ref, "b", "InputValidator"),
        output=_binding(spec.output_contract_ref, "c", "OutputValidator"),
        coverage=_binding("test.recipe.coverage.v1", "d", "CoverageValidator"),
    )


def test_recipe_v2_requires_only_math_and_codec_hooks() -> None:
    assert TrainingRecipeV2.__abstractmethods__ == {
        "batch_adapter",
        "build_modules",
        "checkpoint_codec",
        "metric_plan",
        "optimization_plan",
        "training_step",
        "validation_step",
    }


def test_builder_selects_dedicated_recipe_v2_runtime() -> None:
    spec = make_spec(
        "external_recipe_v2",
        operations=("fit",),
        mode=ExecutionMode.TRAINING_RECIPE_V2,
    )
    descriptor = AlgorithmBuilder.from_training_recipe_v2(
        spec=spec,
        implementation_id="example.recipe_v2.binary_linear",
        implementation_version="1.0.0",
        recipe="tests.support.training_recipe_v2:BinaryLinearRecipeV2",
        environment=EnvironmentSpec(
            environment_id="example.recipe_v2.v1",
            dependencies=("example-recipe-v2==1.0.0", "torch>=2.5"),
        ),
        metric_reducers={"accuracy": MetricReduction.SUM_COUNT},
        supported_worker_range=WorkerRange(1, 8),
        supported_execution_profiles=(
            ExecutionProfile.LOCAL,
            ExecutionProfile.CLUSTER,
        ),
        resources_per_worker=WorkerResources(num_cpus=1),
        package_name="example-recipe-v2",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
        contract_bindings=_contracts(spec),
    )

    registration = descriptor.registration
    assert registration.distribution_spec is not None
    assert (
        registration.distribution_spec.strategy
        is DistributionStrategy.RAY_TRAIN_RECIPE_V2
    )
    assert registration.implementation.runtime_id == "tributo.ray_train_recipe_v2"
    assert registration.implementation.input_compatibility.distribution_policy == (
        RuntimeTopology.RAY_TRAIN_RECIPE_V2,
    )
    assert descriptor.api_version == 2

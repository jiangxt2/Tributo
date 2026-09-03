"""Contract tests for the low-code PyTorch recipe builder."""

from __future__ import annotations

import pytest

from tributo.algorithms import AlgorithmBuilder, TorchRecipe
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MetricReduction,
    ResultPolicy,
    WorkerRange,
    WorkerResources,
)

from .conftest import make_spec


def _descriptor():
    return AlgorithmBuilder.from_torch(
        spec=make_spec(
            "external_torch_recipe",
            operations=("fit",),
            mode=ExecutionMode.RAY_TRAIN_TORCH,
            default_config={
                "model": {"input_features": 2},
                "output": {"bundle_uri": "./bundle"},
            },
        ),
        implementation_id="example.torch_recipe.binary_linear",
        implementation_version="1.0.0",
        recipe="tests.support.torch_recipe:BinaryLinearRecipe",
        environment=EnvironmentSpec(
            environment_id="example.torch_recipe.v1",
            dependencies=(
                "example-torch-recipe==1.0.0",
                "ray==2.55.1",
                "torch>=2.5.0",
            ),
        ),
        metric_reducers={"accuracy": MetricReduction.SUM_COUNT},
        supported_worker_range=WorkerRange(1, 8),
        supported_execution_profiles=(
            ExecutionProfile.LOCAL,
            ExecutionProfile.CLUSTER,
        ),
        resources_per_worker=WorkerResources(num_cpus=1),
        package_name="example-torch-recipe",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
        code_digest="0" * 64,
        descriptor_api_version=1,
    )


def test_torch_recipe_has_only_typed_math_and_export_hooks() -> None:
    abstract_methods = TorchRecipe.__abstractmethods__

    assert abstract_methods == {
        "adapt_batch",
        "artifact_plan",
        "build_modules",
        "configure_optimizers",
        "metric_plan",
        "training_step",
        "validation_step",
    }
    assert "execution_plan" not in abstract_methods


def test_builder_lowers_recipe_to_unified_torch_runtime() -> None:
    descriptor = _descriptor()
    registration = descriptor.registration

    assert registration.implementation.runtime_id == "tributo.ray_train_torch"
    assert str(registration.implementation.implementation_ref) == (
        "tests.support.torch_recipe:BinaryLinearRecipe"
    )
    assert str(registration.implementation.executable_factory_ref) == (
        "tributo.integrations.algorithm_runtimes.ray_train_torch:create_torch_algorithm"
    )
    assert str(registration.implementation.exporter_ref) == (
        "tributo.integrations.algorithm_runtimes.ray_train_torch:"
        "export_ray_train_torch_result"
    )
    assert registration.distribution_spec is not None
    assert registration.distribution_spec.result_policy is ResultPolicy.BUNDLE_REQUIRED
    assert registration.distribution_spec.policy.metric_reducers == {
        "accuracy": MetricReduction.SUM_COUNT,
        "train_loss": MetricReduction.SUM_COUNT,
    }
    assert registration.implementation.allowed_config_keys == (
        "data",
        "loss",
        "metrics",
        "model",
        "optimizer",
        "output",
        "ray",
        "training",
    )


def test_builder_rejects_train_loss_reducer_override() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="train_loss"):
        AlgorithmBuilder.from_torch(
            spec=make_spec(
                "invalid_torch_recipe",
                operations=("fit",),
                mode=ExecutionMode.RAY_TRAIN_TORCH,
            ),
            implementation_id="example.torch_recipe.invalid",
            implementation_version="1.0.0",
            recipe="tests.support.torch_recipe:BinaryLinearRecipe",
            environment=EnvironmentSpec(
                environment_id="example.torch_recipe.invalid",
                dependencies=("example-torch-recipe==1.0.0",),
            ),
            metric_reducers={"train_loss": MetricReduction.MAX},
            supported_worker_range=WorkerRange(1, 2),
            supported_execution_profiles=(ExecutionProfile.LOCAL,),
            resources_per_worker=WorkerResources(),
            package_name="example-torch-recipe",
            package_version="1.0.0",
            tributo_version_spec=">=1,<2",
            code_digest="0" * 64,
        )

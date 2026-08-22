"""Contract tests for the low-code PyTorch recipe builder."""

from __future__ import annotations

import pytest

from tributo.algorithms import AlgorithmBuilder, TorchTrainingRecipe
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
    return AlgorithmBuilder.from_torch_recipe(
        spec=make_spec(
            "external_torch_recipe",
            operations=("fit",),
            mode=ExecutionMode.COLLECTIVE,
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
    )


def test_torch_recipe_is_narrower_than_collective_worker_loop() -> None:
    abstract_methods = TorchTrainingRecipe.__abstractmethods__

    assert abstract_methods == {
        "loss_factory",
        "metric_factories",
        "model_factory",
        "optimizer_factory",
    }
    assert "train_loop_per_worker" not in abstract_methods
    assert "checkpoint_state" not in abstract_methods


def test_builder_lowers_recipe_to_existing_collective_runtime() -> None:
    descriptor = _descriptor()
    registration = descriptor.registration

    assert registration.implementation.runtime_id == "tributo.ray_train_collective"
    assert str(registration.implementation.implementation_ref) == (
        "tests.support.torch_recipe:BinaryLinearRecipe"
    )
    assert str(registration.implementation.executable_factory_ref) == (
        "tributo.integrations.algorithm_runtimes.torch_recipe:"
        "create_torch_recipe_algorithm"
    )
    assert str(registration.implementation.exporter_ref) == (
        "tributo.integrations.algorithm_runtimes.torch_recipe:"
        "export_torch_recipe_result"
    )
    assert registration.distribution_spec is not None
    assert registration.distribution_spec.result_policy is ResultPolicy.BUNDLE_REQUIRED
    assert registration.distribution_spec.policy.metric_reducers == {
        "accuracy": MetricReduction.SUM_COUNT,
        "train_loss": MetricReduction.SUM_COUNT,
    }
    assert registration.implementation.allowed_config_keys == (
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
        AlgorithmBuilder.from_torch_recipe(
            spec=make_spec(
                "invalid_torch_recipe",
                operations=("fit",),
                mode=ExecutionMode.COLLECTIVE,
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
        )

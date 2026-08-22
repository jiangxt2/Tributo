"""Independent low-code PyTorch recipe package used by Tributo conformance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tributo.algorithms import AlgorithmBuilder, TorchTrainingRecipe
from tributo.algorithms.api import (
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MetricReduction,
    WorkerRange,
    WorkerResources,
)
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    Capability,
    ProblemType,
)


class ThirdPartyBinaryLinearRecipe(TorchTrainingRecipe):
    """Define only the four model-level factories required by the recipe SPI."""

    def model_factory(self, config: Mapping[str, Any]) -> object:
        import torch

        return torch.nn.Linear(int(config.get("input_features", 2)), 1)

    def loss_factory(self, config: Mapping[str, Any]) -> object:
        import torch

        del config
        return torch.nn.BCEWithLogitsLoss()

    def optimizer_factory(
        self,
        model: object,
        config: Mapping[str, Any],
    ) -> object:
        import torch

        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be torch.nn.Module")
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config.get("learning_rate", 0.1)),
        )

    def metric_factories(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        del config

        def accuracy(predictions: object, targets: object) -> object:
            import torch

            if not isinstance(predictions, torch.Tensor) or not isinstance(
                targets, torch.Tensor
            ):
                raise TypeError("accuracy requires Tensor values")
            return (torch.sigmoid(predictions) >= 0.5) == targets.bool()

        return {"accuracy": accuracy}


DESCRIPTOR = AlgorithmBuilder.from_torch_recipe(
    spec=AlgorithmSpec(
        name="third_party_binary_linear",
        trainer_cls=None,
        version="0.1.0",
        default_config={},
        supported_tasks=("fit",),
        operations=("fit",),
        problem_types=(ProblemType.BINARY_CLASSIFICATION,),
        capabilities=(Capability.EXPORTABLE, Capability.DISTRIBUTED),
        extras_group="model-export-torch",
        learning_paradigm="supervised",
        model_family="linear_model",
        data_modalities=("tabular",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=(ExecutionMode.COLLECTIVE.value,),
        config_contract_ref="example.torch_recipe.config.v1",
        input_contract_ref="tributo.tabular.dense.v1",
        output_contract_ref="tributo.classification.onnx.v1",
    ),
    implementation_id="example.torch_recipe.binary_linear",
    implementation_version="0.1.0",
    recipe=("tributo_test_torch_recipe_algorithm:ThirdPartyBinaryLinearRecipe"),
    environment=EnvironmentSpec(
        environment_id="example.torch_recipe.binary_linear.v1",
        dependencies=(
            "ray==2.55.1",
            "torch>=2.5.0",
            "tributo-test-torch-recipe-algorithm==0.1.0",
        ),
    ),
    metric_reducers={"accuracy": MetricReduction.SUM_COUNT},
    supported_worker_range=WorkerRange(1, 32),
    supported_execution_profiles=(
        ExecutionProfile.LOCAL,
        ExecutionProfile.CLUSTER,
    ),
    resources_per_worker=WorkerResources(num_cpus=1, num_gpus=0),
    package_name="tributo-test-torch-recipe-algorithm",
    package_version="0.1.0",
    tributo_version_spec=">=1,<2",
    stability="alpha",
    tested=True,
    supported=True,
    validated_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    limitations=("CPU/Gloo dense-tabular conformance fixture; GPU is not validated.",),
    is_default=True,
)
REGISTRATION = DESCRIPTOR.registration

__all__ = [
    "DESCRIPTOR",
    "REGISTRATION",
    "ThirdPartyBinaryLinearRecipe",
]

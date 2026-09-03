"""Independent low-code PyTorch recipe package used by Tributo conformance."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from tributo.algorithms import (
    AlgorithmBuilder,
    TorchArtifactPlan,
    TorchBatch,
    TorchBatchContext,
    TorchLossContribution,
    TorchMetricPlan,
    TorchModuleSet,
    TorchOptimizationPlan,
    TorchRecipe,
    TorchStepResult,
)
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

CODE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ThirdPartyBinaryLinearRecipe(TorchRecipe):
    """A low-dependency out-of-tree implementation of the new Recipe SPI."""

    def build_modules(self, context: object) -> TorchModuleSet:
        import torch

        return TorchModuleSet(
            {"model": torch.nn.Linear(2, 1), "loss": torch.nn.BCEWithLogitsLoss()}
        )

    def adapt_batch(self, batch: object, context: object) -> TorchBatch:
        import torch

        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")
        if not isinstance(context, TorchBatchContext):
            raise TypeError("Torch batch context is invalid")
        features = torch.stack(
            [
                torch.as_tensor(batch[name], dtype=torch.float32)
                for name in context.feature_names
            ],
            dim=1,
        )
        if context.label_name is None:
            raise ValueError("Torch fixture requires a label")
        targets = torch.as_tensor(
            batch[context.label_name], dtype=torch.float32
        ).reshape(-1, 1)
        return TorchBatch(
            positional=(features,), targets=targets, local_rows=len(targets)
        )

    def training_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: object
    ) -> TorchStepResult:
        import torch

        del context
        predictions = modules["model"](batch.positional[0])
        numerator = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, batch.targets.float(), reduction="sum"
        )
        return TorchStepResult(
            outputs={"prediction": predictions},
            loss=TorchLossContribution(numerator, batch.local_rows),
        )

    def validation_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: object
    ) -> TorchStepResult:
        return self.training_step(modules, batch, context)

    def configure_optimizers(
        self, modules: TorchModuleSet, context: object
    ) -> TorchOptimizationPlan:
        import torch

        del context
        return TorchOptimizationPlan(
            torch.optim.SGD(modules["model"].parameters(), lr=0.1)
        )

    def metric_plan(self, context: object) -> TorchMetricPlan:
        del context
        return TorchMetricPlan({"accuracy": "sum_count", "train_loss": "sum_count"})

    def artifact_plan(self, context: object) -> TorchArtifactPlan:
        del context
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {
                    "name": "features",
                    "dtype": "float32",
                    "shape": ("batch", 2),
                },
            ),
            output_signature=(
                {"name": "prediction", "dtype": "float32", "shape": ("batch", 1)},
            ),
            targets=({"name": "onnx-model", "format": "onnx"},),
            roles={"inference": "onnx-model"},
        )


DESCRIPTOR = AlgorithmBuilder.from_torch(
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
        allowed_execution_modes=(ExecutionMode.RAY_TRAIN_TORCH.value,),
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
    code_digest=CODE_DIGEST,
    descriptor_api_version=1,
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

"""Tests for generic Torch recipe checkpoint reconstruction and ONNX export."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from tributo.algorithms import (
    TorchArtifactPlan,
    TorchCheckpointDescriptor,
    TorchCheckpointRef,
    TorchLossContribution,
    TorchMetricPlan,
    TorchModuleSet,
    TorchOptimizationPlan,
    TorchRecipe,
    TorchRuntimeContext,
    TorchStageContext,
    TorchStageRunIdentity,
    TorchStepResult,
)
from tributo.algorithms.spi import (
    RayTorchAdapter,
    TorchArtifactContext,
    TorchCheckpointContext,
    TorchWorkerCheckpointContext,
)
from tributo.exporting.models import BundleOutputConfig, ExportSource, ExportTarget
from tributo.exporting.service import BundleExportService
from tributo.integrations.sources.ray_torch import (
    RayTorchSourceProvider,
    TorchSourceOptions,
)

torch = pytest.importorskip("torch")

CODE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class BinaryLinearRecipe(TorchRecipe):
    def build_modules(self, context):
        return TorchModuleSet(
            {"model": torch.nn.Linear(2, 1), "loss": torch.nn.MSELoss()}
        )

    def adapt_batch(self, batch, context):
        features = torch.as_tensor(batch["features"])
        targets = torch.as_tensor(batch["label"])
        return __import__("tributo.algorithms", fromlist=["TorchBatch"]).TorchBatch(
            positional=(features,), targets=targets, local_rows=len(targets)
        )

    def training_step(self, modules, batch, context):
        predictions = modules["model"](batch.positional[0])
        loss = torch.nn.functional.mse_loss(predictions, batch.targets, reduction="sum")
        return TorchStepResult(
            outputs={"prediction": predictions},
            loss=TorchLossContribution(loss, batch.local_rows),
        )

    def validation_step(self, modules, batch, context):
        return self.training_step(modules, batch, context)

    def configure_optimizers(self, modules, context):
        return TorchOptimizationPlan(
            torch.optim.SGD(modules["model"].parameters(), lr=0.1)
        )

    def metric_plan(self, context):
        return __import__(
            "tributo.algorithms", fromlist=["TorchMetricPlan"]
        ).TorchMetricPlan({"train_loss": "sum_count"})

    def artifact_plan(self, context):
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
            targets=(
                {
                    "name": "onnx-model",
                    "format": "onnx",
                    "exporter_id": "torch-onnx-v1",
                },
            ),
            roles={"inference": "onnx-model"},
        )


class AdapterExportFixture(RayTorchAdapter):
    """Minimal Adapter proving Core attaches its typed Artifact Plan."""

    def validate_environment(self, context: TorchRuntimeContext) -> None:
        del context

    def bind_datasets(self, datasets, context: TorchStageContext):
        del context
        return datasets

    def worker_config(self, context: TorchStageContext):
        del context
        return {}

    def train_loop_per_worker(
        self,
        worker_config,
        checkpoint_context: TorchWorkerCheckpointContext,
    ) -> None:
        del worker_config, checkpoint_context

    def checkpoint_source(
        self, result: object, context: TorchCheckpointContext
    ) -> object:
        del context
        return result.checkpoint

    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan:
        del context
        return TorchMetricPlan({"train_loss": "sum_count"})

    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan:
        if set(context.stage.runtime.algorithm_config) & {"ray", "output"}:
            raise AssertionError("Core control config leaked into Adapter context")
        if "train" not in context.stage.runtime.input_bindings:
            raise AssertionError(
                "InputBinding metadata is missing from Adapter context"
            )
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {"name": "features", "dtype": "float32", "shape": ("batch", 2)},
            ),
            output_signature=(
                {"name": "prediction", "dtype": "float32", "shape": ("batch", 1)},
            ),
            targets=({"name": "model", "format": "onnx"},),
            roles={"inference": "model"},
        )

    @contextmanager
    def open_export_source(
        self,
        checkpoint_ref: TorchCheckpointRef,
        artifact_context: TorchArtifactContext,
    ):
        del checkpoint_ref, artifact_context
        yield ExportSource(
            source_kind="torch_module",
            model_object=torch.nn.Linear(2, 1),
            sample_inputs={"features": torch.zeros((1, 2))},
        )


def _checkpoint(path: Path) -> Path:
    torch.manual_seed(7)
    model = BinaryLinearRecipe().build_modules(None)["model"]
    assert isinstance(model, torch.nn.Module)
    torch.save(model.state_dict(), path / "model.pt")
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "binary",
        "example.binary",
        CODE_DIGEST,
        "1" * 64,
        "2" * 64,
        plan_digest="2" * 64,
    )
    descriptor = TorchCheckpointDescriptor(
        schema_version=1,
        identity=identity,
        run_config_name=identity.run_config_name,
        state_layout="replicated",
        world_size=1,
        completed_step=1,
        policy_digest=identity.policy_digest,
        execution_plan_digest=identity.execution_plan_digest,
        input_binding_digest="3" * 64,
        implementation_code_digest=CODE_DIGEST,
        payload_files={
            "model.pt": hashlib.sha256((path / "model.pt").read_bytes()).hexdigest()
        },
    )
    (path / "torch_checkpoint_descriptor.json").write_text(
        json.dumps(descriptor.to_dict()), encoding="utf-8"
    )
    (path / "metrics.json").write_text(
        json.dumps({"train_loss": 0.5}), encoding="utf-8"
    )
    return path


def _adapter_checkpoint(path: Path) -> Path:
    (path / "model.pt").write_bytes(b"adapter-model")
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "adapter",
        "example.adapter",
        CODE_DIGEST,
        "1" * 64,
        "2" * 64,
        plan_digest="2" * 64,
    )
    descriptor = TorchCheckpointDescriptor(
        schema_version=1,
        identity=identity,
        run_config_name=identity.run_config_name,
        state_layout="component",
        world_size=1,
        completed_step=1,
        policy_digest=identity.policy_digest,
        execution_plan_digest=identity.execution_plan_digest,
        input_binding_digest="3" * 64,
        implementation_code_digest=CODE_DIGEST,
        payload_files={
            "model.pt": hashlib.sha256((path / "model.pt").read_bytes()).hexdigest()
        },
        adapter_identity=identity.implementation_id,
        resume_supported=False,
        same_world_size_resume=None,
    )
    (path / "torch_checkpoint_descriptor.json").write_text(
        json.dumps(descriptor.to_dict()), encoding="utf-8"
    )
    return path


def _options() -> TorchSourceOptions:
    return TorchSourceOptions(
        implementation_ref="tests.training.exporters.test_torch_recipe_source:BinaryLinearRecipe",
        implementation_code_digest=CODE_DIGEST,
        implementation_id="example.binary",
        policy_digest="1" * 64,
        plan_digest="2" * 64,
        input_binding_digest="3" * 64,
        algorithm_config={"model": {"input_features": 2}},
    )


def test_recipe_source_reconstructs_model_and_checkpoint_contract(
    tmp_path: Path,
) -> None:
    provider = RayTorchSourceProvider()

    with provider.open_source(_checkpoint(tmp_path), _options()) as source:
        assert isinstance(source.model_object, torch.nn.Module)
        assert source.source_kind == "torch_module"
        assert source.architecture_id == "example.binary"
        assert source.metadata["artifact_plan"]["roles"] == {"inference": "onnx-model"}


def test_recipe_source_rejects_plan_identity_drift(tmp_path: Path) -> None:
    provider = RayTorchSourceProvider()
    options = _options().model_copy(update={"implementation_id": "other.model"})

    with pytest.raises(Exception, match="identity"):
        with provider.open_source(_checkpoint(tmp_path), options):
            pass


def test_recipe_source_uses_existing_onnx_bundle_pipeline(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    bundle_dir = tmp_path / "bundle"
    provider = RayTorchSourceProvider()

    with provider.open_source(_checkpoint(checkpoint_dir), _options()) as source:
        result = BundleExportService().export_bundle(
            source,
            BundleOutputConfig(
                bundle_uri=str(bundle_dir),
                targets=[
                    ExportTarget(
                        name="onnx-model",
                        format="onnx",
                        exporter_id="torch-onnx-v1",
                        options={"opset": 18, "dynamo": False},
                    )
                ],
                roles={"inference": "onnx-model"},
            ),
            tributo_version="1.0.0",
        )

    assert result.status == "succeeded"
    assert (Path(result.canonical_uri) / "manifest.json").is_file()
    assert any(artifact.format == "onnx" for artifact in result.artifacts)


def test_adapter_source_provider_attaches_typed_artifact_plan(tmp_path: Path) -> None:
    provider = RayTorchSourceProvider()
    options = TorchSourceOptions(
        implementation_ref=(
            "tests.training.exporters.test_torch_recipe_source:AdapterExportFixture"
        ),
        implementation_code_digest=CODE_DIGEST,
        implementation_id="example.adapter",
        policy_digest="1" * 64,
        plan_digest="2" * 64,
        input_binding_digest="3" * 64,
        loop_owner="adapter",
        algorithm_config={
            "model": {"input_features": 2},
            "ray": {"storage_path": "/not-visible-to-adapter"},
            "output": {"bundle_uri": "/not-visible-to-adapter"},
        },
        input_bindings={"train": {"feature_names": ["features"]}},
        output_config={"bundle_uri": "/not-visible-to-worker"},
    )

    with provider.open_source(_adapter_checkpoint(tmp_path), options) as source:
        assert source.metadata["artifact_plan"]["roles"] == {"inference": "model"}

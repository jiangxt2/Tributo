"""Tests for generic Torch recipe checkpoint reconstruction and ONNX export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.torch_recipe import BinaryLinearRecipe
from tributo.exporting.models import BundleOutputConfig, ExportTarget
from tributo.exporting.service import BundleExportService
from tributo.integrations.algorithm_runtimes.torch_recipe import _checkpoint_contract
from tributo.integrations.sources.ray_torch_recipe import (
    RayTorchRecipeSourceProvider,
    TorchRecipeSourceOptions,
)

torch = pytest.importorskip("torch")


def _checkpoint(path: Path) -> Path:
    torch.manual_seed(7)
    model = BinaryLinearRecipe().model_factory({"input_features": 2})
    assert isinstance(model, torch.nn.Module)
    torch.save(model.state_dict(), path / "model.pt")
    payload = _checkpoint_contract(
        config={
            "model": {"input_features": 2},
            "_tributo_implementation_id": "example.binary_linear",
            "_tributo_algorithm": "binary_linear",
            "_tributo_feature_names": ["x1", "x2"],
            "_tributo_recipe_ref": ("tests.support.torch_recipe:BinaryLinearRecipe"),
            "_tributo_recipe_code_digest": None,
        },
        feature_count=2,
        output_shape=(1,),
        framework_version=torch.__version__,
    )
    (path / "model_config.json").write_text(json.dumps(payload), encoding="utf-8")
    (path / "metrics.json").write_text(
        json.dumps({"train_loss": 0.5}), encoding="utf-8"
    )
    return path


def _options() -> TorchRecipeSourceOptions:
    return TorchRecipeSourceOptions(
        recipe_ref="tests.support.torch_recipe:BinaryLinearRecipe",
        recipe_code_digest=None,
        implementation_id="example.binary_linear",
    )


def test_recipe_source_reconstructs_model_and_checkpoint_contract(
    tmp_path: Path,
) -> None:
    provider = RayTorchRecipeSourceProvider()

    with provider.open_source(_checkpoint(tmp_path), _options()) as source:
        assert isinstance(source.model_object, torch.nn.Module)
        assert source.source_kind == "torch_module"
        assert source.architecture_id == "example.binary_linear"
        assert source.model_config_data == {"input_features": 2}
        assert set(source.sample_inputs) == {"x1", "x2"}
        assert source.sample_inputs["x1"].shape == (2,)
        assert source.feature_schema["feature_names"] == ["x1", "x2"]
        assert source.checkpoint_contract is not None
        assert source.checkpoint_contract.trainer_type == "torch_recipe"


def test_recipe_source_rejects_plan_identity_drift(tmp_path: Path) -> None:
    provider = RayTorchRecipeSourceProvider()
    options = _options().model_copy(update={"implementation_id": "other.model"})

    with pytest.raises(ValueError, match="implementation identity"):
        with provider.open_source(_checkpoint(tmp_path), options):
            pass


def test_recipe_source_uses_existing_onnx_bundle_pipeline(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    bundle_dir = tmp_path / "bundle"
    provider = RayTorchRecipeSourceProvider()

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

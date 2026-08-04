"""DNN and PU trainer → Bundle → BundleReader vertical slices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.integration.test_export_helpers import (
    _assert_onnx_signature_matches_manifest,
)

pytestmark = pytest.mark.integration


def _bundle_config(bundle_root: Path, target_name: str) -> Any:
    from tributo.exporting.models import BundleOutputConfig, ExportTarget

    return BundleOutputConfig(
        bundle_uri=str(bundle_root),
        targets=[
            ExportTarget(
                name=target_name,
                format="onnx",
                options={"opset": 18},
            )
        ],
        roles={"inference": target_name},
    )


def _training_config(storage_path: Path) -> dict[str, Any]:
    return {
        "features": [
            {
                "name": "age",
                "type": "dense",
                "dimension": 1,
                "norm": "none",
            }
        ],
        "model": {
            "dnn_hidden_units": [4],
            "dnn_dropout": 0.0,
            "use_batch_norm": False,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "val_size": 0.0,
        },
        "ray": {
            "num_workers": 1,
            "use_gpu": False,
            "storage_path": str(storage_path),
        },
    }


def _write_torch_checkpoint(
    checkpoint_dir: Path,
    *,
    trainer_type: str,
) -> None:
    """Write the smallest contract-compliant DNN-family checkpoint."""
    torch = pytest.importorskip("torch")
    from tributo.training.dnn_trainer import build_export_checkpoint_config
    from tributo.training.features.column_types import features_from_dicts
    from tributo.training.models.dnn import DNNModel

    feature_configs = [{"name": "age", "dimension": 1, "norm": "none"}]
    features = features_from_dicts(feature_configs)
    model_config = {"dnn_hidden_units": [4], "dnn_dropout": 0.0}
    model = DNNModel(features, **model_config)

    checkpoint_dir.mkdir()
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    metadata = build_export_checkpoint_config(
        feature_configs,
        model_config,
        trainer_type=trainer_type,
        task_type="pu_classification" if trainer_type == "pu" else "classification",
        framework_version=torch.__version__,
        extra_metadata=(
            {"pu": {"enabled": True, "class_prior": 0.5}}
            if trainer_type == "pu"
            else None
        ),
    )
    (checkpoint_dir / "model_config.json").write_text(json.dumps(metadata))
    (checkpoint_dir / "preprocessor.json").write_text(
        json.dumps({"features": feature_configs})
    )


def _assert_typed_manifest(manifest: Any) -> None:
    """Require non-empty typed signatures in the final published Manifest."""
    assert manifest.input_signature is not None
    assert manifest.output_signature is not None

    fields = (
        *manifest.input_signature.input_fields,
        *manifest.output_signature.output_fields,
    )
    assert fields
    assert all(field.name and field.dtype and field.shape for field in fields)


@pytest.mark.slow
def test_dnn_trainer_run_publishes_typed_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise DNNTrainerImpl.run through publication and BundleReader."""
    from tributo.exporting.bundle_reader import BundleReader
    from tributo.training.dnn_trainer import DNNTrainerImpl

    checkpoint_dir = tmp_path / "dnn-checkpoint"
    _write_torch_checkpoint(checkpoint_dir, trainer_type="dnn")
    trainer = DNNTrainerImpl(
        datasets={},
        config=_training_config(tmp_path / "dnn-ray-results"),
    )
    monkeypatch.setattr(trainer, "setup", lambda: None)
    monkeypatch.setattr(trainer, "training_loop", lambda: str(checkpoint_dir))

    summary = trainer.run(
        bundle_config=_bundle_config(tmp_path / "dnn-bundles", "dnn")
    )

    assert summary["status"] in ("succeeded", "partial")
    assert summary["canonical_uri"]
    reader = BundleReader()
    manifest = reader.read_manifest(summary["canonical_uri"])
    _assert_typed_manifest(manifest)
    with reader.open_artifact(summary["canonical_uri"], role="inference") as artifact:
        _assert_onnx_signature_matches_manifest(
            artifact.path_for("model.onnx"), manifest
        )


@pytest.mark.slow
def test_pu_trainer_run_publishes_typed_manifest_and_prior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise PUTrainerImpl.run and preserve the effective class prior."""
    from tributo.exporting.bundle_reader import BundleReader
    from tributo.training.pu_trainer import PUTrainerImpl

    config = _training_config(tmp_path / "pu-ray-results")
    config["pu"] = {
        "loss_type": "nnpu",
        "class_prior": 0.5,
        "class_prior_method": "label_frequency",
        "beta": 0.0,
        "gamma": 1.0,
    }

    checkpoint_dir = tmp_path / "pu-checkpoint"
    _write_torch_checkpoint(checkpoint_dir, trainer_type="pu")
    trainer = PUTrainerImpl(datasets={}, config=config)
    monkeypatch.setattr(trainer, "setup", lambda: None)
    monkeypatch.setattr(trainer, "training_loop", lambda: str(checkpoint_dir))
    summary = trainer.run(
        bundle_config=_bundle_config(tmp_path / "pu-bundles", "pu")
    )

    assert summary["status"] in ("succeeded", "partial")
    assert summary["canonical_uri"]
    reader = BundleReader()
    manifest = reader.read_manifest(summary["canonical_uri"])
    _assert_typed_manifest(manifest)
    with reader.open_artifact(summary["canonical_uri"], role="inference") as artifact:
        _assert_onnx_signature_matches_manifest(
            artifact.path_for("model.onnx"), manifest
        )

    with reader.open_artifact(summary["canonical_uri"], role="inference") as artifact:
        model_config = json.loads(
            artifact.path_for("model_config.json").read_text()
        )
    assert model_config["pu"]["class_prior"] == pytest.approx(0.5)

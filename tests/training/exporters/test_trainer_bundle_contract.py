"""Contract tests for the E2 Trainer → Bundle adapters."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tributo.exporting.conftest import ExportSourceProviderConformanceTest
from tributo.exporting.models import CheckpointField, ExportCheckpointV1
from tributo.integrations.sources.huggingface import HuggingFaceSourceProvider
from tributo.integrations.sources.ray_dnn import RayDnnSourceProvider
from tributo.integrations.sources.ray_pu import RayPUSourceProvider
from tributo.integrations.sources.ray_xgboost import RayXGBoostSourceProvider
from tributo.training.base import BaseTrainer
from tributo.training.dnn_trainer import build_export_checkpoint_config


class TestExportCheckpointV1:
    def test_typed_fields_convert_to_manifest_signatures(self) -> None:
        contract = ExportCheckpointV1(
            trainer_type="dnn",
            architecture_id="dnn",
            input_schema=(
                CheckpointField(name="age", dtype="float32", shape=("batch",)),
            ),
            output_schema=(
                CheckpointField(name="output", dtype="float32", shape=("batch",)),
            ),
            preprocessing={"artifact": "preprocessor.json"},
            task_type="classification",
            framework="pytorch",
            framework_version="2.5.0",
            checkpoint_format_version=1,
            required_artifacts=(
                "model.pt",
                "model_config.json",
                "preprocessor.json",
            ),
        )

        input_signature, output_signature = contract.to_manifest_signatures()
        assert input_signature.input_fields[0].name == "age"
        assert input_signature.input_fields[0].dtype == "float32"
        assert input_signature.input_fields[0].shape == ("batch",)
        assert output_signature.output_fields[0].name == "output"

    def test_dnn_checkpoint_metadata_excludes_resume_state(self) -> None:
        metadata = build_export_checkpoint_config(
            [{"name": "age", "dimension": 1, "norm": "none"}],
            {"dnn_hidden_units": [4], "dnn_dropout": 0.0},
            trainer_type="dnn",
            task_type="classification",
            framework_version="2.5.0",
        )

        assert metadata["architecture_id"] == "dnn"
        assert metadata["input_schema"][0]["dtype"] == "float32"
        assert metadata["output_schema"][0]["dtype"] == "float32"
        assert metadata["required_artifacts"] == [
            "model.pt",
            "model_config.json",
            "preprocessor.json",
        ]
        assert "optimizer" not in metadata
        assert "epoch" not in metadata
        assert "rng" not in metadata

    def test_resume_only_fields_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="resume-only"):
            ExportCheckpointV1(
                trainer_type="dnn",
                architecture_id="dnn",
                input_schema=(CheckpointField(name="x", dtype="float32"),),
                output_schema=(CheckpointField(name="y", dtype="float32"),),
                task_type="classification",
                framework="pytorch",
                framework_version="2.5.0",
                optimizer={"type": "adam"},
            )


class TestTrainerTypeDeclarations:
    def test_builtin_trainers_declare_explicit_types(self) -> None:
        from tributo.training.dnn_trainer import DNNTrainerImpl
        from tributo.training.pu_trainer import PUTrainerImpl
        from tributo.training.xgboost_trainer import XGBoostTrainerImpl

        assert DNNTrainerImpl._get_trainer_type() == "dnn"
        assert PUTrainerImpl._get_trainer_type() == "pu"
        assert XGBoostTrainerImpl._get_trainer_type() == "xgboost"

    def test_base_trainer_fails_fast_without_a_type(self) -> None:
        class _UndeclaredTrainer(BaseTrainer):
            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                return None

        with pytest.raises(NotImplementedError, match="_get_trainer_type"):
            _UndeclaredTrainer._get_trainer_type()


def _write_dnn_checkpoint(
    checkpoint_dir: Path,
    *,
    trainer_type: str,
) -> None:
    torch = pytest.importorskip("torch")
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
        extra_metadata={"pu": {"enabled": True}} if trainer_type == "pu" else None,
    )
    (checkpoint_dir / "model_config.json").write_text(json.dumps(metadata))
    (checkpoint_dir / "preprocessor.json").write_text(
        json.dumps({"features": feature_configs})
    )


class TestTorchSourceProviders:
    @pytest.mark.parametrize(
        ("trainer_type", "provider_module", "provider_name"),
        (
            ("dnn", "ray_dnn", "RayDnnSourceProvider"),
            ("pu", "ray_pu", "RayPUSourceProvider"),
        ),
    )
    def test_checkpoint_contract_is_loaded_for_dnn_family(
        self,
        tmp_path: Path,
        trainer_type: str,
        provider_module: str,
        provider_name: str,
    ) -> None:
        torch = pytest.importorskip("torch")
        module = __import__(
            f"tributo.integrations.sources.{provider_module}",
            fromlist=[provider_name],
        )
        provider = getattr(module, provider_name)()
        checkpoint_dir = tmp_path / trainer_type
        _write_dnn_checkpoint(checkpoint_dir, trainer_type=trainer_type)

        with provider.open_source(str(checkpoint_dir)) as source:
            assert source.checkpoint_contract is not None
            assert source.checkpoint_contract.trainer_type == trainer_type
            assert source.checkpoint_contract.architecture_id == "dnn"
            assert source.preprocessing_state == {
                "features": [{"name": "age", "dimension": 1, "norm": "none"}]
            }
            assert source.sample_inputs["age"].dtype == torch.float32
            assert source.sample_inputs["age"].shape == (2,)
            with torch.no_grad():
                output = source.model_object(source.sample_inputs["age"])
            assert output.shape == (2,)


class TestTorchSourceValidation:
    def test_unsupported_checkpoint_dtype_fails_fast(self) -> None:
        pytest.importorskip("torch")

        from tributo.integrations.sources.ray_dnn import _torch_dtype

        with pytest.raises(ValueError, match="Unsupported checkpoint dtype"):
            _torch_dtype("float128")

    def test_missing_model_artifact_is_reported_by_contract(
        self,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("torch")

        from tributo.integrations.sources.ray_dnn import RayDnnSourceProvider

        checkpoint_dir = tmp_path / "missing-model"
        _write_dnn_checkpoint(checkpoint_dir, trainer_type="dnn")
        (checkpoint_dir / "model.pt").unlink()

        with pytest.raises(
            FileNotFoundError,
            match="Required checkpoint artifact 'model.pt' is missing",
        ):
            with RayDnnSourceProvider().open_source(str(checkpoint_dir)):
                pass


class _TemporaryProviderResult:
    """Keep a generated checkpoint alive for one conformance test instance."""

    _temporary_directory: tempfile.TemporaryDirectory[str]

    def checkpoint_dir(self, name: str) -> Path:
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix=f"tributo-{name}-conformance-"
        )
        return Path(self._temporary_directory.name) / name


class TestRayXGBoostSourceProviderConformance(
    _TemporaryProviderResult,
    ExportSourceProviderConformanceTest,
):
    provider_cls = RayXGBoostSourceProvider

    def make_result(self) -> str:
        np = pytest.importorskip("numpy")
        xgb = pytest.importorskip("xgboost")
        matrix = xgb.DMatrix(
            np.array([[0.0], [1.0]], dtype=np.float32),
            label=np.array([0, 1]),
        )
        booster = xgb.train(
            {"objective": "binary:logistic", "nthread": 1},
            matrix,
            num_boost_round=1,
        )
        checkpoint = self.checkpoint_dir("xgboost")
        checkpoint.mkdir()
        booster.save_model(str(checkpoint / "model.json"))
        return str(checkpoint)


class TestRayDnnSourceProviderConformance(
    _TemporaryProviderResult,
    ExportSourceProviderConformanceTest,
):
    provider_cls = RayDnnSourceProvider

    def make_result(self) -> str:
        checkpoint = self.checkpoint_dir("dnn")
        _write_dnn_checkpoint(checkpoint, trainer_type="dnn")
        return str(checkpoint)


class TestRayPUSourceProviderConformance(
    _TemporaryProviderResult,
    ExportSourceProviderConformanceTest,
):
    provider_cls = RayPUSourceProvider

    def make_result(self) -> str:
        checkpoint = self.checkpoint_dir("pu")
        _write_dnn_checkpoint(checkpoint, trainer_type="pu")
        return str(checkpoint)


class _FakeHuggingFaceConfig:
    def to_dict(self) -> dict[str, str]:
        return {"model_type": "test-transformer"}


class TestHuggingFaceSourceProviderConformance(ExportSourceProviderConformanceTest):
    provider_cls = HuggingFaceSourceProvider

    def make_result(self) -> tuple[SimpleNamespace, object]:
        pytest.importorskip("transformers")
        model = SimpleNamespace(
            name_or_path="test-model",
            config=_FakeHuggingFaceConfig(),
        )
        return model, object()

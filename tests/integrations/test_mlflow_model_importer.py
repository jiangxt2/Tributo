"""Unit tests for immutable MLflow version and Alias resolution."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.serving.bundle_fixtures import build_test_bundle
from tributo.exceptions import JobConfigurationError
from tributo.exporting.bundle_reader import BundleReader
from tributo.inference.contracts import RegistryModelReference
from tributo.integrations.model_importers.mlflow import MLflowModelImporter


class _Client:
    calls: list[tuple[str, ...]] = []
    source: str = ""

    def __init__(self, *, tracking_uri=None, registry_uri=None) -> None:
        type(self).calls.append(("init", str(tracking_uri), str(registry_uri)))

    def get_model_version_by_alias(self, name: str, alias: str):
        type(self).calls.append(("alias", name, alias))
        return SimpleNamespace(version="7")

    def get_model_version(self, name: str, version: str):
        type(self).calls.append(("version", name, version))
        return SimpleNamespace(version=version, source=self.source, run_id="run-7")


def _install_fake_mlflow(monkeypatch, source: Path) -> None:
    _Client.calls = []
    _Client.source = str(source)
    mlflow = ModuleType("mlflow")
    mlflow.MlflowClient = _Client
    mlflow.artifacts = SimpleNamespace(
        download_artifacts=lambda **kwargs: kwargs["artifact_uri"]
    )
    models = ModuleType("mlflow.models")
    models.Model = object
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.models", models)


def test_alias_is_frozen_to_numeric_version_before_bundle_import(
    tmp_path: Path, monkeypatch
) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        alias="champion",
        import_bundle_uri=str(tmp_path / "imports"),
        options={"tracking_uri": "http://mlflow.test:5000"},
    )

    bundle_ref = MLflowModelImporter().import_model(reference)
    manifest = BundleReader().read_manifest(bundle_ref.canonical_uri)

    assert _Client.calls[1:] == [
        ("alias", "classifier", "champion"),
        ("version", "classifier", "7"),
    ]
    assert "version=7" in manifest.source_info.source_fingerprint
    assert "run=run-7" in manifest.source_info.source_fingerprint


def test_numeric_version_never_calls_alias_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
    )

    MLflowModelImporter().import_model(reference)

    assert not any(call[0] == "alias" for call in _Client.calls)
    assert ("version", "classifier", "9") in _Client.calls


def test_unimplemented_mlflow_credential_profile_is_not_silently_ignored(
    tmp_path: Path,
) -> None:
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
        storage_profile="registry-credentials",
    )

    with pytest.raises(JobConfigurationError, match="environment chain"):
        MLflowModelImporter().import_model(reference)


@pytest.mark.parametrize(
    "tracking_uri",
    [
        "https://user:must-not-leak@mlflow.example",
        "https://mlflow.example?api_token=must-not-leak",
    ],
)
def test_connection_uri_rejects_plaintext_credentials_before_sdk_access(
    tmp_path: Path, tracking_uri: str
) -> None:
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
        options={"tracking_uri": tracking_uri},
    )

    with pytest.raises(JobConfigurationError, match="must not contain") as error:
        MLflowModelImporter().import_model(reference)

    assert "must-not-leak" not in str(error.value)

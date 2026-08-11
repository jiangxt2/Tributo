"""Unit tests for immutable MLflow version and Alias resolution."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace

import pytest

from tests.serving.bundle_fixtures import build_test_bundle
from tributo.exceptions import JobConfigurationError
from tributo.exporting.bundle_reader import BundleReader
from tributo.inference.contracts import RegistryModelReference
from tributo.integrations.model_importers.mlflow import MLflowModelImporter


class _Client:
    calls: list[tuple[str, ...]] = []
    source: Path

    def __init__(self, *, tracking_uri=None, registry_uri=None) -> None:
        type(self).calls.append(("init", str(tracking_uri), str(registry_uri)))

    def get_model_version_by_alias(self, name: str, alias: str):
        type(self).calls.append(("alias", name, alias))
        return SimpleNamespace(version="7")

    def get_model_version(self, name: str, version: str):
        type(self).calls.append(("version", name, version))
        return SimpleNamespace(
            version=version,
            source="runs:/run-7/model",
            run_id="run-7",
        )

    def list_artifacts(self, run_id: str, path: str):
        type(self).calls.append(("list", run_id, path))
        relative = PurePosixPath(path).relative_to("model")
        local = self.source.joinpath(*relative.parts)
        return [
            SimpleNamespace(
                path=f"{path}/{child.name}",
                is_dir=child.is_dir(),
                file_size=None if child.is_dir() else child.stat().st_size,
            )
            for child in sorted(local.iterdir())
        ]

    def download_artifacts(self, run_id: str, path: str, dst_path: str):
        type(self).calls.append(("download", run_id, path))
        destination = Path(dst_path) / "model"
        shutil.copytree(self.source, destination)
        return str(destination)


def _install_fake_mlflow(monkeypatch, source: Path) -> None:
    _Client.calls = []
    _Client.source = source
    mlflow = ModuleType("mlflow")
    mlflow.MlflowClient = _Client
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

    assert _Client.calls[1:3] == [
        ("alias", "classifier", "champion"),
        ("version", "classifier", "7"),
    ]
    assert len(manifest.source_info.source_fingerprint) == 64
    assert "run-7" not in manifest.source_info.source_fingerprint

    numeric_ref = reference.model_copy(
        update={
            "alias": None,
            "version": "9",
            "import_bundle_uri": str(tmp_path / "numeric-imports"),
        }
    )
    numeric_bundle = MLflowModelImporter().import_model(numeric_ref)
    numeric_manifest = BundleReader().read_manifest(numeric_bundle.canonical_uri)
    assert (
        numeric_manifest.source_info.source_fingerprint
        != manifest.source_info.source_fingerprint
    )


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


def test_artifact_tree_is_bounded_before_download(tmp_path: Path, monkeypatch) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
        options={"max_files": 1},
    )

    with pytest.raises(JobConfigurationError, match="entry limit"):
        MLflowModelImporter().import_model(reference)

    assert not any(call[0] == "download" for call in _Client.calls)


def test_artifact_listing_cannot_escape_requested_root(
    tmp_path: Path, monkeypatch
) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)
    monkeypatch.setattr(
        _Client,
        "list_artifacts",
        lambda self, run_id, path: [
            SimpleNamespace(
                path="outside/secret",
                is_dir=False,
                file_size=1,
            )
        ],
    )
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
    )

    with pytest.raises(JobConfigurationError, match="escaped"):
        MLflowModelImporter().import_model(reference)


def test_download_cannot_escape_staging_directory(tmp_path: Path, monkeypatch) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(
        _Client,
        "download_artifacts",
        lambda self, run_id, path, dst_path: str(outside),
    )
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
    )

    with pytest.raises(JobConfigurationError, match="escaped its staging"):
        MLflowModelImporter().import_model(reference)


def test_download_must_resolve_to_model_directory(tmp_path: Path, monkeypatch) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)

    def _download_file(self, run_id: str, path: str, dst_path: str) -> str:
        del self, run_id, path
        destination = Path(dst_path) / "model.onnx"
        shutil.copy2(
            next(item for item in source.rglob("*") if item.is_file()), destination
        )
        return str(destination)

    monkeypatch.setattr(_Client, "download_artifacts", _download_file)
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
    )

    with pytest.raises(JobConfigurationError, match="model directory"):
        MLflowModelImporter().import_model(reference)


def test_download_tree_rejects_symlinks(tmp_path: Path, monkeypatch) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)
    original_download = _Client.download_artifacts

    def _download_with_symlink(self, run_id: str, path: str, dst_path: str) -> str:
        destination = Path(original_download(self, run_id, path, dst_path))
        target = next(item for item in destination.rglob("*") if item.is_file())
        (destination / "unexpected-link").symlink_to(target)
        return str(destination)

    monkeypatch.setattr(_Client, "download_artifacts", _download_with_symlink)
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
    )

    with pytest.raises(JobConfigurationError, match="must not contain symlinks"):
        MLflowModelImporter().import_model(reference)


def test_download_tree_must_match_preflight_file_set(
    tmp_path: Path, monkeypatch
) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)
    original_download = _Client.download_artifacts

    def _download_with_extra_file(self, run_id: str, path: str, dst_path: str) -> str:
        destination = Path(original_download(self, run_id, path, dst_path))
        (destination / "unexpected.bin").write_bytes(b"unexpected")
        return str(destination)

    monkeypatch.setattr(_Client, "download_artifacts", _download_with_extra_file)
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
    )

    with pytest.raises(JobConfigurationError, match="changed between"):
        MLflowModelImporter().import_model(reference)


def test_download_tree_must_match_preflight_file_sizes(
    tmp_path: Path, monkeypatch
) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)
    original_download = _Client.download_artifacts

    def _download_with_changed_file(self, run_id: str, path: str, dst_path: str) -> str:
        destination = Path(original_download(self, run_id, path, dst_path))
        target = next(item for item in destination.rglob("*") if item.is_file())
        target.write_bytes(target.read_bytes() + b"changed")
        return str(destination)

    monkeypatch.setattr(_Client, "download_artifacts", _download_with_changed_file)
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
    )

    with pytest.raises(JobConfigurationError, match="changed between"):
        MLflowModelImporter().import_model(reference)


def test_download_tree_enforces_actual_size_limit(tmp_path: Path, monkeypatch) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)

    def _list_with_false_zero_sizes(self, run_id: str, path: str):
        del run_id
        relative = PurePosixPath(path).relative_to("model")
        local = self.source.joinpath(*relative.parts)
        return [
            SimpleNamespace(
                path=f"{path}/{child.name}",
                is_dir=child.is_dir(),
                file_size=None if child.is_dir() else 0,
            )
            for child in sorted(local.iterdir())
        ]

    monkeypatch.setattr(_Client, "list_artifacts", _list_with_false_zero_sizes)
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
        options={"max_total_bytes": 1},
    )

    with pytest.raises(JobConfigurationError, match="model size"):
        MLflowModelImporter().import_model(reference)


def test_artifact_tree_depth_is_bounded_before_download(
    tmp_path: Path, monkeypatch
) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
        options={"max_depth": 1},
    )

    with pytest.raises(JobConfigurationError, match="depth limit"):
        MLflowModelImporter().import_model(reference)

    assert not any(call[0] == "download" for call in _Client.calls)


def test_non_runs_model_source_is_rejected_without_echoing_uri(
    tmp_path: Path, monkeypatch
) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)
    secret = "must-not-leak"
    monkeypatch.setattr(
        _Client,
        "get_model_version",
        lambda self, name, version: SimpleNamespace(
            version=version,
            source=f"https://registry.invalid/model?token={secret}",
            run_id="run-7",
        ),
    )
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
    )

    with pytest.raises(JobConfigurationError, match="must not contain") as error:
        MLflowModelImporter().import_model(reference)

    assert secret not in str(error.value)


def test_sdk_error_is_classified_without_secret_details(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    source = build_test_bundle(tmp_path / "source")
    _install_fake_mlflow(monkeypatch, source)
    secret = "eyJhbGciOiJIUzI1NiJ9.must-not-leak.signature"

    def _fail(self, name, version):
        raise RuntimeError(f"permission denied: Authorization: Bearer {secret}")

    monkeypatch.setattr(_Client, "get_model_version", _fail)
    reference = RegistryModelReference(
        provider_id="mlflow.v2",
        model_name="classifier",
        version="9",
        import_bundle_uri=str(tmp_path / "imports"),
    )

    with (
        caplog.at_level(
            logging.WARNING,
            logger="tributo.integrations.model_importers.mlflow",
        ),
        pytest.raises(JobConfigurationError, match="RuntimeError") as error,
    ):
        MLflowModelImporter().import_model(reference)

    assert secret not in str(error.value)
    assert "numeric version resolution" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "permission denied" in caplog.text
    assert secret not in caplog.text


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

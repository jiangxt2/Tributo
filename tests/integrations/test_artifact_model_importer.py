"""Unit tests for explicit local/S3 artifact normalization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tributo.exceptions import JobConfigurationError
from tributo.exporting.bundle_reader import BundleReader
from tributo.inference.contracts import ArtifactModelReference
from tributo.inference.importers import build_default_model_importer_registry
from tributo.integrations.model_importers.artifact import ArtifactModelImporter


def _reference(source: Path, destination: Path, **updates) -> ArtifactModelReference:
    values = {
        "provider_id": "tributo.artifact",
        "uri": str(source),
        "format_id": "onnx",
        "flavor_id": "onnx-runtime-v1",
        "import_bundle_uri": str(destination),
        "expected_sha256": (
            hashlib.sha256(source.read_bytes()).hexdigest()
            if source.is_file()
            else None
        ),
        "options": {
            "variant": "onnx",
            "input_fields": [
                {"name": "float_input", "dtype": "float32", "shape": ["batch", 2]}
            ],
            "output_fields": [
                {"name": "score", "dtype": "float32", "shape": ["batch", 1]}
            ],
        },
    }
    values.update(updates)
    return ArtifactModelReference(**values)


def test_local_artifact_is_published_as_a_pinned_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "weights.bin"
    source.write_bytes(b"explicit-model-bytes")
    reference = _reference(source, tmp_path / "imports")
    monkeypatch.setattr(
        "tributo.integrations.model_importers._bundle._get_tributo_version",
        lambda: "9.8.7",
    )

    bundle_ref = ArtifactModelImporter().import_model(reference)
    manifest = BundleReader().read_manifest(bundle_ref.canonical_uri)

    assert manifest.bundle_id == bundle_ref.bundle_id
    assert manifest.roles == {"inference": "imported-model"}
    assert manifest.artifacts[0].format == "onnx"
    assert manifest.artifacts[0].flavor_id == "onnx-runtime-v1"
    assert manifest.tributo_version == "9.8.7"
    assert manifest.input_signature.input_fields[0].name == "float_input"
    assert (
        hashlib.sha256(
            (Path(bundle_ref.canonical_uri) / "manifest.json").read_bytes()
        ).hexdigest()
        == bundle_ref.manifest_sha256
    )


def test_artifact_digest_mismatch_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "weights.onnx"
    source.write_bytes(b"model")
    reference = _reference(
        source,
        tmp_path / "imports",
        expected_sha256="f" * 64,
    )

    with pytest.raises(JobConfigurationError, match="digest mismatch"):
        ArtifactModelImporter().import_model(reference)


def test_same_bytes_with_different_provenance_get_distinct_bundle_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.onnx"
    source.write_bytes(b"same-model")
    destination = tmp_path / "imports"

    first = ArtifactModelImporter().import_model(
        _reference(source, destination, architecture_id="architecture-a")
    )
    second = ArtifactModelImporter().import_model(
        _reference(source, destination, architecture_id="architecture-b")
    )

    assert first.bundle_id != second.bundle_id
    assert first.canonical_uri != second.canonical_uri
    first_manifest = BundleReader().read_manifest(first.canonical_uri)
    second_manifest = BundleReader().read_manifest(second.canonical_uri)
    assert first_manifest.source_info.architecture_id == "architecture-a"
    assert second_manifest.source_info.architecture_id == "architecture-b"


def test_same_bytes_from_different_sources_get_distinct_bundle_identity(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "source-a" / "weights.onnx"
    second_source = tmp_path / "source-b" / "weights.onnx"
    first_source.parent.mkdir()
    second_source.parent.mkdir()
    first_source.write_bytes(b"same-model")
    second_source.write_bytes(b"same-model")
    destination = tmp_path / "imports"

    first = ArtifactModelImporter().import_model(_reference(first_source, destination))
    second = ArtifactModelImporter().import_model(
        _reference(second_source, destination)
    )

    assert first.bundle_id != second.bundle_id
    first_fingerprint = (
        BundleReader().read_manifest(first.canonical_uri).source_info.source_fingerprint
    )
    second_fingerprint = (
        BundleReader()
        .read_manifest(second.canonical_uri)
        .source_info.source_fingerprint
    )
    assert first_fingerprint != second_fingerprint
    assert str(first_source) not in first_fingerprint


def test_s3_artifact_is_bounded_before_download_and_uses_source_profile(
    tmp_path: Path,
) -> None:
    payload = b"s3-model"
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": len(payload)}
    client.download_file.side_effect = lambda bucket, key, destination: Path(
        destination
    ).write_bytes(payload)
    reference = _reference(
        tmp_path / "not-local.onnx",
        tmp_path / "imports",
        uri="s3://external-models/model.onnx",
        storage_profile="model-source-domain",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    with patch(
        "tributo.integrations.model_importers.artifact.get_boto3_client",
        return_value=client,
    ) as get_client:
        bundle_ref = ArtifactModelImporter().import_model(reference)

    assert Path(bundle_ref.canonical_uri).is_dir()
    client.head_object.assert_called_once_with(
        Bucket="external-models", Key="model.onnx"
    )
    client.download_file.assert_called_once()
    assert get_client.call_args.kwargs["profile_name"] == "model-source-domain"


def test_s3_artifact_size_limit_fails_before_download(tmp_path: Path) -> None:
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 5}
    reference = _reference(
        tmp_path / "not-local.onnx",
        tmp_path / "imports",
        uri="s3://external-models/model.onnx",
        expected_sha256=None,
        options={
            "variant": "onnx",
            "input_fields": [{"name": "x", "dtype": "float32"}],
            "output_fields": [{"name": "y", "dtype": "float32"}],
            "max_bytes": 4,
        },
    )

    with patch(
        "tributo.integrations.model_importers.artifact.get_boto3_client",
        return_value=client,
    ):
        with pytest.raises(JobConfigurationError, match="exceeds configured limit"):
            ArtifactModelImporter().import_model(reference)

    client.download_file.assert_not_called()


def test_s3_artifact_size_is_rechecked_after_download(tmp_path: Path) -> None:
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 4}
    client.download_file.side_effect = lambda bucket, key, destination: Path(
        destination
    ).write_bytes(b"12345")
    reference = _reference(
        tmp_path / "not-local.onnx",
        tmp_path / "imports",
        uri="s3://external-models/model.onnx",
        expected_sha256=None,
        options={
            "variant": "onnx",
            "input_fields": [{"name": "x", "dtype": "float32"}],
            "output_fields": [{"name": "y", "dtype": "float32"}],
            "max_bytes": 4,
        },
    )

    with patch(
        "tributo.integrations.model_importers.artifact.get_boto3_client",
        return_value=client,
    ):
        with pytest.raises(JobConfigurationError, match="Acquired artifact size"):
            ArtifactModelImporter().import_model(reference)


def test_safetensors_without_architecture_is_rejected_before_acquisition(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.safetensors"
    reference = _reference(
        missing,
        tmp_path / "imports",
        expected_sha256=None,
        format_id="safetensors",
        flavor_id="safetensors-v1",
        options={
            "input_fields": [{"name": "x", "dtype": "float32"}],
            "output_fields": [{"name": "y", "dtype": "float32"}],
        },
    )

    with pytest.raises(JobConfigurationError, match="weights-only"):
        ArtifactModelImporter().import_model(reference)


def test_xgboost_native_requires_canonical_input_name_before_acquisition(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.ubj"
    reference = _reference(
        missing,
        tmp_path / "imports",
        expected_sha256=None,
        format_id="xgboost",
        flavor_id="xgboost-native-v1",
        architecture_id="xgboost",
        options={
            "variant": "ubj",
            "input_fields": [
                {"name": "features", "dtype": "float32", "shape": ["batch", 2]}
            ],
            "output_fields": [
                {"name": "prediction", "dtype": "float32", "shape": ["batch", 1]}
            ],
        },
    )

    with pytest.raises(JobConfigurationError, match="named 'float_input'"):
        ArtifactModelImporter().import_model(reference)


def test_default_registry_uses_exact_first_party_ids() -> None:
    registry = build_default_model_importer_registry()

    assert registry.get("tributo.artifact") is ArtifactModelImporter
    assert registry.get("mlflow.v2").provider_id == "mlflow.v2"
    with pytest.raises(ValueError, match="Unknown ModelImporter"):
        registry.get("artifact")

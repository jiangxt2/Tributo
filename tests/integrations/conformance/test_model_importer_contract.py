"""Conformance harness for explicit external ModelImporter adapters."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from tributo.exceptions import JobConfigurationError, UnsupportedArtifactFormat
from tributo.exporting.models import BundleRef
from tributo.inference.contracts import ArtifactModelReference
from tributo.integrations.model_importers import (
    ModelImporter,
    ModelImporterRegistry,
    build_default_model_importer_registry,
)


class _Options(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    conversion: str = "onnx"


class _FakeImporter:
    api_version = 1
    provider_id = "conformance.artifact"
    options_model = _Options
    reference_kinds = ("artifact",)
    uri_schemes = ("file", "s3")
    credential_profile_types = ("source-storage", "bundle-storage")
    capabilities = ("acquire", "content-digest", "bundle-import")

    def import_model(self, reference):
        assert reference.expected_sha256 is not None
        return BundleRef(
            canonical_uri=reference.import_bundle_uri,
            bundle_id="bundle-conformance",
            manifest_sha256="b" * 64,
        )


def assert_model_importer_conformance(importer_cls: type[ModelImporter]) -> None:
    assert importer_cls.api_version == 1
    assert importer_cls.provider_id
    assert importer_cls.reference_kinds
    assert importer_cls.uri_schemes
    assert importer_cls.credential_profile_types
    assert importer_cls.capabilities
    assert importer_cls.options_model.model_config.get("extra") == "forbid"

    registry = ModelImporterRegistry()
    registry.register(importer_cls)
    reference = ArtifactModelReference(
        provider_id=importer_cls.provider_id,
        uri="file:///models/model.onnx",
        format_id="onnx",
        flavor_id="onnx-runtime-v1",
        expected_sha256="a" * 64,
        import_bundle_uri="file:///bundles/imported",
        options={"conversion": "onnx"},
    )
    result = registry.import_model(reference)
    repeated = registry.import_model(reference)
    assert isinstance(result, BundleRef)
    assert result.manifest_sha256 == "b" * 64
    assert repeated == result
    assert "credential" not in result.model_dump_json()

    invalid = reference.model_copy(update={"options": {"unknown": True}})
    try:
        registry.import_model(invalid)
    except ValidationError:
        pass
    else:
        raise AssertionError("Importer options model must reject unknown fields")


def test_fake_model_importer_runs_full_conformance_suite() -> None:
    assert_model_importer_conformance(_FakeImporter)


def test_artifact_uri_scheme_matching_is_case_insensitive() -> None:
    registry = ModelImporterRegistry()
    registry.register(_FakeImporter)
    reference = ArtifactModelReference(
        provider_id=_FakeImporter.provider_id,
        uri="S3://models/model.onnx",
        format_id="onnx",
        flavor_id="onnx-runtime-v1",
        expected_sha256="a" * 64,
        import_bundle_uri="file:///bundles/imported",
        options={"conversion": "onnx"},
    )

    result = registry.import_model(reference)

    assert result.bundle_id == "bundle-conformance"


def test_unsupported_scheme_is_explicitly_classified() -> None:
    registry = ModelImporterRegistry()
    registry.register(_FakeImporter)
    reference = ArtifactModelReference(
        provider_id=_FakeImporter.provider_id,
        uri="https://models.invalid/model.onnx",
        format_id="onnx",
        flavor_id="onnx-runtime-v1",
        import_bundle_uri="file:///bundles/imported",
    )

    with pytest.raises(ValueError, match="does not support URI scheme"):
        registry.import_model(reference)


def test_importer_failure_and_unsupported_are_distinct() -> None:
    class _FailingImporter(_FakeImporter):
        provider_id = "conformance.failure"

        def import_model(self, reference):
            del reference
            raise JobConfigurationError("classified acquisition failure")

    class _UnsupportedImporter(_FakeImporter):
        provider_id = "conformance.unsupported"

        def import_model(self, reference):
            del reference
            raise UnsupportedArtifactFormat("classified unsupported artifact")

    reference = ArtifactModelReference(
        provider_id=_FailingImporter.provider_id,
        uri="file:///models/model.onnx",
        format_id="onnx",
        flavor_id="onnx-runtime-v1",
        import_bundle_uri="file:///bundles/imported",
    )
    failing = ModelImporterRegistry()
    failing.register(_FailingImporter)
    with pytest.raises(JobConfigurationError, match="acquisition"):
        failing.import_model(reference)

    unsupported = ModelImporterRegistry()
    unsupported.register(_UnsupportedImporter)
    with pytest.raises(UnsupportedArtifactFormat, match="unsupported"):
        unsupported.import_model(
            reference.model_copy(
                update={"provider_id": _UnsupportedImporter.provider_id}
            )
        )


def test_importer_releases_temporary_acquisition_state() -> None:
    class _CleanupImporter(_FakeImporter):
        provider_id = "conformance.cleanup"
        released_path: Path | None = None

        def import_model(self, reference):
            with tempfile.TemporaryDirectory(prefix="importer-conformance-") as raw:
                path = Path(raw)
                type(self).released_path = path
                assert path.is_dir()
            return BundleRef(
                canonical_uri=reference.import_bundle_uri,
                bundle_id="bundle-cleanup",
                manifest_sha256="d" * 64,
            )

    registry = ModelImporterRegistry()
    registry.register(_CleanupImporter)
    registry.import_model(
        ArtifactModelReference(
            provider_id=_CleanupImporter.provider_id,
            uri="file:///models/model.onnx",
            format_id="onnx",
            flavor_id="onnx-runtime-v1",
            import_bundle_uri="file:///bundles/imported",
        )
    )

    assert _CleanupImporter.released_path is not None
    assert not _CleanupImporter.released_path.exists()


def test_first_party_importers_reuse_registry_metadata() -> None:
    registry = build_default_model_importer_registry()

    for provider_id in ("tributo.artifact", "mlflow.v2"):
        importer = registry.get(provider_id)
        assert importer.api_version == 1
        assert importer.provider_id == provider_id
        assert importer.options_model.model_config.get("extra") == "forbid"
        assert importer.credential_profile_types
        assert importer.capabilities

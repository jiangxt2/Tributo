"""Conformance harness for explicit external ModelImporter adapters."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationError

from tributo.exporting.models import BundleRef
from tributo.inference.contracts import ArtifactModelReference
from tributo.inference.importers import ModelImporter, ModelImporterRegistry


class _Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    assert isinstance(result, BundleRef)
    assert result.manifest_sha256 == "b" * 64

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

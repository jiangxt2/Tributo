"""Manifest v1/v2 compatibility and digest contract tests."""

from __future__ import annotations

import pytest

from tributo.explainability.contracts import ExplainabilityDescriptor
from tributo.exporting.manifest import (
    ExportManifest,
    ExportManifestV2,
    ManifestExecution,
    ManifestSignature,
    ManifestSourceInfo,
    _read_manifest_v1,
    compute_bundle_digest,
    get_schema_registry,
)
from tributo.exporting.models import ArtifactFile, LogicalArtifact, ProducerInfo


def _manifest_kwargs() -> dict[str, object]:
    return {
        "bundle_id": "bundle-1",
        "status": "succeeded",
        "canonical_uri": "/bundles/bundle-1",
        "tributo_version": "1.0.0",
        "source_info": ManifestSourceInfo(source_kind="test"),
        "input_signature": ManifestSignature(),
        "output_signature": ManifestSignature(),
        "execution": ManifestExecution(execution_id="execution-1"),
    }


def test_disabled_manifest_remains_v1_and_descriptor_changes_digest() -> None:
    v1 = ExportManifest(**_manifest_kwargs())
    descriptor = ExplainabilityDescriptor(
        adapter_id="shap-v1",
        backend="tree",
        exactness="exact",
        feature_view="raw",
        output_target="model_output",
        reference_policy="optional",
    )
    v2 = ExportManifestV2(**_manifest_kwargs(), explainability=descriptor)
    assert v1.schema_version == 1
    assert v2.schema_version == 2
    base_digest = compute_bundle_digest((), {})
    explain_digest = compute_bundle_digest((), {}, explainability=descriptor)
    assert base_digest != explain_digest


def test_schema_registry_reads_v1_and_v2() -> None:
    descriptor = ExplainabilityDescriptor(
        adapter_id="shap-v1",
        backend="tree",
        exactness="exact",
        feature_view="raw",
        output_target="model_output",
        reference_policy="optional",
    )
    v1 = ExportManifest(**_manifest_kwargs())
    v2 = ExportManifestV2(**_manifest_kwargs(), explainability=descriptor)
    registry = get_schema_registry()
    assert isinstance(
        registry.read(1, v1.model_dump(mode="json"), v1.canonical_json()),
        ExportManifest,
    )
    parsed = registry.read(2, v2.model_dump(mode="json"), v2.canonical_json())
    assert isinstance(parsed, ExportManifestV2)
    assert parsed.explainability == descriptor


def test_v1_reader_rejects_v2_payload_explicitly() -> None:
    descriptor = ExplainabilityDescriptor(
        adapter_id="shap-v1",
        backend="tree",
        exactness="exact",
        feature_view="raw",
        output_target="model_output",
        reference_policy="optional",
    )
    raw = ExportManifestV2(**_manifest_kwargs(), explainability=descriptor).model_dump(
        mode="json"
    )
    with pytest.raises(ValueError, match="v1 reader cannot read"):
        _read_manifest_v1(raw, b"manifest")


def test_v2_reader_rejects_descriptor_references_to_missing_role() -> None:
    descriptor = ExplainabilityDescriptor(
        adapter_id="shap-v1",
        backend="tree",
        exactness="exact",
        model_roles=("explainability_model",),
        feature_view="raw",
        output_target="model_output",
        reference_policy="optional",
    )
    raw = ExportManifestV2(
        **_manifest_kwargs(),
        explainability=descriptor,
    ).model_dump(mode="json")
    with pytest.raises(ValueError, match="missing roles"):
        get_schema_registry().read(2, raw, b"manifest")


def test_v2_reader_rejects_backend_and_artifact_flavor_mismatch() -> None:
    file_entry = ArtifactFile(
        relative_path="model.onnx",
        sha256="a" * 64,
        size_bytes=1,
        role="model",
    )
    artifact = LogicalArtifact(
        name="model",
        format="onnx",
        flavor_id="onnx-runtime-v1",
        files=(file_entry,),
        entrypoint="model.onnx",
        tree_digest=LogicalArtifact.compute_tree_digest((file_entry,)),
        producer=ProducerInfo(exporter_id="test"),
    )
    descriptor = ExplainabilityDescriptor(
        adapter_id="shap-v1",
        backend="tree",
        exactness="exact",
        model_roles=("explainability_model",),
        required_artifacts=("model",),
        feature_view="raw",
        output_target="model_output",
        reference_policy="optional",
    )
    raw = ExportManifestV2(
        **_manifest_kwargs(),
        artifacts=(artifact,),
        roles={"explainability_model": "model"},
        explainability=descriptor,
    ).model_dump(mode="json")
    with pytest.raises(ValueError, match="xgboost-native-v1"):
        get_schema_registry().read(2, raw, b"manifest")

"""E1 acceptance matrix: schema-v1 manifest compatibility and digest semantics.

Digests are computed over the raw manifest bytes as published — never over
a re-serialisation of the parsed model, which would diverge once new
optional fields with defaults are added to the schema.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tributo.exporting import BundleRef, load_bundle
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.manifest import (
    ExportManifest,
    ManifestExecution,
    ManifestSignature,
    ManifestSourceInfo,
    SignatureField,
    compute_bundle_digest,
)
from tributo.exporting.models import (
    ArtifactFile,
    LogicalArtifact,
    ProducerInfo,
)

_FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "manifest-v1-golden.json"
#: SHA-256 of the raw fixture bytes — change the fixture, change the constant.
_GOLDEN_SHA256 = "778ff3e62754d8681fea171054a4da540ce9dab2433d4fe148c70ab8cccbaa43"
_FIXED_TS = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


def _make_artifact(name: str = "model") -> LogicalArtifact:
    files = (
        ArtifactFile(
            relative_path="model.onnx",
            sha256="a" * 64,
            size_bytes=42,
            role="model",
        ),
    )
    return LogicalArtifact(
        name=name,
        format="onnx",
        flavor_id="onnx-runtime-v1",
        entrypoint="model.onnx",
        files=files,
        tree_digest=LogicalArtifact.compute_tree_digest(files),
        producer=ProducerInfo(exporter_id="xgboost-onnx-v1"),
    )


def _make_manifest(signature: ManifestSignature | None = None) -> ExportManifest:
    return ExportManifest(
        schema_version=1,
        bundle_id="b1",
        status="succeeded",
        created_at=_FIXED_TS,
        canonical_uri="s3://b/pre/b1",
        tributo_version="1.0.0",
        source_info=ManifestSourceInfo(source_kind="xgboost_result"),
        input_signature=signature or ManifestSignature(),
        execution=ManifestExecution(execution_id="exec-1"),
    )


class TestV1GoldenFixture:
    """Old v1 manifests (as published by 1.0.0) remain readable and verifiable."""

    def test_golden_fixture_bytes_unchanged(self) -> None:
        assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == _GOLDEN_SHA256

    def test_new_reader_reads_v1_golden(self) -> None:
        manifest = BundleReader().read_manifest(str(_FIXTURE))
        assert manifest.schema_version == 1
        assert manifest.input_signature.input_names == ("f0", "f1")
        # New optional fields default to empty — no breakage for old manifests.
        assert manifest.input_signature.input_fields == ()
        # v1 artifacts lack artifact_kind — the reader injects "model".
        assert manifest.artifacts[0].artifact_kind == "model"
        assert manifest.roles == {"inference": "onnx-model"}
        assert manifest.execution.nodes[0].status == "succeeded"

    def test_load_bundle_verifies_raw_bytes(self) -> None:
        ref = BundleRef(
            canonical_uri=str(_FIXTURE),
            bundle_id="bundle-golden-v1",
            manifest_sha256=_GOLDEN_SHA256,
        )
        result = load_bundle(ref)
        assert result["bundle_id"] == "bundle-golden-v1"

    def test_load_bundle_rejects_wrong_digest(self) -> None:
        ref = BundleRef(
            canonical_uri=str(_FIXTURE),
            bundle_id="bundle-golden-v1",
            manifest_sha256="f" * 64,
        )
        with pytest.raises(ValueError, match="integrity check failed"):
            load_bundle(ref)

    @pytest.mark.parametrize(
        ("variant", "expected_format", "expected_flavor"),
        (
            ("ubj", "ubj", "xgboost-ubj-v1"),
            ("json", "xgboost-json", "xgboost-json-v1"),
        ),
    )
    def test_legacy_xgboost_artifact_normalises_without_digest_drift(
        self,
        tmp_path: Path,
        variant: str,
        expected_format: str,
        expected_flavor: str,
    ) -> None:
        payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        payload["artifacts"][0].update(
            {
                "format": "xgboost",
                "flavor_id": "xgboost-native-v1",
                "variant": variant,
            }
        )
        raw_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        manifest_path = tmp_path / f"legacy-{variant}.json"
        manifest_path.write_bytes(raw_bytes)
        expected_sha256 = hashlib.sha256(raw_bytes).hexdigest()

        manifest = BundleReader().read_manifest(str(manifest_path))
        artifact = manifest.artifacts[0]

        assert artifact.format == expected_format
        assert artifact.flavor_id == expected_flavor
        assert manifest_path.read_bytes() == raw_bytes
        result = load_bundle(
            BundleRef(
                canonical_uri=str(manifest_path),
                bundle_id=payload["bundle_id"],
                manifest_sha256=expected_sha256,
            )
        )
        assert result["artifacts"][0]["format"] == expected_format


class TestDigestSemantics:
    """manifest_sha256 vs bundle_digest behave differently by design."""

    def test_manifest_rejects_naive_created_at(self) -> None:
        payload = _make_manifest().model_dump()
        payload["created_at"] = datetime(2026, 7, 30, 12, 0, 0)
        with pytest.raises(ValueError, match="timezone offset"):
            ExportManifest.model_validate(payload)

    def test_manifest_sha256_changes_with_new_fields(self) -> None:
        plain = _make_manifest(ManifestSignature(input_names=("f0",)))
        with_fields = _make_manifest(
            ManifestSignature(
                input_names=("f0",),
                input_fields=(SignatureField(name="f0", dtype="float32"),),
            )
        )
        assert plain.compute_sha256() != with_fields.compute_sha256()

    def test_bundle_digest_changes_with_signature(self) -> None:
        artifact = _make_artifact()
        without = compute_bundle_digest((artifact,), {"inference": "onnx"})
        with_sig = compute_bundle_digest(
            (artifact,),
            {"inference": "onnx"},
            input_sig=ManifestSignature(input_names=("x",)),
        )
        assert without != with_sig

    def test_bundle_digest_stable_for_same_logical_content(self) -> None:
        artifact = _make_artifact()
        first = compute_bundle_digest((artifact,), {"inference": "onnx"})
        second = compute_bundle_digest((artifact,), {"inference": "onnx"})
        assert first == second


class TestSignatureFieldValidation:
    """Frozen field rules: strict types, no blanks, unique ordered names."""

    def test_shape_must_be_positive_int_or_nonempty_str(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SignatureField(name="x", dtype="float32", shape=(0,))
        with pytest.raises(ValueError, match="non-empty"):
            SignatureField(name="x", dtype="float32", shape=(" ",))

    def test_shape_rejects_coerced_floats_and_bools(self) -> None:
        # Pydantic must not coerce 3.0 → 3 or True → 1 (strict mode).
        with pytest.raises(ValueError):
            SignatureField(name="x", dtype="float32", shape=(3.0,))
        with pytest.raises(ValueError):
            SignatureField(name="x", dtype="float32", shape=(True,))

    def test_name_and_dtype_required_and_non_blank(self) -> None:
        with pytest.raises(ValueError):
            SignatureField(name="", dtype="float32")
        with pytest.raises(ValueError):
            SignatureField(name="x", dtype="")
        with pytest.raises(ValueError, match="whitespace"):
            SignatureField(name=" ", dtype="float32")
        with pytest.raises(ValueError, match="whitespace"):
            SignatureField(name="x", dtype=" float32")

    def test_fields_and_names_conflict_fail_fast(self) -> None:
        with pytest.raises(ValueError, match="disagree"):
            ManifestSignature(
                input_names=("a", "b"),
                input_fields=(SignatureField(name="a", dtype="float32"),),
            )

    def test_fields_and_names_order_must_match(self) -> None:
        # Same set, different order — rejected: order is part of the contract.
        with pytest.raises(ValueError, match="order"):
            ManifestSignature(
                input_names=("a", "b"),
                input_fields=(
                    SignatureField(name="b", dtype="float32"),
                    SignatureField(name="a", dtype="float32"),
                ),
            )

    def test_duplicate_field_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            ManifestSignature(
                input_fields=(
                    SignatureField(name="a", dtype="float32"),
                    SignatureField(name="a", dtype="float32"),
                ),
            )

    def test_fields_consistent_with_names_accepted(self) -> None:
        sig = ManifestSignature(
            input_names=("a", "b"),
            input_fields=(
                SignatureField(name="a", dtype="float32"),
                SignatureField(name="b", dtype="float32"),
            ),
        )
        assert len(sig.input_fields) == 2

    def test_dynamic_axes_must_match_shape(self) -> None:
        # Shape says dim 0 is dynamic "batch" — dynamic_axes must agree.
        ok = ManifestSignature(
            input_fields=(
                SignatureField(name="x", dtype="float32", shape=("batch", 3)),
            ),
            dynamic_axes={"x": {0: "batch"}},
        )
        assert ok.input_fields[0].shape == ("batch", 3)
        with pytest.raises(ValueError, match="dynamic axis 0"):
            ManifestSignature(
                input_fields=(
                    SignatureField(name="x", dtype="float32", shape=("seq", 3)),
                ),
                dynamic_axes={"x": {0: "batch"}},
            )
        # Fixed dim declared dynamic in dynamic_axes → rejected.
        with pytest.raises(ValueError, match="fixed in shape"):
            ManifestSignature(
                input_fields=(SignatureField(name="x", dtype="float32", shape=(3, 4)),),
                dynamic_axes={"x": {0: "batch"}},
            )

    def test_dynamic_axes_axis_out_of_range_rejected(self) -> None:
        # axis 1 declared but shape has only one dimension.
        with pytest.raises(ValueError, match="out of range"):
            ManifestSignature(
                input_fields=(SignatureField(name="x", dtype="float32", shape=(3,)),),
                dynamic_axes={"x": {1: "batch"}},
            )
        # Negative axis index.
        with pytest.raises(ValueError, match="out of range"):
            ManifestSignature(
                input_fields=(SignatureField(name="x", dtype="float32", shape=(3,)),),
                dynamic_axes={"x": {-1: "batch"}},
            )

    def test_dynamic_axes_undeclared_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="undeclared field"):
            ManifestSignature(
                input_fields=(SignatureField(name="x", dtype="float32"),),
                dynamic_axes={"y": {0: "batch"}},
            )

    def test_dynamic_axes_empty_shape_with_axis_rejected(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            ManifestSignature(
                input_fields=(SignatureField(name="x", dtype="float32"),),
                dynamic_axes={"x": {0: "batch"}},
            )

    def test_mixed_input_output_fields_share_dynamic_axes(self) -> None:
        # dynamic_axes is one shared map across both sides — a signature
        # mixing input and output fields must validate as a whole.
        sig = ManifestSignature(
            input_fields=(SignatureField(name="x", dtype="float32", shape=("batch",)),),
            output_fields=(
                SignatureField(name="y", dtype="float32", shape=("batch",)),
            ),
            dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        )
        assert len(sig.input_fields) == 1
        assert len(sig.output_fields) == 1

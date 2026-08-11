"""Tests for fail-closed inference plan resolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from tests.serving.bundle_fixtures import build_test_bundle
from tributo.data import IngestionDescriptor, IngestionRequest, SelectColumns
from tributo.data.source_config import ParquetSourceConfig
from tributo.exceptions import JobConfigurationError, UnsupportedArtifactFormat
from tributo.exporting.models import BundleRef
from tributo.inference.contracts import (
    ArtifactModelReference,
    BundleModelReference,
    InferenceRequest,
    InputBindingSpec,
    OutputBindingSpec,
    ParquetResultSinkRequest,
    ResolvedInputSelection,
    TensorInputBinding,
    TensorOutputBinding,
)
from tributo.inference.importers import ModelImporterRegistry
from tributo.inference.resolver import InferenceResolver


def _manifest_digest(bundle: Path) -> str:
    return hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()


def _resolved_input(request: IngestionRequest) -> ResolvedInputSelection:
    binding_id = request.binding_id or "tributo.ray.parquet"
    pinned = request.model_copy(update={"binding_id": binding_id})
    return ResolvedInputSelection(
        request=pinned,
        descriptor=IngestionDescriptor(
            request_digest="1" * 64,
            source_ref="2" * 64,
            dataset_ref="3" * 64,
            logical_plan_digest="4" * 64,
            engine_id="tributo.ray_data",
            provider_id="tributo.parquet",
            connector_id="parquet",
            binding_id=binding_id,
            scan_kind="file",
            handle_kind="ray_data",
            binding_distribution="tributo",
            binding_distribution_version="1.0.0",
            capability_version=1,
        ),
    )


def _request(
    bundle: Path,
    *,
    expected_digest: str | None = None,
    role: str = "inference",
    unsafe: bool = False,
    input_names: tuple[str, ...] = ("float_input",),
    output_names: tuple[str, ...] = ("label", "probabilities"),
    model_storage_profile: str | None = None,
    source_storage_profile: str | None = None,
    sink_storage_profile: str | None = None,
    run_id: str | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        model=BundleModelReference(
            uri=str(bundle),
            role=role,
            expected_manifest_sha256=expected_digest,
            storage_profile=model_storage_profile,
            unsafe=unsafe,
        ),
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"),
            engine="ray",
            storage_profile=source_storage_profile,
        ),
        input_binding=InputBindingSpec(
            tensors=tuple(
                TensorInputBinding(
                    tensor_name=name,
                    columns=("feature_a", "feature_b"),
                    dtype="float32",
                )
                for name in input_names
            ),
            passthrough_columns=("entity_id",),
        ),
        output_binding=OutputBindingSpec(
            tensors=tuple(
                TensorOutputBinding(
                    tensor_name=name,
                    column=f"result_{name}",
                    semantic="label" if name == "label" else "probability",
                )
                for name in output_names
            )
        ),
        result_sink=ParquetResultSinkRequest(
            uri="/data/output", storage_profile=sink_storage_profile
        ),
        run_id=run_id,
    )


class TestInferenceResolver:
    def test_resolves_pinned_bundle_projection_and_identity(
        self, tmp_path: Path
    ) -> None:
        bundle = build_test_bundle(tmp_path)
        request = _request(bundle, expected_digest=_manifest_digest(bundle))

        plan = InferenceResolver().resolve(request)

        assert plan.model.bundle_ref.bundle_id == "bundle-e3-test"
        assert plan.model.bundle_ref.manifest_sha256 == _manifest_digest(bundle)
        assert plan.model.role == "inference"
        assert plan.model.flavor_id == "onnx-runtime-v1"
        projection = plan.input.request.transforms.steps[-1]
        assert isinstance(projection, SelectColumns)
        assert projection.columns == ("feature_a", "feature_b", "entity_id")
        assert plan.input.request.binding_id == plan.input.descriptor.binding_id
        assert len(plan.plan_digest) == 64
        assert plan.attempt_id == "attempt-1"
        assert plan.submission_id.startswith("tributo-infer-")

    def test_plan_digest_excludes_run_identity(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)

        first = InferenceResolver().resolve(_request(bundle, run_id="run-a"))
        second = InferenceResolver().resolve(_request(bundle, run_id="run-b"))

        assert first.plan_digest == second.plan_digest
        assert first.submission_id != second.submission_id

    def test_expected_manifest_digest_is_fail_closed(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)

        with pytest.raises(JobConfigurationError, match="digest mismatch"):
            InferenceResolver().resolve(_request(bundle, expected_digest="f" * 64))

    def test_unknown_role_is_not_bypassed_by_unsafe(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)

        with pytest.raises(JobConfigurationError, match="Role 'serve' not found"):
            InferenceResolver().resolve(_request(bundle, role="serve", unsafe=True))

    def test_artifact_kind_mismatch_is_not_bypassed_by_unsafe(
        self, tmp_path: Path
    ) -> None:
        bundle = build_test_bundle(tmp_path)
        path = bundle / "manifest.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["artifacts"][0]["artifact_kind"] = "report"
        path.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(UnsupportedArtifactFormat, match="requires 'model'"):
            InferenceResolver().resolve(_request(bundle, unsafe=True))

    def test_unsupported_flavor_fails_before_execution(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path, flavor_id="safetensors-v1")

        with pytest.raises(
            UnsupportedArtifactFormat, match="batch inference capability"
        ):
            InferenceResolver().resolve(_request(bundle))

    def test_empty_signature_requires_explicit_unsafe(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path, with_signature=False)

        with pytest.raises(UnsupportedArtifactFormat, match="typed input signature"):
            InferenceResolver().resolve(_request(bundle))

        plan = InferenceResolver().resolve(_request(bundle, unsafe=True))
        assert plan.model.unsafe is True

    def test_binding_names_must_exactly_match_manifest(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)

        with pytest.raises(JobConfigurationError, match="missing=.*probabilities"):
            InferenceResolver().resolve(_request(bundle, output_names=("label",)))

        with pytest.raises(JobConfigurationError, match="unknown=.*unknown"):
            InferenceResolver().resolve(
                _request(bundle, output_names=("label", "unknown"))
            )

    def test_binding_dtype_must_match_manifest(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)
        request = _request(bundle)
        bad_binding = request.input_binding.model_copy(
            update={
                "tensors": (
                    TensorInputBinding(
                        tensor_name="float_input",
                        columns=("feature_a", "feature_b"),
                        dtype="float64",
                    ),
                )
            }
        )

        with pytest.raises(JobConfigurationError, match="manifest declares 'float32'"):
            InferenceResolver().resolve(
                request.model_copy(update={"input_binding": bad_binding})
            )

    def test_each_storage_profile_stays_in_its_domain(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)

        class Inputs:
            calls: list[str | None] = []

            def describe(self, request):
                self.calls.append(request.storage_profile)
                return _resolved_input(request)

        inputs = Inputs()
        plan = InferenceResolver(input_resolver=inputs).resolve(
            _request(
                bundle,
                model_storage_profile="model-domain",
                source_storage_profile="source-domain",
                sink_storage_profile="sink-domain",
            )
        )

        assert inputs.calls == ["source-domain"]
        assert plan.model.storage_profile == "model-domain"
        assert plan.result_sink.storage_profile == "sink-domain"

    def test_training_environment_is_parent_context_not_inference_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = build_test_bundle(tmp_path)
        monkeypatch.setenv("TRIBUTO_RUN_ID", "training-run")
        monkeypatch.setenv("TRIBUTO_ATTEMPT_ID", "attempt-9")

        plan = InferenceResolver().resolve(_request(bundle))

        assert plan.run_id != "training-run"
        assert plan.attempt_id == "attempt-1"

        monkeypatch.setenv("TRIBUTO_JOB_KIND", "inference")
        inference_plan = InferenceResolver().resolve(
            _request(bundle, run_id="training-run")
        )
        assert inference_plan.run_id == "training-run"
        assert inference_plan.attempt_id == "attempt-9"


class _StrictOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    convert: bool = True


class _FakeImporter:
    api_version = 1
    provider_id = "external.artifact"
    options_model = _StrictOptions
    reference_kinds = ("artifact",)
    uri_schemes = ("file",)
    credential_profile_types = ("source-storage", "bundle-storage")
    capabilities = ("acquire", "bundle-import")
    bundle_ref: BundleRef
    calls = 0

    def import_model(self, reference):
        type(self).calls += 1
        assert reference.expected_sha256 == "b" * 64
        return type(self).bundle_ref


class TestExternalImportResolution:
    def test_explicit_importer_normalizes_to_pinned_bundle(
        self, tmp_path: Path
    ) -> None:
        bundle = build_test_bundle(tmp_path)
        _FakeImporter.bundle_ref = BundleRef(
            canonical_uri=str(bundle),
            bundle_id="bundle-e3-test",
            manifest_sha256=_manifest_digest(bundle),
        )
        registry = ModelImporterRegistry()
        registry.register(_FakeImporter)
        _FakeImporter.calls = 0
        base = _request(bundle)
        request = base.model_copy(
            update={
                "model": ArtifactModelReference(
                    provider_id="external.artifact",
                    uri="/models/raw.onnx",
                    format_id="onnx",
                    flavor_id="onnx-runtime-v1",
                    expected_sha256="b" * 64,
                    import_bundle_uri=str(bundle),
                    import_storage_profile="imported-bundle-domain",
                    options={"convert": True},
                )
            }
        )

        plan = InferenceResolver(importers=registry).resolve(request)

        assert plan.model.bundle_ref == _FakeImporter.bundle_ref
        assert plan.model.storage_profile == "imported-bundle-domain"
        assert plan.model.source_provenance.startswith("artifact:external.artifact")
        assert _FakeImporter.calls == 1

    def test_post_construction_credential_mutation_fails_before_import(
        self, tmp_path: Path
    ) -> None:
        bundle = build_test_bundle(tmp_path)
        _FakeImporter.bundle_ref = BundleRef(
            canonical_uri=str(bundle),
            bundle_id="bundle-e3-test",
            manifest_sha256=_manifest_digest(bundle),
        )
        registry = ModelImporterRegistry()
        registry.register(_FakeImporter)
        _FakeImporter.calls = 0
        base = _request(bundle)
        model = ArtifactModelReference(
            provider_id="external.artifact",
            uri="/models/raw.onnx",
            format_id="onnx",
            flavor_id="onnx-runtime-v1",
            expected_sha256="b" * 64,
            import_bundle_uri=str(bundle),
            options={"convert": True},
        )
        request = base.model_copy(update={"model": model})
        model.options["api_token"] = "must-not-leak"

        with pytest.raises(ValueError, match="plaintext credentials") as error:
            InferenceResolver(importers=registry).resolve(request)

        assert _FakeImporter.calls == 0
        assert "must-not-leak" not in str(error.value)

    def test_unregistered_external_source_is_not_probed(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)
        base = _request(bundle)
        request = base.model_copy(
            update={
                "model": ArtifactModelReference(
                    provider_id="external.unknown",
                    uri="/models/raw.onnx",
                    format_id="onnx",
                    flavor_id="onnx-runtime-v1",
                    import_bundle_uri=str(bundle),
                )
            }
        )

        with pytest.raises(ValueError, match="Unknown ModelImporter"):
            InferenceResolver().resolve(request)

"""Resolve user inference intent into an immutable executable plan."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable

from tributo._common.submission_id import generate_submission_id
from tributo.data import IngestionRequest, SelectColumns, TransformPipeline
from tributo.exceptions import JobConfigurationError, UnsupportedArtifactFormat
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.manifest import ExportManifest, SignatureField
from tributo.exporting.models import BundleRef, LogicalArtifact
from tributo.exporting.runtime import FLAVOR_SUPPORT_MATRIX
from tributo.inference.contracts import (
    ArtifactModelReference,
    BundleModelReference,
    InferenceRequest,
    RegistryModelReference,
    ResolvedInference,
    ResolvedInputSelection,
    ResolvedModelSelection,
)
from tributo.inference.input_resolver import (
    IngestionGatewayInputResolver,
    InputResolverPort,
)
from tributo.integrations.model_importers import (
    ModelImporterRegistry,
    build_default_model_importer_registry,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class InferenceResolver:
    """Pin model, data identity, bindings, and execution identity once."""

    def __init__(
        self,
        *,
        bundle_reader: BundleReader | None = None,
        input_resolver: InputResolverPort | None = None,
        importers: ModelImporterRegistry | None = None,
    ) -> None:
        self._bundles = bundle_reader or BundleReader()
        self._inputs = input_resolver or IngestionGatewayInputResolver()
        self._importers = importers or build_default_model_importer_registry()

    def resolve(self, request: InferenceRequest) -> ResolvedInference:
        """Return a credential-free immutable plan for one request."""
        # Pydantic's frozen models are shallow: provider/importer option
        # mappings can still be mutated after construction.  Rebuild from a
        # plain payload before any importer, Bundle reader, or Data Gateway
        # side effect so the aggregate credential and JSON gates run again.
        request = InferenceRequest.model_validate(request.model_dump(mode="python"))
        model_reference, provenance = self._normalize_model(request)
        manifest, manifest_bytes = self._bundles.read_manifest_with_bytes(
            model_reference.uri,
            storage_profile=model_reference.storage_profile,
        )
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if (
            model_reference.expected_manifest_sha256 is not None
            and manifest_sha256 != model_reference.expected_manifest_sha256
        ):
            raise JobConfigurationError(
                "Bundle manifest digest mismatch: expected "
                f"{model_reference.expected_manifest_sha256[:16]}..., got "
                f"{manifest_sha256[:16]}..."
            )

        artifact = _select_artifact(
            manifest,
            role=model_reference.role,
            unsafe=model_reference.unsafe,
        )
        if isinstance(request.model, ArtifactModelReference):
            if artifact.flavor_id != request.model.flavor_id:
                raise UnsupportedArtifactFormat(
                    f"Imported artifact flavor {artifact.flavor_id!r} does not "
                    f"match requested flavor {request.model.flavor_id!r}"
                )
        _validate_bindings(request, manifest)

        resolved_input = self._inputs.describe(_bind_input_projection(request))

        bundle_ref = BundleRef(
            canonical_uri=manifest.canonical_uri,
            bundle_id=manifest.bundle_id,
            manifest_sha256=manifest_sha256,
        )
        source_provenance = (
            f"{provenance};source_kind={manifest.source_info.source_kind};"
            f"source_fingerprint={manifest.source_info.source_fingerprint}"
        )
        selection = ResolvedModelSelection(
            bundle_ref=bundle_ref,
            role=model_reference.role,
            flavor_id=artifact.flavor_id,
            storage_profile=model_reference.storage_profile,
            source_provenance=source_provenance,
            unsafe=model_reference.unsafe,
        )
        plan_digest = _plan_digest(
            request=request,
            bundle_ref=bundle_ref,
            role=model_reference.role,
            flavor_id=artifact.flavor_id,
            resolved_input=resolved_input,
        )
        run_id = _resolve_run_id(request, plan_digest)
        attempt_id = (
            os.environ.get("TRIBUTO_ATTEMPT_ID")
            if os.environ.get("TRIBUTO_JOB_KIND") == "inference"
            else None
        ) or "attempt-1"
        submission_id = generate_submission_id("infer", run_id, attempt_id, plan_digest)

        return ResolvedInference(
            plan_digest=plan_digest,
            model=selection,
            input=resolved_input,
            input_signature=manifest.input_signature,
            output_signature=manifest.output_signature,
            input_binding=request.input_binding,
            output_binding=request.output_binding,
            result_sink=request.result_sink,
            execution=request.execution,
            run_id=run_id,
            attempt_id=attempt_id,
            submission_id=submission_id,
            parent_run_id=request.parent_run_id,
        )

    def _normalize_model(
        self, request: InferenceRequest
    ) -> tuple[BundleModelReference, str]:
        reference = request.model
        if isinstance(reference, BundleModelReference):
            return reference, "tributo-bundle"

        bundle_ref = self._importers.import_model(reference)
        if isinstance(reference, RegistryModelReference):
            selector = reference.version or reference.alias
            provenance = (
                f"registry:{reference.provider_id}:{reference.model_name}:{selector}"
            )
        elif isinstance(reference, ArtifactModelReference):
            provenance = (
                f"artifact:{reference.provider_id}:{reference.format_id}:"
                f"{reference.expected_sha256 or 'imported'}"
            )
        else:  # pragma: no cover - discriminated union makes this unreachable.
            raise AssertionError(type(reference).__name__)
        return (
            BundleModelReference.from_bundle_ref(
                bundle_ref,
                storage_profile=reference.import_storage_profile,
            ),
            provenance,
        )


def _select_artifact(
    manifest: ExportManifest, *, role: str, unsafe: bool
) -> LogicalArtifact:
    artifact_name = manifest.roles.get(role)
    if artifact_name is None:
        raise JobConfigurationError(
            f"Role {role!r} not found in bundle. Available roles: "
            f"{sorted(manifest.roles)}"
        )
    artifact = next(
        (item for item in manifest.artifacts if item.name == artifact_name), None
    )
    if artifact is None:
        raise JobConfigurationError(
            f"Role {role!r} references missing artifact {artifact_name!r}"
        )
    matrix_entry = next(
        (
            entry
            for entry in FLAVOR_SUPPORT_MATRIX
            if entry.flavor_id == artifact.flavor_id
        ),
        None,
    )
    if matrix_entry is None:
        raise UnsupportedArtifactFormat(
            f"Flavor {artifact.flavor_id!r} is not in the capability matrix"
        )
    if not matrix_entry.batch_inference_capable:
        raise UnsupportedArtifactFormat(
            f"Flavor {artifact.flavor_id!r} is readable but does not declare "
            "batch inference capability"
        )
    if artifact.artifact_kind != matrix_entry.artifact_role:
        raise UnsupportedArtifactFormat(
            f"Artifact {artifact.name!r} has kind {artifact.artifact_kind!r} but "
            f"flavor {artifact.flavor_id!r} requires "
            f"{matrix_entry.artifact_role!r}"
        )
    if matrix_entry.signature_required and not unsafe:
        if not manifest.input_signature.input_fields:
            raise UnsupportedArtifactFormat(
                f"Artifact {artifact.name!r} requires a typed input signature"
            )
        if not manifest.output_signature.output_fields:
            raise UnsupportedArtifactFormat(
                f"Artifact {artifact.name!r} requires a typed output signature"
            )
    return artifact


def _validate_bindings(request: InferenceRequest, manifest: ExportManifest) -> None:
    _validate_side(
        side="input",
        bindings=(
            (binding.tensor_name, binding.dtype)
            for binding in request.input_binding.tensors
        ),
        fields=manifest.input_signature.input_fields,
        unsafe=(
            request.model.unsafe
            if isinstance(request.model, BundleModelReference)
            else False
        ),
    )
    _validate_side(
        side="output",
        bindings=(
            (binding.tensor_name, binding.dtype)
            for binding in request.output_binding.tensors
        ),
        fields=manifest.output_signature.output_fields,
        unsafe=(
            request.model.unsafe
            if isinstance(request.model, BundleModelReference)
            else False
        ),
    )


def _validate_side(
    *,
    side: str,
    bindings: Iterable[tuple[str, str | None]],
    fields: tuple[SignatureField, ...],
    unsafe: bool,
) -> None:
    binding_map = dict(bindings)
    if not fields:
        if unsafe:
            return
        raise UnsupportedArtifactFormat(f"Bundle has no typed {side} signature")
    field_map = {field.name: field for field in fields}
    if set(binding_map) != set(field_map):
        missing = sorted(set(field_map) - set(binding_map))
        unknown = sorted(set(binding_map) - set(field_map))
        raise JobConfigurationError(
            f"{side} binding names do not match manifest signature; "
            f"missing={missing}, unknown={unknown}"
        )
    for name, dtype in binding_map.items():
        if dtype is not None and dtype != field_map[name].dtype:
            raise JobConfigurationError(
                f"{side} binding {name!r} declares dtype {dtype!r}, but the "
                f"manifest declares {field_map[name].dtype!r}"
            )


def _plan_digest(
    *,
    request: InferenceRequest,
    bundle_ref: BundleRef,
    role: str,
    flavor_id: str,
    resolved_input: ResolvedInputSelection,
) -> str:
    payload = {
        "schema_version": request.schema_version,
        "bundle": bundle_ref.model_dump(mode="json"),
        "role": role,
        "flavor_id": flavor_id,
        "input_descriptor": resolved_input.descriptor.model_dump(mode="json"),
        "input_binding": request.input_binding.model_dump(mode="json"),
        "output_binding": request.output_binding.model_dump(mode="json"),
        "result_sink": request.result_sink.model_dump(
            mode="json", exclude={"storage_profile"}
        ),
        "execution": request.execution.model_dump(mode="json"),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bind_input_projection(request: InferenceRequest) -> IngestionRequest:
    """Append inference's final table projection to the public Transform IR."""
    columns = request.input_binding.projected_columns()
    projection = SelectColumns(columns=columns)
    steps = request.input.transforms.steps
    if steps and steps[-1] == projection:
        return request.input
    return request.input.model_copy(
        update={"transforms": TransformPipeline(steps=(*steps, projection))}
    )


def _resolve_run_id(request: InferenceRequest, plan_digest: str) -> str:
    env_run_id = (
        os.environ.get("TRIBUTO_RUN_ID")
        if os.environ.get("TRIBUTO_JOB_KIND") == "inference"
        else None
    )
    if request.run_id is not None and env_run_id not in (None, request.run_id):
        raise JobConfigurationError(
            "InferenceRequest.run_id conflicts with TRIBUTO_RUN_ID"
        )
    return (
        request.run_id
        or env_run_id
        or generate_submission_id("infer-run", plan_digest, request.parent_run_id or "")
    )


__all__ = ["InferenceResolver"]

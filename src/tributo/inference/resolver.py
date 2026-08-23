"""Resolve user inference intent into an immutable executable plan."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from tributo._common.submission_id import generate_submission_id
from tributo.data import IngestionRequest, SelectColumns, TransformPipeline
from tributo.exceptions import JobConfigurationError, UnsupportedArtifactFormat
from tributo.exporting.manifest import ManifestSignature, SignatureField
from tributo.inference.contracts import (
    InferenceRequest,
    ModelReference,
    ResolvedInference,
    ResolvedInputSelection,
    ResolvedModelBinding,
    ResolvedModelSelection,
)
from tributo.inference.input_resolver import (
    IngestionGatewayInputResolver,
    InputResolverPort,
)
from tributo.util.annotations import PublicAPI


@runtime_checkable
@PublicAPI(stability="alpha")
class ModelReferenceResolver(Protocol):
    """Resolve a logical model reference without exposing artifact formats."""

    def resolve(self, reference: ModelReference) -> ResolvedModelBinding: ...


@PublicAPI(stability="alpha")
class InferenceResolver:
    """Pin model, data identity, bindings, and execution identity once."""

    def __init__(
        self,
        *,
        model_resolver: ModelReferenceResolver | None = None,
        bundle_reader: Any | None = None,
        input_resolver: InputResolverPort | None = None,
        importers: Any | None = None,
    ) -> None:
        if model_resolver is not None and (
            bundle_reader is not None or importers is not None
        ):
            raise ValueError(
                "model_resolver cannot be combined with legacy bundle/importer "
                "constructor arguments"
            )
        self._models = model_resolver or _default_model_reference_resolver(
            bundle_reader=bundle_reader,
            importers=importers,
        )
        self._inputs = input_resolver or IngestionGatewayInputResolver()

    def resolve(self, request: InferenceRequest) -> ResolvedInference:
        """Return a credential-free immutable plan for one request."""
        # Pydantic's frozen models are shallow: provider/importer option
        # mappings can still be mutated after construction.  Rebuild from a
        # plain payload before any importer, Bundle reader, or Data Gateway
        # side effect so the aggregate credential and JSON gates run again.
        request = InferenceRequest.model_validate(request.model_dump(mode="python"))
        resolved_model = self._models.resolve(request.model)
        _validate_bindings(
            request,
            input_signature=resolved_model.input_signature,
            output_signature=resolved_model.output_signature,
            unsafe=resolved_model.selection.unsafe,
        )

        resolved_input = self._inputs.describe(_bind_input_projection(request))

        plan_digest = _plan_digest(
            request=request,
            model=resolved_model.selection,
            resolved_input=resolved_input,
        )
        run_id = _resolve_run_id(request, plan_digest)
        attempt_id = (
            os.environ.get("TRIBUTO_ATTEMPT_ID")
            if os.environ.get("TRIBUTO_JOB_KIND") == "inference"
            else None
        ) or "attempt-1"
        submission_id = generate_submission_id("infer", run_id, attempt_id)

        return ResolvedInference(
            plan_digest=plan_digest,
            model=resolved_model.selection,
            input=resolved_input,
            input_signature=resolved_model.input_signature,
            output_signature=resolved_model.output_signature,
            input_binding=request.input_binding,
            output_binding=request.output_binding,
            result_sink=request.result_sink,
            execution=request.execution,
            run_id=run_id,
            attempt_id=attempt_id,
            submission_id=submission_id,
            parent_run_id=request.parent_run_id,
        )


def _validate_bindings(
    request: InferenceRequest,
    *,
    input_signature: ManifestSignature,
    output_signature: ManifestSignature,
    unsafe: bool,
) -> None:
    _validate_side(
        side="input",
        bindings=(
            (binding.tensor_name, binding.dtype)
            for binding in request.input_binding.tensors
        ),
        fields=input_signature.input_fields,
        unsafe=unsafe,
    )
    _validate_side(
        side="output",
        bindings=(
            (binding.tensor_name, binding.dtype)
            for binding in request.output_binding.tensors
        ),
        fields=output_signature.output_fields,
        unsafe=unsafe,
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
    model: ResolvedModelSelection,
    resolved_input: ResolvedInputSelection,
) -> str:
    payload = {
        "schema_version": request.schema_version,
        "bundle": model.bundle_ref.model_dump(mode="json"),
        "role": model.role,
        "flavor_id": model.flavor_id,
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


def _default_model_reference_resolver(
    *,
    bundle_reader: Any | None,
    importers: Any | None,
) -> ModelReferenceResolver:
    """Resolve the compatibility adapter through top-level composition."""
    if bundle_reader is None and importers is None:
        from tributo.runtime import default_model_reference_resolver

        return default_model_reference_resolver()

    from tributo.integrations.model_runtimes import BundleModelReferenceResolver

    return BundleModelReferenceResolver(
        bundle_reader=bundle_reader,
        importers=importers,
    )


__all__ = ["InferenceResolver", "ModelReferenceResolver"]

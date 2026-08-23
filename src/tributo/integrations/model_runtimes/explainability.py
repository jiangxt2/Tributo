"""Bundle-backed explainability model provider."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np

from tributo.exceptions import UnsupportedArtifactFormat
from tributo.explainability.contracts import Exactness, ExplainabilityRequest
from tributo.explainability.protocols import (
    ExplainabilityModelBinding,
    ExplainableModelContext,
    NativeAttributionModel,
    ReferenceProvider,
)
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.capabilities import get_default_capability_registry
from tributo.exporting.manifest import compute_bundle_digest
from tributo.exporting.runtime import BundleModelLoader, BundleModelRuntime
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class BundleExplainabilityModelSession:
    """Own one verified Bundle model context and its temporary resources."""

    def __init__(
        self,
        context: ExplainableModelContext,
        *,
        output_count_upper_bound: int,
        runtime: BundleModelRuntime,
    ) -> None:
        self._context = context
        self._output_count_upper_bound = output_count_upper_bound
        self._runtime = runtime
        self._closed = False

    @property
    def context(self) -> ExplainableModelContext:
        return self._context

    @property
    def output_count_upper_bound(self) -> int:
        return self._output_count_upper_bound

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runtime.close()


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class BundleExplainabilityModelSessionFactory:
    """Open one verified Bundle model session inside a Ray worker."""

    factory_id: ClassVar[str] = "tributo.bundle-explainability-session-v1"
    _request: ExplainabilityRequest
    _manifest_bytes: bytes = field(repr=False)

    def create(
        self, reference_provider: ReferenceProvider
    ) -> BundleExplainabilityModelSession:
        return BundleExplainabilityModelProvider().open(
            self._request,
            self._manifest_bytes,
            reference_provider,
        )


@PublicAPI(stability="alpha")
class BundleExplainabilityModelProvider:
    """Load Bundle model capabilities without exposing formats to the worker."""

    provider_id: ClassVar[str] = "tributo.bundle-explainability-v1"

    def resolve(self, request: ExplainabilityRequest) -> ExplainabilityModelBinding:
        """Resolve Bundle metadata without exposing Bundle types to the domain."""
        manifest, manifest_bytes = BundleReader().read_manifest_with_bytes(
            request.bundle_uri,
            storage_profile=request.storage_profile,
        )
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if (
            request.expected_manifest_sha256 is not None
            and request.expected_manifest_sha256 != manifest_sha256
        ):
            raise ValueError(
                "expected_manifest_sha256 does not match the Bundle manifest"
            )
        if request.bundle_id is not None and request.bundle_id != manifest.bundle_id:
            raise ValueError("bundle_id does not match the Bundle manifest")
        validate_explainability_request(manifest, request)
        descriptor = getattr(manifest, "explainability", None)
        if descriptor is None:  # validated above; retained for static narrowing.
            raise ValueError("Bundle explainability descriptor is missing")
        model_role = selected_model_role(manifest, request)
        artifact = selected_artifact(manifest, request)
        backend, exactness = selected_backend(manifest, request)
        return ExplainabilityModelBinding(
            bundle_id=manifest.bundle_id,
            bundle_digest=compute_bundle_digest(
                artifacts=tuple(manifest.artifacts),
                roles=dict(manifest.roles),
                input_sig=manifest.input_signature,
                output_sig=manifest.output_signature,
                explainability=descriptor,
            ),
            manifest_sha256=manifest_sha256,
            model_role=model_role,
            model_digest=str(artifact.tree_digest),
            preprocessor_digest=manifest_role_digest(manifest, "preprocessor"),
            feature_map_digest=manifest_role_digest(manifest, "feature_map"),
            descriptor=descriptor,
            backend=backend,
            exactness=exactness,
            output_count_upper_bound=_output_count_upper_bound(
                manifest,
                request,
                native=backend == "tree",
            ),
            session_factory=BundleExplainabilityModelSessionFactory(
                _request=request,
                _manifest_bytes=manifest_bytes,
            ),
            dependency_versions=_dependency_versions(backend),
        )

    def open(
        self,
        request: ExplainabilityRequest,
        manifest_bytes: bytes,
        reference_provider: ReferenceProvider,
    ) -> BundleExplainabilityModelSession:
        reader = BundleReader()
        manifest = reader._parse_manifest_bytes(manifest_bytes)
        model_role = selected_model_role(manifest, request)
        runtime: BundleModelRuntime | None = None
        try:
            runtime = BundleModelLoader().open(
                request.bundle_uri,
                role=model_role,
                storage_profile=request.storage_profile,
                expected_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                use_case="batch",
            )
            artifact = runtime.artifact
            capability = get_default_capability_registry().for_flavor(
                artifact.flavor_id
            )
            native_model = (
                runtime.model
                if isinstance(runtime.model, NativeAttributionModel)
                else None
            )
            native = bool(
                native_model is not None
                and native_model.native_attribution_id is not None
                and "attribution.tree-shap" in capability.conditional_operations
            )
            requested_backend, _ = selected_backend(manifest, request)
            if requested_backend == "tree" and not native:
                raise UnsupportedArtifactFormat(
                    "The loaded model does not satisfy the native TreeSHAP "
                    "booster/objective capability contract"
                )
            if native:
                context = _native_runtime_context(
                    runtime,
                    manifest=manifest,
                    artifact=artifact,
                    request=request,
                    model_role=model_role,
                    reference_provider=reference_provider,
                )
            else:
                context = _prediction_runtime_context(
                    runtime,
                    manifest=manifest,
                    artifact=artifact,
                    request=request,
                    model_role=model_role,
                    reference_provider=reference_provider,
                )
            return BundleExplainabilityModelSession(
                context,
                output_count_upper_bound=_output_count_upper_bound(
                    manifest, request, native=native
                ),
                runtime=runtime,
            )
        except BaseException:
            if runtime is not None:
                runtime.close()
            raise


def _prediction_runtime_context(
    runtime: BundleModelRuntime,
    *,
    manifest: Any,
    artifact: Any,
    request: ExplainabilityRequest,
    model_role: str,
    reference_provider: ReferenceProvider,
) -> ExplainableModelContext:
    input_names = tuple(runtime.model.input_names)
    input_dtypes = tuple(np.dtype(dtype) for dtype in runtime.model.input_dtypes)
    input_shapes = tuple(runtime.model.input_shapes)
    output_names = tuple(runtime.model.output_names)
    preprocessor = load_onnx_preprocessor(runtime)
    feature_map = load_onnx_feature_map(runtime)
    if preprocessor is not None and request.feature_view == "raw":
        if feature_map is None:
            raise ValueError(
                "raw DNN/PU explainability requires a verified feature_map.json"
            )
        validate_feature_map(
            feature_map,
            tuple(feature.name for feature in preprocessor.features),
            input_names,
        )
    feature_names = onnx_feature_names(request, input_names, preprocessor)

    def predict(values: np.ndarray) -> np.ndarray:
        inputs = build_onnx_inputs(
            values,
            input_names=input_names,
            input_dtypes=input_dtypes,
            input_shapes=input_shapes,
            feature_names=feature_names,
            preprocessor=preprocessor,
            feature_view=request.feature_view,
        )
        outputs = runtime.predict(inputs)
        result = select_onnx_output(
            outputs, output_names, output_target=request.output_target
        )
        return result[:, 0] if result.ndim == 2 and result.shape[1] == 1 else result

    return ExplainableModelContext(
        bundle_uri=request.bundle_uri,
        model_role=model_role,
        artifact_name=artifact.name,
        artifact_format=artifact.format,
        flavor_id=artifact.flavor_id,
        artifact_path=None,
        feature_names=feature_names,
        predict=predict,
        model_digest=artifact.tree_digest,
        preprocessor_digest=manifest_role_digest(manifest, "preprocessor"),
        feature_map_digest=manifest_role_digest(manifest, "feature_map"),
        metadata={
            "reference_data": _load_reference(request, reference_provider),
            "feature_map": feature_map,
        },
    )


def _native_runtime_context(
    runtime: BundleModelRuntime,
    *,
    manifest: Any,
    artifact: Any,
    request: ExplainabilityRequest,
    model_role: str,
    reference_provider: ReferenceProvider,
) -> ExplainableModelContext:
    model = runtime.model
    if not isinstance(model, NativeAttributionModel):
        raise TypeError("Runtime did not provide native attribution capability")
    native_model = model.native_model_object
    return ExplainableModelContext(
        bundle_uri=request.bundle_uri,
        model_role=model_role,
        artifact_name=artifact.name,
        artifact_format=artifact.format,
        flavor_id=artifact.flavor_id,
        artifact_path=None,
        model_object=native_model,
        feature_names=resolve_xgboost_feature_names(
            runtime.resolved_artifact,
            native_model,
            request,
        )
        or model.native_feature_names,
        objective=model.native_objective,
        model_digest=artifact.tree_digest,
        preprocessor_digest=manifest_role_digest(manifest, "preprocessor"),
        feature_map_digest=manifest_role_digest(manifest, "feature_map"),
        native_attribution_id=model.native_attribution_id,
        metadata={
            "reference_data": _load_reference(request, reference_provider),
        },
    )


def selected_model_role(manifest: Any, request: ExplainabilityRequest) -> str:
    if request.model_role is not None:
        return request.model_role
    descriptor = getattr(manifest, "explainability", None)
    if descriptor is not None:
        for role in descriptor.model_roles:
            if role in manifest.roles:
                return str(role)
    if request.backend in {"auto", "tree"} and "explainability_model" in manifest.roles:
        return "explainability_model"
    return "inference"


def selected_artifact(manifest: Any, request: ExplainabilityRequest) -> Any:
    role = selected_model_role(manifest, request)
    target_name = manifest.roles.get(role)
    if target_name is None:
        raise ValueError(f"Bundle role {role!r} is not present")
    try:
        return next(
            artifact for artifact in manifest.artifacts if artifact.name == target_name
        )
    except StopIteration as exc:
        raise ValueError(
            f"Bundle role {role!r} points to missing artifact {target_name!r}"
        ) from exc


def manifest_role_digest(manifest: Any, role: str) -> str | None:
    target_name = manifest.roles.get(role)
    if target_name is not None:
        for artifact in manifest.artifacts:
            if artifact.name == target_name:
                return str(artifact.tree_digest)
        raise ValueError(
            f"Bundle role {role!r} points to missing artifact {target_name!r}"
        )
    matches = [
        file
        for artifact in manifest.artifacts
        for file in artifact.files
        if file.role == role
        or (role == "feature_map" and file.relative_path == "feature_map.json")
    ]
    if len(matches) > 1:
        raise ValueError(f"Bundle contains multiple files with role {role!r}")
    return str(matches[0].sha256) if matches else None


def selected_backend(
    manifest: Any, request: ExplainabilityRequest
) -> tuple[str, Exactness]:
    descriptor = getattr(manifest, "explainability", None)
    if request.backend != "auto":
        backend = request.backend
    elif descriptor is not None and descriptor.backend != "auto":
        backend = descriptor.backend
    else:
        raise ValueError(
            "Explainability backend must be explicit in the request or Bundle "
            "descriptor"
        )
    exactness: Exactness = cast(
        Exactness,
        {"tree": "exact", "model_agnostic": "approximate"}.get(backend, "conditional"),
    )
    if descriptor is not None and descriptor.backend == backend:
        exactness = descriptor.exactness
    return backend, exactness


def validate_explainability_request(
    manifest: Any, request: ExplainabilityRequest
) -> None:
    descriptor = getattr(manifest, "explainability", None)
    if descriptor is None:
        raise ValueError(
            "Bundle does not declare an explainability descriptor; enable "
            "explainability during Bundle export before submitting a request"
        )
    expected_adapter = descriptor.adapter_id
    if request.explainer != expected_adapter.removesuffix("-v1"):
        raise ValueError(
            f"request explainer {request.explainer!r} does not match Bundle "
            f"descriptor adapter {expected_adapter!r}"
        )
    if request.backend not in {"auto", descriptor.backend}:
        raise ValueError(
            f"request backend {request.backend!r} is not declared by Bundle "
            f"descriptor ({descriptor.backend!r})"
        )
    if request.feature_view != descriptor.feature_view:
        raise ValueError(
            f"request feature_view {request.feature_view!r} does not match "
            f"Bundle descriptor ({descriptor.feature_view!r})"
        )
    if request.output_target != descriptor.output_target:
        raise ValueError(
            f"request output_target {request.output_target!r} does not match "
            f"Bundle descriptor ({descriptor.output_target!r})"
        )
    if descriptor.reference_policy == "required" and request.reference is None:
        raise ValueError("Bundle descriptor requires a reference binding")
    role_targets = {
        manifest.roles[role]
        for role in descriptor.model_roles
        if role in manifest.roles
    }
    selected_role = selected_model_role(manifest, request)
    request_target = manifest.roles.get(selected_role)
    if not role_targets:
        raise ValueError(
            "Bundle explainability descriptor does not resolve any declared model role"
        )
    if request_target not in role_targets:
        raise ValueError(
            f"request model role {selected_role!r} is not one of the descriptor's "
            "declared model roles"
        )


def load_onnx_preprocessor(runtime: BundleModelRuntime) -> Any | None:
    path = runtime.resolved_artifact.path_for("preprocessor.json")
    if not path.is_file():
        return None
    from tributo.training.features.transformer import FeatureTransformer

    return FeatureTransformer.load(path)


def load_onnx_feature_map(runtime: BundleModelRuntime) -> dict[str, Any] | None:
    path = runtime.resolved_artifact.path_for("feature_map.json")
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("feature_map.json must contain a JSON object")
    return raw


def validate_feature_map(
    feature_map: dict[str, Any],
    raw_names: tuple[str, ...],
    input_names: tuple[str, ...],
) -> None:
    if feature_map.get("schema_version") != 1:
        raise ValueError("unsupported feature_map.json schema_version")
    mappings = feature_map.get("mappings")
    if not isinstance(mappings, list):
        raise ValueError("feature_map.json mappings must be a list")
    pairs = {
        (item.get("raw_feature"), item.get("model_input"))
        for item in mappings
        if isinstance(item, dict)
    }
    expected = {(name, name) for name in raw_names}
    if pairs != expected or set(input_names) != set(raw_names):
        raise ValueError(
            "feature_map.json does not provide a one-to-one raw/model input map"
        )


def onnx_feature_names(
    request: ExplainabilityRequest,
    input_names: tuple[str, ...],
    preprocessor: Any | None,
) -> tuple[str, ...]:
    if request.feature_columns:
        return tuple(request.feature_columns)
    if preprocessor is not None:
        return tuple(feature.name for feature in preprocessor.features)
    return input_names


def build_onnx_inputs(
    values: np.ndarray,
    *,
    input_names: tuple[str, ...],
    input_dtypes: tuple[np.dtype[Any], ...],
    input_shapes: tuple[tuple[int | None, ...], ...],
    feature_names: tuple[str, ...],
    preprocessor: Any | None,
    feature_view: str,
) -> dict[str, np.ndarray]:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"ONNX explainability input must be 2-D, got {values.shape}")
    if len(input_names) != len(input_dtypes) or len(input_names) != len(input_shapes):
        raise ValueError("ONNX input metadata cardinality is inconsistent")

    transformed: dict[str, np.ndarray] | None = None
    if preprocessor is not None and feature_view == "raw":
        positions = {name: index for index, name in enumerate(feature_names)}
        missing = [
            feature.name
            for feature in preprocessor.features
            if feature.name not in positions
        ]
        if missing:
            raise ValueError(
                f"raw explainability input is missing preprocessor features {missing}"
            )
        raw = {
            feature.name: values[:, positions[feature.name]]
            for feature in preprocessor.features
        }
        transformed = preprocessor.transform(raw)

    if transformed is not None:
        missing = [name for name in input_names if name not in transformed]
        if missing:
            raise ValueError(f"preprocessor output is missing ONNX inputs {missing}")
        arrays = [transformed[name] for name in input_names]
    elif len(input_names) == 1:
        arrays = [values]
    else:
        positions = {name: index for index, name in enumerate(feature_names)}
        if all(name in positions for name in input_names):
            arrays = [values[:, positions[name]] for name in input_names]
        elif values.shape[1] == len(input_names):
            arrays = [values[:, index] for index in range(len(input_names))]
        else:
            raise ValueError(
                "multi-input ONNX explanation requires feature columns matching "
                "the model input names"
            )

    bound: dict[str, np.ndarray] = {}
    for name, array, dtype, shape in zip(
        input_names, arrays, input_dtypes, input_shapes, strict=True
    ):
        candidate = np.asarray(array, dtype=dtype)
        if candidate.ndim == 1 and len(shape) > 1 and shape[-1] not in (None, 1):
            raise ValueError(
                f"ONNX input {name!r} requires shape {shape}, got {candidate.shape}"
            )
        bound[name] = candidate
    return bound


def select_onnx_output(
    outputs: dict[str, np.ndarray],
    output_names: tuple[str, ...],
    *,
    output_target: str,
) -> np.ndarray:
    if tuple(outputs) != output_names:
        raise ValueError(
            f"ONNX runtime outputs {tuple(outputs)!r} do not match verified "
            f"signature {output_names!r}"
        )
    if output_target in outputs:
        return np.asarray(outputs[output_target])
    if len(output_names) == 1 and output_target == "model_output":
        return np.asarray(outputs[output_names[0]])
    if output_target == "probability":
        candidates = tuple(
            name
            for name in output_names
            if name.lower()
            in {"probability", "probabilities", "proba", "score", "scores"}
        )
        if len(candidates) == 1:
            return np.asarray(outputs[candidates[0]])
    raise ValueError(
        f"ONNX output_target={output_target!r} is ambiguous for outputs "
        f"{output_names!r}; use a declared output name or a single-output model"
    )


def resolve_xgboost_feature_names(
    resolved: Any, booster: Any, request: ExplainabilityRequest
) -> tuple[str, ...]:
    sidecar_names: tuple[str, ...] = ()
    sidecar = resolved.path_for("feature_names.json")
    if sidecar.is_file():
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("XGBoost feature_names.json must contain a string list")
        sidecar_names = tuple(raw)

    booster_names = tuple(booster.feature_names or ())
    requested_names = tuple(request.feature_columns)
    if requested_names and booster_names and requested_names != booster_names:
        raise ValueError(
            "XGBoost request feature names do not match the booster feature order"
        )
    for label, names in (("request", requested_names), ("booster", booster_names)):
        if names and sidecar_names and names != sidecar_names:
            raise ValueError(
                f"XGBoost {label} feature names do not match feature_names.json"
            )
    resolved_names = requested_names or sidecar_names or booster_names
    if resolved_names and len(resolved_names) != booster.num_features():
        raise ValueError("XGBoost feature names count does not match the booster")
    return resolved_names


def _load_reference(
    request: ExplainabilityRequest,
    reference_provider: ReferenceProvider,
) -> np.ndarray | None:
    if request.reference is None:
        return None
    return reference_provider.resolve(request.reference, request.limits).data


def _output_count_upper_bound(
    manifest: Any,
    request: ExplainabilityRequest,
    *,
    native: bool,
) -> int:
    if not native:
        return 1 if request.output_target in {"raw", "raw_margin"} else 2
    signature = getattr(manifest, "output_signature", None)
    fields = tuple(getattr(signature, "output_fields", ()))
    probability_fields = tuple(
        field
        for field in fields
        if str(getattr(field, "name", "")).lower()
        in {"probability", "probabilities", "proba", "scores"}
    )
    prediction_fields = tuple(
        field
        for field in fields
        if str(getattr(field, "name", "")).lower() in {"prediction", "predictions"}
    )
    task_type = getattr(getattr(manifest, "source_info", None), "task_type", None)
    if task_type == "regression":
        candidates = prediction_fields
    elif task_type == "classification":
        candidates = probability_fields
    else:
        candidates = probability_fields or prediction_fields
    if len(candidates) != 1:
        raise ValueError(
            "Native tree explainability requires one typed probability or "
            "prediction output signature"
        )
    shape = tuple(getattr(candidates[0], "shape", ()))
    if len(shape) != 2 or not isinstance(shape[1], int) or shape[1] < 1:
        raise ValueError(
            "Native tree explainability requires a fixed output dimension "
            "in the typed manifest signature"
        )
    return shape[1]


def _dependency_versions(backend: str) -> tuple[tuple[str, str], ...]:
    from importlib.metadata import PackageNotFoundError, version

    packages = ["shap", "numpy"]
    if backend == "tree":
        packages.append("xgboost")
    elif backend == "model_agnostic":
        packages.append("onnxruntime")
    versions: list[tuple[str, str]] = []
    for package in packages:
        try:
            versions.append((package, version(package)))
        except PackageNotFoundError:
            continue
    return tuple(versions)


__all__ = [
    "BundleExplainabilityModelProvider",
    "BundleExplainabilityModelSession",
    "BundleExplainabilityModelSessionFactory",
]

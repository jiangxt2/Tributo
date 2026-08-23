"""Ray Data batch execution for explainability operations.

The executor intentionally mirrors the existing batch-inference skeleton:
input resolution is delegated to the ingestion boundary, model state lives in
Ray actors, and output is streamed through the existing Parquet sink.  SHAP is
never imported on the driver or in the disabled path.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any, Literal, cast
from urllib.parse import unquote, urlsplit, urlunsplit
from uuid import uuid4

import numpy as np

from tributo.data.persistence import (
    default_object_store,
    inspect_parquet_output,
)
from tributo.exceptions import ResultMaterializationError
from tributo.explainability.contracts import (
    Exactness,
    ExplainabilityOperationRecord,
    ExplainabilityReceipt,
    ExplainabilityRequest,
    ReferencePolicy,
)
from tributo.explainability.planner import ExplainabilityPlanner
from tributo.explainability.protocols import (
    ExplainableModelContext,
    ReferenceProvider,
)
from tributo.explainability.reference import FileReferenceProvider, ResolvedReference
from tributo.explainability.registry import ExplainerRegistry
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.manifest import compute_bundle_digest
from tributo.exporting.records import InMemoryOperationStore, OperationStore
from tributo.exporting.runtime import BundleModelLoader, BundleModelRuntime
from tributo.inference.contracts import ParquetResultSinkRequest
from tributo.inference.input_resolver import (
    IngestionGatewayInputResolver,
    InputResolverPort,
    OpenedInferenceInput,
)
from tributo.integrations.sinks.parquet import ParquetResultSink
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


class _ExplainabilityLimitExceeded(ValueError):
    """A policy rejection that must clean materialized output."""


class _LeaseHeartbeat:
    """Renew one operation lease while the Ray Data job is active."""

    def __init__(
        self,
        store: OperationStore,
        *,
        operation_id: str,
        idempotency_key: str,
        lease_token: str,
        lease_seconds: int,
    ) -> None:
        self._store = store
        self._operation_id = operation_id
        self._idempotency_key = idempotency_key
        self._lease_token = lease_token
        self._lease_seconds = lease_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        self._thread = Thread(
            target=self._run,
            name=f"tributo-explainability-lease-{self._operation_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "explainability operation lease renewal failed; increase "
                f"operation_lease_seconds (currently {self._lease_seconds}) for "
                "slow or paused environments"
            ) from self._error

    def _run(self) -> None:
        interval = max(0.5, self._lease_seconds / 3)
        while not self._stop.wait(interval):
            now = datetime.now(timezone.utc)
            try:
                self._store.renew_explainability(
                    self._operation_id,
                    idempotency_key=self._idempotency_key,
                    lease_token=self._lease_token,
                    lease_expires_at=now + timedelta(seconds=self._lease_seconds),
                )
            except Exception as exc:
                self._error = exc
                self._stop.set()
                return


@PublicAPI(stability="alpha")
def run_batch_explainability(
    request: ExplainabilityRequest,
    *,
    input_resolver: InputResolverPort | None = None,
    explainer_registry: ExplainerRegistry | None = None,
    operation_store: OperationStore | None = None,
    reference_provider: ReferenceProvider | None = None,
) -> ExplainabilityReceipt:
    """Execute one bounded explainability request through Ray Data."""
    import ray.data

    operation_id = request.operation_id or request.request_id
    resolver = input_resolver or IngestionGatewayInputResolver()
    store = operation_store or _operation_store_for_request(request)
    references = reference_provider or FileReferenceProvider()
    manifest, manifest_bytes = BundleReader().read_manifest_with_bytes(
        request.bundle_uri,
        storage_profile=request.storage_profile,
    )
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if (
        request.expected_manifest_sha256 is not None
        and request.expected_manifest_sha256 != actual_manifest_sha256
    ):
        raise ValueError("expected_manifest_sha256 does not match the Bundle manifest")
    if request.bundle_id is not None and request.bundle_id != manifest.bundle_id:
        raise ValueError("bundle_id does not match the Bundle manifest")
    bundle_digest = _bundle_digest(manifest)
    idempotency_key = _operation_idempotency_key(
        manifest,
        request,
        bundle_digest=bundle_digest,
        reference_provider=references,
    )
    existing = store.get_explainability(operation_id)
    if existing is not None:
        if existing.idempotency_key != idempotency_key:
            raise ValueError(
                f"operation {operation_id!r} was previously submitted with "
                "different inputs"
            )
        if existing.status in {"succeeded", "partial"}:
            cached = _read_receipt(
                existing.receipt_uri or _receipt_uri(existing.result_uri),
                storage_profile=request.result_storage_profile,
            )
            if cached is not None:
                return cached
            raise ValueError(
                f"operation {operation_id!r} is terminal but its receipt is missing"
            )
        expired_running = existing.status == "running" and (
            existing.lease_expires_at is not None
            and existing.lease_expires_at <= datetime.now(timezone.utc)
        )
        if existing.status == "running" and not expired_running:
            raise ValueError(f"operation {operation_id!r} is already running")
        if existing.status == "running" and not request.force_resume:
            raise ValueError(
                f"operation {operation_id!r} lease expired; set force_resume=true "
                "only after confirming the previous driver is no longer active"
            )
        if existing.status != "running" and not existing.retryable:
            raise ValueError(f"operation {operation_id!r} is not retryable")
    retry_count = existing.retries + 1 if existing is not None else 0
    lease_token = uuid4().hex
    attempt_result_uri = _attempt_result_uri(request.result_uri, lease_token)
    _record_operation(
        store,
        ExplainabilityOperationRecord(
            operation_id=operation_id,
            request_id=request.request_id,
            bundle_id=manifest.bundle_id,
            bundle_digest=bundle_digest,
            result_uri=attempt_result_uri,
            reference_digest=_safe_reference_digest(request, references),
            idempotency_key=idempotency_key,
            status="running",
            retries=retry_count,
            lease_token=lease_token,
            lease_expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=request.operation_lease_seconds),
        ),
    )
    heartbeat = _LeaseHeartbeat(
        store,
        operation_id=operation_id,
        idempotency_key=idempotency_key,
        lease_token=lease_token,
        lease_seconds=request.operation_lease_seconds,
    )
    heartbeat.start()

    opened: OpenedInferenceInput | None = None
    worker: type[ExplainabilityBatchWorker] = ExplainabilityBatchWorker
    input_rows = 0
    explanation_rows = 0
    result_digest: str | None = None
    result_bytes = 0
    try:
        _validate_request_against_descriptor(manifest, request)
        output_count = _explanation_output_count_upper_bound(manifest, request)
        selection = resolver.describe(request.input)
        opened = resolver.open(selection)
        dataset = opened.dataset
        input_rows = dataset.count()
        if request.limits.max_rows is not None:
            if input_rows > request.limits.max_rows:
                raise ValueError(
                    f"input rows {input_rows} exceed limits.max_rows="
                    f"{request.limits.max_rows}"
                )
        ExplainabilityPlanner.preflight_limits(
            request,
            input_rows=input_rows,
            output_count=output_count,
        )

        explained = dataset.map_batches(
            worker,
            fn_constructor_args=(
                request,
                manifest_bytes,
                explainer_registry,
                references,
            ),
            batch_format="pandas",
            batch_size=request.resource_policy.batch_size,
            compute=ray.data.ActorPoolStrategy(
                size=request.resource_policy.concurrency
            ),
            num_cpus=request.resource_policy.num_cpus_per_actor,
            num_gpus=request.resource_policy.num_gpus_per_actor,
        )
        sink_request = ParquetResultSinkRequest(
            uri=attempt_result_uri,
            storage_profile=request.result_storage_profile,
            max_bytes=request.limits.max_explanation_bytes,
        )
        ParquetResultSink().write(
            explained,
            sink_request,
            run_id=operation_id,
            plan_digest=bundle_digest,
        )
        result_digest, result_bytes, explanation_rows = _result_stats(
            attempt_result_uri,
            storage_profile=request.result_storage_profile,
        )
        if (
            request.limits.max_explanation_rows is not None
            and explanation_rows > request.limits.max_explanation_rows
        ):
            raise _ExplainabilityLimitExceeded(
                f"explanation output rows {explanation_rows} exceed "
                f"limits.max_explanation_rows={request.limits.max_explanation_rows}"
            )
        heartbeat.raise_if_failed()
        selected_backend, exactness = _selected_backend(manifest, request)
        receipt = _make_receipt(
            manifest=manifest,
            request=request,
            operation_id=operation_id,
            bundle_digest=bundle_digest,
            selected_backend=selected_backend,
            exactness=exactness,
            input_rows=input_rows,
            explanation_rows=explanation_rows,
            result_digest=result_digest,
            result_bytes=result_bytes,
            result_uri=attempt_result_uri,
            status="succeeded",
            reference_provider=references,
        )
        _write_receipt(
            attempt_result_uri,
            receipt,
            storage_profile=request.result_storage_profile,
        )
        _record_operation(
            store,
            ExplainabilityOperationRecord(
                operation_id=operation_id,
                request_id=request.request_id,
                bundle_id=manifest.bundle_id,
                bundle_digest=bundle_digest,
                result_uri=attempt_result_uri,
                receipt_uri=_receipt_uri(attempt_result_uri),
                reference_digest=_safe_reference_digest(request, references),
                idempotency_key=idempotency_key,
                status="succeeded",
                input_rows=receipt.input_rows,
                explanation_rows=receipt.explanation_rows,
                retries=retry_count,
                lease_token=lease_token,
                lease_expires_at=None,
                completed_at=datetime.now(timezone.utc),
            ),
        )
        return receipt
    except Exception as exc:
        logger.exception("Explainability operation failed")
        limit_failure = isinstance(exc, _ExplainabilityLimitExceeded)
        partial = (
            explanation_rows > 0 and result_digest is not None and not limit_failure
        )
        if not partial:
            _cleanup_result_uri(
                attempt_result_uri,
                storage_profile=request.result_storage_profile,
            )
        if partial:
            try:
                selected_backend, exactness = _selected_backend(manifest, request)
                _write_receipt(
                    attempt_result_uri,
                    _make_receipt(
                        manifest=manifest,
                        request=request,
                        operation_id=operation_id,
                        bundle_digest=bundle_digest,
                        selected_backend=selected_backend,
                        exactness=exactness,
                        input_rows=input_rows,
                        explanation_rows=explanation_rows,
                        result_digest=result_digest,
                        result_bytes=result_bytes,
                        result_uri=attempt_result_uri,
                        status="partial",
                        failure_code=type(exc).__name__,
                        reference_provider=references,
                    ),
                    storage_profile=request.result_storage_profile,
                )
            except Exception:
                logger.warning(
                    "Failed to write partial explainability receipt",
                    exc_info=True,
                )
        _record_operation(
            store,
            ExplainabilityOperationRecord(
                operation_id=operation_id,
                request_id=request.request_id,
                bundle_id=manifest.bundle_id,
                bundle_digest=bundle_digest,
                result_uri=attempt_result_uri,
                receipt_uri=_receipt_uri(attempt_result_uri),
                reference_digest=_safe_reference_digest(request, references),
                idempotency_key=idempotency_key,
                status="partial" if partial else "failed",
                failure_phase="limit" if limit_failure else "execution",
                failure_code=type(exc).__name__,
                retryable=_is_retryable_error(exc),
                input_rows=input_rows,
                explanation_rows=explanation_rows,
                retries=retry_count,
                lease_token=lease_token,
                lease_expires_at=None,
                completed_at=datetime.now(timezone.utc),
            ),
        )
        raise RuntimeError(
            f"Explainability operation {operation_id!r} failed: {type(exc).__name__}"
        ) from exc
    finally:
        heartbeat.stop()
        if opened is not None:
            try:
                opened.close()
            except Exception:
                logger.warning("Explainability input close failed", exc_info=True)


def _make_receipt(
    *,
    manifest: Any,
    request: ExplainabilityRequest,
    operation_id: str,
    bundle_digest: str,
    selected_backend: str,
    exactness: Exactness,
    input_rows: int,
    explanation_rows: int,
    result_digest: str | None,
    result_bytes: int,
    result_uri: str,
    status: Literal["succeeded", "partial", "failed"],
    reference_provider: ReferenceProvider,
    failure_code: str | None = None,
) -> ExplainabilityReceipt:
    if result_digest is None:
        raise ValueError("cannot create a receipt without a result digest")
    return ExplainabilityReceipt(
        request_id=request.request_id,
        operation_id=operation_id,
        bundle_id=manifest.bundle_id,
        bundle_digest=bundle_digest,
        model_digest=_model_digest(manifest, request),
        preprocessor_digest=_manifest_role_digest(manifest, "preprocessor"),
        feature_map_digest=_manifest_role_digest(manifest, "feature_map"),
        reference_digest=_safe_reference_digest(request, reference_provider),
        reference_rows=_reference_rows(request, reference_provider),
        reference_privacy_level=(
            request.reference.privacy_level if request.reference else None
        ),
        reference_ttl_seconds=(
            request.reference.ttl_seconds if request.reference else None
        ),
        adapter_id=f"{request.explainer}-v1",
        adapter_version="1",
        backend=selected_backend,
        exactness=exactness,
        feature_view=request.feature_view,
        output_target=request.output_target,
        output_selection=request.output_selection,
        input_rows=input_rows,
        explanation_rows=explanation_rows,
        result_uri=result_uri,
        result_digest=result_digest,
        result_bytes=result_bytes,
        result_access_scope=request.result_policy.access_scope,
        result_privacy_level=request.result_policy.privacy_level,
        result_retention_seconds=request.result_policy.retention_seconds,
        reference_policy=_reference_policy(manifest, request),
        dependency_versions=_dependency_versions(selected_backend),
        schema_signature=_schema_signature(),
        resource_summary={
            "batch_size": request.resource_policy.batch_size,
            "concurrency": request.resource_policy.concurrency,
            "num_cpus_per_actor": request.resource_policy.num_cpus_per_actor,
            "num_gpus_per_actor": request.resource_policy.num_gpus_per_actor,
        },
        status=status,
        failure_code=failure_code,
    )


@PublicAPI(stability="alpha")
class ExplainabilityBatchWorker:
    """Stateful Ray actor that owns one verified model and SHAP object."""

    def __init__(
        self,
        request: ExplainabilityRequest,
        manifest_bytes: bytes,
        registry: ExplainerRegistry | None,
        reference_provider: ReferenceProvider,
    ) -> None:
        self._request = request
        self._manifest_bytes = manifest_bytes
        self._reader = BundleReader()
        self._manifest = self._reader._parse_manifest_bytes(manifest_bytes)
        self._reference_provider = reference_provider
        self._runtime: BundleModelRuntime | None = None
        self._artifact_stack = ExitStack()
        self._context = self._load_context(request)
        plan = ExplainabilityPlanner(registry).plan(
            self._context,
            request,
            output_count=_explanation_output_count_upper_bound(
                self._manifest,
                request,
            ),
        )
        self._plan = plan
        self._prepared = plan.adapter().prepare(self._context, request)

    def __call__(self, batch: Any) -> Any:
        import pandas as pd

        columns = request_columns(self._request, self._context)
        required = list(columns)
        if self._request.label_column is not None:
            required.append(self._request.label_column)
        missing = [column for column in required if column not in batch]
        if missing:
            raise ValueError(f"explainability input is missing columns {missing}")
        values = batch.loc[:, list(columns)].to_numpy()
        values = np.asarray(values)
        if self._context.flavor_id == "xgboost-native-v1":
            values = np.asarray(values, dtype=np.float32)
        labels = None
        if self._request.label_column is not None:
            labels = batch[self._request.label_column].to_numpy()
        input_ids = _input_ids(batch, self._request, values)
        rows = self._plan.adapter().explain_batch(
            self._prepared,
            values,
            input_ids=input_ids,
            model_digest=self._context.model_digest,
            request=self._request,
            labels=labels,
        )
        return pd.DataFrame([row.model_dump() for row in rows])

    def _load_context(self, request: ExplainabilityRequest) -> ExplainableModelContext:
        manifest = self._manifest
        model_role = _selected_model_role(manifest, request)
        target_name = manifest.roles.get(model_role)
        if target_name is None:
            raise ValueError(f"Bundle role {model_role!r} is not present")
        try:
            artifact = next(a for a in manifest.artifacts if a.name == target_name)
        except StopIteration as exc:
            raise ValueError(
                f"Bundle role {model_role!r} points to missing artifact {target_name!r}"
            ) from exc
        if artifact.flavor_id == "xgboost-native-v1":
            resolved = self._artifact_stack.enter_context(
                self._reader.open_artifact(
                    request.bundle_uri,
                    role=model_role,
                    storage_profile=request.storage_profile,
                    manifest=manifest,
                    manifest_bytes=self._manifest_bytes,
                )
            )
            import json as _json

            import xgboost

            booster = xgboost.Booster()
            booster.load_model(str(resolved.entrypoint_path))
            objective = None
            try:
                objective = _json.loads(booster.save_config())["learner"]["objective"][
                    "name"
                ]
            except (KeyError, TypeError, ValueError):
                pass
            return ExplainableModelContext(
                bundle_uri=request.bundle_uri,
                model_role=model_role,
                artifact_name=artifact.name,
                artifact_format=artifact.format,
                flavor_id=artifact.flavor_id,
                artifact_path=resolved.entrypoint_path,
                model_object=booster,
                feature_names=_resolve_xgboost_feature_names(
                    resolved, booster, request
                ),
                objective=objective,
                model_digest=artifact.tree_digest,
                preprocessor_digest=_manifest_role_digest(manifest, "preprocessor"),
                feature_map_digest=_manifest_role_digest(manifest, "feature_map"),
                metadata={
                    "reference_data": _load_reference(request, self._reference_provider)
                },
            )

        runtime = BundleModelLoader().open(
            request.bundle_uri,
            role=model_role,
            storage_profile=request.storage_profile,
            expected_manifest_sha256=hashlib.sha256(self._manifest_bytes).hexdigest(),
            use_case="batch",
        )
        self._runtime = runtime
        input_names = tuple(runtime.model.input_names)
        input_dtypes = tuple(np.dtype(dtype) for dtype in runtime.model.input_dtypes)
        input_shapes = tuple(runtime.model.input_shapes)
        output_names = tuple(runtime.model.output_names)
        preprocessor = _load_onnx_preprocessor(runtime)
        feature_map = _load_onnx_feature_map(runtime)
        if preprocessor is not None and request.feature_view == "raw":
            if feature_map is None:
                raise ValueError(
                    "raw DNN/PU explainability requires a verified feature_map.json"
                )
            _validate_feature_map(
                feature_map,
                tuple(feature.name for feature in preprocessor.features),
                input_names,
            )
        feature_names = _onnx_feature_names(request, input_names, preprocessor)

        def predict(values: np.ndarray) -> np.ndarray:
            inputs = _build_onnx_inputs(
                values,
                input_names=input_names,
                input_dtypes=input_dtypes,
                input_shapes=input_shapes,
                feature_names=feature_names,
                preprocessor=preprocessor,
                feature_view=request.feature_view,
            )
            outputs = runtime.predict(inputs)
            result = _select_onnx_output(
                outputs, output_names, output_target=request.output_target
            )
            return result[:, 0] if result.ndim == 2 and result.shape[1] == 1 else result

        reference_data = _load_reference(request, self._reference_provider)
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
            preprocessor_digest=_manifest_role_digest(manifest, "preprocessor"),
            feature_map_digest=_manifest_role_digest(manifest, "feature_map"),
            metadata={"reference_data": reference_data, "feature_map": feature_map},
        )

    def __del__(self) -> None:
        try:
            self._artifact_stack.close()
            if self._runtime is not None:
                self._runtime.close()
        except Exception:
            pass


def request_columns(
    request: ExplainabilityRequest, context: ExplainableModelContext
) -> tuple[str, ...]:
    columns = request.feature_columns or context.feature_names
    if not columns:
        raise ValueError(
            "feature_columns must be explicit or present in model metadata"
        )
    return tuple(columns)


def _load_onnx_preprocessor(runtime: BundleModelRuntime) -> Any | None:
    """Load the optional trusted DNN/PU preprocessor sidecar."""
    path = runtime.resolved_artifact.path_for("preprocessor.json")
    if not path.is_file():
        return None
    from tributo.training.features.transformer import FeatureTransformer

    return FeatureTransformer.load(path)


def _load_onnx_feature_map(runtime: BundleModelRuntime) -> dict[str, Any] | None:
    path = runtime.resolved_artifact.path_for("feature_map.json")
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("feature_map.json must contain a JSON object")
    return raw


def _validate_feature_map(
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


def _onnx_feature_names(
    request: ExplainabilityRequest,
    input_names: tuple[str, ...],
    preprocessor: Any | None,
) -> tuple[str, ...]:
    if request.feature_columns:
        return tuple(request.feature_columns)
    if preprocessor is not None:
        return tuple(feature.name for feature in preprocessor.features)
    return input_names


def _build_onnx_inputs(
    values: np.ndarray,
    *,
    input_names: tuple[str, ...],
    input_dtypes: tuple[np.dtype[Any], ...],
    input_shapes: tuple[tuple[int | None, ...], ...],
    feature_names: tuple[str, ...],
    preprocessor: Any | None,
    feature_view: str,
) -> dict[str, np.ndarray]:
    """Bind raw or model-input columns to the verified ONNX signature.

    DNN/PU ONNX artifacts expose one named input per feature and carry the
    FeatureTransformer state as a trusted bundle sidecar.  Generic ONNX
    artifacts continue to support the existing one-matrix input contract;
    multi-input artifacts without a sidecar are bound by declared names.
    """
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


def _select_onnx_output(
    outputs: dict[str, np.ndarray],
    output_names: tuple[str, ...],
    *,
    output_target: str,
) -> np.ndarray:
    """Select an ONNX output without silently changing output semantics."""
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


def _resolve_xgboost_feature_names(
    resolved: Any, booster: Any, request: ExplainabilityRequest
) -> tuple[str, ...]:
    """Resolve and cross-check the native export's feature-name sidecar."""
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
    for label, names in (
        ("request", requested_names),
        ("booster", booster_names),
    ):
        if names and sidecar_names and names != sidecar_names:
            raise ValueError(
                f"XGBoost {label} feature names do not match feature_names.json"
            )
    resolved_names = requested_names or sidecar_names or booster_names
    if resolved_names and len(resolved_names) != booster.num_features():
        raise ValueError("XGBoost feature names count does not match the booster")
    return resolved_names


def _input_ids(
    batch: Any, request: ExplainabilityRequest, values: np.ndarray
) -> tuple[str, ...]:
    if request.input_id_column:
        if request.input_id_column not in batch:
            raise ValueError(f"input_id_column {request.input_id_column!r} is missing")
        return tuple(str(value) for value in batch[request.input_id_column].tolist())
    return tuple(
        "row-"
        + hashlib.sha256(
            json.dumps(np.asarray(row).tolist(), sort_keys=True, default=str).encode()
        ).hexdigest()[:24]
        for row in values
    )


def _resolve_reference(
    request: ExplainabilityRequest,
    reference_provider: ReferenceProvider,
) -> ResolvedReference:
    if request.reference is None:
        raise ValueError("reference binding is required")
    return reference_provider.resolve(request.reference, request.limits)


def _load_reference(
    request: ExplainabilityRequest,
    reference_provider: ReferenceProvider | None = None,
) -> np.ndarray | None:
    if request.reference is None:
        return None
    return _resolve_reference(
        request, reference_provider or FileReferenceProvider()
    ).data


def _reference_digest(
    request: ExplainabilityRequest,
    reference_provider: ReferenceProvider | None = None,
) -> str | None:
    if request.reference is None:
        return None
    if request.reference.digest:
        return request.reference.digest
    return (reference_provider or FileReferenceProvider()).digest(request.reference)


def _safe_reference_digest(
    request: ExplainabilityRequest,
    reference_provider: ReferenceProvider,
) -> str | None:
    try:
        return _reference_digest(request, reference_provider)
    except FileNotFoundError:
        return request.reference.digest if request.reference else None


def _receipt_uri(result_uri: str) -> str:
    return result_uri.rstrip("/") + "/receipt.json"


def _attempt_result_uri(result_uri: str, lease_token: str) -> str:
    """Return an isolated result location for one operation attempt.

    ``request.result_uri`` is a stable output root.  Every lease owner writes
    below a unique attempt directory so a stale driver cannot overwrite or
    clean up a replacement driver's result files or receipt.
    """
    parsed = urlsplit(result_uri)
    if parsed.scheme:
        attempt_path = f"{parsed.path.rstrip('/')}/attempts/{lease_token}"
        return urlunsplit(
            (parsed.scheme, parsed.netloc, attempt_path, parsed.query, parsed.fragment)
        )
    return str(Path(result_uri) / "attempts" / lease_token)


def _operation_store_for_request(request: ExplainabilityRequest) -> OperationStore:
    """Resolve the request's durable store when the caller did not inject one."""
    if request.operation_store_uri is None:
        return InMemoryOperationStore()
    from tributo.integrations.storage.json_operation_store import JsonFileOperationStore

    parsed = urlsplit(request.operation_store_uri)
    if parsed.netloc:
        raise ValueError("operation_store_uri file URI must not contain a host")
    path = Path(
        unquote(parsed.path if parsed.scheme == "file" else request.operation_store_uri)
    )
    return JsonFileOperationStore(path)


def _bundle_digest(manifest: Any) -> str:
    return compute_bundle_digest(
        artifacts=tuple(manifest.artifacts),
        roles=dict(manifest.roles),
        input_sig=manifest.input_signature,
        output_sig=manifest.output_signature,
        explainability=getattr(manifest, "explainability", None),
    )


def _selected_model_role(manifest: Any, request: ExplainabilityRequest) -> str:
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


def _selected_artifact(manifest: Any, request: ExplainabilityRequest) -> Any:
    role = _selected_model_role(manifest, request)
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


def _explanation_output_count_upper_bound(
    manifest: Any,
    request: ExplainabilityRequest,
) -> int:
    """Resolve a safe attribution-output bound from a verified manifest."""
    artifact = _selected_artifact(manifest, request)
    if artifact.flavor_id != "xgboost-native-v1":
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
            "XGBoost native explainability requires one typed probability or "
            "prediction output signature"
        )
    shape = tuple(getattr(candidates[0], "shape", ()))
    if len(shape) != 2 or not isinstance(shape[1], int) or shape[1] < 1:
        raise ValueError(
            "XGBoost native explainability requires a fixed output dimension "
            "in the typed manifest signature"
        )
    return shape[1]


def _model_digest(manifest: Any, request: ExplainabilityRequest) -> str:
    return str(_selected_artifact(manifest, request).tree_digest)


def _manifest_role_digest(manifest: Any, role: str) -> str | None:
    target_name = manifest.roles.get(role)
    if target_name is not None:
        for artifact in manifest.artifacts:
            if artifact.name == target_name:
                return str(artifact.tree_digest)
        raise ValueError(
            f"Bundle role {role!r} points to missing artifact {target_name!r}"
        )

    # DNN/PU preprocessors and feature maps are file-level roles inside the
    # model artifact.  Preserve their independent digest in the receipt even
    # though they are not standalone manifest target roles.
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


def _selected_backend(
    manifest: Any, request: ExplainabilityRequest
) -> tuple[str, Exactness]:
    descriptor = getattr(manifest, "explainability", None)
    if request.backend != "auto":
        backend = request.backend
    elif descriptor is not None and descriptor.backend != "auto":
        backend = descriptor.backend
    else:
        artifact = _selected_artifact(manifest, request)
        backend = (
            "tree" if artifact.flavor_id == "xgboost-native-v1" else "model_agnostic"
        )
    exactness: Exactness = cast(
        Exactness,
        {
            "tree": "exact",
            "model_agnostic": "approximate",
        }.get(backend, "conditional"),
    )
    if descriptor is not None and descriptor.backend == backend:
        exactness = descriptor.exactness
    return backend, exactness


def _reference_policy(manifest: Any, request: ExplainabilityRequest) -> ReferencePolicy:
    descriptor = getattr(manifest, "explainability", None)
    if descriptor is not None:
        return cast(ReferencePolicy, descriptor.reference_policy)
    if request.reference is not None:
        return "required"
    return "optional" if request.backend in {"auto", "tree"} else "none"


def _dependency_versions(backend: str) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    packages = ["shap", "numpy"]
    if backend == "tree":
        packages.append("xgboost")
    elif backend == "model_agnostic":
        packages.append("onnxruntime")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def _result_stats(
    uri: str,
    *,
    storage_profile: str | None = None,
) -> tuple[str, int, int]:
    inspection = inspect_parquet_output(uri, storage_profile=storage_profile)
    return inspection.digest, inspection.total_bytes, inspection.rows


def _write_receipt(
    uri: str,
    receipt: ExplainabilityReceipt,
    *,
    storage_profile: str | None = None,
) -> None:
    default_object_store().write_bytes(
        _receipt_uri(uri),
        receipt.model_dump_json().encode(),
        storage_profile=storage_profile,
        content_type="application/json",
    )


def _schema_signature() -> str:
    from tributo.explainability.contracts import FeatureAttribution

    fields = tuple(
        (name, str(field.annotation))
        for name, field in FeatureAttribution.model_fields.items()
    )
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest()


def _record_operation(
    store: OperationStore, record: ExplainabilityOperationRecord
) -> None:
    store.record_explainability(record)


def _operation_idempotency_key(
    manifest: Any,
    request: ExplainabilityRequest,
    *,
    bundle_digest: str,
    reference_provider: ReferenceProvider,
) -> str:
    payload = {
        "bundle_digest": bundle_digest,
        "request": request.model_dump(
            mode="json",
            exclude={
                "operation_id",
                "operation_store_uri",
                "operation_lease_seconds",
                "force_resume",
            },
        ),
        "reference_digest": _safe_reference_digest(request, reference_provider),
        "adapter_version": f"{request.explainer}-v1",
    }
    del manifest
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_request_against_descriptor(
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
    selected_role = _selected_model_role(manifest, request)
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


def _is_retryable_error(exc: Exception) -> bool:
    ray_task_error: type[BaseException] | None = None
    try:
        from ray.exceptions import RayTaskError

        ray_task_error = RayTaskError
    except ImportError:
        pass
    retryable_types: tuple[type[BaseException], ...] = (
        OSError,
        TimeoutError,
        ConnectionError,
        ResultMaterializationError,
    )
    if ray_task_error is not None:
        retryable_types += (ray_task_error,)
    return isinstance(exc, retryable_types)


def _read_receipt(
    uri: str,
    *,
    storage_profile: str | None = None,
) -> ExplainabilityReceipt | None:
    receipt_uri = (
        uri if uri.rstrip("/").endswith("/receipt.json") else _receipt_uri(uri)
    )
    try:
        payload = default_object_store().read_bytes(
            receipt_uri,
            storage_profile=storage_profile,
        )
        return ExplainabilityReceipt.model_validate_json(payload)
    except (FileNotFoundError, OSError, ValueError):
        return None


def _reference_rows(
    request: ExplainabilityRequest,
    reference_provider: ReferenceProvider | None = None,
) -> int | None:
    if request.reference is None:
        return None
    data = _load_reference(request, reference_provider)
    return int(data.shape[0]) if data is not None else None


def _cleanup_result_uri(
    uri: str,
    *,
    storage_profile: str | None = None,
) -> None:
    try:
        default_object_store().delete_tree(uri, storage_profile=storage_profile)
    except (FileNotFoundError, OSError, ValueError):
        logger.warning("Failed to clean explainability result %s", uri, exc_info=True)


__all__ = ["ExplainabilityBatchWorker", "run_batch_explainability"]

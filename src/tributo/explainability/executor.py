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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any, Literal
from urllib.parse import unquote, urlsplit, urlunsplit
from uuid import uuid4

import numpy as np

from tributo.exceptions import ResultMaterializationError
from tributo.explainability.contracts import (
    Exactness,
    ExplainabilityOperationRecord,
    ExplainabilityReceipt,
    ExplainabilityRequest,
)
from tributo.explainability.planner import ExplainabilityPlanner
from tributo.explainability.protocols import (
    ExplainabilityModelBinding,
    ExplainabilityModelProvider,
    ExplainabilityModelSession,
    ExplainabilityModelSessionFactory,
    ExplainabilityResultStore,
    ExplainableModelContext,
    ReferenceProvider,
    ResolvedReference,
)
from tributo.explainability.registry import ExplainerRegistry
from tributo.exporting.records import InMemoryOperationStore, OperationStore
from tributo.inference.input_resolver import (
    IngestionGatewayInputResolver,
    InputResolverPort,
    OpenedInferenceInput,
)
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
    model_provider: ExplainabilityModelProvider | None = None,
    result_store: ExplainabilityResultStore | None = None,
) -> ExplainabilityReceipt:
    """Execute one bounded explainability request through Ray Data."""
    import ray.data

    operation_id = request.operation_id or request.request_id
    resolver = input_resolver or IngestionGatewayInputResolver()
    store = operation_store or _operation_store_for_request(request)
    references = reference_provider or _default_explainability_reference_provider()
    models = model_provider or _default_explainability_model_provider()
    results = result_store or _default_explainability_result_store()
    model_binding = models.resolve(request)
    bundle_digest = model_binding.bundle_digest
    idempotency_key = _operation_idempotency_key(
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
            cached = results.read_receipt(
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
            bundle_id=model_binding.bundle_id,
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
            output_count=model_binding.output_count_upper_bound,
        )

        explained = dataset.map_batches(
            worker,
            fn_constructor_args=(
                request,
                model_binding.session_factory,
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
        materialization = results.materialize(
            explained,
            uri=attempt_result_uri,
            storage_profile=request.result_storage_profile,
            max_bytes=request.limits.max_explanation_bytes,
            run_id=operation_id,
            plan_digest=bundle_digest,
        )
        result_digest = materialization.digest
        result_bytes = materialization.total_bytes
        explanation_rows = materialization.rows
        if (
            request.limits.max_explanation_rows is not None
            and explanation_rows > request.limits.max_explanation_rows
        ):
            raise _ExplainabilityLimitExceeded(
                f"explanation output rows {explanation_rows} exceed "
                f"limits.max_explanation_rows={request.limits.max_explanation_rows}"
            )
        heartbeat.raise_if_failed()
        selected_backend = model_binding.backend
        exactness = model_binding.exactness
        receipt = _make_receipt(
            model_binding=model_binding,
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
        results.write_receipt(
            attempt_result_uri,
            receipt,
            storage_profile=request.result_storage_profile,
        )
        _record_operation(
            store,
            ExplainabilityOperationRecord(
                operation_id=operation_id,
                request_id=request.request_id,
                bundle_id=model_binding.bundle_id,
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
            results.cleanup(
                attempt_result_uri,
                storage_profile=request.result_storage_profile,
            )
        if partial:
            try:
                selected_backend = model_binding.backend
                exactness = model_binding.exactness
                results.write_receipt(
                    attempt_result_uri,
                    _make_receipt(
                        model_binding=model_binding,
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
                bundle_id=model_binding.bundle_id,
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
    model_binding: ExplainabilityModelBinding,
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
        bundle_id=model_binding.bundle_id,
        bundle_digest=bundle_digest,
        model_digest=model_binding.model_digest,
        preprocessor_digest=model_binding.preprocessor_digest,
        feature_map_digest=model_binding.feature_map_digest,
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
        reference_policy=model_binding.descriptor.reference_policy,
        dependency_versions=dict(model_binding.dependency_versions),
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
        model_factory: ExplainabilityModelSessionFactory,
        registry: ExplainerRegistry | None,
        reference_provider: ReferenceProvider,
    ) -> None:
        self._request = request
        self._model_session: ExplainabilityModelSession = model_factory.create(
            reference_provider
        )
        self._context = self._model_session.context
        plan = ExplainabilityPlanner(registry).plan(
            self._context,
            request,
            output_count=self._model_session.output_count_upper_bound,
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
        if self._context.native_attribution_id is not None:
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

    def __del__(self) -> None:
        try:
            self._model_session.close()
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
        request, reference_provider or _default_explainability_reference_provider()
    ).data


def _reference_digest(
    request: ExplainabilityRequest,
    reference_provider: ReferenceProvider | None = None,
) -> str | None:
    if request.reference is None:
        return None
    if request.reference.digest:
        return request.reference.digest
    return (reference_provider or _default_explainability_reference_provider()).digest(
        request.reference
    )


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
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def _reference_rows(
    request: ExplainabilityRequest,
    reference_provider: ReferenceProvider | None = None,
) -> int | None:
    if request.reference is None:
        return None
    data = _load_reference(request, reference_provider)
    return int(data.shape[0]) if data is not None else None


def _default_explainability_model_provider() -> ExplainabilityModelProvider:
    """Resolve the default model provider through top-level composition."""
    from tributo.runtime import default_explainability_model_provider

    return default_explainability_model_provider()


def _default_explainability_reference_provider() -> ReferenceProvider:
    """Resolve the default reference adapter through top-level composition."""
    from tributo.runtime import default_explainability_reference_provider

    return default_explainability_reference_provider()


def _default_explainability_result_store() -> ExplainabilityResultStore:
    """Resolve the default result adapter through top-level composition."""
    from tributo.runtime import default_explainability_result_store

    return default_explainability_result_store()


__all__ = ["ExplainabilityBatchWorker", "run_batch_explainability"]

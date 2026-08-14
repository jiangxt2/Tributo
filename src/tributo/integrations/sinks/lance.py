"""Lance implementation of the Inference ResultSink contract."""

from __future__ import annotations

import hashlib
import json
import logging
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Protocol
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.compute as pc

from tributo._common.storage_profiles import StorageProfile, StorageProfileResolver
from tributo.data.base import WriteMode
from tributo.data.contracts.handles import RayDataHandle
from tributo.data.refs import schema_fingerprint
from tributo.data.writing.builtins import default_write_gateway
from tributo.data.writing.contracts import WriteCapabilityError, WriteRequest
from tributo.exceptions import ResultMaterializationError, ResultWriteError
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    from tributo.inference.contracts import (
        LanceResultSinkRequest,
        ResultSinkReceipt,
        ResultSinkRequest,
    )

logger = logging.getLogger(__name__)


class _StorageProfileResolverLike(Protocol):
    def resolve(self, profile: str | None) -> StorageProfile: ...


@PublicAPI(stability="alpha")
class LanceResultSink:
    """Write a Ray Dataset through the shared native write Gateway.

    The sink is deliberately explicit: it always writes Lance, regardless of
    whether the output contains a vector column.  Vector schema checks apply
    only to columns declared by ``request.vector_columns``; model semantics
    remain the responsibility of the caller's Predictor.  The Gateway selects
    the stable Ray Lance Binding; the selected provider owns all data-plane and
    save-mode behavior.  The compatibility Connector uses the same boundary.
    """

    api_version: ClassVar[int] = 1
    sink_id: ClassVar[str] = "lance-v1"

    def __init__(
        self, storage_resolver: _StorageProfileResolverLike | None = None
    ) -> None:
        self._storage_resolver = storage_resolver or StorageProfileResolver()

    def write(
        self,
        dataset: Any,
        request: ResultSinkRequest,
        *,
        run_id: str,
        plan_digest: str,
    ) -> ResultSinkReceipt:
        """Validate the declared Arrow schema and materialize a Lance table."""
        from tributo.inference.contracts import (
            LanceResultSinkRequest,
            ResultSinkReceipt,
        )

        if not isinstance(request, LanceResultSinkRequest):
            raise ResultWriteError(
                f"Lance result sink cannot write {request.sink_id!r}"
            )
        arrow_schema = _arrow_schema(dataset.schema())
        _validate_vector_schema(arrow_schema, request)
        runtime_s3 = _runtime_s3(
            self._storage_resolver, request.storage_profile, request.uri
        )
        try:
            if request.vector_columns:
                dataset = dataset.map_batches(
                    partial(validate_vector_batch, request=request),
                    batch_format="pyarrow",
                )
            options: dict[str, Any] = {
                "min_rows_per_file": request.min_rows_per_file,
                "max_rows_per_file": request.max_rows_per_file,
            }
            if request.data_storage_version is not None:
                options["data_storage_version"] = request.data_storage_version
            runtime_options: dict[str, Any] = {}
            if runtime_s3 is not None:
                runtime_options["s3"] = runtime_s3
            default_write_gateway().execute(
                WriteRequest(
                    engine="ray",
                    target_kind="lance",
                    target=request.uri,
                    binding_id="tributo.ray.lance",
                    mode=WriteMode(request.mode),
                    options=options,
                    runtime_options=runtime_options,
                ),
                RayDataHandle(dataset),
            )
        except ResultWriteError:
            raise
        except WriteCapabilityError:
            raise ResultWriteError(
                "Lance result sink cannot satisfy the requested write capability"
            ) from None
        except Exception as exc:
            from tributo.inference._credential_safety import safe_exception_summary

            source_error_type = getattr(exc, "source_error_type", None)
            logger.warning(
                "Lance result materialization failed (%s): %s",
                source_error_type or type(exc).__name__,
                safe_exception_summary(exc),
            )
            raise ResultMaterializationError(
                source_error_type or type(exc).__name__
            ) from None

        fingerprint = schema_fingerprint(arrow_schema)
        result_id = _result_id(
            run_id=run_id,
            plan_digest=plan_digest,
            request=request,
            schema_fp=fingerprint,
        )
        metadata = {
            "format": "lance",
            "mode": request.mode,
            "schema_fingerprint": fingerprint,
        }
        if request.data_storage_version is not None:
            metadata["data_storage_version"] = request.data_storage_version
        return ResultSinkReceipt(
            sink_id=self.sink_id,
            result_id=result_id,
            uri=request.uri,
            rows_written=None,
            metadata=metadata,
        )


def _arrow_schema(schema: Any) -> pa.Schema:
    base_schema = getattr(schema, "base_schema", schema)
    if not isinstance(base_schema, pa.Schema):
        raise ResultWriteError("Lance result sink requires an Arrow dataset schema")
    return base_schema


def _runtime_s3(
    resolver: _StorageProfileResolverLike,
    profile_name: str | None,
    uri: str,
) -> StorageProfile | None:
    if urlsplit(uri).scheme.lower() != "s3":
        return None
    profile = resolver.resolve(profile_name)
    if profile.profile_name is not None:
        raise ResultWriteError(
            "Ray result-sink profiles must resolve inside the cluster; set "
            f"TRIBUTO_STORAGE_PROFILE_{profile.profile_name.upper()} or use the "
            "default IAM/environment chain"
        )
    return profile


def _validate_vector_schema(schema: pa.Schema, request: LanceResultSinkRequest) -> None:
    for spec in request.vector_columns:
        if spec.name not in schema.names:
            raise ResultWriteError(
                f"Lance vector column {spec.name!r} is missing from the output schema"
            )
        field = schema.field(spec.name)
        data_type = field.type
        if not pa.types.is_fixed_size_list(data_type):
            raise ResultWriteError(
                f"Lance vector column {spec.name!r} must use fixed_size_list"
            )
        if data_type.list_size != spec.dimension:
            raise ResultWriteError(
                f"Lance vector column {spec.name!r} has dimension "
                f"{data_type.list_size}, expected {spec.dimension}"
            )
        expected_type = getattr(pa, spec.dtype)()
        if data_type.value_type != expected_type:
            raise ResultWriteError(
                f"Lance vector column {spec.name!r} has dtype "
                f"{data_type.value_type}, expected {spec.dtype}"
            )


def validate_vector_batch(batch: pa.Table, request: LanceResultSinkRequest) -> pa.Table:
    """Validate vector values without collecting the Ray Dataset on the driver."""
    for spec in request.vector_columns:
        column = batch.column(spec.name)
        for chunk in column.chunks:
            if chunk.null_count:
                raise ResultWriteError(
                    f"Lance vector column {spec.name!r} contains null or non-finite values"
                )
            values = pc.list_flatten(chunk)
            if values.null_count or (
                len(values) > 0 and not bool(pc.all(pc.is_finite(values)).as_py())
            ):
                raise ResultWriteError(
                    f"Lance vector column {spec.name!r} contains null or non-finite values"
                )
    return batch


def _result_id(
    *,
    run_id: str,
    plan_digest: str,
    request: LanceResultSinkRequest,
    schema_fp: str,
) -> str:
    payload = json.dumps(
        {
            "run_id": run_id,
            "plan_digest": plan_digest,
            "sink_id": LanceResultSink.sink_id,
            "uri": request.uri,
            "mode": request.mode,
            "data_storage_version": request.data_storage_version,
            "vector_columns": [
                spec.model_dump(mode="json") for spec in request.vector_columns
            ],
            "schema_fingerprint": schema_fp,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["LanceResultSink"]

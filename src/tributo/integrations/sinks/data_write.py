"""Generic inference sink backed by the unified data-writing Gateway."""

from __future__ import annotations

import hashlib
import json
from typing import Any, ClassVar

from tributo.data import DataWriteTargetRequest
from tributo.data.contracts.handles import RayDataHandle
from tributo.data.writing.builtins import default_write_gateway
from tributo.data.writing.gateway import WriteGateway
from tributo.exceptions import ResultMaterializationError, ResultWriteError
from tributo.inference.contracts import ResultSinkReceipt, ResultSinkRequest
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class DataWriteResultSink:
    """Write Ray inference results through a registered data Binding."""

    api_version: ClassVar[int] = 1
    sink_id: ClassVar[str] = "data-write-v1"

    def __init__(self, gateway: WriteGateway | None = None) -> None:
        self._gateway = gateway or default_write_gateway()

    def write(
        self,
        dataset: Any,
        request: ResultSinkRequest,
        *,
        run_id: str,
        plan_digest: str,
    ) -> ResultSinkReceipt:
        """Delegate one generic result write to ``WriteGateway``."""
        if not isinstance(request, DataWriteTargetRequest):
            raise ResultWriteError(
                f"Data write result sink cannot write {request.sink_id!r}"
            )
        try:
            receipt = self._gateway.execute(
                request.as_write_request(),
                RayDataHandle(dataset),
            )
        except ResultWriteError:
            raise
        except Exception as exc:
            source_error_type = getattr(exc, "source_error_type", None)
            raise ResultMaterializationError(
                source_error_type or type(exc).__name__
            ) from None

        result_id = _result_id(
            run_id=run_id,
            plan_digest=plan_digest,
            request=request,
        )
        return ResultSinkReceipt(
            sink_id=self.sink_id,
            result_id=result_id,
            uri=request.target,
            rows_written=receipt.rows_written,
            metadata={
                "target_kind": request.target_kind,
                "binding_id": receipt.binding_id,
                "committed": str(receipt.committed).lower(),
            },
        )


def _result_id(
    *,
    run_id: str,
    plan_digest: str,
    request: DataWriteTargetRequest,
) -> str:
    payload = json.dumps(
        {
            "run_id": run_id,
            "plan_digest": plan_digest,
            "sink_id": DataWriteResultSink.sink_id,
            "target_kind": request.target_kind,
            "target": request.target,
            "binding_id": request.binding_id,
            "mode": request.mode.value,
            "options": request.options,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["DataWriteResultSink"]

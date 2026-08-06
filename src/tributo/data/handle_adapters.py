"""Explicit conversions between native ingestion handle types."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tributo.data.ingestion import (
    DaftDataFrameHandle,
    IngestionOpenResult,
    RayDataHandle,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class HandleConversionReceipt(BaseModel):
    """Credential-free evidence for one explicit native-handle conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversion_version: Literal[1] = 1
    adapter_id: Literal["tributo.daft_to_ray"] = "tributo.daft_to_ray"
    adapter_api: Literal["daft.DataFrame.to_ray_dataset"] = (
        "daft.DataFrame.to_ray_dataset"
    )
    source_engine_id: Literal["tributo.daft"] = "tributo.daft"
    target_engine_id: Literal["tributo.ray_data"] = "tributo.ray_data"
    source_dataset_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_engine_version: str = Field(min_length=1)
    execution_may_be_triggered: Literal[True] = True
    full_driver_materialization: Literal[False] = False
    order_preserved: Literal[False] = False


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class RayHandleAdaptation:
    """Ray handle plus evidence produced by an explicit handle adapter."""

    handle: RayDataHandle
    receipt: HandleConversionReceipt


@PublicAPI(stability="alpha")
def adapt_daft_result_to_ray(result: IngestionOpenResult) -> RayHandleAdaptation:
    """Convert a live Daft result through Daft's public Ray Dataset adapter.

    This is an explicit consumer-boundary operation, never an engine-routing
    fallback. Daft may execute its lazy plan while transferring partitions to
    Ray, but the adapter does not collect the complete dataset on the driver
    and does not promise row-order preservation.
    """
    if result.closed:
        raise ValueError("Cannot adapt a closed ingestion result")
    if not isinstance(result.handle, DaftDataFrameHandle):
        raise TypeError("Daft-to-Ray adapter requires a DaftDataFrameHandle")
    if result.receipt.engine_id != "tributo.daft":
        raise ValueError("Daft handle and ingestion receipt engine do not match")

    dataset = result.handle.dataframe.to_ray_dataset()
    return RayHandleAdaptation(
        handle=RayDataHandle(dataset),
        receipt=HandleConversionReceipt(
            source_dataset_ref=result.receipt.dataset_ref,
            source_request_digest=result.receipt.request_digest,
            target_engine_version=importlib.metadata.version("ray"),
        ),
    )


__all__ = [
    "HandleConversionReceipt",
    "RayHandleAdaptation",
    "adapt_daft_result_to_ray",
]

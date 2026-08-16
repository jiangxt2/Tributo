"""Gateway-backed input fixtures shared by algorithm bridge tests."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from tributo.data import (
    DaftDataFrameHandle,
    IngestionDescriptor,
    IngestionOpenResult,
    IngestionPlanReceipt,
    IngestionRequest,
    ParquetSourceConfig,
    RayDataHandle,
)
from tributo.data.scan_plan import ScanKind
from tributo.integrations.algorithm_inputs import IngestionInputInvocation


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class StubRayDataset:
    columns: Mapping[str, Sequence[object]]
    selected: tuple[str, ...] = ()
    row_limit: int | None = None
    streaming_split_calls: list[tuple[int, bool]] = field(default_factory=list)
    count_calls: int = 0

    def count(self) -> int:
        self.count_calls += 1
        return len(next(iter(self.columns.values())))

    def select_columns(self, columns: list[str]) -> StubRayDataset:
        missing = sorted(set(columns) - set(self.columns))
        if missing:
            raise ValueError(f"missing columns: {missing}")
        self.selected = tuple(columns)
        return self

    def limit(self, count: int) -> StubRayDataset:
        self.row_limit = count
        return self

    def iter_batches(self, *, batch_format: str) -> Iterable[dict[str, list[object]]]:
        if batch_format != "numpy":
            raise ValueError("test fixture requires numpy batches")
        yield {
            name: list(self.columns[name][: self.row_limit]) for name in self.selected
        }

    def streaming_split(
        self,
        split_count: int,
        *,
        equal: bool,
    ) -> tuple[StubRayDataset, ...]:
        """Mirror Ray Data's public contiguous streaming split for bridge tests."""
        self.streaming_split_calls.append((split_count, equal))
        row_count = len(next(iter(self.columns.values())))
        return tuple(
            StubRayDataset(
                {
                    name: values[
                        row_count * rank // split_count : row_count
                        * (rank + 1)
                        // split_count
                    ]
                    for name, values in self.columns.items()
                },
                selected=tuple(self.columns),
            )
            for rank in range(split_count)
        )


@dataclass
class StubDaftFrame:
    columns: Mapping[str, Sequence[object]]
    selected: tuple[str, ...] = ()
    row_limit: int | None = None
    ray_conversion_calls: int = 0

    def select(self, *columns: str) -> StubDaftFrame:
        missing = sorted(set(columns) - set(self.columns))
        if missing:
            raise ValueError(f"missing columns: {missing}")
        self.selected = tuple(columns)
        return self

    def limit(self, count: int) -> StubDaftFrame:
        self.row_limit = count
        return self

    def to_pydict(self) -> dict[str, list[object]]:
        return {
            name: list(self.columns[name][: self.row_limit]) for name in self.selected
        }

    def to_ray_dataset(self) -> object:
        self.ray_conversion_calls += 1
        raise AssertionError("Daft input must not convert implicitly to Ray Data")


@dataclass
class StubIngestionGateway:
    """Return typed handles while preserving the real Gateway contracts."""

    columns: Mapping[str, Sequence[object]]
    lifecycle_events: list[str] = field(default_factory=list)
    describe_calls: int = 0
    open_calls: int = 0
    last_daft_frame: StubDaftFrame | None = None

    def describe(self, request: IngestionRequest) -> IngestionDescriptor:
        self.describe_calls += 1
        request_digest = _digest(request.model_dump_json())
        handle_kind: Literal["ray_data", "daft_dataframe"] = (
            "ray_data" if request.engine == "tributo.ray_data" else "daft_dataframe"
        )
        distribution = "ray" if handle_kind == "ray_data" else "daft"
        return IngestionDescriptor(
            request_digest=request_digest,
            source_ref=_digest(f"source:{request_digest}"),
            dataset_ref=_digest(f"dataset:{request_digest}"),
            logical_plan_digest=_digest(f"plan:{request_digest}"),
            engine_id=request.engine,
            provider_id="tributo.parquet",
            connector_id="parquet",
            binding_id=f"tests.{distribution}.parquet",
            scan_kind=ScanKind.FILE,
            handle_kind=handle_kind,
            deferred_validations=("runtime_connectivity", "worker_versions"),
            binding_distribution=distribution,
            binding_distribution_version="2.55.1"
            if distribution == "ray"
            else "0.7.23",
            capability_version=1,
        )

    def open(
        self,
        request: IngestionRequest,
        runtime_context: object = None,
    ) -> IngestionOpenResult:
        del runtime_context
        self.open_calls += 1
        descriptor = self.describe(request)
        if descriptor.handle_kind == "ray_data":
            handle = RayDataHandle(StubRayDataset(self.columns))
            engine_version = "2.55.1"
        else:
            self.last_daft_frame = StubDaftFrame(self.columns)
            handle = DaftDataFrameHandle(self.last_daft_frame)
            engine_version = "0.7.23"
        receipt = IngestionPlanReceipt(
            request_digest=descriptor.request_digest,
            engine_id=descriptor.engine_id,
            engine_version=engine_version,
            provider_id=descriptor.provider_id,
            connector_id=descriptor.connector_id,
            binding_id=descriptor.binding_id,
            scan_kind=descriptor.scan_kind,
            logical_plan_version=1,
            logical_plan_digest=descriptor.logical_plan_digest,
            source_ref=descriptor.source_ref,
            dataset_ref=descriptor.dataset_ref,
            transform_ir_version=1,
            transform_digest=_digest("empty-transform"),
            binding_distribution=descriptor.binding_distribution,
            binding_distribution_version=descriptor.binding_distribution_version,
            reader_api=f"tests.read_{descriptor.connector_id}",
            transport_id="tests.in_process",
        )
        return IngestionOpenResult(
            handle=handle,
            receipt=receipt,
            close_callback=lambda: self.lifecycle_events.append("closed"),
            cancel_callback=lambda: self.lifecycle_events.append("cancelled"),
        )


def ingestion_invocation(
    *,
    engine: Literal["ray", "daft"] = "ray",
    credentials: bool = False,
) -> IngestionInputInvocation:
    s3 = (
        {
            "access_key_id": "fixture-access-key",
            "secret_access_key": "fixture-secret-key",
        }
        if credentials
        else None
    )
    return IngestionInputInvocation(
        IngestionRequest(
            source=ParquetSourceConfig(
                path="s3://fixture-bucket/algorithm-input.parquet",
                s3=s3,
            ),
            engine=engine,
            trace_context={"trace_id": "algorithm-input-test"},
        )
    )


__all__ = [
    "StubDaftFrame",
    "StubIngestionGateway",
    "StubRayDataset",
    "ingestion_invocation",
]

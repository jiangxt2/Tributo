"""Unit tests for the public IngestionGateway inference adapter."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tributo.data import (
    DaftDataFrameHandle,
    IngestionDescriptor,
    IngestionOpenResult,
    IngestionPlanReceipt,
    IngestionRequest,
    IngestionRuntimeContext,
    RayDataHandle,
)
from tributo.data.source_config import ParquetSourceConfig
from tributo.exceptions import DataSourceError, JobConfigurationError
from tributo.inference.contracts import ResolvedInputSelection
from tributo.inference.input_resolver import IngestionGatewayInputResolver


def _request(*, engine: str = "ray", binding_id: str | None = None) -> IngestionRequest:
    return IngestionRequest(
        source=ParquetSourceConfig(path="/data/input.parquet"),
        engine=engine,
        binding_id=binding_id,
        storage_profile="source-domain",
    )


def _descriptor(
    marker: str = "1",
    *,
    binding_id: str = "tributo.ray.parquet",
    distribution: str = "tributo",
    distribution_version: str = "1.0.0",
) -> IngestionDescriptor:
    return IngestionDescriptor(
        request_digest=marker * 64,
        source_ref=marker * 64,
        dataset_ref=marker * 64,
        logical_plan_digest=marker * 64,
        engine_id="tributo.ray_data",
        provider_id="tributo.parquet",
        connector_id="parquet",
        binding_id=binding_id,
        scan_kind="file",
        handle_kind="ray_data",
        binding_distribution=distribution,
        binding_distribution_version=distribution_version,
        capability_version=1,
    )


def _receipt(descriptor: IngestionDescriptor) -> IngestionPlanReceipt:
    return IngestionPlanReceipt(
        request_digest=descriptor.request_digest,
        engine_id=descriptor.engine_id,
        engine_version="2.55.1",
        provider_id=descriptor.provider_id,
        connector_id=descriptor.connector_id,
        binding_id=descriptor.binding_id,
        scan_kind=descriptor.scan_kind,
        logical_plan_version=1,
        logical_plan_digest=descriptor.logical_plan_digest,
        source_ref=descriptor.source_ref,
        dataset_ref=descriptor.dataset_ref,
        transform_ir_version=1,
        transform_digest="f" * 64,
        binding_distribution=descriptor.binding_distribution,
        binding_distribution_version=descriptor.binding_distribution_version,
        reader_api="ray.data.read_parquet",
        transport_id="ray-data",
    )


class _Gateway:
    def __init__(
        self,
        descriptors: list[IngestionDescriptor],
        result: IngestionOpenResult | None = None,
    ) -> None:
        self._descriptors = iter(descriptors)
        self.result = result
        self.describe_calls: list[IngestionRequest] = []
        self.open_calls: list[
            tuple[IngestionRequest, IngestionRuntimeContext | None]
        ] = []

    def describe(self, request: IngestionRequest) -> IngestionDescriptor:
        self.describe_calls.append(request)
        return next(self._descriptors)

    def open(
        self,
        request: IngestionRequest,
        runtime_context: IngestionRuntimeContext | None = None,
    ) -> IngestionOpenResult:
        self.open_calls.append((request, runtime_context))
        assert self.result is not None
        return self.result


def _resolver(
    gateway: _Gateway,
    runtime_context_factory: Callable[[], IngestionRuntimeContext] | None = None,
) -> IngestionGatewayInputResolver:
    return IngestionGatewayInputResolver(
        gateway,
        runtime_context_factory=runtime_context_factory
        or (lambda: IngestionRuntimeContext()),
    )


class TestIngestionGatewayInputResolver:
    def test_describe_pins_selected_binding_and_rechecks_the_route(self) -> None:
        gateway = _Gateway([_descriptor("1"), _descriptor("2")])

        selection = _resolver(gateway).describe(_request())

        assert selection.request.binding_id == "tributo.ray.parquet"
        assert selection.descriptor.request_digest == "2" * 64
        assert [call.binding_id for call in gateway.describe_calls] == [
            None,
            "tributo.ray.parquet",
        ]
        assert selection.request.storage_profile == "source-domain"

    def test_route_drift_while_pinning_fails_closed(self) -> None:
        gateway = _Gateway(
            [
                _descriptor("1"),
                _descriptor("2", binding_id="third.party.ray.parquet"),
            ]
        )

        with pytest.raises(DataSourceError, match="binding selection changed"):
            _resolver(gateway).describe(_request())

    def test_daft_request_is_rejected_before_gateway_access(self) -> None:
        gateway = _Gateway([])

        with pytest.raises(JobConfigurationError, match="Daft-to-Ray conversion"):
            _resolver(gateway).describe(_request(engine="daft"))

        assert gateway.describe_calls == []

    def test_open_allows_topology_specific_identity_but_verifies_worker_receipt(
        self,
    ) -> None:
        planning = _descriptor("1")
        worker = _descriptor("2", distribution_version="1.1.0")
        dataset = object()
        close_calls: list[str] = []
        result = IngestionOpenResult(
            handle=RayDataHandle(dataset),
            receipt=_receipt(worker),
            close_callback=lambda: close_calls.append("close"),
        )
        gateway = _Gateway([worker], result)
        context = IngestionRuntimeContext()
        resolver = _resolver(gateway, lambda: context)
        selection = ResolvedInputSelection(
            request=_request(binding_id=planning.binding_id),
            descriptor=planning,
        )

        opened = resolver.open(selection)

        assert opened.dataset is dataset
        assert opened.receipt == _receipt(worker)
        assert gateway.open_calls == [(selection.request, context)]
        opened.close()
        opened.close()
        assert close_calls == ["close"]

    def test_open_cancels_when_worker_receipt_does_not_match_descriptor(self) -> None:
        descriptor = _descriptor("2")
        bad_receipt = _receipt(descriptor).model_copy(update={"source_ref": "3" * 64})
        cancel_calls: list[str] = []
        result = IngestionOpenResult(
            handle=RayDataHandle(object()),
            receipt=bad_receipt,
            cancel_callback=lambda: cancel_calls.append("cancel"),
        )
        gateway = _Gateway([descriptor], result)
        selection = ResolvedInputSelection(
            request=_request(binding_id=descriptor.binding_id),
            descriptor=descriptor,
        )

        with pytest.raises(DataSourceError, match="open receipt differ"):
            _resolver(gateway).open(selection)

        assert cancel_calls == ["cancel"]
        assert result.closed is True

    def test_open_rejects_non_ray_handle_without_implicit_conversion(self) -> None:
        descriptor = _descriptor("2")
        cancel_calls: list[str] = []
        result = IngestionOpenResult(
            handle=DaftDataFrameHandle(object()),
            receipt=_receipt(descriptor),
            cancel_callback=lambda: cancel_calls.append("cancel"),
        )
        gateway = _Gateway([descriptor], result)
        selection = ResolvedInputSelection(
            request=_request(binding_id=descriptor.binding_id),
            descriptor=descriptor,
        )

        with pytest.raises(DataSourceError, match="implicit engine conversion"):
            _resolver(gateway).open(selection)

        assert cancel_calls == ["cancel"]

    def test_open_rejects_planning_to_worker_route_drift_before_read(self) -> None:
        planning = _descriptor("1")
        worker = _descriptor("2", binding_id="third.party.ray.parquet")
        gateway = _Gateway([worker])
        selection = ResolvedInputSelection(
            request=_request(binding_id=planning.binding_id),
            descriptor=planning,
        )

        with pytest.raises(DataSourceError, match="planning and execution"):
            _resolver(gateway).open(selection)

        assert gateway.open_calls == []

    def test_open_rejects_third_party_binding_version_drift(self) -> None:
        planning = _descriptor(
            "1",
            binding_id="vendor.ray.parquet",
            distribution="vendor-ingestion",
            distribution_version="2.0.0",
        )
        worker = _descriptor(
            "2",
            binding_id="vendor.ray.parquet",
            distribution="vendor-ingestion",
            distribution_version="2.1.0",
        )
        gateway = _Gateway([worker])
        selection = ResolvedInputSelection(
            request=_request(binding_id=planning.binding_id),
            descriptor=planning,
        )

        with pytest.raises(DataSourceError, match="binding_distribution_version"):
            _resolver(gateway).open(selection)

        assert gateway.open_calls == []

"""Public ingestion request, gateway, receipt, and lifecycle tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping, cast

import pytest

from tributo._common.storage_profiles import StorageProfile
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingDescriptor,
    BindingKey,
    classify_transform_decisions,
)
from tributo.data.handle_adapters import adapt_daft_result_to_ray
from tributo.data.ingestion import (
    DaftDataFrameHandle,
    DistributionVersionEvidence,
    HandleOwnership,
    IngestionGateway,
    IngestionOpenResult,
    IngestionRequest,
    IngestionRuntimeContext,
    PhysicalSplitSummary,
    RayDataHandle,
    ReadOptions,
    SchemaContract,
    TransformDecision,
)
from tributo.data.provider_registry import resolve_provider
from tributo.data.scan_plan import ScanKind, SourceCapability
from tributo.data.source_config import (
    IcebergSourceConfig,
    ParquetSourceConfig,
    ProviderSourceConfig,
)
from tributo.data.transform_ir import SelectColumns, TransformPipeline
from tributo.exceptions import DataSourceError


def _descriptor() -> BindingDescriptor:
    return BindingDescriptor(
        key=BindingKey(
            "tributo.ray_data",
            ScanKind.FILE,
            "parquet",
            "test.ray.parquet",
        ),
        factory=lambda: object(),
        capabilities=frozenset({SourceCapability.PROJECTION}),
        distribution_name="tributo",
        distribution_version="1.0.0",
        engine_version_spec="==2.55.1",
    )


class _Bindings:
    def __init__(self, schema: str = "a" * 64) -> None:
        self.schema = schema
        self.calls: list[dict[str, Any]] = []

    def compile(
        self, **kwargs: Any
    ) -> tuple[
        BindingCompilation,
        BindingDescriptor,
        tuple[DistributionVersionEvidence, ...],
    ]:
        self.calls.append(kwargs)
        return (
            BindingCompilation(
                handle=RayDataHandle(object()),
                engine_version="2.55.1",
                reader_api="fake.read_parquet",
                transport_id="test",
                transform_decisions=classify_transform_decisions(kwargs["transforms"]),
                schema_fingerprint=self.schema,
                metadata_fetched=True,
            ),
            _descriptor(),
            (
                DistributionVersionEvidence(
                    distribution_name="ray", driver_version="2.55.1"
                ),
                DistributionVersionEvidence(
                    distribution_name="tributo", driver_version="1.0.0"
                ),
            ),
        )

    def describe(
        self, **kwargs: Any
    ) -> tuple[BindingDescriptor, frozenset[SourceCapability]]:
        plan = kwargs["plan"]
        return _descriptor(), plan.required_capabilities


def _request(**kwargs: Any) -> IngestionRequest:
    return IngestionRequest(
        source=ParquetSourceConfig(path="/tmp/input.parquet"),
        engine="ray",
        **kwargs,
    )


def test_request_requires_explicit_supported_engine() -> None:
    source = ParquetSourceConfig(path="/tmp/input.parquet")

    assert IngestionRequest(source=source, engine="ray").engine == "tributo.ray_data"
    with pytest.raises(ValueError, match="Field required"):
        IngestionRequest(source=source)
    with pytest.raises(ValueError, match="engine must be one of"):
        IngestionRequest(source=source, engine="auto")


def test_request_trace_context_is_credential_free_and_copied() -> None:
    trace_context = {"trace_id": "abc"}
    request = _request(trace_context=trace_context)

    trace_context["trace_id"] = "changed"
    assert request.trace_context == {"trace_id": "abc"}
    assert "trace_id" in request.model_dump_json()
    with pytest.raises(ValueError, match="credential field"):
        _request(trace_context={"access_token": "hidden"})


def test_request_copies_source_and_freezes_nested_mappings() -> None:
    source = ParquetSourceConfig(path="/tmp/input.parquet")
    resource_hints = {"CPU": 1.0}
    trace_context = {"trace_id": "abc"}
    request = IngestionRequest(
        source=source,
        engine="ray",
        read_options=ReadOptions(resource_hints=resource_hints),
        trace_context=trace_context,
    )

    source.path = "/tmp/changed.parquet"
    resource_hints["CPU"] = 2.0
    trace_context["trace_id"] = "changed"

    assert request.source.path == "/tmp/input.parquet"
    assert request.read_options.model_dump(mode="json")["resource_hints"] == {
        "CPU": 1.0
    }
    assert request.model_dump(mode="json")["trace_context"] == {"trace_id": "abc"}
    with pytest.raises(TypeError):
        cast(MutableMapping[str, float], request.read_options.resource_hints)["CPU"] = 3
    with pytest.raises(TypeError):
        cast(MutableMapping[str, str], request.trace_context)["trace_id"] = "other"


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf")])
def test_read_options_rejects_non_finite_resource_hints(amount: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        ReadOptions(resource_hints={"CPU": amount})


def test_receipt_metadata_contract_rejects_credentials() -> None:
    with pytest.raises(ValueError, match="credential-free"):
        PhysicalSplitSummary(detail="password=top-secret")
    with pytest.raises(ValueError, match="credential-free"):
        TransformDecision(
            ordinal=0,
            transform_type="filter_eq",
            pushdown_level="none",
            residual_required=True,
            compiled_result="residual",
            diagnostic="token=top-secret",
        )


def test_gateway_builds_complete_credential_free_receipt() -> None:
    bindings = _Bindings()
    source = ParquetSourceConfig(
        path="s3://bucket/input.parquet",
        s3={
            "access_key_id": "access-secret",
            "secret_access_key": "top-secret",
        },
    )
    request = IngestionRequest(
        source=source,
        engine="ray",
        transforms=TransformPipeline(steps=[SelectColumns(columns=["id", "score"])]),
    )

    result = IngestionGateway(bindings).open(request)

    receipt = result.receipt
    assert receipt.engine_id == "tributo.ray_data"
    assert receipt.provider_id == "tributo.parquet"
    assert receipt.connector_id == "parquet"
    assert receipt.scan_kind is ScanKind.FILE
    assert receipt.metadata_fetched is True
    assert receipt.reader_api == "fake.read_parquet"
    assert receipt.transport_id == "test"
    assert receipt.binding_id == "test.ray.parquet"
    assert len(receipt.transform_digest) == 64
    assert len(receipt.logical_plan_digest) == 64
    assert receipt.transform_decisions[0].compiled_result == "residual"
    assert {item.distribution_name for item in receipt.component_versions} == {
        "ray",
        "tributo",
    }
    serialized = receipt.model_dump_json()
    assert "access-secret" not in serialized
    assert "top-secret" not in serialized
    assert bindings.calls[0]["engine_id"] == "tributo.ray_data"


def test_runtime_context_is_passed_only_to_binding() -> None:
    bindings = _Bindings()
    context = IngestionRuntimeContext()

    IngestionGateway(bindings).open(_request(), context)

    assert bindings.calls[0]["runtime_context"] is context


@pytest.mark.parametrize(
    "source",
    [
        ParquetSourceConfig(path="data/input.parquet"),
        ProviderSourceConfig(provider="tributo.parquet", uri="data/input.parquet"),
    ],
)
def test_gateway_resolves_relative_file_sources_before_planning(
    source: ParquetSourceConfig | ProviderSourceConfig,
    tmp_path: Path,
) -> None:
    bindings = _Bindings()
    request = IngestionRequest(source=source, engine="ray")

    IngestionGateway(bindings, project_root_path=tmp_path).open(request)

    assert bindings.calls[0]["plan"].uri == str(
        tmp_path.resolve() / "data/input.parquet"
    )


def test_storage_profile_endpoint_affects_identity_but_credentials_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.data.ingestion.StorageProfileResolver.resolve",
        lambda self, name: StorageProfile(
            endpoint="http://user:password@minio:9000",
            region="us-east-1",
            access_key_id="profile-key",
            secret_access_key="profile-secret",
        ),
    )
    request = IngestionRequest(
        source=ParquetSourceConfig(path="s3://bucket/input.parquet"),
        engine="ray",
        storage_profile="production",
    )
    bindings = _Bindings()

    descriptor = IngestionGateway(bindings).describe(request)
    result = IngestionGateway(bindings).open(request)

    assert descriptor.request_digest == result.receipt.request_digest
    assert "minio:9000" in repr(bindings.calls[0]["plan"].options)
    serialized = result.receipt.model_dump_json()
    assert "profile-key" not in serialized
    assert "profile-secret" not in serialized


def test_storage_profile_rejects_non_s3_source() -> None:
    request = IngestionRequest(
        source=ParquetSourceConfig(path="/tmp/input.parquet"),
        engine="ray",
        storage_profile="production",
    )

    with pytest.raises(DataSourceError, match="requires an S3 file"):
        IngestionGateway(_Bindings()).describe(request)


def test_storage_profile_supports_s3_backed_iceberg_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = StorageProfile(
        endpoint="http://minio:9000",
        region="us-east-1",
        access_key_id="profile-key",
        secret_access_key="profile-secret",
    )
    monkeypatch.setattr(
        "tributo.data.ingestion.StorageProfileResolver.resolve",
        lambda self, name: profile,
    )
    request = IngestionRequest(
        source=IcebergSourceConfig(
            catalog="default",
            table="analytics.events",
            catalog_properties={"type": "sql", "uri": "sqlite:///:memory:"},
        ),
        engine="ray",
        storage_profile="production",
    )
    resolved = resolve_provider(request.source).normalize(request.source)

    attached = IngestionGateway._attach_storage_profile(request, resolved)

    assert attached.identity_options["s3"] == {
        "endpoint": "http://minio:9000",
        "region": "us-east-1",
    }
    assert attached.runtime_options["s3_profile"] is profile
    assert "profile-key" not in attached.ref_id()


def test_schema_contract_mismatch_releases_before_handoff() -> None:
    calls: list[str] = []

    class _ClosingBindings(_Bindings):
        def compile(
            self, **kwargs: Any
        ) -> tuple[
            BindingCompilation,
            BindingDescriptor,
            tuple[DistributionVersionEvidence, ...],
        ]:
            compilation, descriptor, versions = super().compile(**kwargs)
            return (
                BindingCompilation(
                    handle=compilation.handle,
                    engine_version=compilation.engine_version,
                    reader_api=compilation.reader_api,
                    transport_id=compilation.transport_id,
                    schema_fingerprint=compilation.schema_fingerprint,
                    close_callback=lambda: calls.append("close"),
                ),
                descriptor,
                versions,
            )

    request = _request(schema_contract=SchemaContract(fingerprint="b" * 64))

    with pytest.raises(DataSourceError, match="does not match"):
        IngestionGateway(_ClosingBindings(schema="a" * 64)).open(request)

    assert calls == ["close"]


def test_describe_is_deterministic_and_does_not_compile() -> None:
    bindings = _Bindings()
    request = _request()

    first = IngestionGateway(bindings).describe(request)
    second = IngestionGateway(bindings).describe(request)

    assert first == second
    assert first.handle_kind == "ray_data"
    assert first.provider_id == "tributo.parquet"
    assert first.capability_version == 1
    assert first.deferred_validations == (
        "runtime_connectivity",
        "engine_schema",
        "worker_versions",
    )
    assert bindings.calls == []


def test_owned_open_result_lifecycle_is_idempotent() -> None:
    calls: list[str] = []
    receipt = IngestionGateway(_Bindings()).open(_request()).receipt
    result = IngestionOpenResult(
        handle=RayDataHandle(object()),
        receipt=receipt,
        close_callback=lambda: calls.append("close"),
    )

    result.close()
    result.close()

    assert result.closed is True
    assert calls == ["close"]


def test_daft_to_ray_adapter_is_explicit_and_records_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_dataset = object()

    class _DaftFrame:
        def to_ray_dataset(self) -> object:
            return ray_dataset

    monkeypatch.setattr(
        "tributo.data.handle_adapters.importlib.metadata.version",
        lambda name: "2.55.1",
    )
    source_receipt = (
        IngestionGateway(_Bindings())
        .open(_request())
        .receipt.model_copy(update={"engine_id": "tributo.daft"})
    )
    source = IngestionOpenResult(
        handle=DaftDataFrameHandle(_DaftFrame()),
        receipt=source_receipt,
    )

    converted = adapt_daft_result_to_ray(source)

    assert converted.handle.dataset is ray_dataset
    assert converted.receipt.source_dataset_ref == source_receipt.dataset_ref
    assert converted.receipt.source_request_digest == source_receipt.request_digest
    assert converted.receipt.adapter_api == "daft.DataFrame.to_ray_dataset"
    assert converted.receipt.execution_may_be_triggered is True
    assert converted.receipt.full_driver_materialization is False
    assert converted.receipt.order_preserved is False


def test_daft_to_ray_adapter_rejects_wrong_or_closed_handle() -> None:
    ray_result = IngestionOpenResult(
        handle=RayDataHandle(object()),
        receipt=IngestionGateway(_Bindings()).open(_request()).receipt,
    )
    with pytest.raises(TypeError, match="DaftDataFrameHandle"):
        adapt_daft_result_to_ray(ray_result)

    daft_receipt = ray_result.receipt.model_copy(update={"engine_id": "tributo.daft"})
    daft_result = IngestionOpenResult(
        handle=DaftDataFrameHandle(object()),
        receipt=daft_receipt,
    )
    daft_result.close()
    with pytest.raises(ValueError, match="closed"):
        adapt_daft_result_to_ray(daft_result)


@pytest.mark.parametrize(
    "ownership", [HandleOwnership.BORROWED, HandleOwnership.SESSION_SCOPED]
)
def test_non_owned_open_result_is_idempotent_without_callbacks(
    ownership: HandleOwnership,
) -> None:
    result = IngestionOpenResult(
        handle=RayDataHandle(object()),
        receipt=IngestionGateway(_Bindings()).open(_request()).receipt,
        ownership=ownership,
    )

    result.close()
    result.close()

    assert result.closed is True


@pytest.mark.parametrize(
    "ownership", [HandleOwnership.BORROWED, HandleOwnership.SESSION_SCOPED]
)
def test_non_owned_open_result_rejects_callbacks(
    ownership: HandleOwnership,
) -> None:
    with pytest.raises(ValueError, match="Non-owned handles"):
        IngestionOpenResult(
            handle=RayDataHandle(object()),
            receipt=IngestionGateway(_Bindings()).open(_request()).receipt,
            ownership=ownership,
            close_callback=lambda: None,
        )


def test_cancel_prefers_owned_cancel_callback() -> None:
    calls: list[str] = []
    result = IngestionOpenResult(
        handle=RayDataHandle(object()),
        receipt=IngestionGateway(_Bindings()).open(_request()).receipt,
        cancel_callback=lambda: calls.append("cancel"),
        close_callback=lambda: calls.append("close"),
    )

    result.cancel()

    assert calls == ["cancel"]

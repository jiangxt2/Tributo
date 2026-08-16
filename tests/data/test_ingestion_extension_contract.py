"""End-to-end contract for installable catalog and format extensions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tributo.data import (
    DaftDataFrameHandle,
    IngestionGateway,
    IngestionRequest,
    ProviderSourceConfig,
    RayDataHandle,
    apply_source_projection,
)
from tributo.data._source_paths import resolve_file_source_path
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingDescriptor,
    BindingKey,
    BindingPlanConstraints,
    EngineBindings,
    classify_transform_decisions,
)
from tributo.data.provider import DataSourceProvider, ResolvedSource
from tributo.data.provider_registry import register_provider, unregister_provider
from tributo.data.scan_plan import (
    CatalogTableRef,
    ScanKind,
    SourceCapability,
    TableScan,
)
from tributo.data.source_config import CanonicalSourceInput
from tributo.exceptions import JobConfigurationError


class _ExternalHiveProvider(DataSourceProvider):
    provider_id = "example.hive"
    aliases = frozenset({"example_hive"})
    projection_option_name = "columns"

    def normalize(self, source: CanonicalSourceInput) -> ResolvedSource:
        if not isinstance(source, ProviderSourceConfig):
            raise JobConfigurationError("example.hive requires provider/uri input")
        allowed = {"columns", "database", "storage_format", "table"}
        unknown = sorted(set(source.options) - allowed)
        if unknown:
            raise JobConfigurationError(f"example.hive unknown options: {unknown}")
        database = source.options.get("database")
        table = source.options.get("table")
        storage_format = source.options.get("storage_format")
        if not all(isinstance(value, str) and value for value in (database, table)):
            raise JobConfigurationError("example.hive requires database and table")
        if storage_format not in {"parquet", "orc", "iceberg"}:
            raise JobConfigurationError(
                "example.hive storage_format must be parquet, orc, or iceberg"
            )
        columns = source.options.get("columns")
        if columns is not None and (
            not isinstance(columns, list)
            or any(not isinstance(column, str) or not column for column in columns)
        ):
            raise JobConfigurationError("example.hive columns must be strings")
        identity = {
            "database": database,
            "table": table,
            "storage_format": storage_format,
        }
        if columns:
            identity["columns"] = list(columns)
        return ResolvedSource(
            provider_id=self.provider_id,
            canonical_uri=source.uri,
            identity_options=identity,
        )

    def plan(self, resolved: ResolvedSource) -> TableScan:
        columns = tuple(resolved.identity_options.get("columns", ()))
        required = (
            frozenset(
                {
                    SourceCapability.PROJECTION,
                    SourceCapability.PARTITION_PRUNING,
                }
            )
            if columns
            else frozenset({SourceCapability.PARTITION_PRUNING})
        )
        return TableScan(
            provider_id=self.provider_id,
            connector_id="hive",
            table=CatalogTableRef(
                "hive",
                (str(resolved.identity_options["database"]),),
                str(resolved.identity_options["table"]),
            ),
            storage_format_id=str(resolved.identity_options["storage_format"]),
            required_capabilities=required,
            options={"columns": columns},
        )


class _ExternalOrcProvider(DataSourceProvider):
    provider_id = "example.orc"
    relative_uri_is_path = True

    def normalize(self, source: CanonicalSourceInput) -> ResolvedSource:
        if not isinstance(source, ProviderSourceConfig):
            raise JobConfigurationError("example.orc requires provider/uri input")
        return ResolvedSource(
            provider_id=self.provider_id,
            canonical_uri=source.uri,
        )


class _ExternalHiveBinding:
    def __init__(self, engine_id: str) -> None:
        self.engine_id = engine_id
        self.seen_plan: TableScan | None = None

    def compile(self, request: Any) -> BindingCompilation:
        if not isinstance(request.plan, TableScan):
            raise AssertionError("Hive binding requires TableScan")
        self.seen_plan = request.plan
        if self.engine_id == "tributo.ray_data":
            handle = RayDataHandle(object())
            engine_version = "2.55.1"
        else:
            handle = DaftDataFrameHandle(object())
            engine_version = "0.7.23"
        return BindingCompilation(
            handle=handle,
            engine_version=engine_version,
            transform_decisions=classify_transform_decisions(request.transforms),
            reader_api="external_hive.read_table",
            transport_id="external-hive",
            diagnostics=("catalog and storage resolution delegated to connector",),
        )


@pytest.mark.parametrize(
    ("engine_id", "expected_handle"),
    [
        ("tributo.ray_data", RayDataHandle),
        ("tributo.daft", DaftDataFrameHandle),
    ],
)
def test_new_hive_provider_and_bindings_require_no_consumer_change(
    monkeypatch: pytest.MonkeyPatch,
    engine_id: str,
    expected_handle: type[RayDataHandle] | type[DaftDataFrameHandle],
) -> None:
    versions = {
        "daft": "0.7.23",
        "external-hive-connector": "1.0.0",
        "ray": "2.55.1",
    }
    monkeypatch.setattr(
        "tributo.data.engine_binding.importlib.metadata.version",
        lambda name: versions[name],
    )
    binding = _ExternalHiveBinding(engine_id)
    bindings = EngineBindings()
    bindings.register(
        BindingDescriptor(
            key=BindingKey(
                engine_id,
                ScanKind.TABLE,
                "hive",
                f"external_hive.{engine_id.rsplit('.', 1)[-1]}.hive",
            ),
            factory=lambda: binding,
            capabilities=frozenset(
                {
                    SourceCapability.PROJECTION,
                    SourceCapability.PARTITION_PRUNING,
                }
            ),
            distribution_name="external-hive-connector",
            distribution_version="1.0.0",
            engine_version_spec=(
                "==2.55.1" if engine_id.endswith("ray_data") else "==0.7.23"
            ),
            constraints=BindingPlanConstraints(
                catalog_ids=frozenset({"hive"}),
                storage_format_ids=frozenset({"parquet", "orc", "iceberg"}),
            ),
        )
    )
    register_provider(_ExternalHiveProvider)
    try:
        source = apply_source_projection(
            ProviderSourceConfig(
                provider="example.hive",
                uri="thrift://hive-metastore:9083",
                options={
                    "database": "analytics",
                    "table": "events",
                    "storage_format": "orc",
                },
            ),
            ["id", "score"],
        )
        request = IngestionRequest(source=source, engine=engine_id)
        gateway = IngestionGateway(bindings)

        descriptor = gateway.describe(request)
        result = gateway.open(request)

        assert descriptor.provider_id == "example.hive"
        assert descriptor.connector_id == "hive"
        assert isinstance(result.handle, expected_handle)
        assert result.receipt.provider_id == "example.hive"
        assert result.receipt.reader_api == "external_hive.read_table"
        assert binding.seen_plan is not None
        assert binding.seen_plan.storage_format_id == "orc"
        assert binding.seen_plan.table == CatalogTableRef(
            "hive", ("analytics",), "events"
        )
        result.close()
    finally:
        unregister_provider("example.hive")


def test_new_file_provider_declares_relative_path_semantics(tmp_path: Path) -> None:
    register_provider(_ExternalOrcProvider)
    try:
        source = ProviderSourceConfig(provider="example.orc", uri="data/events.orc")

        resolved = resolve_file_source_path(source, tmp_path)

        assert isinstance(resolved, ProviderSourceConfig)
        assert resolved.uri == str(tmp_path / "data/events.orc")
    finally:
        unregister_provider("example.orc")

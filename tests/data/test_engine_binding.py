"""Controlled engine-binding registration and compilation tests."""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any

import pytest

import tributo.data.bindings as builtin_bindings
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingDescriptor,
    BindingKey,
    BindingPlanConstraints,
    BindingStageError,
    EngineBindings,
    classify_transform_decisions,
)
from tributo.data.ingestion import (
    HandleOwnership,
    IngestionRuntimeContext,
    RayDataHandle,
    ReadOptions,
    TransformDecision,
)
from tributo.data.scan_plan import (
    CatalogTableRef,
    FileScan,
    ScanKind,
    SourceCapability,
    TableScan,
)
from tributo.data.transform_ir import FilterEq, Limit, SelectColumns, TransformPipeline
from tributo.exceptions import (
    DataSourceError,
    EngineNotAvailableError,
    JobConfigurationError,
)

_TEST_BINDING_ID = "test.ray.parquet"


@pytest.fixture(autouse=True)
def installed_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "daft": "0.7.23",
        "pyiceberg": "0.11.1",
        "ray": "2.55.1",
        "test-binding": "1.2.3",
        "test-helper": "4.5.6",
        "tributo": "1.0.0",
    }
    monkeypatch.setattr(
        "tributo.data.engine_binding.importlib.metadata.version",
        lambda name: versions[name],
    )


class _Binding:
    def compile(self, request: Any) -> BindingCompilation:
        return BindingCompilation(
            handle=RayDataHandle(object()),
            engine_version="2.55.1",
            transform_decisions=classify_transform_decisions(request.transforms),
            reader_api="test.native_reader",
            transport_id="test",
        )


class _WrongReturnBinding:
    def compile(self, request: object) -> object:
        return object()


def _descriptor(
    *,
    factory: Any = _Binding,
    capabilities: frozenset[SourceCapability] = frozenset(),
    distribution_name: str = "test-binding",
    distribution_version: str = "1.2.3",
    binding_id: str = _TEST_BINDING_ID,
    constraints: BindingPlanConstraints | None = None,
    dependency_distributions: tuple[str, ...] = (),
) -> BindingDescriptor:
    return BindingDescriptor(
        key=BindingKey("tributo.ray_data", ScanKind.FILE, "parquet", binding_id),
        factory=factory,
        capabilities=capabilities,
        distribution_name=distribution_name,
        distribution_version=distribution_version,
        engine_version_spec="==2.55.1",
        constraints=constraints or BindingPlanConstraints(),
        dependency_distributions=dependency_distributions,
    )


def _plan(*, columns: tuple[str, ...] = ()) -> FileScan:
    return FileScan(
        provider_id="tributo.parquet",
        connector_id="parquet",
        uri="/tmp/input.parquet",
        filesystem_id="local",
        options={"columns": columns},
    )


def _compile(
    bindings: EngineBindings,
    *,
    plan: FileScan | None = None,
    transforms: TransformPipeline | None = None,
    context: IngestionRuntimeContext | None = None,
) -> tuple[BindingCompilation, BindingDescriptor, tuple[Any, ...]]:
    return bindings.compile(
        engine_id="tributo.ray_data",
        binding_id=None,
        plan=plan or _plan(),
        runtime_options={},
        transforms=transforms or TransformPipeline(),
        read_options=ReadOptions(),
        source_ref="0" * 64,
        runtime_context=context or IngestionRuntimeContext(),
    )


def test_duplicate_registration_fails_atomically() -> None:
    bindings = EngineBindings()
    descriptor = _descriptor()
    bindings.register(descriptor)

    with pytest.raises(JobConfigurationError, match="already registered"):
        bindings.register(descriptor)


def test_explicit_third_party_descriptor_is_resolved() -> None:
    bindings = EngineBindings()
    bindings.register(_descriptor())

    assert bindings.resolve(_descriptor().key).distribution_name == "test-binding"


def test_plan_constraints_select_non_conflicting_bindings() -> None:
    bindings = EngineBindings()
    local = _descriptor(
        binding_id="test.ray.parquet.local",
        constraints=BindingPlanConstraints(
            filesystem_ids=frozenset({"local", "file", "s3"})
        ),
    )
    hdfs = _descriptor(
        binding_id="test.ray.parquet.hdfs",
        constraints=BindingPlanConstraints(filesystem_ids=frozenset({"hdfs"})),
    )
    bindings.register(local)
    bindings.register(hdfs)

    selected_local, _ = bindings.describe(engine_id="tributo.ray_data", plan=_plan())
    selected_hdfs, _ = bindings.describe(
        engine_id="tributo.ray_data",
        plan=FileScan(
            provider_id="tributo.parquet",
            connector_id="parquet",
            uri="hdfs://namenode/data/input.parquet",
            filesystem_id="hdfs",
        ),
    )

    assert selected_local.key.binding_id == "test.ray.parquet.local"
    assert selected_hdfs.key.binding_id == "test.ray.parquet.hdfs"


def test_catalog_and_storage_format_constraints_select_hive_binding() -> None:
    bindings = EngineBindings()
    for storage_format_id in ("parquet", "orc"):
        bindings.register(
            BindingDescriptor(
                key=BindingKey(
                    "tributo.ray_data",
                    ScanKind.TABLE,
                    "hive",
                    f"test.ray.hive.{storage_format_id}",
                ),
                factory=_Binding,
                capabilities=frozenset(),
                distribution_name="test-binding",
                distribution_version="1.2.3",
                engine_version_spec="==2.55.1",
                constraints=BindingPlanConstraints(
                    catalog_ids=frozenset({"hive"}),
                    storage_format_ids=frozenset({storage_format_id}),
                ),
            )
        )
    plan = TableScan(
        provider_id="example.hive",
        connector_id="hive",
        table=CatalogTableRef("hive", ("analytics",), "events"),
        storage_format_id="orc",
    )

    selected, _ = bindings.describe(engine_id="tributo.ray_data", plan=plan)

    assert selected.key.binding_id == "test.ray.hive.orc"


def test_table_constraints_do_not_match_unknown_catalog_storage() -> None:
    constraints = BindingPlanConstraints(
        catalog_ids=frozenset({"hive"}),
        storage_format_ids=frozenset({"orc"}),
    )

    assert not constraints.matches(
        TableScan(
            provider_id="example.hive",
            connector_id="hive",
            table=CatalogTableRef("glue", ("analytics",), "events"),
            storage_format_id="orc",
        )
    )
    assert not constraints.matches(
        TableScan(
            provider_id="example.hive",
            connector_id="hive",
            table=CatalogTableRef("hive", ("analytics",), "events"),
        )
    )


def test_ambiguous_bindings_require_explicit_binding_id() -> None:
    bindings = EngineBindings()
    first = _descriptor(binding_id="test.ray.parquet.first")
    second = _descriptor(binding_id="test.ray.parquet.second")
    bindings.register(first)
    bindings.register(second)

    with pytest.raises(JobConfigurationError, match="Multiple installed bindings"):
        bindings.describe(engine_id="tributo.ray_data", plan=_plan())

    selected, _ = bindings.describe(
        engine_id="tributo.ray_data",
        binding_id="test.ray.parquet.second",
        plan=_plan(),
    )
    assert selected.key == second.key


def test_requirement_constraints_report_only_matching_install_options() -> None:
    bindings = EngineBindings()
    bindings.register_requirement(
        BindingKey(
            "tributo.ray_data",
            ScanKind.FILE,
            "parquet",
            "third_party.ray.parquet.hdfs",
        ),
        "pip install third-party-hdfs",
        constraints=BindingPlanConstraints(filesystem_ids=frozenset({"hdfs"})),
    )

    with pytest.raises(EngineNotAvailableError, match="third-party-hdfs"):
        bindings.describe(
            engine_id="tributo.ray_data",
            plan=FileScan(
                provider_id="tributo.parquet",
                connector_id="parquet",
                uri="hdfs://namenode/data/input.parquet",
                filesystem_id="hdfs",
            ),
        )

    with pytest.raises(EngineNotAvailableError) as exc_info:
        bindings.describe(engine_id="tributo.ray_data", plan=_plan())
    assert "third-party-hdfs" not in str(exc_info.value)


def test_declared_connector_distribution_must_match_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.data.engine_binding.importlib.metadata.version",
        lambda name: "2.55.1" if name == "ray" else "9.9.9",
    )

    with pytest.raises(EngineNotAvailableError, match="declares distribution version"):
        EngineBindings().register(_descriptor())


def test_missing_binding_reports_install_hint() -> None:
    key = BindingKey("tributo.daft", ScanKind.SQL, "doris", "test.daft.doris")
    bindings = EngineBindings()
    bindings.register_requirement(key, "pip install tributo-doris")

    with pytest.raises(EngineNotAvailableError, match="tributo-doris"):
        bindings.resolve(key)


def test_required_projection_capability_fails_closed() -> None:
    bindings = EngineBindings()
    bindings.register(_descriptor())

    with pytest.raises(JobConfigurationError, match="projection"):
        _compile(bindings, plan=_plan(columns=("id",)))


def test_binding_factory_receives_engine_neutral_request_and_returns_evidence() -> None:
    bindings = EngineBindings()
    bindings.register(_descriptor())

    compilation, descriptor, versions = _compile(bindings)

    assert isinstance(compilation.handle, RayDataHandle)
    assert descriptor.distribution_name == "test-binding"
    assert {(item.distribution_name, item.driver_version) for item in versions} == {
        ("ray", "2.55.1"),
        ("test-binding", "1.2.3"),
    }
    assert all(not item.worker_validation_complete for item in versions)


def test_binding_dependency_distributions_are_part_of_version_evidence() -> None:
    bindings = EngineBindings()
    bindings.register(_descriptor(dependency_distributions=("test-helper",)))

    _, _, versions = _compile(bindings)

    assert {(item.distribution_name, item.driver_version) for item in versions} == {
        ("ray", "2.55.1"),
        ("test-binding", "1.2.3"),
        ("test-helper", "4.5.6"),
    }


def test_worker_distribution_versions_are_proved() -> None:
    bindings = EngineBindings()
    bindings.register(_descriptor())
    context = IngestionRuntimeContext(
        distribution_probe=lambda names: {
            name: (("2.55.1",) if name == "ray" else ("1.2.3",)) for name in names
        },
        require_worker_validation=True,
    )

    _, _, versions = _compile(bindings, context=context)

    assert all(item.worker_validation_complete for item in versions)


def test_worker_distribution_mismatch_fails_closed() -> None:
    calls: list[str] = []

    class _ClosingBinding(_Binding):
        def compile(self, request: Any) -> BindingCompilation:
            compilation = super().compile(request)
            return BindingCompilation(
                handle=compilation.handle,
                engine_version=compilation.engine_version,
                reader_api=compilation.reader_api,
                transport_id=compilation.transport_id,
                close_callback=lambda: calls.append("close"),
            )

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_ClosingBinding))
    context = IngestionRuntimeContext(
        distribution_probe=lambda names: {
            name: (("2.54.0",) if name == "ray" else ("1.2.3",)) for name in names
        }
    )

    with pytest.raises(
        EngineNotAvailableError,
        match=r"\[worker_version_mismatch\].*Driver and worker versions differ",
    ) as exc_info:
        _compile(bindings, context=context)

    assert calls == ["close"]
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_invalid_worker_distribution_version_is_engine_unavailable() -> None:
    calls: list[str] = []

    class _ClosingBinding(_Binding):
        def compile(self, request: Any) -> BindingCompilation:
            compilation = super().compile(request)
            return BindingCompilation(
                handle=compilation.handle,
                engine_version=compilation.engine_version,
                reader_api=compilation.reader_api,
                transport_id=compilation.transport_id,
                close_callback=lambda: calls.append("close"),
            )

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_ClosingBinding))
    context = IngestionRuntimeContext(
        distribution_probe=lambda names: {
            name: (("not-a-version",) if name == "ray" else ("1.2.3",))
            for name in names
        }
    )

    with pytest.raises(
        EngineNotAvailableError,
        match=r"\[worker_version_invalid\].*invalid version",
    ) as exc_info:
        _compile(bindings, context=context)

    assert calls == ["close"]
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize("probe_error_type", [RuntimeError, DataSourceError])
def test_worker_distribution_probe_error_is_normalized_without_secret(
    probe_error_type: type[Exception],
) -> None:
    calls: list[str] = []

    class _ClosingBinding(_Binding):
        def compile(self, request: Any) -> BindingCompilation:
            compilation = super().compile(request)
            return BindingCompilation(
                handle=compilation.handle,
                engine_version=compilation.engine_version,
                transform_decisions=compilation.transform_decisions,
                reader_api=compilation.reader_api,
                transport_id=compilation.transport_id,
                close_callback=lambda: calls.append("close"),
            )

    def _probe(_names: tuple[str, ...]) -> Any:
        raise probe_error_type("token=top-secret")

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_ClosingBinding))

    with pytest.raises(DataSourceError, match="during compile") as exc_info:
        _compile(
            bindings,
            context=IngestionRuntimeContext(distribution_probe=_probe),
        )

    assert calls == ["close"]
    assert "top-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_binding_return_type_fails_closed() -> None:
    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_WrongReturnBinding))

    with pytest.raises(JobConfigurationError, match="expected BindingCompilation"):
        _compile(bindings)


def test_every_transform_requires_a_decision() -> None:
    class _IncompleteBinding:
        def compile(self, request: object) -> BindingCompilation:
            return BindingCompilation(
                handle=RayDataHandle(object()),
                engine_version="2.55.1",
                reader_api="test.native_reader",
                transport_id="test",
            )

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_IncompleteBinding))

    with pytest.raises(
        JobConfigurationError,
        match=r"\[transform_decision_count_mismatch\].*classified 0 of 1",
    ) as exc_info:
        _compile(
            bindings,
            transforms=TransformPipeline(steps=[SelectColumns(columns=["id"])]),
        )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_native_exception_is_normalized_without_secret() -> None:
    class _ExplodingBinding:
        def compile(self, request: object) -> BindingCompilation:
            raise RuntimeError("top-secret credential")

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_ExplodingBinding))

    with pytest.raises(
        DataSourceError, match=r"during compile \[unexpected\] with RuntimeError"
    ) as exc_info:
        _compile(bindings)

    assert "top-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_binding_domain_exception_keeps_category_without_secret_cause() -> None:
    class _ExplodingBinding:
        def compile(self, request: object) -> BindingCompilation:
            raise JobConfigurationError("password=top-secret")

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_ExplodingBinding))

    with pytest.raises(
        JobConfigurationError,
        match=r"during compile \[invalid_configuration\] with JobConfigurationError",
    ) as exc_info:
        _compile(bindings)

    assert "top-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_binding_engine_error_keeps_category_without_secret_cause() -> None:
    class _ExplodingBinding:
        def compile(self, request: object) -> BindingCompilation:
            raise EngineNotAvailableError("token=top-secret")

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_ExplodingBinding))

    with pytest.raises(
        EngineNotAvailableError,
        match=r"during compile \[engine_not_available\]",
    ) as exc_info:
        _compile(bindings)

    assert "top-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_binding_factory_exception_is_normalized_without_secret() -> None:
    def _factory() -> Any:
        raise RuntimeError("password=top-secret")

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_factory))

    with pytest.raises(
        DataSourceError, match=r"during compile \[unexpected\] with RuntimeError"
    ) as exc_info:
        _compile(bindings)

    assert "top-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_third_party_binding_cannot_publish_framework_diagnostic() -> None:
    class _ExplodingBinding:
        def compile(self, request: object) -> BindingCompilation:
            raise BindingStageError.framework_diagnostic(
                "validate_capabilities",
                error_type=JobConfigurationError,
                diagnostic_code="third_party_detail",
                diagnostic="internal-only detail",
            )

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_ExplodingBinding))

    with pytest.raises(JobConfigurationError) as exc_info:
        _compile(bindings)

    assert "third_party_detail" not in str(exc_info.value)
    assert "internal-only detail" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_first_party_binding_can_publish_framework_diagnostic() -> None:
    class _ExplodingBinding:
        def compile(self, request: object) -> BindingCompilation:
            raise BindingStageError.framework_diagnostic(
                "validate_capabilities",
                error_type=JobConfigurationError,
                diagnostic_code="unsupported_read_hints",
                diagnostic="Ray binding does not support read hint: concurrency",
            )

    _ExplodingBinding.__module__ = "tributo.data.bindings.test"

    bindings = EngineBindings()
    bindings.register(
        _descriptor(
            factory=_ExplodingBinding,
            distribution_name="tributo",
            distribution_version="1.0.0",
        )
    )

    with pytest.raises(
        JobConfigurationError,
        match=r"\[unsupported_read_hints\].*concurrency",
    ) as exc_info:
        _compile(bindings)

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    ("engine_id", "descriptor_factory"),
    [
        ("tributo.ray_data", builtin_bindings._ray_iceberg_descriptor),
        ("tributo.daft", builtin_bindings._daft_iceberg_descriptor),
    ],
)
def test_builtin_iceberg_file_io_rejection_preserves_diagnostic(
    engine_id: str,
    descriptor_factory: Any,
) -> None:
    bindings = EngineBindings()
    bindings.register(descriptor_factory())
    plan = TableScan(
        provider_id="tributo.iceberg",
        connector_id="iceberg",
        table=CatalogTableRef(
            catalog_id="prod",
            namespace=("analytics",),
            table="events",
        ),
    )

    with pytest.raises(
        JobConfigurationError,
        match=(
            r"\[unsupported_iceberg_file_io\].*"
            r"Built-in Iceberg bindings require PyArrowFileIO"
        ),
    ) as exc_info:
        bindings.compile(
            engine_id=engine_id,
            binding_id=None,
            plan=plan,
            runtime_options={
                "catalog_properties": {"py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO"}
            },
            transforms=TransformPipeline(),
            read_options=ReadOptions(),
            source_ref="0" * 64,
            runtime_context=IngestionRuntimeContext(),
        )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_framework_binding_diagnostic_rejects_credentials() -> None:
    with pytest.raises(ValueError, match="credential-free"):
        BindingStageError.framework_diagnostic(
            "compile",
            error_type=JobConfigurationError,
            diagnostic_code="unsafe_detail",
            diagnostic="token=top-secret",
        )


def test_binding_metadata_and_install_hints_reject_credentials() -> None:
    with pytest.raises(ValueError, match="credential-free"):
        BindingCompilation(
            handle=RayDataHandle(object()),
            engine_version="2.55.1",
            reader_api="test.native_reader",
            transport_id="test",
            diagnostics=("password=top-secret",),
        )

    bindings = EngineBindings()
    with pytest.raises(ValueError, match="credential-free"):
        bindings.register_requirement(
            BindingKey("tributo.daft", ScanKind.SQL, "doris", "test.daft.doris"),
            "pip install connector --token=top-secret",
        )


@pytest.mark.parametrize(
    "ownership", [HandleOwnership.BORROWED, HandleOwnership.SESSION_SCOPED]
)
def test_non_owned_binding_compilation_rejects_callbacks(
    ownership: HandleOwnership,
) -> None:
    with pytest.raises(ValueError, match="Non-owned BindingCompilation"):
        BindingCompilation(
            handle=RayDataHandle(object()),
            engine_version="2.55.1",
            reader_api="test.native_reader",
            transport_id="test",
            ownership=ownership,
            close_callback=lambda: None,
        )


def test_pushdown_decisions_are_structured_and_limit_is_residual() -> None:
    pipeline = TransformPipeline(steps=[FilterEq(column="id", value=1), Limit(count=1)])

    decisions = classify_transform_decisions(pipeline, {0: "inexact"})

    assert decisions[0].compiled_result == "pushed_and_residual"
    assert decisions[0].residual_required is True
    assert decisions[1].pushdown_level == "none"
    assert decisions[1].compiled_result == "residual"
    with pytest.raises(JobConfigurationError, match="Limit"):
        classify_transform_decisions(pipeline, {1: "exact"})


def test_transform_decision_rejects_inconsistent_state() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        TransformDecision(
            ordinal=0,
            transform_type="filter_eq",
            pushdown_level="exact",
            residual_required=True,
            compiled_result="residual",
        )


def test_daft_sql_descriptors_use_new_package_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builtin_bindings,
        "_distribution_version",
        lambda name: {
            "daft-clickhouse": "1.0",
            "daft-doris": "1.0",
        }.get(name),
    )

    clickhouse = builtin_bindings._daft_clickhouse_descriptor()
    doris = builtin_bindings._daft_doris_descriptor()

    assert clickhouse.key.binding_id == "daft_clickhouse.daft.clickhouse"
    assert clickhouse.distribution_name == "daft-clickhouse"
    assert clickhouse.dependency_distributions == ("clickhouse-connect",)
    assert doris.key.binding_id == "daft_doris.daft.doris"
    assert doris.distribution_name == "daft-doris"
    assert doris.dependency_distributions == ("PyMySQL",)


def test_missing_daft_sql_packages_report_new_install_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builtin_bindings, "_DEFAULT_BINDINGS", None)
    versions = {
        "daft": "0.7.23",
        "ray": "2.55.1",
        "tributo": "1.0.0",
    }
    monkeypatch.setattr(
        builtin_bindings, "_distribution_version", lambda name: versions.get(name)
    )

    bindings = builtin_bindings.default_engine_bindings()

    with pytest.raises(
        EngineNotAvailableError,
        match=r"daft_clickhouse\.daft\.clickhouse.*daft-clickhouse.*tributo\[clickhouse\]",
    ):
        bindings.resolve(
            BindingKey(
                "tributo.daft",
                ScanKind.SQL,
                "clickhouse",
                "daft_clickhouse.daft.clickhouse",
            )
        )
    with pytest.raises(
        EngineNotAvailableError,
        match=r"daft_doris\.daft\.doris.*daft-doris.*tributo\[mysql\]",
    ):
        bindings.resolve(
            BindingKey(
                "tributo.daft",
                ScanKind.SQL,
                "doris",
                "daft_doris.daft.doris",
            )
        )


def test_incompatible_optional_daft_does_not_disable_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builtin_bindings, "_DEFAULT_BINDINGS", None)
    monkeypatch.setattr(
        builtin_bindings,
        "_distribution_version",
        lambda name: {
            "daft": "0.6.0",
            "ray": "2.55.1",
            "tributo": "1.0.0",
        }.get(name),
    )
    versions = {"ray": "2.55.1", "tributo": "1.0.0"}
    monkeypatch.setattr(
        "tributo.data.engine_binding.importlib.metadata.version",
        lambda name: versions[name],
    )

    bindings = builtin_bindings.default_engine_bindings()

    assert (
        bindings.resolve(
            BindingKey(
                "tributo.ray_data",
                ScanKind.FILE,
                "parquet",
                "tributo.ray.parquet",
            )
        ).key.engine_id
        == "tributo.ray_data"
    )
    with pytest.raises(EngineNotAvailableError, match="data-daft"):
        bindings.resolve(
            BindingKey(
                "tributo.daft",
                ScanKind.FILE,
                "parquet",
                "tributo.daft.parquet",
            )
        )


def test_incompatible_ray_does_not_disable_daft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builtin_bindings, "_DEFAULT_BINDINGS", None)
    versions = {"ray": "2.54.0", "daft": "0.7.23", "tributo": "1.0.0"}
    monkeypatch.setattr(
        builtin_bindings, "_distribution_version", lambda name: versions.get(name)
    )
    monkeypatch.setattr(
        "tributo.data.engine_binding.importlib.metadata.version",
        lambda name: versions[name],
    )

    bindings = builtin_bindings.default_engine_bindings()

    assert (
        bindings.resolve(
            BindingKey(
                "tributo.daft",
                ScanKind.FILE,
                "parquet",
                "tributo.daft.parquet",
            )
        ).key.engine_id
        == "tributo.daft"
    )
    with pytest.raises(
        EngineNotAvailableError,
        match=r"installed ray 2\.54\.0; required ==2\.55\.1",
    ):
        bindings.resolve(
            BindingKey(
                "tributo.ray_data",
                ScanKind.FILE,
                "parquet",
                "tributo.ray.parquet",
            )
        )


def test_missing_postgresql_dependency_does_not_disable_daft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builtin_bindings, "_DEFAULT_BINDINGS", None)
    versions = {
        "ray": "2.55.1",
        "daft": "0.7.23",
        "tributo": "1.0.0",
        "psycopg": "3.3.4",
        "psycopg-binary": "3.3.4",
        "SQLAlchemy": "2.0.51",
    }
    monkeypatch.setattr(
        builtin_bindings, "_distribution_version", lambda name: versions.get(name)
    )
    monkeypatch.setattr(
        "tributo.data.engine_binding.importlib.metadata.version",
        lambda name: versions[name],
    )

    bindings = builtin_bindings.default_engine_bindings()

    assert (
        bindings.resolve(
            BindingKey(
                "tributo.daft",
                ScanKind.FILE,
                "parquet",
                "tributo.daft.parquet",
            )
        ).key.engine_id
        == "tributo.daft"
    )
    with pytest.raises(EngineNotAvailableError, match=r"tributo\[postgresql\]"):
        bindings.resolve(
            BindingKey(
                "tributo.daft",
                ScanKind.SQL,
                "postgresql",
                "tributo.daft.postgresql",
            )
        )


def test_binding_reported_engine_version_must_match_installed() -> None:
    calls: list[str] = []

    class _WrongVersionBinding:
        def compile(self, request: Any) -> BindingCompilation:
            return BindingCompilation(
                handle=RayDataHandle(object()),
                engine_version="2.54.0",
                transform_decisions=classify_transform_decisions(request.transforms),
                reader_api="test.native_reader",
                transport_id="test",
                close_callback=lambda: calls.append("close"),
            )

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_WrongVersionBinding))

    with pytest.raises(
        EngineNotAvailableError,
        match=r"\[engine_version_mismatch\].*reported engine version 2.54.0, "
        r"installed 2.55.1",
    ) as exc_info:
        _compile(bindings)

    assert calls == ["close"]
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_engine_distribution_disappearing_after_compile_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _ClosingBinding(_Binding):
        def compile(self, request: Any) -> BindingCompilation:
            compilation = super().compile(request)
            return BindingCompilation(
                handle=compilation.handle,
                engine_version=compilation.engine_version,
                transform_decisions=compilation.transform_decisions,
                reader_api=compilation.reader_api,
                transport_id=compilation.transport_id,
                close_callback=lambda: calls.append("close"),
            )

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_ClosingBinding))

    def _missing_engine(name: str) -> str:
        if name == "ray":
            raise importlib.metadata.PackageNotFoundError("token=top-secret")
        return "1.2.3"

    monkeypatch.setattr(
        "tributo.data.engine_binding.importlib.metadata.version", _missing_engine
    )

    with pytest.raises(
        EngineNotAvailableError,
        match=r"\[engine_distribution_missing\].*no longer installed",
    ) as exc_info:
        _compile(bindings)

    assert calls == ["close"]
    assert "top-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_cleanup_exception_is_not_retained_by_contract_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []

    def _close() -> None:
        calls.append("close")
        raise RuntimeError("password=top-secret")

    class _WrongVersionBinding:
        def compile(self, request: Any) -> BindingCompilation:
            return BindingCompilation(
                handle=RayDataHandle(object()),
                engine_version="2.54.0",
                transform_decisions=classify_transform_decisions(request.transforms),
                reader_api="test.native_reader",
                transport_id="test",
                close_callback=_close,
            )

    bindings = EngineBindings()
    bindings.register(_descriptor(factory=_WrongVersionBinding))

    with caplog.at_level(logging.WARNING, logger="tributo.data.ingestion"):
        with pytest.raises(EngineNotAvailableError) as exc_info:
            _compile(bindings)

    assert calls == ["close"]
    assert "Failed to close invalid owned ingestion handle" in caplog.text
    assert "exception_type=RuntimeError" in caplog.text
    assert "top-secret" not in caplog.text
    cleanup_records = [
        record
        for record in caplog.records
        if record.name == "tributo.data.ingestion"
        and "invalid owned ingestion handle" in record.getMessage()
    ]
    assert len(cleanup_records) == 1
    assert cleanup_records[0].exc_info is None
    assert cleanup_records[0].exc_text is None
    assert cleanup_records[0].args == ("close", "RuntimeError")
    assert "top-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_builtin_descriptor_uses_loaded_tributo_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tributo.__version__", "7.8.9")

    assert builtin_bindings._distribution_version("tributo") == "7.8.9"

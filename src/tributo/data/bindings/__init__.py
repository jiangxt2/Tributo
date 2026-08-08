"""Built-in thin bindings for public Ray Data and Daft reader APIs."""

from __future__ import annotations

import importlib.metadata
import threading
from collections.abc import Callable

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from tributo.data.engine_binding import (
    BindingDescriptor,
    BindingKey,
    BindingPlanConstraints,
    EngineBindings,
)
from tributo.data.ingestion import ReadHint
from tributo.data.scan_plan import ScanKind, SourceCapability

_DEFAULT_BINDINGS: EngineBindings | None = None
_DEFAULT_BINDINGS_LOCK = threading.Lock()
_RAY_VERSION_SPEC = "==2.55.1"
_DAFT_VERSION_SPEC = ">=0.7.0,<0.8.0"
_RAY_INSTALL_HINT = "pip install 'ray[default,serve,tune]==2.55.1'"
_DAFT_INSTALL_HINT = "pip install 'tributo[data-daft]'"
_DAFT_LANCE_INSTALL_HINT = "pip install 'tributo[data,data-daft]'"
_DATA_INSTALL_HINT = "pip install 'tributo[data]'"
_DAFT_OLAP_INSTALL_HINT = "pip install 'daft-olap-connectors[clickhouse,doris]'"
_RAY_DORIS_INSTALL_HINT = "pip install 'ray-doris[mysql,flight]'"
_POSTGRESQL_INSTALL_HINT = "pip install 'tributo[postgresql]'"


def _distribution_version(name: str) -> str | None:
    if name == "tributo":
        from tributo import __version__

        return __version__
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _ray_parquet_descriptor() -> BindingDescriptor:
    from tributo.data.bindings.ray_parquet import RayParquetBinding

    return BindingDescriptor(
        key=BindingKey(
            "tributo.ray_data",
            ScanKind.FILE,
            "parquet",
            "tributo.ray.parquet",
        ),
        factory=RayParquetBinding,
        capabilities=frozenset({SourceCapability.PROJECTION}),
        distribution_name="tributo",
        distribution_version=_distribution_version("tributo") or "1.0.0",
        engine_version_spec=_RAY_VERSION_SPEC,
        constraints=BindingPlanConstraints(
            filesystem_ids=frozenset({"local", "file", "s3"})
        ),
        supported_read_hints=frozenset(
            {
                ReadHint.TARGET_PARALLELISM,
                ReadHint.BATCH_SIZE,
                ReadHint.CONCURRENCY,
            }
        ),
        install_hint=_RAY_INSTALL_HINT,
    )


def _daft_parquet_descriptor() -> BindingDescriptor:
    from tributo.data.bindings.daft_parquet import DaftParquetBinding

    return BindingDescriptor(
        key=BindingKey(
            "tributo.daft",
            ScanKind.FILE,
            "parquet",
            "tributo.daft.parquet",
        ),
        factory=DaftParquetBinding,
        capabilities=frozenset({SourceCapability.PROJECTION}),
        distribution_name="tributo",
        distribution_version=_distribution_version("tributo") or "1.0.0",
        engine_version_spec=_DAFT_VERSION_SPEC,
        constraints=BindingPlanConstraints(
            filesystem_ids=frozenset({"local", "file", "s3"})
        ),
        install_hint=_DAFT_INSTALL_HINT,
    )


def _ray_csv_descriptor() -> BindingDescriptor:
    from tributo.data.bindings.ray_csv import RayCsvBinding

    return BindingDescriptor(
        key=BindingKey("tributo.ray_data", ScanKind.FILE, "csv", "tributo.ray.csv"),
        factory=RayCsvBinding,
        capabilities=frozenset({SourceCapability.PROJECTION}),
        distribution_name="tributo",
        distribution_version=_distribution_version("tributo") or "1.0.0",
        engine_version_spec=_RAY_VERSION_SPEC,
        constraints=BindingPlanConstraints(
            filesystem_ids=frozenset({"local", "file", "s3"})
        ),
        supported_read_hints=frozenset(
            {ReadHint.TARGET_PARALLELISM, ReadHint.CONCURRENCY}
        ),
        install_hint=_RAY_INSTALL_HINT,
    )


def _daft_csv_descriptor() -> BindingDescriptor:
    from tributo.data.bindings.daft_csv import DaftCsvBinding

    return BindingDescriptor(
        key=BindingKey("tributo.daft", ScanKind.FILE, "csv", "tributo.daft.csv"),
        factory=DaftCsvBinding,
        capabilities=frozenset({SourceCapability.PROJECTION}),
        distribution_name="tributo",
        distribution_version=_distribution_version("tributo") or "1.0.0",
        engine_version_spec=_DAFT_VERSION_SPEC,
        constraints=BindingPlanConstraints(
            filesystem_ids=frozenset({"local", "file", "s3"})
        ),
        install_hint=_DAFT_INSTALL_HINT,
    )


def _ray_hdfs_descriptor(connector_id: str) -> BindingDescriptor:
    from tributo.data.bindings.ray_hdfs import (
        RayHdfsCsvBinding,
        RayHdfsParquetBinding,
    )

    factory = RayHdfsParquetBinding if connector_id == "parquet" else RayHdfsCsvBinding
    supported_hints = {ReadHint.TARGET_PARALLELISM, ReadHint.CONCURRENCY}
    if connector_id == "parquet":
        supported_hints.add(ReadHint.BATCH_SIZE)
    return BindingDescriptor(
        key=BindingKey(
            "tributo.ray_data",
            ScanKind.FILE,
            connector_id,
            f"tributo.ray.{connector_id}.hdfs",
        ),
        factory=factory,
        capabilities=frozenset({SourceCapability.PROJECTION}),
        distribution_name="tributo",
        distribution_version=_distribution_version("tributo") or "1.0.0",
        engine_version_spec=_RAY_VERSION_SPEC,
        dependency_distributions=("pyarrow",),
        constraints=BindingPlanConstraints(filesystem_ids=frozenset({"hdfs"})),
        supported_read_hints=frozenset(supported_hints),
        install_hint=_RAY_INSTALL_HINT,
    )


def _ray_iceberg_descriptor() -> BindingDescriptor:
    from tributo.data.bindings.ray_iceberg import RayIcebergBinding

    return BindingDescriptor(
        key=BindingKey(
            "tributo.ray_data",
            ScanKind.TABLE,
            "iceberg",
            "tributo.ray.iceberg",
        ),
        factory=RayIcebergBinding,
        capabilities=frozenset(
            {
                SourceCapability.PROJECTION,
                SourceCapability.PREDICATE_PUSHDOWN,
                SourceCapability.SNAPSHOT,
                SourceCapability.SCHEMA_EVOLUTION,
                SourceCapability.PARTITION_PRUNING,
            }
        ),
        distribution_name="tributo",
        distribution_version=_distribution_version("tributo") or "1.0.0",
        engine_version_spec=_RAY_VERSION_SPEC,
        dependency_distributions=("pyiceberg",),
        supported_read_hints=frozenset({ReadHint.TARGET_PARALLELISM}),
        install_hint=_DATA_INSTALL_HINT,
    )


def _daft_iceberg_descriptor() -> BindingDescriptor:
    from tributo.data.bindings.daft_iceberg import DaftIcebergBinding

    return BindingDescriptor(
        key=BindingKey(
            "tributo.daft",
            ScanKind.TABLE,
            "iceberg",
            "tributo.daft.iceberg",
        ),
        factory=DaftIcebergBinding,
        capabilities=frozenset(
            {
                SourceCapability.PROJECTION,
                SourceCapability.SNAPSHOT,
                SourceCapability.SCHEMA_EVOLUTION,
                SourceCapability.PARTITION_PRUNING,
            }
        ),
        distribution_name="tributo",
        distribution_version=_distribution_version("tributo") or "1.0.0",
        engine_version_spec=_DAFT_VERSION_SPEC,
        dependency_distributions=("pyiceberg",),
        install_hint=_DATA_INSTALL_HINT,
    )


def _ray_lance_descriptor() -> BindingDescriptor:
    from tributo.data.bindings.ray_lance import RayLanceBinding

    return BindingDescriptor(
        key=BindingKey(
            "tributo.ray_data", ScanKind.TABLE, "lance", "tributo.ray.lance"
        ),
        factory=RayLanceBinding,
        capabilities=frozenset(
            {
                SourceCapability.PROJECTION,
                SourceCapability.PREDICATE_PUSHDOWN,
                SourceCapability.SNAPSHOT,
            }
        ),
        distribution_name="tributo",
        distribution_version=_distribution_version("tributo") or "1.0.0",
        engine_version_spec=_RAY_VERSION_SPEC,
        dependency_distributions=("pylance",),
        supported_read_hints=frozenset(
            {ReadHint.TARGET_PARALLELISM, ReadHint.CONCURRENCY}
        ),
        install_hint=_DATA_INSTALL_HINT,
    )


def _daft_lance_descriptor() -> BindingDescriptor:
    from tributo.data.bindings.daft_lance import DaftLanceBinding

    return BindingDescriptor(
        key=BindingKey("tributo.daft", ScanKind.TABLE, "lance", "tributo.daft.lance"),
        factory=DaftLanceBinding,
        capabilities=frozenset(
            {
                SourceCapability.PROJECTION,
                SourceCapability.PREDICATE_PUSHDOWN,
                SourceCapability.SNAPSHOT,
            }
        ),
        distribution_name="tributo",
        distribution_version=_distribution_version("tributo") or "1.0.0",
        engine_version_spec=_DAFT_VERSION_SPEC,
        dependency_distributions=("pylance", "daft-lance"),
        install_hint=_DAFT_LANCE_INSTALL_HINT,
    )


def _daft_olap_descriptor(connector_id: str) -> BindingDescriptor:
    from tributo.data.bindings.daft_olap import (
        DaftClickHouseBinding,
        DaftDorisBinding,
    )

    factory = (
        DaftClickHouseBinding if connector_id == "clickhouse" else DaftDorisBinding
    )
    return BindingDescriptor(
        key=BindingKey(
            "tributo.daft",
            ScanKind.SQL,
            connector_id,
            f"daft_olap.daft.{connector_id}",
        ),
        factory=factory,
        capabilities=frozenset({SourceCapability.PROJECTION}),
        distribution_name="daft-olap-connectors",
        distribution_version=(
            _distribution_version("daft-olap-connectors") or "0.1.0a1"
        ),
        engine_version_spec=_DAFT_VERSION_SPEC,
        supported_read_hints=frozenset(
            {ReadHint.TARGET_PARALLELISM, ReadHint.BATCH_SIZE}
        ),
        install_hint=_DAFT_OLAP_INSTALL_HINT,
    )


def _ray_doris_descriptor() -> BindingDescriptor:
    from tributo.data.bindings.ray_doris import RayDorisBinding

    return BindingDescriptor(
        key=BindingKey(
            "tributo.ray_data",
            ScanKind.SQL,
            "doris",
            "ray_doris.ray.doris",
        ),
        factory=RayDorisBinding,
        capabilities=frozenset({SourceCapability.PROJECTION}),
        distribution_name="ray-doris",
        distribution_version=_distribution_version("ray-doris") or "0.1.0a1",
        engine_version_spec=_RAY_VERSION_SPEC,
        supported_read_hints=frozenset(
            {
                ReadHint.TARGET_PARALLELISM,
                ReadHint.BATCH_SIZE,
                ReadHint.CONCURRENCY,
            }
        ),
        install_hint=_RAY_DORIS_INSTALL_HINT,
    )


def _ray_postgresql_descriptor() -> BindingDescriptor:
    from tributo.data.bindings.ray_postgresql import RayPostgreSqlBinding

    return BindingDescriptor(
        key=BindingKey(
            "tributo.ray_data",
            ScanKind.SQL,
            "postgresql",
            "tributo.ray.postgresql",
        ),
        factory=RayPostgreSqlBinding,
        capabilities=frozenset({SourceCapability.PROJECTION}),
        distribution_name="tributo",
        distribution_version=_distribution_version("tributo") or "1.0.0",
        engine_version_spec=_RAY_VERSION_SPEC,
        dependency_distributions=("psycopg", "psycopg-binary"),
        supported_read_hints=frozenset({ReadHint.CONCURRENCY}),
        install_hint=_POSTGRESQL_INSTALL_HINT,
    )


def _daft_postgresql_descriptor() -> BindingDescriptor:
    from tributo.data.bindings.daft_postgresql import DaftPostgreSqlBinding

    return BindingDescriptor(
        key=BindingKey(
            "tributo.daft",
            ScanKind.SQL,
            "postgresql",
            "tributo.daft.postgresql",
        ),
        factory=DaftPostgreSqlBinding,
        capabilities=frozenset({SourceCapability.PROJECTION}),
        distribution_name="tributo",
        distribution_version=_distribution_version("tributo") or "1.0.0",
        engine_version_spec=_DAFT_VERSION_SPEC,
        dependency_distributions=(
            "psycopg",
            "psycopg-binary",
            "SQLAlchemy",
            "sqlglot",
        ),
        install_hint=_POSTGRESQL_INSTALL_HINT,
    )


def _is_compatible_distribution(name: str, version_spec: str) -> bool:
    version = _distribution_version(name)
    if version is None:
        return False
    try:
        return Version(version) in SpecifierSet(version_spec)
    except InvalidVersion:
        return False


def _register_optional(
    bindings: EngineBindings,
    *,
    descriptor_factory: Callable[[], BindingDescriptor],
    key: BindingKey,
    engine_distribution: str,
    engine_version_spec: str,
    install_hint: str,
    constraints: BindingPlanConstraints | None = None,
    dependencies: tuple[str, ...] = (),
) -> None:
    installed_engine = _distribution_version(engine_distribution)
    engine_compatible = _is_compatible_distribution(
        engine_distribution, engine_version_spec
    )
    missing_dependencies = tuple(
        name for name in dependencies if _distribution_version(name) is None
    )
    available = engine_compatible and not missing_dependencies
    if available:
        bindings.register(descriptor_factory())
    else:
        details: list[str] = []
        if installed_engine is not None and not engine_compatible:
            details.append(
                f"installed {engine_distribution} {installed_engine}; "
                f"required {engine_version_spec}"
            )
        if missing_dependencies:
            details.append("missing " + ", ".join(sorted(missing_dependencies)))
        diagnostic_hint = (
            f"{install_hint} ({'; '.join(details)})" if details else install_hint
        )
        bindings.register_requirement(
            key,
            diagnostic_hint,
            constraints=constraints,
        )


def default_engine_bindings() -> EngineBindings:
    """Return the process-wide registry without importing optional engines."""
    global _DEFAULT_BINDINGS
    if _DEFAULT_BINDINGS is not None:
        return _DEFAULT_BINDINGS
    with _DEFAULT_BINDINGS_LOCK:
        if _DEFAULT_BINDINGS is not None:
            return _DEFAULT_BINDINGS
        bindings = EngineBindings()
        file_constraints = BindingPlanConstraints(
            filesystem_ids=frozenset({"local", "file", "s3"})
        )
        hdfs_constraints = BindingPlanConstraints(filesystem_ids=frozenset({"hdfs"}))
        specs: tuple[
            tuple[
                Callable[[], BindingDescriptor],
                BindingKey,
                str,
                str,
                str,
                BindingPlanConstraints | None,
                tuple[str, ...],
            ],
            ...,
        ] = (
            (
                _ray_parquet_descriptor,
                BindingKey(
                    "tributo.ray_data",
                    ScanKind.FILE,
                    "parquet",
                    "tributo.ray.parquet",
                ),
                "ray",
                _RAY_VERSION_SPEC,
                _RAY_INSTALL_HINT,
                file_constraints,
                (),
            ),
            (
                _daft_parquet_descriptor,
                BindingKey(
                    "tributo.daft",
                    ScanKind.FILE,
                    "parquet",
                    "tributo.daft.parquet",
                ),
                "daft",
                _DAFT_VERSION_SPEC,
                _DAFT_INSTALL_HINT,
                file_constraints,
                (),
            ),
            (
                _ray_csv_descriptor,
                BindingKey(
                    "tributo.ray_data",
                    ScanKind.FILE,
                    "csv",
                    "tributo.ray.csv",
                ),
                "ray",
                _RAY_VERSION_SPEC,
                _RAY_INSTALL_HINT,
                file_constraints,
                (),
            ),
            (
                _daft_csv_descriptor,
                BindingKey(
                    "tributo.daft",
                    ScanKind.FILE,
                    "csv",
                    "tributo.daft.csv",
                ),
                "daft",
                _DAFT_VERSION_SPEC,
                _DAFT_INSTALL_HINT,
                file_constraints,
                (),
            ),
            *(
                (
                    lambda connector_id=connector_id: _ray_hdfs_descriptor(
                        connector_id
                    ),
                    BindingKey(
                        "tributo.ray_data",
                        ScanKind.FILE,
                        connector_id,
                        f"tributo.ray.{connector_id}.hdfs",
                    ),
                    "ray",
                    _RAY_VERSION_SPEC,
                    _RAY_INSTALL_HINT,
                    hdfs_constraints,
                    ("pyarrow",),
                )
                for connector_id in ("parquet", "csv")
            ),
            (
                _ray_iceberg_descriptor,
                BindingKey(
                    "tributo.ray_data",
                    ScanKind.TABLE,
                    "iceberg",
                    "tributo.ray.iceberg",
                ),
                "ray",
                _RAY_VERSION_SPEC,
                _DATA_INSTALL_HINT,
                None,
                ("pyiceberg",),
            ),
            (
                _daft_iceberg_descriptor,
                BindingKey(
                    "tributo.daft",
                    ScanKind.TABLE,
                    "iceberg",
                    "tributo.daft.iceberg",
                ),
                "daft",
                _DAFT_VERSION_SPEC,
                _DATA_INSTALL_HINT,
                None,
                ("pyiceberg",),
            ),
            (
                _ray_lance_descriptor,
                BindingKey(
                    "tributo.ray_data",
                    ScanKind.TABLE,
                    "lance",
                    "tributo.ray.lance",
                ),
                "ray",
                _RAY_VERSION_SPEC,
                _DATA_INSTALL_HINT,
                None,
                ("pylance",),
            ),
            (
                _daft_lance_descriptor,
                BindingKey(
                    "tributo.daft",
                    ScanKind.TABLE,
                    "lance",
                    "tributo.daft.lance",
                ),
                "daft",
                _DAFT_VERSION_SPEC,
                _DAFT_LANCE_INSTALL_HINT,
                None,
                ("pylance", "daft-lance"),
            ),
            *(
                (
                    lambda connector_id=connector_id: _daft_olap_descriptor(
                        connector_id
                    ),
                    BindingKey(
                        "tributo.daft",
                        ScanKind.SQL,
                        connector_id,
                        f"daft_olap.daft.{connector_id}",
                    ),
                    "daft",
                    _DAFT_VERSION_SPEC,
                    _DAFT_OLAP_INSTALL_HINT,
                    None,
                    ("daft-olap-connectors",),
                )
                for connector_id in ("clickhouse", "doris")
            ),
            (
                _ray_doris_descriptor,
                BindingKey(
                    "tributo.ray_data",
                    ScanKind.SQL,
                    "doris",
                    "ray_doris.ray.doris",
                ),
                "ray",
                _RAY_VERSION_SPEC,
                _RAY_DORIS_INSTALL_HINT,
                None,
                ("ray-doris",),
            ),
            (
                _ray_postgresql_descriptor,
                BindingKey(
                    "tributo.ray_data",
                    ScanKind.SQL,
                    "postgresql",
                    "tributo.ray.postgresql",
                ),
                "ray",
                _RAY_VERSION_SPEC,
                _POSTGRESQL_INSTALL_HINT,
                None,
                ("psycopg", "psycopg-binary"),
            ),
            (
                _daft_postgresql_descriptor,
                BindingKey(
                    "tributo.daft",
                    ScanKind.SQL,
                    "postgresql",
                    "tributo.daft.postgresql",
                ),
                "daft",
                _DAFT_VERSION_SPEC,
                _POSTGRESQL_INSTALL_HINT,
                None,
                ("psycopg", "psycopg-binary", "SQLAlchemy", "sqlglot"),
            ),
        )
        for (
            descriptor_factory,
            key,
            engine_distribution,
            engine_version_spec,
            install_hint,
            constraints,
            dependencies,
        ) in specs:
            _register_optional(
                bindings,
                descriptor_factory=descriptor_factory,
                key=key,
                engine_distribution=engine_distribution,
                engine_version_spec=engine_version_spec,
                install_hint=install_hint,
                constraints=constraints,
                dependencies=dependencies,
            )
        from tributo.data.binding_plugins import register_discovered_bindings

        register_discovered_bindings(bindings)
        _DEFAULT_BINDINGS = bindings
        return bindings

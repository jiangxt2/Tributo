"""Serializable, engine-neutral logical scan contracts for bounded ingestion."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias, cast

from tributo.data.refs import _credential_paths, _ensure_credential_free_uri, digest
from tributo.util.annotations import DeveloperAPI

_FILE_PROJECTION_OPTION = "columns"


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, frozenset)):
        return [_thaw_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _freeze_value(value)
    if not isinstance(frozen, Mapping):
        raise AssertionError("mapping freeze must preserve the outer mapping")
    return frozen


def _validate_plan_options(options: Mapping[str, Any]) -> None:
    if not isinstance(options, Mapping):
        raise ValueError("Logical scan options must be a mapping")
    if any(not isinstance(key, str) or not key for key in options):
        raise ValueError("Logical scan option keys must be non-empty strings")
    leaked = _credential_paths(options)
    if leaked:
        raise ValueError(
            "Logical scan options must not contain credential field(s): "
            f"{sorted(leaked)}"
        )
    try:
        digest(options)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Logical scan options must contain only serializable, engine-neutral values"
        ) from exc


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError(f"{field_name} must be a non-empty identifier")


def _validate_plan_version(value: int, plan_name: str) -> None:
    if type(value) is not int or value != 1:
        raise ValueError(f"Unsupported {plan_name} version {value!r}")


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _reject_unknown_fields(
    value: Mapping[str, Any], allowed: frozenset[str], field_name: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unknown field(s): {unknown}")


def _require_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence")
    return tuple(value)


def _require_string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    items = _require_sequence(value, field_name)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{field_name} values must be non-empty strings")
    return cast(tuple[str, ...], items)


@DeveloperAPI
class ScanKind(str, Enum):
    FILE = "file"
    SQL = "sql"
    TABLE = "table"


@DeveloperAPI
class SourceCapability(str, Enum):
    PROJECTION = "projection"
    PREDICATE_PUSHDOWN = "predicate_pushdown"
    SNAPSHOT = "snapshot"
    SCHEMA_EVOLUTION = "schema_evolution"
    PARTITION_PRUNING = "partition_pruning"


def _freeze_capabilities(
    capabilities: frozenset[SourceCapability] | set[SourceCapability],
) -> frozenset[SourceCapability]:
    frozen = frozenset(capabilities)
    if any(not isinstance(item, SourceCapability) for item in frozen):
        raise ValueError("required_capabilities must contain SourceCapability values")
    return frozen


@DeveloperAPI
class FileDiscoveryMode(str, Enum):
    EXACT = "exact"
    DIRECTORY = "directory"
    GLOB = "glob"


@DeveloperAPI
@dataclass(frozen=True)
class FileDiscoveryStrategy:
    """Logical file discovery requirements delegated to the native reader."""

    mode: FileDiscoveryMode = FileDiscoveryMode.EXACT
    recursive: bool = False
    extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, FileDiscoveryMode):
            raise ValueError("File discovery mode must be FileDiscoveryMode")
        if type(self.recursive) is not bool:
            raise ValueError("File discovery recursive must be a bool")
        extensions = _require_string_sequence(
            self.extensions, "File discovery extensions"
        )
        if any(not item.startswith(".") for item in extensions):
            raise ValueError("File discovery extensions must start with '.'")
        if len(set(extensions)) != len(extensions):
            raise ValueError("File discovery extensions must not contain duplicates")
        object.__setattr__(self, "extensions", extensions)


@DeveloperAPI
class PartitioningKind(str, Enum):
    NONE = "none"
    HIVE = "hive"


@DeveloperAPI
@dataclass(frozen=True)
class PartitioningRule:
    """Logical path partition interpretation, not a filesystem implementation."""

    kind: PartitioningKind = PartitioningKind.NONE
    field_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PartitioningKind):
            raise ValueError("Partitioning kind must be PartitioningKind")
        field_names = _require_string_sequence(
            self.field_names, "Partition field names"
        )
        if len(set(field_names)) != len(field_names):
            raise ValueError("Partition field names must not contain duplicates")
        if self.kind is PartitioningKind.NONE and field_names:
            raise ValueError("Partition field names require a partitioning kind")
        object.__setattr__(self, "field_names", field_names)


@DeveloperAPI
@dataclass(frozen=True)
class FileScan:
    """Logical file read delegated to an engine or installed Connector."""

    provider_id: str
    connector_id: str
    uri: str
    filesystem_id: str
    discovery: FileDiscoveryStrategy = field(default_factory=FileDiscoveryStrategy)
    partitioning: PartitioningRule = field(default_factory=PartitioningRule)
    required_capabilities: frozenset[SourceCapability] = field(
        default_factory=frozenset
    )
    options: Mapping[str, Any] = field(default_factory=dict)
    input_schema_fingerprint: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        _validate_identifier(self.provider_id, "FileScan.provider_id")
        _validate_identifier(self.connector_id, "FileScan.connector_id")
        _validate_identifier(self.filesystem_id, "FileScan.filesystem_id")
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("FileScan.uri must be a non-empty string")
        _validate_plan_version(self.version, "FileScan")
        _ensure_credential_free_uri(self.uri)
        _validate_plan_options(self.options)
        if not isinstance(self.discovery, FileDiscoveryStrategy):
            raise ValueError("FileScan.discovery must be FileDiscoveryStrategy")
        if not isinstance(self.partitioning, PartitioningRule):
            raise ValueError("FileScan.partitioning must be PartitioningRule")
        capabilities = set(self.required_capabilities)
        # Built-in file Providers normalize their projection into this one
        # engine-neutral option before capability negotiation.
        if self.options.get(_FILE_PROJECTION_OPTION):
            capabilities.add(SourceCapability.PROJECTION)
        if self.partitioning.kind is not PartitioningKind.NONE:
            capabilities.add(SourceCapability.PARTITION_PRUNING)
        object.__setattr__(
            self, "required_capabilities", _freeze_capabilities(capabilities)
        )
        object.__setattr__(self, "options", _freeze_mapping(self.options))
        _validate_fingerprint(self.input_schema_fingerprint, "input_schema_fingerprint")

    @property
    def scan_kind(self) -> ScanKind:
        return ScanKind.FILE


@DeveloperAPI
class SqlPredicateOperator(str, Enum):
    EQ = "eq"
    NOT_EQ = "not_eq"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    IS_IN = "is_in"


SqlPredicateScalar: TypeAlias = str | int | float | bool


@DeveloperAPI
@dataclass(frozen=True)
class SqlPredicate:
    """Small structured SQL predicate compiled by a dialect whitelist."""

    column: str
    operator: SqlPredicateOperator
    values: tuple[SqlPredicateScalar, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("SqlPredicate.column must be non-empty")
        if not isinstance(self.operator, SqlPredicateOperator):
            raise ValueError("SqlPredicate.operator must be SqlPredicateOperator")
        values = _require_sequence(self.values, "SqlPredicate.values")
        if any(type(value) not in {str, int, float, bool} for value in values):
            raise ValueError("SQL predicate values must be scalar values")
        if any(
            isinstance(value, float) and not math.isfinite(value) for value in values
        ):
            raise ValueError("SQL predicate floats must be finite")
        unary = {SqlPredicateOperator.IS_NULL, SqlPredicateOperator.IS_NOT_NULL}
        if self.operator in unary and values:
            raise ValueError("NULL predicates do not accept values")
        if self.operator is SqlPredicateOperator.IS_IN:
            if not values:
                raise ValueError("IS IN requires at least one value")
        elif self.operator not in unary and len(values) != 1:
            raise ValueError("Comparison predicates require exactly one value")
        object.__setattr__(self, "values", values)


@DeveloperAPI
@dataclass(frozen=True)
class SqlTableRead:
    """Structured SQL table target; no raw SQL text or connection state."""

    table: str
    schema: str | None = None
    catalog: str | None = None
    projection: tuple[str, ...] = ()
    predicates: tuple[SqlPredicate, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.table, str) or not self.table:
            raise ValueError("SqlTableRead.table must be a non-empty string")
        for field_name, value in (("schema", self.schema), ("catalog", self.catalog)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"SqlTableRead.{field_name} must be non-empty")
        projection = _require_string_sequence(
            self.projection, "SqlTableRead.projection"
        )
        if len(set(projection)) != len(projection):
            raise ValueError("SqlTableRead.projection must not contain duplicates")
        predicates = _require_sequence(self.predicates, "SqlTableRead.predicates")
        if any(not isinstance(item, SqlPredicate) for item in predicates):
            raise ValueError("SqlTableRead.predicates must contain SqlPredicate values")
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "predicates", predicates)


@DeveloperAPI
@dataclass(frozen=True)
class ParameterizedQuery:
    """Compatibility SQL target identified by a credential-free query digest."""

    query_digest: str

    def __post_init__(self) -> None:
        _validate_fingerprint(self.query_digest, "ParameterizedQuery.query_digest")


SqlReadTarget: TypeAlias = SqlTableRead | ParameterizedQuery


@DeveloperAPI
class SqlShardMode(str, Enum):
    SINGLE = "single"
    AUTO = "auto"
    PARALLEL = "parallel"


@DeveloperAPI
@dataclass(frozen=True)
class SqlShardRequirement:
    """Logical parallelism requirement; physical splits remain Connector-owned."""

    mode: SqlShardMode = SqlShardMode.SINGLE
    columns: tuple[str, ...] = ()
    target_partitions: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, SqlShardMode):
            raise ValueError("SQL shard mode must be SqlShardMode")
        if self.mode is SqlShardMode.SINGLE and (
            self.columns or self.target_partitions
        ):
            raise ValueError("Single SQL reads cannot declare shard hints")
        columns = _require_string_sequence(self.columns, "SQL shard columns")
        if len(set(columns)) != len(columns):
            raise ValueError("SQL shard columns must not contain duplicates")
        if self.mode is SqlShardMode.PARALLEL and not columns:
            raise ValueError("Parallel SQL reads require at least one shard column")
        if self.mode is SqlShardMode.AUTO and columns:
            raise ValueError("Automatic SQL reads cannot declare shard columns")
        if self.target_partitions is not None and (
            type(self.target_partitions) is not int or self.target_partitions < 1
        ):
            raise ValueError("SQL target_partitions must be a positive integer")
        object.__setattr__(self, "columns", columns)


@DeveloperAPI
class ConsistencyRequirement(str, Enum):
    BEST_EFFORT = "best_effort"
    STATEMENT = "statement_consistent"
    SNAPSHOT = "snapshot_consistent"


@DeveloperAPI
@dataclass(frozen=True)
class SqlScan:
    """Logical database read whose physical plan belongs to the Connector."""

    provider_id: str
    connector_id: str
    target: SqlReadTarget
    sharding: SqlShardRequirement = field(default_factory=SqlShardRequirement)
    consistency: ConsistencyRequirement = ConsistencyRequirement.BEST_EFFORT
    required_capabilities: frozenset[SourceCapability] = field(
        default_factory=frozenset
    )
    options: Mapping[str, Any] = field(default_factory=dict)
    input_schema_fingerprint: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        _validate_identifier(self.provider_id, "SqlScan.provider_id")
        _validate_identifier(self.connector_id, "SqlScan.connector_id")
        _validate_plan_version(self.version, "SqlScan")
        if not isinstance(self.target, (SqlTableRead, ParameterizedQuery)):
            raise ValueError("SqlScan.target must be a structured SQL target")
        if not isinstance(self.sharding, SqlShardRequirement):
            raise ValueError("SqlScan.sharding must be SqlShardRequirement")
        if not isinstance(self.consistency, ConsistencyRequirement):
            raise ValueError("SqlScan.consistency must be ConsistencyRequirement")
        capabilities = set(self.required_capabilities)
        if isinstance(self.target, SqlTableRead):
            if self.target.projection:
                capabilities.add(SourceCapability.PROJECTION)
            if self.target.predicates:
                capabilities.add(SourceCapability.PREDICATE_PUSHDOWN)
        object.__setattr__(
            self, "required_capabilities", _freeze_capabilities(capabilities)
        )
        _validate_plan_options(self.options)
        object.__setattr__(self, "options", _freeze_mapping(self.options))
        _validate_fingerprint(self.input_schema_fingerprint, "input_schema_fingerprint")

    @property
    def scan_kind(self) -> ScanKind:
        return ScanKind.SQL


@DeveloperAPI
@dataclass(frozen=True)
class CatalogTableRef:
    catalog_id: str
    namespace: tuple[str, ...]
    table: str

    def __post_init__(self) -> None:
        _validate_identifier(self.catalog_id, "CatalogTableRef.catalog_id")
        if not isinstance(self.table, str) or not self.table:
            raise ValueError("CatalogTableRef.table must be a non-empty string")
        namespace = _require_string_sequence(
            self.namespace, "CatalogTableRef.namespace"
        )
        object.__setattr__(self, "namespace", namespace)


@DeveloperAPI
@dataclass(frozen=True)
class UriTableRef:
    uri: str

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("UriTableRef.uri must be a non-empty string")
        _ensure_credential_free_uri(self.uri)


TableReference: TypeAlias = CatalogTableRef | UriTableRef


@DeveloperAPI
@dataclass(frozen=True)
class SnapshotVersionRef:
    snapshot_id: int

    def __post_init__(self) -> None:
        if type(self.snapshot_id) is not int or self.snapshot_id < 0:
            raise ValueError("Snapshot ID must be a non-negative integer")


@DeveloperAPI
@dataclass(frozen=True)
class NumericVersionRef:
    version: int

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 0:
            raise ValueError("Table version must be a non-negative integer")


@DeveloperAPI
@dataclass(frozen=True)
class TagVersionRef:
    tag: str

    def __post_init__(self) -> None:
        if not isinstance(self.tag, str) or not self.tag:
            raise ValueError("Table tag must be non-empty")


@DeveloperAPI
@dataclass(frozen=True)
class AsOfVersionRef:
    timestamp: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime) or self.timestamp.tzinfo is None:
            raise ValueError("As-of timestamp must be timezone-aware")


VersionRef: TypeAlias = (
    SnapshotVersionRef | NumericVersionRef | TagVersionRef | AsOfVersionRef
)


@DeveloperAPI
@dataclass(frozen=True)
class TableScan:
    """Logical catalog or versioned-table read delegated to a native reader.

    ``storage_format_id`` is an optional assertion for catalog-resolved tables,
    such as a Hive external table expected to resolve to Parquet or ORC.  The
    selected Binding still owns catalog lookup and physical file discovery.
    """

    provider_id: str
    connector_id: str
    table: TableReference
    storage_format_id: str | None = None
    version_ref: VersionRef | None = None
    required_capabilities: frozenset[SourceCapability] = field(
        default_factory=frozenset
    )
    options: Mapping[str, Any] = field(default_factory=dict)
    input_schema_fingerprint: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        _validate_identifier(self.provider_id, "TableScan.provider_id")
        _validate_identifier(self.connector_id, "TableScan.connector_id")
        _validate_plan_version(self.version, "TableScan")
        if not isinstance(self.table, (CatalogTableRef, UriTableRef)):
            raise ValueError("TableScan.table must be a table reference")
        if self.storage_format_id is not None:
            _validate_identifier(self.storage_format_id, "TableScan.storage_format_id")
        version_types = (
            SnapshotVersionRef,
            NumericVersionRef,
            TagVersionRef,
            AsOfVersionRef,
        )
        if self.version_ref is not None and not isinstance(
            self.version_ref, version_types
        ):
            raise ValueError("TableScan.version_ref must be a typed VersionRef")
        capabilities = set(self.required_capabilities)
        if self.version_ref is not None:
            capabilities.add(SourceCapability.SNAPSHOT)
        object.__setattr__(
            self, "required_capabilities", _freeze_capabilities(capabilities)
        )
        _validate_plan_options(self.options)
        object.__setattr__(self, "options", _freeze_mapping(self.options))
        _validate_fingerprint(self.input_schema_fingerprint, "input_schema_fingerprint")

    @property
    def scan_kind(self) -> ScanKind:
        return ScanKind.TABLE


LogicalScanPlan: TypeAlias = FileScan | SqlScan | TableScan


def _validate_fingerprint(value: str | None, field_name: str) -> None:
    if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be lowercase SHA-256 hex")


def _capabilities_to_list(plan: LogicalScanPlan) -> list[str]:
    return sorted(item.value for item in plan.required_capabilities)


@DeveloperAPI
def logical_scan_plan_to_dict(plan: LogicalScanPlan) -> dict[str, Any]:
    """Serialize a logical plan without credentials or engine-native objects."""
    common = {
        "version": plan.version,
        "provider_id": plan.provider_id,
        "connector_id": plan.connector_id,
        "required_capabilities": _capabilities_to_list(plan),
        "options": _thaw_value(plan.options),
        "input_schema_fingerprint": plan.input_schema_fingerprint,
    }
    if isinstance(plan, FileScan):
        return {
            "scan_kind": ScanKind.FILE.value,
            **common,
            "uri": plan.uri,
            "filesystem_id": plan.filesystem_id,
            "discovery": {
                "mode": plan.discovery.mode.value,
                "recursive": plan.discovery.recursive,
                "extensions": list(plan.discovery.extensions),
            },
            "partitioning": {
                "kind": plan.partitioning.kind.value,
                "field_names": list(plan.partitioning.field_names),
            },
        }
    if isinstance(plan, SqlScan):
        target: dict[str, Any]
        if isinstance(plan.target, SqlTableRead):
            target = {
                "type": "table",
                "table": plan.target.table,
                "schema": plan.target.schema,
                "catalog": plan.target.catalog,
                "projection": list(plan.target.projection),
                "predicates": [
                    {
                        "column": predicate.column,
                        "operator": predicate.operator.value,
                        "values": list(predicate.values),
                    }
                    for predicate in plan.target.predicates
                ],
            }
        else:
            target = {
                "type": "parameterized_query",
                "query_digest": plan.target.query_digest,
            }
        return {
            "scan_kind": ScanKind.SQL.value,
            **common,
            "target": target,
            "sharding": {
                "mode": plan.sharding.mode.value,
                "columns": list(plan.sharding.columns),
                "target_partitions": plan.sharding.target_partitions,
            },
            "consistency": plan.consistency.value,
        }
    table: dict[str, Any]
    if isinstance(plan.table, CatalogTableRef):
        table = {
            "type": "catalog",
            "catalog_id": plan.table.catalog_id,
            "namespace": list(plan.table.namespace),
            "table": plan.table.table,
        }
    else:
        table = {"type": "uri", "uri": plan.table.uri}
    version_ref: dict[str, Any] | None = None
    if isinstance(plan.version_ref, SnapshotVersionRef):
        version_ref = {"type": "snapshot", "snapshot_id": plan.version_ref.snapshot_id}
    elif isinstance(plan.version_ref, NumericVersionRef):
        version_ref = {"type": "version", "version": plan.version_ref.version}
    elif isinstance(plan.version_ref, TagVersionRef):
        version_ref = {"type": "tag", "tag": plan.version_ref.tag}
    elif isinstance(plan.version_ref, AsOfVersionRef):
        version_ref = {
            "type": "as_of",
            "timestamp": plan.version_ref.timestamp.isoformat(),
        }
    encoded = {
        "scan_kind": ScanKind.TABLE.value,
        **common,
        "table": table,
        "version_ref": version_ref,
    }
    if plan.storage_format_id is not None:
        encoded["storage_format_id"] = plan.storage_format_id
    return encoded


@DeveloperAPI
def logical_scan_plan_from_dict(value: Mapping[str, Any]) -> LogicalScanPlan:
    """Deserialize a version-1 logical plan and reject unknown scan kinds."""
    data = dict(_require_mapping(value, "LogicalScanPlan"))
    kind = data.get("scan_kind")
    common_fields = frozenset(
        {
            "scan_kind",
            "version",
            "provider_id",
            "connector_id",
            "required_capabilities",
            "options",
            "input_schema_fingerprint",
        }
    )
    if "version" not in data:
        raise ValueError("LogicalScanPlan.version is required")
    raw_capabilities = data.get("required_capabilities", ())
    if not isinstance(raw_capabilities, (list, tuple, set, frozenset)):
        raise ValueError("required_capabilities must be a sequence")
    provider_id = cast(str, data.get("provider_id"))
    connector_id = cast(str, data.get("connector_id"))
    required_capabilities = frozenset(
        SourceCapability(item) for item in raw_capabilities
    )
    options = _require_mapping(data.get("options", {}), "LogicalScanPlan.options")
    input_schema_fingerprint = cast(str | None, data.get("input_schema_fingerprint"))
    version = cast(int, data.get("version"))
    if kind == ScanKind.FILE.value:
        _reject_unknown_fields(
            data,
            common_fields
            | frozenset({"uri", "filesystem_id", "discovery", "partitioning"}),
            "FileScan",
        )
        discovery = _require_mapping(data.get("discovery", {}), "FileScan.discovery")
        _reject_unknown_fields(
            discovery,
            frozenset({"mode", "recursive", "extensions"}),
            "FileScan.discovery",
        )
        partitioning = _require_mapping(
            data.get("partitioning", {}), "FileScan.partitioning"
        )
        _reject_unknown_fields(
            partitioning,
            frozenset({"kind", "field_names"}),
            "FileScan.partitioning",
        )
        return FileScan(
            provider_id=provider_id,
            connector_id=connector_id,
            required_capabilities=required_capabilities,
            options=options,
            input_schema_fingerprint=input_schema_fingerprint,
            version=version,
            uri=cast(str, data.get("uri")),
            filesystem_id=cast(str, data.get("filesystem_id")),
            discovery=FileDiscoveryStrategy(
                mode=FileDiscoveryMode(discovery.get("mode", "exact")),
                recursive=cast(bool, discovery.get("recursive", False)),
                extensions=_require_string_sequence(
                    discovery.get("extensions", ()), "FileScan.discovery.extensions"
                ),
            ),
            partitioning=PartitioningRule(
                kind=PartitioningKind(partitioning.get("kind", "none")),
                field_names=_require_string_sequence(
                    partitioning.get("field_names", ()),
                    "FileScan.partitioning.field_names",
                ),
            ),
        )
    if kind == ScanKind.SQL.value:
        _reject_unknown_fields(
            data,
            common_fields | frozenset({"target", "sharding", "consistency"}),
            "SqlScan",
        )
        target_data = _require_mapping(data.get("target", {}), "SqlScan.target")
        if target_data.get("type") == "table":
            _reject_unknown_fields(
                target_data,
                frozenset(
                    {
                        "type",
                        "table",
                        "schema",
                        "catalog",
                        "projection",
                        "predicates",
                    }
                ),
                "SqlScan.target",
            )
            predicate_values = target_data.get("predicates", ())
            if not isinstance(predicate_values, (list, tuple)):
                raise ValueError("SqlScan.target.predicates must be a sequence")
            predicates: list[SqlPredicate] = []
            for raw_predicate in predicate_values:
                predicate = _require_mapping(raw_predicate, "SqlScan.target.predicate")
                _reject_unknown_fields(
                    predicate,
                    frozenset({"column", "operator", "values"}),
                    "SqlScan.target.predicate",
                )
                raw_values = predicate.get("values", ())
                if not isinstance(raw_values, (list, tuple)):
                    raise ValueError("SqlScan predicate values must be a sequence")
                predicates.append(
                    SqlPredicate(
                        column=cast(str, predicate.get("column")),
                        operator=SqlPredicateOperator(predicate.get("operator")),
                        values=tuple(raw_values),
                    )
                )
            target: SqlReadTarget = SqlTableRead(
                table=cast(str, target_data.get("table")),
                schema=cast(str | None, target_data.get("schema")),
                catalog=cast(str | None, target_data.get("catalog")),
                projection=_require_string_sequence(
                    target_data.get("projection", ()), "SqlScan.target.projection"
                ),
                predicates=tuple(predicates),
            )
        elif target_data.get("type") == "parameterized_query":
            _reject_unknown_fields(
                target_data,
                frozenset({"type", "query_digest"}),
                "SqlScan.target",
            )
            target = ParameterizedQuery(cast(str, target_data.get("query_digest")))
        else:
            raise ValueError("Unknown SqlScan target type")
        sharding = _require_mapping(data.get("sharding", {}), "SqlScan.sharding")
        _reject_unknown_fields(
            sharding,
            frozenset({"mode", "columns", "target_partitions"}),
            "SqlScan.sharding",
        )
        return SqlScan(
            provider_id=provider_id,
            connector_id=connector_id,
            required_capabilities=required_capabilities,
            options=options,
            input_schema_fingerprint=input_schema_fingerprint,
            version=version,
            target=target,
            sharding=SqlShardRequirement(
                mode=SqlShardMode(sharding.get("mode", "single")),
                columns=_require_string_sequence(
                    sharding.get("columns", ()), "SqlScan.sharding.columns"
                ),
                target_partitions=cast(int | None, sharding.get("target_partitions")),
            ),
            consistency=ConsistencyRequirement(data.get("consistency", "best_effort")),
        )
    if kind == ScanKind.TABLE.value:
        _reject_unknown_fields(
            data,
            common_fields | frozenset({"table", "storage_format_id", "version_ref"}),
            "TableScan",
        )
        table_data = _require_mapping(data.get("table", {}), "TableScan.table")
        if table_data.get("type") == "catalog":
            _reject_unknown_fields(
                table_data,
                frozenset({"type", "catalog_id", "namespace", "table"}),
                "TableScan.table",
            )
            table: TableReference = CatalogTableRef(
                catalog_id=cast(str, table_data.get("catalog_id")),
                namespace=_require_string_sequence(
                    table_data.get("namespace", ()), "TableScan.table.namespace"
                ),
                table=cast(str, table_data.get("table")),
            )
        elif table_data.get("type") == "uri":
            _reject_unknown_fields(
                table_data, frozenset({"type", "uri"}), "TableScan.table"
            )
            table = UriTableRef(cast(str, table_data.get("uri")))
        else:
            raise ValueError("Unknown TableScan reference type")
        ref_data = data.get("version_ref")
        version_ref: VersionRef | None = None
        if ref_data is not None:
            ref_data = _require_mapping(ref_data, "TableScan.version_ref")
            ref_type = ref_data.get("type")
            if ref_type == "snapshot":
                _reject_unknown_fields(
                    ref_data,
                    frozenset({"type", "snapshot_id"}),
                    "TableScan.version_ref",
                )
                version_ref = SnapshotVersionRef(cast(int, ref_data.get("snapshot_id")))
            elif ref_type == "version":
                _reject_unknown_fields(
                    ref_data,
                    frozenset({"type", "version"}),
                    "TableScan.version_ref",
                )
                version_ref = NumericVersionRef(cast(int, ref_data.get("version")))
            elif ref_type == "tag":
                _reject_unknown_fields(
                    ref_data,
                    frozenset({"type", "tag"}),
                    "TableScan.version_ref",
                )
                version_ref = TagVersionRef(cast(str, ref_data.get("tag")))
            elif ref_type == "as_of":
                _reject_unknown_fields(
                    ref_data,
                    frozenset({"type", "timestamp"}),
                    "TableScan.version_ref",
                )
                try:
                    timestamp = datetime.fromisoformat(
                        cast(str, ref_data.get("timestamp"))
                    )
                except (TypeError, ValueError):
                    raise ValueError(
                        "TableScan.version_ref.timestamp must be ISO-8601"
                    ) from None
                version_ref = AsOfVersionRef(timestamp)
            else:
                raise ValueError("Unknown TableScan version reference type")
        return TableScan(
            provider_id=provider_id,
            connector_id=connector_id,
            required_capabilities=required_capabilities,
            options=options,
            input_schema_fingerprint=input_schema_fingerprint,
            version=version,
            table=table,
            storage_format_id=cast(str | None, data.get("storage_format_id")),
            version_ref=version_ref,
        )
    raise ValueError(f"Unknown logical scan kind {kind!r}")


__all__ = [
    "AsOfVersionRef",
    "CatalogTableRef",
    "ConsistencyRequirement",
    "FileDiscoveryMode",
    "FileDiscoveryStrategy",
    "FileScan",
    "NumericVersionRef",
    "ParameterizedQuery",
    "PartitioningKind",
    "PartitioningRule",
    "ScanKind",
    "SnapshotVersionRef",
    "SourceCapability",
    "SqlPredicate",
    "SqlPredicateOperator",
    "SqlScan",
    "SqlShardMode",
    "SqlShardRequirement",
    "SqlTableRead",
    "TableScan",
    "TagVersionRef",
    "UriTableRef",
    "logical_scan_plan_from_dict",
    "logical_scan_plan_to_dict",
]

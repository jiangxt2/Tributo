"""Versioned engine-neutral scan plan contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any, MutableMapping, cast

import pytest

from tributo.data.scan_plan import (
    AsOfVersionRef,
    CatalogTableRef,
    ConsistencyRequirement,
    FileDiscoveryMode,
    FileDiscoveryStrategy,
    FileScan,
    NumericVersionRef,
    ParameterizedQuery,
    PartitioningKind,
    PartitioningRule,
    ScanKind,
    SnapshotVersionRef,
    SourceCapability,
    SqlPredicate,
    SqlPredicateOperator,
    SqlScan,
    SqlShardMode,
    SqlShardRequirement,
    SqlTableRead,
    TableScan,
    TagVersionRef,
    UriTableRef,
    logical_scan_plan_from_dict,
    logical_scan_plan_to_dict,
)


def test_file_scan_is_deeply_immutable() -> None:
    plan = FileScan(
        provider_id="tributo.parquet",
        connector_id="parquet",
        uri="s3://bucket/input.parquet",
        filesystem_id="s3",
        options={"columns": ["id"], "partition": {"year": 2026}},
    )

    assert plan.scan_kind is ScanKind.FILE
    assert plan.options["columns"] == ("id",)
    nested = cast(MutableMapping[str, Any], plan.options["partition"])
    with pytest.raises(TypeError):
        nested["year"] = 2027
    with pytest.raises(FrozenInstanceError):
        plan.__setattr__("uri", "s3://other/input.parquet")


def test_file_scan_round_trip_preserves_discovery_partitioning_and_capabilities() -> (
    None
):
    plan = FileScan(
        provider_id="tributo.parquet",
        connector_id="parquet",
        uri="s3://bucket/year=*/input.parquet",
        filesystem_id="s3",
        discovery=FileDiscoveryStrategy(
            mode=FileDiscoveryMode.GLOB,
            recursive=True,
            extensions=(".parquet",),
        ),
        partitioning=PartitioningRule(
            kind=PartitioningKind.HIVE, field_names=("year",)
        ),
        options={"columns": ["id"]},
    )

    encoded = logical_scan_plan_to_dict(plan)
    restored = logical_scan_plan_from_dict(encoded)

    assert logical_scan_plan_to_dict(restored) == encoded
    assert restored.required_capabilities == frozenset(
        {SourceCapability.PROJECTION, SourceCapability.PARTITION_PRUNING}
    )


def test_structured_sql_scan_round_trip_and_capabilities() -> None:
    plan = SqlScan(
        provider_id="tributo.postgresql",
        connector_id="postgresql",
        target=SqlTableRead(
            catalog="warehouse",
            schema="public",
            table="events",
            projection=("id", "score"),
            predicates=(
                SqlPredicate("score", SqlPredicateOperator.GTE, (0.5,)),
                SqlPredicate("deleted_at", SqlPredicateOperator.IS_NULL),
            ),
        ),
        sharding=SqlShardRequirement(
            mode=SqlShardMode.PARALLEL,
            columns=("id",),
            target_partitions=8,
        ),
        consistency=ConsistencyRequirement.STATEMENT,
    )

    encoded = logical_scan_plan_to_dict(plan)
    restored = logical_scan_plan_from_dict(encoded)

    assert logical_scan_plan_to_dict(restored) == encoded
    assert restored.required_capabilities == frozenset(
        {SourceCapability.PROJECTION, SourceCapability.PREDICATE_PUSHDOWN}
    )


def test_automatic_sql_sharding_round_trip() -> None:
    plan = SqlScan(
        provider_id="tributo.clickhouse",
        connector_id="clickhouse",
        target=SqlTableRead(schema="analytics", table="events"),
        sharding=SqlShardRequirement(
            mode=SqlShardMode.AUTO,
            target_partitions=8,
        ),
    )

    encoded = logical_scan_plan_to_dict(plan)
    restored = logical_scan_plan_from_dict(encoded)

    assert logical_scan_plan_to_dict(restored) == encoded
    assert restored.sharding.mode is SqlShardMode.AUTO


@pytest.mark.parametrize(
    "version_ref",
    [
        SnapshotVersionRef(123),
        NumericVersionRef(7),
        TagVersionRef("production"),
        AsOfVersionRef(datetime(2026, 8, 5, tzinfo=UTC)),
    ],
)
def test_lance_compatible_table_scan_round_trip(version_ref: Any) -> None:
    plan = TableScan(
        provider_id="tributo.lance",
        connector_id="lance",
        table=UriTableRef("s3://bucket/features.lance"),
        version_ref=version_ref,
    )

    restored = logical_scan_plan_from_dict(logical_scan_plan_to_dict(plan))

    assert logical_scan_plan_to_dict(restored) == logical_scan_plan_to_dict(plan)
    assert restored.required_capabilities == frozenset({SourceCapability.SNAPSHOT})


def test_catalog_table_reference_is_not_reduced_to_a_file_path() -> None:
    plan = TableScan(
        provider_id="tributo.iceberg",
        connector_id="iceberg",
        table=CatalogTableRef("hive", ("analytics",), "events"),
    )

    encoded = logical_scan_plan_to_dict(plan)

    assert encoded["table"] == {
        "type": "catalog",
        "catalog_id": "hive",
        "namespace": ["analytics"],
        "table": "events",
    }


@pytest.mark.parametrize("storage_format_id", ["parquet", "orc", "iceberg"])
def test_hive_catalog_storage_format_hint_round_trip(
    storage_format_id: str,
) -> None:
    plan = TableScan(
        provider_id="example.hive",
        connector_id="hive",
        table=CatalogTableRef("hive", ("analytics",), "events"),
        storage_format_id=storage_format_id,
        required_capabilities=frozenset({SourceCapability.PARTITION_PRUNING}),
    )

    encoded = logical_scan_plan_to_dict(plan)
    restored = logical_scan_plan_from_dict(encoded)

    assert encoded["storage_format_id"] == storage_format_id
    assert logical_scan_plan_to_dict(restored) == encoded


def test_table_scan_omits_absent_storage_format_for_v1_compatibility() -> None:
    plan = TableScan(
        provider_id="tributo.iceberg",
        connector_id="iceberg",
        table=CatalogTableRef("hive", ("analytics",), "events"),
    )

    assert "storage_format_id" not in logical_scan_plan_to_dict(plan)


def test_parameterized_query_round_trip_uses_digest_only() -> None:
    plan = SqlScan(
        provider_id="tributo.doris",
        connector_id="doris",
        target=ParameterizedQuery("a" * 64),
    )

    encoded = logical_scan_plan_to_dict(plan)

    assert encoded["target"] == {
        "type": "parameterized_query",
        "query_digest": "a" * 64,
    }
    assert logical_scan_plan_to_dict(logical_scan_plan_from_dict(encoded)) == encoded


def test_unknown_file_scan_version_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported FileScan version"):
        FileScan(
            provider_id="tributo.parquet",
            connector_id="parquet",
            uri="/tmp/input.parquet",
            filesystem_id="local",
            version=2,
        )


def test_unknown_scan_kind_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown logical scan kind"):
        logical_scan_plan_from_dict({"scan_kind": "custom", "version": 1})


def test_scan_plan_deserialization_rejects_missing_version_and_unknown_fields() -> None:
    encoded = logical_scan_plan_to_dict(
        FileScan(
            provider_id="tributo.parquet",
            connector_id="parquet",
            uri="/tmp/input.parquet",
            filesystem_id="local",
        )
    )
    missing_version = dict(encoded)
    del missing_version["version"]
    with pytest.raises(ValueError, match="version is required"):
        logical_scan_plan_from_dict(missing_version)

    encoded["future_field"] = True
    with pytest.raises(ValueError, match="unknown field"):
        logical_scan_plan_from_dict(encoded)


def test_scan_plan_nested_contracts_are_strict() -> None:
    encoded = logical_scan_plan_to_dict(
        FileScan(
            provider_id="tributo.parquet",
            connector_id="parquet",
            uri="/tmp/input.parquet",
            filesystem_id="local",
        )
    )
    encoded["discovery"]["future_field"] = True
    with pytest.raises(ValueError, match="unknown field"):
        logical_scan_plan_from_dict(encoded)

    encoded = logical_scan_plan_to_dict(
        FileScan(
            provider_id="tributo.parquet",
            connector_id="parquet",
            uri="/tmp/input.parquet",
            filesystem_id="local",
        )
    )
    encoded["discovery"]["recursive"] = "false"
    with pytest.raises(ValueError, match="recursive must be a bool"):
        logical_scan_plan_from_dict(encoded)


def test_scan_plan_rejects_credentials_and_engine_objects() -> None:
    with pytest.raises(ValueError, match="credential field"):
        FileScan(
            provider_id="tributo.parquet",
            connector_id="parquet",
            uri="s3://bucket/input.parquet",
            filesystem_id="s3",
            options={"secret_access_key": "hidden"},
        )
    with pytest.raises(ValueError, match="engine-neutral"):
        FileScan(
            provider_id="tributo.parquet",
            connector_id="parquet",
            uri="/tmp/input.parquet",
            filesystem_id="local",
            options={"filesystem": object()},
        )


def test_sql_predicate_rejects_non_finite_float() -> None:
    with pytest.raises(ValueError, match="finite"):
        SqlPredicate("score", SqlPredicateOperator.EQ, (float("nan"),))


def test_sql_contract_rejects_runtime_type_coercion() -> None:
    with pytest.raises(ValueError, match="scalar"):
        SqlPredicate("score", SqlPredicateOperator.EQ, ([1],))
    with pytest.raises(ValueError, match="projection"):
        SqlTableRead(table="events", projection=(1,))
    with pytest.raises(ValueError, match="positive integer"):
        SqlShardRequirement(
            mode=SqlShardMode.PARALLEL,
            columns=("id",),
            target_partitions=True,
        )
    with pytest.raises(ValueError, match="shard columns.*duplicates"):
        SqlShardRequirement(
            mode=SqlShardMode.PARALLEL,
            columns=("id", "id"),
            target_partitions=2,
        )
    with pytest.raises(ValueError, match="at least one shard column"):
        SqlShardRequirement(
            mode=SqlShardMode.PARALLEL,
            target_partitions=2,
        )
    with pytest.raises(ValueError, match="cannot declare shard columns"):
        SqlShardRequirement(
            mode=SqlShardMode.AUTO,
            columns=("id",),
        )


def test_catalog_namespace_may_repeat_segments() -> None:
    reference = CatalogTableRef("hive", ("analytics", "analytics"), "events")

    assert reference.namespace == ("analytics", "analytics")


def test_as_of_timestamp_deserialization_has_field_context() -> None:
    encoded = logical_scan_plan_to_dict(
        TableScan(
            provider_id="tributo.iceberg",
            connector_id="iceberg",
            table=CatalogTableRef("hive", ("analytics",), "events"),
        )
    )
    encoded["version_ref"] = {"type": "as_of", "timestamp": "not-a-timestamp"}

    with pytest.raises(
        ValueError, match="TableScan.version_ref.timestamp must be ISO-8601"
    ):
        logical_scan_plan_from_dict(encoded)


def test_sequence_contracts_do_not_split_strings_into_characters() -> None:
    with pytest.raises(ValueError, match="sequence"):
        SqlPredicate("kind", SqlPredicateOperator.EQ, "x")
    with pytest.raises(ValueError, match="sequence"):
        SqlTableRead(table="events", projection="id")
    with pytest.raises(ValueError, match="sequence"):
        CatalogTableRef("hive", "analytics", "events")

    encoded = logical_scan_plan_to_dict(
        SqlScan(
            provider_id="tributo.postgresql",
            connector_id="postgresql",
            target=SqlTableRead(table="events", projection=("id",)),
        )
    )
    encoded["target"]["projection"] = "id"
    with pytest.raises(ValueError, match="sequence"):
        logical_scan_plan_from_dict(encoded)


@pytest.mark.parametrize("value", ["A" * 64, "0" * 63, "z" * 64])
def test_parameterized_query_requires_canonical_digest(value: str) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ParameterizedQuery(value)

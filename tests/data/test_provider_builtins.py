"""Built-in provider contract tests: normalize semantics + real local reads."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from tributo.data.provider import DataSourceProvider
from tributo.data.provider_builtins import (
    ClickHouseProvider,
    CsvProvider,
    DorisProvider,
    IcebergProvider,
    ParquetProvider,
    PostgreSqlProvider,
)
from tributo.data.refs import digest
from tributo.data.source_config import (
    CanonicalSourceInput,
    ProviderSourceConfig,
    RayReadTaskOptions,
    SqlSourceConfig,
)
from tributo.exceptions import JobConfigurationError


def cfg(source: dict) -> CanonicalSourceInput:
    return TypeAdapter(CanonicalSourceInput).validate_python(source)


class TestFileProviderNormalize:
    """Parquet/CSV: both shapes normalize to credential-safe ResolvedSource."""

    @pytest.mark.parametrize(
        ("provider", "source", "uri", "columns"),
        [
            (
                ParquetProvider(),
                {"type": "parquet", "path": "s3://bkt/a.parquet", "columns": ["x"]},
                "s3://bkt/a.parquet",
                ("x",),
            ),
            (
                CsvProvider(),
                {"type": "csv", "path": "local/a.csv"},
                "local/a.csv",
                None,
            ),
        ],
    )
    def test_builtin_shape(
        self, provider: DataSourceProvider, source: dict, uri: str, columns: object
    ) -> None:
        resolved = provider.normalize(cfg(source))
        assert resolved.provider_id == provider.provider_id
        assert resolved.canonical_uri == uri
        assert resolved.identity_options.get("columns") == columns
        assert "s3" in resolved.runtime_options

    def test_provider_shape_splits_identity_and_runtime(self) -> None:
        provider = ParquetProvider()
        resolved = provider.normalize(
            ProviderSourceConfig(
                provider="tributo.parquet",
                uri="s3://bkt/a.parquet",
                options={
                    "columns": ["x"],
                    "s3": {"access_key_id": "s3cr3t"},
                },
            )
        )
        assert resolved.identity_options == {"columns": ("x",)}
        assert resolved.runtime_options["s3"] == {"access_key_id": "s3cr3t"}
        # Credential value must not leak through repr.
        assert "s3cr3t" not in repr(resolved)

    @pytest.mark.parametrize(
        "source",
        [
            {"type": "parquet", "path": "S3://bkt/a.parquet"},
            {
                "provider": "tributo.parquet",
                "uri": "S3://bkt/a.parquet",
            },
        ],
    )
    def test_s3_scheme_is_canonicalized_for_identity_and_execution(
        self, source: dict
    ) -> None:
        resolved = ParquetProvider().normalize(cfg(source))

        assert resolved.canonical_uri == "s3://bkt/a.parquet"
        assert resolved.runtime_options["uri"] == "s3://bkt/a.parquet"
        assert ParquetProvider().plan(resolved).filesystem_id == "s3"

    def test_provider_shape_uri_userinfo_is_rejected(self) -> None:
        with pytest.raises(JobConfigurationError, match="userinfo"):
            ParquetProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.parquet",
                    uri="s3://user:secret@host/bucket/a.parquet",
                )
            )

    def test_provider_shape_unknown_option_rejected(self) -> None:
        with pytest.raises(JobConfigurationError, match="unknown option"):
            ParquetProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.parquet",
                    uri="s3://bkt/a.parquet",
                    options={"delimiter": ";"},
                )
            )

    def test_provider_shape_unknown_sql_option_rejected(self) -> None:
        with pytest.raises(JobConfigurationError, match="unknown option"):
            ClickHouseProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.clickhouse",
                    uri="clickhouse://x",
                    options={"sql": "SELECT 1", "timeout": 30},
                )
            )

    def test_provider_shape_accepts_sql_projection(self) -> None:
        resolved = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://x/db",
                options={"sql": "SELECT * FROM events", "columns": ["id", "user-name"]},
            )
        )
        assert resolved.identity_options["columns"] == ("id", "user-name")

    def test_provider_mismatch_rejected(self) -> None:
        with pytest.raises(JobConfigurationError, match="cannot be normalized"):
            ParquetProvider().normalize(
                ProviderSourceConfig(provider="tributo.csv", uri="x")
            )

    def test_wrong_builtin_type_rejected(self) -> None:
        with pytest.raises(JobConfigurationError, match="unsupported source"):
            ParquetProvider().normalize(cfg({"type": "csv", "path": "a.csv"}))

    def test_provider_shape_s3_endpoint_enters_ref_id(self) -> None:
        # endpoint selects *which* storage endpoint the data comes from —
        # different endpoints, different identity.
        a = ParquetProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.parquet",
                uri="s3://bkt/a.parquet",
                options={"s3": {"endpoint": "http://endpoint-a"}},
            )
        )
        b = ParquetProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.parquet",
                uri="s3://bkt/a.parquet",
                options={"s3": {"endpoint": "http://endpoint-b"}},
            )
        )
        assert a.ref_id() != b.ref_id()
        assert a.identity_options["s3"] == {"endpoint": "http://endpoint-a"}
        # Credentials stay runtime only — never in the identity.
        c = ParquetProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.parquet",
                uri="s3://bkt/a.parquet",
                options={"s3": {"endpoint": "http://x", "secret_access_key": "s3cr3t"}},
            )
        )
        assert c.identity_options["s3"] == {"endpoint": "http://x"}
        assert c.runtime_options["s3"] == {
            "endpoint": "http://x",
            "secret_access_key": "s3cr3t",
        }
        assert "s3cr3t" not in repr(c)

    def test_builtin_shape_s3_endpoint_enters_ref_id(self) -> None:
        a = ParquetProvider().normalize(
            cfg(
                {
                    "type": "parquet",
                    "path": "s3://bkt/a.parquet",
                    "s3": {"endpoint": "http://endpoint-a"},
                }
            )
        )
        b = ParquetProvider().normalize(
            cfg(
                {
                    "type": "parquet",
                    "path": "s3://bkt/a.parquet",
                    "s3": {"endpoint": "http://endpoint-b"},
                }
            )
        )
        assert a.ref_id() != b.ref_id()
        assert a.identity_options["s3"] == {"endpoint": "http://endpoint-a"}
        # Credentials never enter the identity from the S3Config either.
        c = ParquetProvider().normalize(
            cfg(
                {
                    "type": "parquet",
                    "path": "s3://bkt/a.parquet",
                    "s3": {"endpoint": "http://x", "secret_access_key": "s3cr3t"},
                }
            )
        )
        assert c.identity_options["s3"] == {"endpoint": "http://x"}
        assert "s3cr3t" not in repr(c)

    def test_builtin_shape_signed_url_is_rejected_during_normalize(self) -> None:
        with pytest.raises(JobConfigurationError, match="query parameters"):
            ParquetProvider().normalize(
                cfg(
                    {
                        "type": "parquet",
                        "path": "s3://bkt/data.parquet?X-Amz-Signature=abc",
                    }
                )
            )

    def test_s3_none_does_not_import_environment_into_identity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("S3_ENDPOINT", "http://unrelated.example")
        monkeypatch.setenv("AWS_REGION", "unrelated-region")
        resolved = ParquetProvider().normalize(
            cfg({"type": "parquet", "path": str(tmp_path / "data.parquet")})
        )
        assert "s3" not in resolved.identity_options

    def test_s3_uri_without_config_uses_environment_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("S3_ENDPOINT", "http://implicit-s3.example")
        resolved = ParquetProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.parquet", uri="s3://bucket/data.parquet"
            )
        )
        assert resolved.identity_options["s3"] == {
            "endpoint": "http://implicit-s3.example"
        }

    def test_s3_uri_query_is_rejected_during_normalize(self) -> None:
        with pytest.raises(JobConfigurationError, match="query parameters"):
            ParquetProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.parquet",
                    uri="s3://bucket/data.parquet?versionId=42",
                )
            )

    def test_columns_none_equals_omitted(self) -> None:
        # An explicit None reads all columns, same as omitting it — the
        # ref_id must not distinguish the two spellings.
        with_none = ParquetProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.parquet",
                uri="s3://bkt/a.parquet",
                options={"columns": None},
            )
        )
        omitted = ParquetProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.parquet",
                uri="s3://bkt/a.parquet",
                options={},
            )
        )
        assert with_none.identity_options == omitted.identity_options
        assert with_none.ref_id() == omitted.ref_id()

    def test_columns_entries_must_be_str(self) -> None:
        # A numeric entry would either read columns named "1" or crash in
        # the connector — fail here, not at read time.
        with pytest.raises(JobConfigurationError, match="entries must be str"):
            ParquetProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.parquet",
                    uri="s3://b/x",
                    options={"columns": [1]},
                )
            )
        with pytest.raises(JobConfigurationError, match="entries must be str"):
            IcebergProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.iceberg",
                    uri="warehouse://prod/db.t",
                    options={"selected_fields": [1]},
                )
            )

    def test_s3_unknown_key_rejected(self) -> None:
        # A typo'd s3 key (e.g. "bucket") must fail at normalize time
        # instead of silently reading the wrong endpoint at the connector.
        with pytest.raises(JobConfigurationError, match="unknown s3 option"):
            ParquetProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.parquet",
                    uri="s3://b/x",
                    options={"s3": {"region": "us-east-1", "bucket": "x"}},
                )
            )
        with pytest.raises(JobConfigurationError, match="unknown s3 option"):
            IcebergProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.iceberg",
                    uri="warehouse://prod/db.t",
                    options={"s3": {"foo": "bar"}},
                )
            )

    def test_endpoint_userinfo_stripped_from_identity(self) -> None:
        # An endpoint can carry userinfo itself ("https://user:pass@host") —
        # it must not leak into the identity; runtime keeps the full value
        # for the actual read.
        resolved = ParquetProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.parquet",
                uri="s3://bkt/a.parquet",
                options={"s3": {"endpoint": "https://user:password@host"}},
            )
        )
        assert resolved.identity_options["s3"] == {"endpoint": "https://host"}
        assert resolved.runtime_options["s3"]["endpoint"] == (
            "https://user:password@host"
        )
        assert "password" not in repr(resolved)

    def test_builtin_shape_endpoint_userinfo_stripped(self) -> None:
        resolved = ParquetProvider().normalize(
            cfg(
                {
                    "type": "parquet",
                    "path": "s3://bkt/a.parquet",
                    "s3": {"endpoint": "https://user:password@host"},
                }
            )
        )
        assert resolved.identity_options["s3"] == {"endpoint": "https://host"}
        assert "password" not in repr(resolved)


class TestSqlProviderNormalize:
    """ClickHouse/Doris: digests in identity, credentials in runtime only."""

    def test_invalid_params_do_not_survive_in_public_error_context(self) -> None:
        class SecretValue:
            def __repr__(self) -> str:
                return "password=top-secret"

        with pytest.raises(JobConfigurationError) as exc_info:
            ClickHouseProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.clickhouse",
                    uri="clickhouse://host/analytics",
                    options={"table": "events", "params": {"value": SecretValue()}},
                )
            )

        error = exc_info.value
        assert "top-secret" not in str(error)
        assert error.__cause__ is None
        assert error.__context__ is None

    def test_provider_shape_uri_drives_connection(self) -> None:
        # The uri is the connection address: host/port/database come from it,
        # options may override.
        resolved = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://ch.example:9443/analytics",
                options={
                    "sql": "SELECT 1",
                    "user": "reader",
                    "password": "p",
                    "host": "override.example",
                },
            )
        )
        assert resolved.runtime_options["host"] == "override.example"
        assert resolved.runtime_options["port"] == 9443
        assert resolved.runtime_options["database"] == "analytics"
        assert resolved.runtime_options["user"] == "reader"
        # Canonical uri is rebuilt from the *effective* connection values.
        assert resolved.canonical_uri == "clickhouse://override.example:9443/analytics"

    def test_provider_shape_uri_scheme_mismatch_rejected(self) -> None:
        with pytest.raises(JobConfigurationError, match="scheme"):
            ClickHouseProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.clickhouse",
                    uri="doris://other.example/db",
                    options={"sql": "SELECT 1"},
                )
            )

    def test_provider_shape_options_override_enters_ref_id(self) -> None:
        # Explicit options override the uri; the effective endpoint must be
        # part of the identity.
        opts = {"sql": "SELECT 1"}
        a = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://A/db",
                options={**opts, "host": "A"},
            )
        )
        b = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://A/db",
                options={**opts, "host": "B"},
            )
        )
        assert a.runtime_options["host"] == "A"
        assert b.runtime_options["host"] == "B"
        assert a.ref_id() != b.ref_id()

    def test_provider_shape_iceberg_catalog_override_enters_ref_id(self) -> None:
        a = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.t",
                options={"catalog_name": "A"},
            )
        )
        b = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.t",
                options={"catalog_name": "B"},
            )
        )
        assert a.runtime_options["catalog_name"] == "A"
        assert a.ref_id() != b.ref_id()

    def test_builtin_shape_uri_userinfo_is_rejected(self) -> None:
        with pytest.raises(JobConfigurationError, match="userinfo"):
            ParquetProvider().normalize(
                cfg({"type": "parquet", "path": "s3://user:secret@host/b.parquet"})
            )

    def test_provider_shape_signed_url_is_rejected_during_normalize(self) -> None:
        with pytest.raises(JobConfigurationError, match="query parameters"):
            ParquetProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.parquet",
                    uri="s3://bucket/data.parquet?X-Amz-Signature=abc",
                )
            )

    def test_provider_shape_option_type_validation(self) -> None:
        with pytest.raises(JobConfigurationError, match="must be list"):
            ParquetProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.parquet",
                    uri="s3://b/x",
                    options={"columns": "id"},
                )
            )
        with pytest.raises(JobConfigurationError, match="must be dict"):
            ParquetProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.parquet",
                    uri="s3://b/x",
                    options={"s3": "bad"},
                )
            )
        with pytest.raises(JobConfigurationError, match="must be int"):
            ClickHouseProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.clickhouse",
                    uri="clickhouse://h/db",
                    options={"sql": "SELECT 1", "port": "9443"},
                )
            )

    def test_provider_shape_iceberg_columns_rejected(self) -> None:
        # columns is not an Iceberg option — use selected_fields.
        with pytest.raises(JobConfigurationError, match="unknown option"):
            IcebergProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.iceberg",
                    uri="warehouse://prod/db.t",
                    options={"columns": ["a"]},
                )
            )

    def test_provider_shape_uri_credentials_stripped(self) -> None:
        resolved = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://user:secret@ch.example:9443/analytics",
                options={"sql": "SELECT 1"},
            )
        )
        assert resolved.canonical_uri == "clickhouse://ch.example:9443/analytics"
        assert "secret" not in repr(resolved)

    def test_provider_shape_uri_userinfo_kept_in_runtime(self) -> None:
        # uri userinfo carries the connection credentials — stripping them
        # from canonical_uri must not silently drop them: runtime keeps
        # user/password for the actual read.
        resolved = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://reader:s3cr3t@ch.example/analytics",
                options={"sql": "SELECT 1"},
            )
        )
        assert resolved.runtime_options["user"] == "reader"
        assert resolved.runtime_options["password"] == "s3cr3t"
        assert resolved.canonical_uri == "clickhouse://ch.example/analytics"
        assert "s3cr3t" not in repr(resolved)

    def test_provider_shape_percent_encoded_uri_userinfo_is_decoded(self) -> None:
        resolved = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://u%40ser:p%40ss@ch.example/analytics",
                options={"sql": "SELECT 1"},
            )
        )
        assert resolved.runtime_options["user"] == "u@ser"
        assert resolved.runtime_options["password"] == "p@ss"

    def test_provider_shape_options_override_uri_userinfo(self) -> None:
        resolved = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://uri-user:uri-pass@ch.example/analytics",
                options={
                    "sql": "SELECT 1",
                    "user": "opt-user",
                    "password": "opt-pass",
                },
            )
        )
        assert resolved.runtime_options["user"] == "opt-user"
        assert resolved.runtime_options["password"] == "opt-pass"
        assert resolved.canonical_uri == "clickhouse://ch.example/analytics"

    def test_provider_shape_port_bool_rejected(self) -> None:
        # bool is an int subclass — a bare isinstance check would accept
        # port=True and connect to port 1.  Fail here instead.
        with pytest.raises(JobConfigurationError, match="must be int"):
            ClickHouseProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.clickhouse",
                    uri="clickhouse://h/db",
                    options={"sql": "SELECT 1", "port": True},
                )
            )

    def test_provider_shape_host_none_falls_back_to_uri(self) -> None:
        # None means "not provided" — fall back to the uri host, never to an
        # env-var fallback that disagrees with the canonical_uri.
        resolved = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://uri-host.example/analytics",
                options={"sql": "SELECT 1", "host": None},
            )
        )
        assert resolved.runtime_options["host"] == "uri-host.example"
        assert resolved.canonical_uri == "clickhouse://uri-host.example/analytics"

    def test_provider_shape_params_datetime_supported(self) -> None:
        from datetime import datetime

        ts = datetime(2026, 1, 1, 8, 30)
        resolved = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://ch.example/db",
                options={"sql": "SELECT 1", "params": {"ts": ts}},
            )
        )
        assert resolved.identity_options["params_digest"] == digest({"ts": ts})
        assert len(resolved.ref_id()) == 64

    def test_provider_shape_params_unserializable_rejected(self) -> None:
        # A non-JSON-serializable bound parameter is a configuration error —
        # surfaced with a clear message, not a raw TypeError at ref_id time.
        with pytest.raises(JobConfigurationError, match="JSON-serializable"):
            ClickHouseProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.clickhouse",
                    uri="clickhouse://ch.example/db",
                    options={"sql": "SELECT 1", "params": {"p": object()}},
                )
            )

    def test_provider_shape_s3_query_is_rejected_before_identity(self) -> None:
        with pytest.raises(JobConfigurationError, match="query parameters"):
            ParquetProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.parquet",
                    uri="s3://bucket/data.parquet?token=secret&versionId=42",
                )
            )

    def test_provider_shape_iceberg_uri_drives_table(self) -> None:
        resolved = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.events",
                options={"snapshot_id": 7},
            )
        )
        assert resolved.runtime_options["catalog_name"] == "prod"
        assert resolved.runtime_options["table_identifier"] == "db.events"
        assert resolved.identity_options["snapshot_id"] == 7

    def test_provider_shape_iceberg_uri_without_table_rejected(self) -> None:
        with pytest.raises(JobConfigurationError, match="table"):
            IcebergProvider().normalize(
                ProviderSourceConfig(provider="tributo.iceberg", uri="warehouse://prod")
            )

    @pytest.mark.parametrize("uri", ["prod/db.events", "warehouse:///db.events"])
    def test_provider_shape_iceberg_uri_requires_scheme_and_catalog(
        self, uri: str
    ) -> None:
        with pytest.raises(JobConfigurationError, match="<scheme>://<catalog>"):
            IcebergProvider().normalize(
                ProviderSourceConfig(provider="tributo.iceberg", uri=uri)
            )

    def test_iceberg_provider_shape_builds_catalog_plan(self) -> None:
        from tributo.data.scan_plan import CatalogTableRef, SourceCapability, TableScan

        resolved = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.events",
                options={"row_filter": "event_id > 1"},
            )
        )
        plan = IcebergProvider().plan(resolved)

        assert isinstance(plan, TableScan)
        assert plan.table == CatalogTableRef(
            catalog_id="prod", namespace=("db",), table="events"
        )
        assert plan.options["row_filter"] == "event_id > 1"
        assert SourceCapability.PREDICATE_PUSHDOWN not in plan.required_capabilities
        with pytest.raises(JobConfigurationError, match="no legacy Ray-only reader"):
            IcebergProvider().open(resolved)

    def test_sql_canonical_uri_includes_port(self) -> None:
        resolved = ClickHouseProvider().normalize(
            cfg(
                {
                    "type": "sql",
                    "dialect": "clickhouse",
                    "host": "ch.example",
                    "port": 8443,
                    "sql": "SELECT 1",
                }
            )
        )
        assert resolved.canonical_uri == "clickhouse://ch.example:8443/"

    def test_clickhouse_builtin_shape(self) -> None:
        resolved = ClickHouseProvider().normalize(
            cfg(
                {
                    "type": "sql",
                    "dialect": "clickhouse",
                    "host": "ch.example",
                    "user": "reader",
                    "password": "p@ss",
                    "sql": "SELECT * FROM t",
                    "params": {"x": 1},
                    "columns": ["id", "score"],
                }
            )
        )
        assert resolved.provider_id == "tributo.clickhouse"
        assert resolved.identity_options["sql_digest"] == digest("SELECT * FROM t")
        assert resolved.identity_options["params_digest"] == digest({"x": 1})
        assert resolved.identity_options["columns"] == ("id", "score")
        assert "password" not in resolved.identity_options
        assert resolved.runtime_options["password"] == "p@ss"
        assert resolved.runtime_options["sql"] == "SELECT * FROM t"
        # Credential-free URI: host/database only.
        assert "p@ss" not in resolved.canonical_uri

    @pytest.mark.parametrize(
        ("provider", "dialect"),
        [
            (ClickHouseProvider(), "clickhouse"),
            (DorisProvider(), "doris"),
            (PostgreSqlProvider(), "postgresql"),
        ],
    )
    def test_structured_table_builds_sql_plan(
        self, provider: DataSourceProvider, dialect: str
    ) -> None:
        from tributo.data.scan_plan import SqlScan, SqlShardMode, SqlTableRead

        resolved = provider.normalize(
            cfg(
                {
                    "type": "sql",
                    "dialect": dialect,
                    "host": "db.example",
                    "database": "analytics",
                    "table": "events",
                    "columns": ["id"],
                }
            )
        )
        plan = provider.plan(resolved)

        assert isinstance(plan, SqlScan)
        assert plan.connector_id == dialect
        expected_schema = "public" if dialect == "postgresql" else "analytics"
        assert plan.target == SqlTableRead(
            schema=expected_schema, table="events", projection=("id",)
        )
        assert plan.sharding.mode is (
            SqlShardMode.SINGLE if dialect == "postgresql" else SqlShardMode.AUTO
        )

    def test_sql_partitioning_modes_are_preserved_in_plan(self) -> None:
        from tributo.data.scan_plan import SqlShardMode

        auto_source = cfg(
            {
                "type": "sql",
                "dialect": "clickhouse",
                "database": "analytics",
                "table": "events",
                "partitioning": {"mode": "auto", "num_partitions": 6},
            }
        )
        single_source = cfg(
            {
                "type": "sql",
                "dialect": "clickhouse",
                "database": "analytics",
                "table": "events",
                "partitioning": {"mode": "single"},
            }
        )

        auto_plan = ClickHouseProvider().plan(
            ClickHouseProvider().normalize(auto_source)
        )
        single_plan = ClickHouseProvider().plan(
            ClickHouseProvider().normalize(single_source)
        )

        assert auto_plan.sharding.mode is SqlShardMode.AUTO
        assert auto_plan.sharding.target_partitions == 6
        assert single_plan.sharding.mode is SqlShardMode.SINGLE

    def test_postgresql_database_schema_is_part_of_structured_target(self) -> None:
        from tributo.data.scan_plan import SqlScan, SqlTableRead

        resolved = PostgreSqlProvider().normalize(
            cfg(
                {
                    "type": "sql",
                    "dialect": "postgresql",
                    "database": "analytics",
                    "database_schema": "feature_store",
                    "table": "events",
                }
            )
        )
        plan = PostgreSqlProvider().plan(resolved)

        assert isinstance(plan, SqlScan)
        assert plan.target == SqlTableRead(schema="feature_store", table="events")

    @pytest.mark.parametrize("provider", [ClickHouseProvider(), DorisProvider()])
    def test_raw_sql_has_actionable_migration_error(
        self, provider: DataSourceProvider
    ) -> None:
        raw_sql = "SELECT * FROM events WHERE password = 'top-secret'"
        resolved = provider.normalize(
            cfg(
                {
                    "type": "sql",
                    "dialect": provider.provider_id.removeprefix("tributo."),
                    "sql": raw_sql,
                    "params": {"id": 7},
                }
            )
        )

        with pytest.raises(
            JobConfigurationError, match="structured 'table' source"
        ) as exc_info:
            provider.plan(resolved)

        assert raw_sql not in str(exc_info.value)
        assert "top-secret" not in str(exc_info.value)

    def test_doris_builtin_shape(self) -> None:
        resolved = DorisProvider().normalize(
            cfg(
                {
                    "type": "sql",
                    "dialect": "doris",
                    "host": "fe.example",
                    "sql": "SELECT 1",
                }
            )
        )
        assert resolved.provider_id == "tributo.doris"
        assert resolved.runtime_options["host"] == "fe.example"

    def test_doris_provider_shape_preserves_validated_read_options_in_runtime(
        self,
    ) -> None:
        resolved = DorisProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.doris",
                uri="doris://fe.example/analytics",
                options={
                    "table": "events",
                    "tablet_size": 128,
                    "on_query_plan_error": "error",
                    "ray_remote_args": {
                        "num_cpus": 1.5,
                        "scheduling_strategy": "SPREAD",
                        "max_retries": 3,
                    },
                },
            )
        )

        assert resolved.runtime_options["tablet_size"] == 128
        assert resolved.runtime_options["on_query_plan_error"] == "error"
        assert resolved.runtime_options["ray_remote_args"] == {
            "num_cpus": 1.5,
            "scheduling_strategy": "SPREAD",
            "max_retries": 3,
        }

    def test_doris_builtin_shape_preserves_validated_read_options_in_runtime(
        self,
    ) -> None:
        resolved = DorisProvider().normalize(
            SqlSourceConfig(
                dialect="doris",
                host="fe.example",
                database="analytics",
                table="events",
                tablet_size=128,
                on_query_plan_error="single_task",
                ray_remote_args=RayReadTaskOptions(
                    num_cpus=1,
                    scheduling_strategy="SPREAD",
                    max_retries=0,
                ),
            )
        )

        assert resolved.runtime_options["tablet_size"] == 128
        assert resolved.runtime_options["on_query_plan_error"] == "single_task"
        assert resolved.runtime_options["ray_remote_args"] == {
            "num_cpus": 1,
            "scheduling_strategy": "SPREAD",
            "max_retries": 0,
        }

    @pytest.mark.parametrize(
        "option",
        [
            {"tablet_size": 0},
            {"tablet_size": True},
            {"tablet_size": 1.5},
            {"on_query_plan_error": "fallback"},
            {"ray_remote_args": {"max_retries": -1}},
            {"ray_remote_args": {"scheduling_strategy": "DEFAULT"}},
            {"ray_remote_args": {"resources": {"olap_worker": 1}}},
        ],
    )
    def test_doris_provider_shape_rejects_invalid_read_options(
        self, option: dict[str, object]
    ) -> None:
        with pytest.raises(JobConfigurationError):
            DorisProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.doris",
                    uri="doris://fe.example/analytics",
                    options={"table": "events", **option},
                )
            )

    def test_doris_read_options_are_rejected_for_clickhouse_provider(self) -> None:
        with pytest.raises(JobConfigurationError, match="Doris-only"):
            ClickHouseProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.clickhouse",
                    uri="clickhouse://ch.example/analytics",
                    options={"table": "events", "tablet_size": 128},
                )
            )

    def test_provider_shape_requires_sql(self) -> None:
        with pytest.raises(JobConfigurationError, match="required"):
            ClickHouseProvider().normalize(
                ProviderSourceConfig(
                    provider="tributo.clickhouse", uri="clickhouse://x"
                )
            )

    def test_ref_id_uses_identity_only(self) -> None:
        resolved = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://x/analytics",
                options={
                    "sql": "SELECT * FROM events WHERE p = %(p)s",
                    "params": {"p": 7},
                    "password": "s3cr3t",
                },
            )
        )
        ref_id = resolved.ref_id()
        assert len(ref_id) == 64
        # Stable and credential-free.
        assert resolved.ref_id() == ref_id
        assert "s3cr3t" not in ref_id

    def test_provider_shape_digests(self) -> None:
        resolved = ClickHouseProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.clickhouse",
                uri="clickhouse://x/analytics",
                options={
                    "sql": "SELECT * FROM events WHERE p = %(p)s",
                    "params": {"p": 7},
                    "password": "s3cr3t",
                },
            )
        )
        assert resolved.identity_options["sql_digest"] == digest(
            "SELECT * FROM events WHERE p = %(p)s"
        )
        assert resolved.identity_options["params_digest"] == digest({"p": 7})
        assert "s3cr3t" not in repr(resolved)
        assert resolved.runtime_options["password"] == "s3cr3t"


class TestIcebergNormalize:
    def test_builtin_shape(self) -> None:
        resolved = IcebergProvider().normalize(
            cfg(
                {
                    "type": "iceberg",
                    "catalog": "prod",
                    "table": "db.events",
                    "snapshot_id": 42,
                    "row_filter": "id > 1",
                    "selected_fields": ["id", "name"],
                    "s3": {"endpoint": "http://127.0.0.1:9000"},
                }
            )
        )
        assert resolved.provider_id == "tributo.iceberg"
        assert resolved.canonical_uri == "prod/db.events"
        assert resolved.identity_options["snapshot_id"] == 42
        assert resolved.identity_options["row_filter"] == "id > 1"
        assert resolved.identity_options["selected_fields"] == ("id", "name")
        assert resolved.runtime_options["table_identifier"] == "db.events"
        assert resolved.runtime_options["catalog_name"] == "prod"
        assert resolved.runtime_options["s3"]["endpoint"] == "http://127.0.0.1:9000"

    def test_provider_shape(self) -> None:
        resolved = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.events",
                options={"snapshot_id": 7, "catalog_properties": {"type": "rest"}},
            )
        )
        assert resolved.identity_options["snapshot_id"] == 7
        assert resolved.identity_options["catalog_scheme"] == "warehouse"
        assert resolved.identity_options["catalog_properties"] == {"type": "rest"}
        assert resolved.runtime_options["catalog_properties"] == {"type": "rest"}

    def test_catalog_properties_classified(self) -> None:
        # Non-credential properties select the catalog/table — they belong
        # in the identity (userinfo stripped); credential-keyed properties
        # stay in runtime for execution.
        resolved = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.events",
                options={
                    "catalog_properties": {
                        "type": "rest",
                        "warehouse": "s3://wh/",
                        "uri": "https://user:secret@host:8443/api",
                        "password": "p@ss",
                        "token": "tok",
                    }
                },
            )
        )
        assert resolved.identity_options["catalog_properties"] == {
            "type": "rest",
            "warehouse": "s3://wh/",
            "uri": "https://host:8443/api",
        }
        # Runtime keeps the full dict for the actual read.
        assert resolved.runtime_options["catalog_properties"] == {
            "type": "rest",
            "warehouse": "s3://wh/",
            "uri": "https://user:secret@host:8443/api",
            "password": "p@ss",
            "token": "tok",
        }
        assert "p@ss" not in repr(resolved)
        assert "secret" not in repr(resolved)

    def test_builtin_shape_catalog_scheme_and_s3_identity(self) -> None:
        resolved = IcebergProvider().normalize(
            cfg(
                {
                    "type": "iceberg",
                    "catalog": "prod",
                    "table": "db.events",
                    "catalog_properties": {"type": "rest"},
                    "s3": {"endpoint": "http://endpoint-a"},
                }
            )
        )
        assert resolved.identity_options["catalog_scheme"] == "rest"
        assert resolved.identity_options["s3"] == {"endpoint": "http://endpoint-a"}
        assert resolved.runtime_options["catalog_properties"] == {"type": "rest"}
        # No type in catalog_properties → no scheme key (not a None entry).
        no_type = IcebergProvider().normalize(
            cfg(
                {
                    "type": "iceberg",
                    "catalog": "prod",
                    "table": "db.events",
                }
            )
        )
        assert "catalog_scheme" not in no_type.identity_options

    def test_builtin_shape_s3_unknown_option_rejected(self) -> None:
        with pytest.raises(JobConfigurationError, match="unknown s3 option"):
            IcebergProvider().normalize(
                cfg(
                    {
                        "type": "iceberg",
                        "catalog": "prod",
                        "table": "db.events",
                        "s3": {"bucket": "not-supported"},
                    }
                )
            )

    def test_s3_endpoint_enters_ref_id(self) -> None:
        a = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.events",
                options={"s3": {"endpoint": "http://endpoint-a"}},
            )
        )
        b = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.events",
                options={"s3": {"endpoint": "http://endpoint-b"}},
            )
        )
        assert a.ref_id() != b.ref_id()

    def test_catalog_properties_none_ok(self) -> None:
        # An explicit None must behave like an omitted catalog_properties —
        # not crash in the identity classification.
        resolved = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.events",
                options={"catalog_properties": None},
            )
        )
        assert resolved.runtime_options["catalog_properties"] == {}
        assert "catalog_properties" not in resolved.identity_options

    def test_catalog_name_none_falls_back_to_uri(self) -> None:
        # None means "not provided" — fall back to the uri catalog part.
        resolved = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.events",
                options={"catalog_name": None},
            )
        )
        assert resolved.runtime_options["catalog_name"] == "prod"
        assert resolved.canonical_uri == "prod/db.events"

    def test_catalog_property_namespaced_credentials_stay_runtime(self) -> None:
        # PyIceberg catalogs use namespaced credential keys (rest.token,
        # session_token, api_key, client_password, ...) — none of them may
        # reach the identity; non-credential properties still do.
        resolved = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.events",
                options={
                    "catalog_properties": {
                        "type": "rest",
                        "s3.endpoint": "http://endpoint-a",
                        "rest.token": "tok",
                        "session_token": "sess",
                        "sessionToken": "camel-sess",
                        "api_key": "key",
                        "apiKey": "camel-key",
                        "client_password": "pw",
                        "signature": "sig",
                        "aws.secret-access-key": "sk",
                    }
                },
            )
        )
        assert resolved.identity_options["catalog_properties"] == {
            "type": "rest",
            "s3.endpoint": "http://endpoint-a",
        }
        rt = resolved.runtime_options["catalog_properties"]
        assert rt["rest.token"] == "tok"
        assert rt["session_token"] == "sess"
        assert rt["sessionToken"] == "camel-sess"
        assert rt["api_key"] == "key"
        assert rt["apiKey"] == "camel-key"
        assert rt["client_password"] == "pw"
        assert rt["signature"] == "sig"
        assert rt["aws.secret-access-key"] == "sk"
        assert "tok" not in repr(resolved)

    def test_catalog_scheme_changes_ref_id(self) -> None:
        # warehouse:// vs glue:// may resolve to different catalogs for the
        # same catalog/table path — the scheme is part of the identity.
        a = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="warehouse://prod/db.events",
                options={},
            )
        )
        b = IcebergProvider().normalize(
            ProviderSourceConfig(
                provider="tributo.iceberg",
                uri="glue://prod/db.events",
                options={},
            )
        )
        assert a.ref_id() != b.ref_id()


class TestFileProviderPlans:
    """File Providers describe reads; only Engine Bindings execute them."""

    @pytest.mark.parametrize(
        ("provider", "source_type", "connector_id"),
        [
            (ParquetProvider(), "parquet", "parquet"),
            (CsvProvider(), "csv", "csv"),
        ],
    )
    def test_local_file_plan_and_legacy_open_rejection(
        self,
        provider: DataSourceProvider,
        source_type: str,
        connector_id: str,
        tmp_path: Path,
    ) -> None:
        from tributo.data.scan_plan import FileScan

        resolved = provider.normalize(
            cfg({"type": source_type, "path": str(tmp_path / f"data.{source_type}")})
        )
        plan = provider.plan(resolved)

        assert isinstance(plan, FileScan)
        assert plan.connector_id == connector_id
        assert plan.filesystem_id == "local"
        with pytest.raises(JobConfigurationError, match="no legacy Ray-only reader"):
            provider.open(resolved)


class TestConnectionResolution:
    """Explicit value > env var > dialect default (connection-time fallback)."""

    def test_explicit_values_win(self) -> None:
        from tributo.data.provider_builtins import _resolve_connection

        rt = {
            "host": "db.example",
            "port": 8443,
            "user": "reader",
            "password": "p",
            "database": "analytics",
        }
        assert _resolve_connection("clickhouse", rt) == (
            "db.example",
            8443,
            "reader",
            "p",
            "analytics",
        )

    def test_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tributo.data.provider_builtins import _resolve_connection

        monkeypatch.setenv("TRIBUTO_CLICKHOUSE_HOST", "env.example")
        monkeypatch.setenv("TRIBUTO_CLICKHOUSE_PORT", "9443")
        monkeypatch.setenv("TRIBUTO_CLICKHOUSE_PASSWORD", "env-pass")
        host, port, user, password, database = _resolve_connection("clickhouse", {})
        assert host == "env.example"
        assert port == 9443
        assert password == "env-pass"

    def test_invalid_environment_port_is_structured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tributo.data.provider_builtins import _resolve_connection

        monkeypatch.setenv("TRIBUTO_CLICKHOUSE_PORT", "not-a-port")

        with pytest.raises(JobConfigurationError, match="port must be an integer"):
            _resolve_connection("clickhouse", {})

    def test_dialect_defaults(self) -> None:
        from tributo.data.provider_builtins import _resolve_connection

        host, port, user, password, database = _resolve_connection("doris", {})
        assert port == 9030
        assert user == "root"
        assert host == "localhost"

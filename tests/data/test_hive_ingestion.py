"""Hive Provider and ray-hive Binding contracts."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pyarrow as pa
import pytest

import tributo.data.bindings as builtin_bindings
from tributo.data.bindings import _ray_hive_descriptor
from tributo.data.bindings.ray_hive import RayHiveBinding
from tributo.data.engine_binding import BindingCompileRequest, BindingKey
from tributo.data.ingestion import (
    IngestionRuntimeContext,
    RayDataHandle,
    ReadOptions,
)
from tributo.data.provider_builtins import HiveProvider
from tributo.data.provider_registry import resolve_provider
from tributo.data.scan_plan import (
    ConsistencyRequirement,
    ScanKind,
    SqlScan,
    SqlTableRead,
)
from tributo.data.source_config import ProviderSourceConfig
from tributo.data.transform_ir import TransformPipeline
from tributo.exceptions import EngineNotAvailableError, JobConfigurationError


def _source(**options: Any) -> ProviderSourceConfig:
    return ProviderSourceConfig(
        provider="tributo.hive",
        uri="hive://hs2.example:10000/analytics/events",
        options=options,
    )


def test_hive_provider_builds_structured_credential_safe_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVE_IT_PASSWORD", "do-not-leak")
    provider = HiveProvider()
    resolved = provider.normalize(
        _source(
            columns=["id", "category"],
            username="reader",
            password_env="HIVE_IT_PASSWORD",
            auth="ldap",
            transport="binary",
            fetch_rows=128,
            target_batch_bytes=4096,
            query_timeout=30,
            connect_timeout=4,
            rpc_timeout=10,
            session_options={"hive.local.time.zone": "UTC"},
        )
    )
    plan = provider.plan(resolved)

    assert resolved.canonical_uri == "hive://hs2.example:10000/analytics/events"
    assert resolved.identity_options == {
        "columns": ("id", "category"),
        "database": "analytics",
        "session_options": {"hive.local.time.zone": "UTC"},
        "table": "events",
    }
    assert resolved.runtime_options["password_env"] == "HIVE_IT_PASSWORD"
    assert "do-not-leak" not in repr(resolved)
    assert "do-not-leak" not in resolved.ref_id()
    assert isinstance(plan, SqlScan)
    assert plan.connector_id == "hive"
    assert plan.consistency is ConsistencyRequirement.STATEMENT
    assert plan.target == SqlTableRead(
        schema="analytics",
        table="events",
        projection=("id", "category"),
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            ProviderSourceConfig(
                provider="tributo.hive",
                uri="hive://user:secret@hs2/analytics/events",
            ),
            "must not contain userinfo",
        ),
        (
            ProviderSourceConfig(
                provider="tributo.hive",
                uri="hive://hs2/analytics/events?token=secret",
            ),
            "must not contain query",
        ),
        (_source(sql="SELECT 1"), "unknown option"),
        (_source(transport="http"), "requires binary transport"),
        (_source(auth="ldap", username="reader"), "LDAP requires"),
        (_source(session_options={"hive.exec.scratchdir": "/tmp"}), "session option"),
        (
            _source(password_env="BAD-NAME", auth="ldap", username="reader"),
            "environment variable",
        ),
        (
            ProviderSourceConfig(
                provider="tributo.hive",
                uri="hive://hs2:0/analytics/events",
            ),
            "port must be between",
        ),
        (
            ProviderSourceConfig(
                provider="tributo.hive",
                uri="hive://hs2/analytics//events",
            ),
            "exactly database/table",
        ),
        (
            ProviderSourceConfig(
                provider="tributo.hive",
                uri="hive://hs2/analytics/events/",
            ),
            "exactly database/table",
        ),
    ],
)
def test_hive_provider_rejects_unsafe_or_unsupported_configuration(
    source: ProviderSourceConfig,
    message: str,
) -> None:
    with pytest.raises(JobConfigurationError, match=message):
        HiveProvider().normalize(source)


def test_hive_provider_is_registered_with_projection_metadata() -> None:
    provider = resolve_provider(_source(columns=["id"]))
    assert isinstance(provider, HiveProvider)
    assert provider.projection_option_name == "columns"


def test_hive_provider_preserves_encoded_identifier_boundaries() -> None:
    provider = HiveProvider()

    resolved = provider.normalize(
        ProviderSourceConfig(
            provider="tributo.hive",
            uri="hive://hs2/analytics%3Ftenant/events%23daily",
        )
    )
    plan = provider.plan(resolved)

    assert (
        resolved.canonical_uri == "hive://hs2:10000/analytics%3Ftenant/events%23daily"
    )
    assert isinstance(plan, SqlScan)
    assert plan.target == SqlTableRead(
        schema="analytics?tenant",
        table="events#daily",
    )


def test_ray_hive_binding_delegates_to_public_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class _Connection:
        def __init__(self, **kwargs: Any) -> None:
            calls["connection"] = kwargs
            self.transport = kwargs["transport"]

    class _Read:
        def __init__(self, **kwargs: Any) -> None:
            calls["read"] = kwargs

    class _Table:
        def __init__(self, database: str, table: str) -> None:
            self.database = database
            self.table = table

    class _Secret:
        @classmethod
        def env(cls, variable: str) -> tuple[str, str]:
            return ("env", variable)

    class _Dataset:
        def schema(self) -> pa.Schema:
            return pa.schema([("id", pa.int32()), ("category", pa.string())])

    def _read_hive(**kwargs: Any) -> _Dataset:
        calls["reader"] = kwargs
        return _Dataset()

    module = ModuleType("ray_hive")
    module.HiveConnectionOptions = _Connection
    module.HiveReadOptions = _Read
    module.HiveTableIdentifier = _Table
    module.SecretRef = _Secret
    module.read_hive = _read_hive
    monkeypatch.setitem(sys.modules, "ray_hive", module)
    monkeypatch.setattr(
        "tributo.data.bindings.ray_hive.importlib.metadata.version",
        lambda name: "2.55.1" if name == "ray" else "1.0",
    )

    plan = SqlScan(
        provider_id="tributo.hive",
        connector_id="hive",
        target=SqlTableRead(
            schema="analytics",
            table="events",
            projection=("id", "category"),
        ),
        consistency=ConsistencyRequirement.STATEMENT,
    )
    compilation = RayHiveBinding().compile(
        BindingCompileRequest(
            plan=plan,
            runtime_options={
                "host": "hs2.example",
                "port": 10000,
                "username": "reader",
                "password_env": "HIVE_PASSWORD",
                "auth": "LDAP",
                "transport": "binary",
                "fetch_rows": 2,
                "target_batch_bytes": 1024,
                "query_timeout": 30.0,
                "connect_timeout": 4.0,
                "rpc_timeout": 10.0,
                "session_options": {"hive.local.time.zone": "UTC"},
            },
            transforms=TransformPipeline(),
            read_options=ReadOptions(),
            source_ref="a" * 64,
            runtime_context=IngestionRuntimeContext(),
        )
    )

    assert isinstance(compilation.handle, RayDataHandle)
    assert compilation.reader_api == "ray_hive.read_hive"
    assert compilation.transport_id == "hiveserver2.binary"
    assert calls["connection"]["credentials"] == ("env", "HIVE_PASSWORD")
    assert calls["read"]["columns"] == ("id", "category")
    assert calls["reader"]["connection"].transport == "binary"


def test_ray_hive_descriptor_is_optional_and_versioned() -> None:
    descriptor = _ray_hive_descriptor()
    assert descriptor.key.scan_kind is ScanKind.SQL
    assert descriptor.key.connector_id == "hive"
    assert descriptor.key.binding_id == "tributo.ray.hive"
    assert descriptor.distribution_name == "ray-hive"
    assert descriptor.distribution_version == "1.0"
    assert descriptor.dependency_distributions == ("thrift",)


@pytest.mark.parametrize("ray_hive_version", [None, "0.9", "1.1"])
def test_ray_hive_registration_rejects_missing_or_incompatible_distribution(
    monkeypatch: pytest.MonkeyPatch,
    ray_hive_version: str | None,
) -> None:
    versions = {
        "ray": "2.55.1",
        "ray-hive": ray_hive_version,
        "thrift": "0.16.0",
        "tributo": "1.0.0",
    }
    monkeypatch.setattr(builtin_bindings, "_DEFAULT_BINDINGS", None)
    monkeypatch.setattr(
        builtin_bindings,
        "_distribution_version",
        lambda name: versions.get(name),
    )

    bindings = builtin_bindings.default_engine_bindings()

    with pytest.raises(EngineNotAvailableError, match="ray-hive"):
        bindings.resolve(
            BindingKey(
                "tributo.ray_data",
                ScanKind.SQL,
                "hive",
                "tributo.ray.hive",
            )
        )


def test_ray_hive_registration_accepts_exact_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "ray": "2.55.1",
        "ray-hive": "1.0",
        "thrift": "0.16.0",
        "tributo": "1.0.0",
    }
    monkeypatch.setattr(builtin_bindings, "_DEFAULT_BINDINGS", None)
    monkeypatch.setattr(
        builtin_bindings,
        "_distribution_version",
        lambda name: versions.get(name),
    )
    monkeypatch.setattr(
        "tributo.data.engine_binding.importlib.metadata.version",
        lambda name: versions[name],
    )

    descriptor = builtin_bindings.default_engine_bindings().resolve(
        BindingKey(
            "tributo.ray_data",
            ScanKind.SQL,
            "hive",
            "tributo.ray.hive",
        )
    )

    assert descriptor.distribution_version == "1.0"

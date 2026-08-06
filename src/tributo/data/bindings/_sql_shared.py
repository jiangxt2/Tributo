"""Credential-isolated parameter mapping for external SQL reader packages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from tributo.data.scan_plan import LogicalScanPlan, SqlScan, SqlTableRead
from tributo.exceptions import JobConfigurationError

_PORT_DEFAULTS = {"clickhouse": 8123, "doris": 9030, "postgresql": 5432}
_USER_DEFAULTS = {
    "clickhouse": "default",
    "doris": "root",
    "postgresql": "postgres",
}


@dataclass(frozen=True)
class SqlBindingTarget:
    host: str
    port: int
    database: str
    table: str
    username: str
    password: str
    columns: tuple[str, ...]


def require_sql_table(plan: LogicalScanPlan, connector_id: str) -> SqlScan:
    if (
        not isinstance(plan, SqlScan)
        or plan.connector_id != connector_id
        or not isinstance(plan.target, SqlTableRead)
    ):
        raise JobConfigurationError(
            f"{connector_id} Binding requires a structured SqlTableRead"
        )
    return plan


def resolve_sql_target(
    plan: SqlScan,
    runtime_options: Mapping[str, Any],
) -> SqlBindingTarget:
    if not isinstance(plan.target, SqlTableRead):
        raise JobConfigurationError("Structured SQL table target is required")
    dialect = plan.connector_id
    prefix = f"TRIBUTO_{dialect.upper()}"
    host = str(runtime_options.get("host") or os.getenv(f"{prefix}_HOST", "localhost"))
    port_value = runtime_options.get("port")
    port = (
        int(port_value)
        if port_value is not None
        else int(os.getenv(f"{prefix}_PORT", str(_PORT_DEFAULTS[dialect])))
    )
    database = str(
        runtime_options.get("database")
        or os.getenv(f"{prefix}_DB", "")
        or (plan.target.schema if dialect != "postgresql" else "")
    )
    username = str(
        runtime_options.get("user")
        or os.getenv(f"{prefix}_USER", _USER_DEFAULTS[dialect])
    )
    password = str(
        runtime_options.get("password") or os.getenv(f"{prefix}_PASSWORD", "")
    )
    if not host or not database or not plan.target.table:
        raise JobConfigurationError(
            f"{dialect} Binding requires host, database, and table"
        )
    return SqlBindingTarget(
        host=host,
        port=port,
        database=database,
        table=plan.target.table,
        username=username,
        password=password,
        columns=plan.target.projection,
    )

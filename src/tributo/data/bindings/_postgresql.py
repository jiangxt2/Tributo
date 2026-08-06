"""Safe structured-query and connection factories for PostgreSQL readers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tributo.data.bindings._sql_shared import SqlBindingTarget
from tributo.data.scan_plan import SqlScan, SqlTableRead
from tributo.exceptions import JobConfigurationError


def _quote_identifier(identifier: str) -> str:
    if not identifier or "\x00" in identifier:
        raise JobConfigurationError(
            "PostgreSQL table and column identifiers must be non-empty and NUL-free"
        )
    quote = '"'
    return f"{quote}{identifier.replace(quote, quote * 2)}{quote}"


def compile_table_query(plan: SqlScan) -> str:
    """Compile the bounded structured target; arbitrary SQL is never accepted."""
    if not isinstance(plan.target, SqlTableRead):
        raise JobConfigurationError(
            "PostgreSQL Binding requires a structured SqlTableRead"
        )
    projection = (
        ", ".join(_quote_identifier(column) for column in plan.target.projection)
        if plan.target.projection
        else "*"
    )
    qualified = _quote_identifier(plan.target.table)
    if plan.target.schema:
        qualified = f"{_quote_identifier(plan.target.schema)}.{qualified}"
    return f"SELECT {projection} FROM {qualified}"


@dataclass(frozen=True)
class PsycopgConnectionFactory:
    """Picklable DB-API connection factory consumed by Ray Data workers."""

    target: SqlBindingTarget = field(repr=False)

    def __call__(self) -> Any:
        import psycopg

        return psycopg.connect(
            host=self.target.host,
            port=self.target.port,
            dbname=self.target.database,
            user=self.target.username,
            password=self.target.password,
        )

    def __repr__(self) -> str:
        return (
            "PsycopgConnectionFactory("
            f"host={self.target.host!r}, port={self.target.port!r}, "
            f"database={self.target.database!r}, username={self.target.username!r})"
        )


@dataclass(frozen=True)
class SqlAlchemyConnectionFactory:
    """Picklable SQLAlchemy connection factory consumed by Daft."""

    target: SqlBindingTarget = field(repr=False)

    def __call__(self) -> Any:
        from sqlalchemy import URL, create_engine
        from sqlalchemy.pool import NullPool

        url = URL.create(
            "postgresql+psycopg",
            username=self.target.username,
            password=self.target.password,
            host=self.target.host,
            port=self.target.port,
            database=self.target.database,
        )
        # The factory is invoked independently by Daft planning and workers.
        # NullPool makes closing the returned SQLAlchemy Connection close the
        # underlying DB-API connection instead of leaving it in an orphaned
        # per-call Engine pool.
        return create_engine(url, poolclass=NullPool).connect()

    def __repr__(self) -> str:
        return (
            "SqlAlchemyConnectionFactory("
            f"host={self.target.host!r}, port={self.target.port!r}, "
            f"database={self.target.database!r}, username={self.target.username!r})"
        )

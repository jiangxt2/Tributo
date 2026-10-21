"""Test-only observation of successful real SQL-driver execution calls.

No connector, connection, response, or data block is replaced. Ray installs
these observers in its worker processes through the public runtime-env hook.
Only statements against synthetic fixture tables are recorded.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def _record(sql: Any, parameters: Any = None, query_id: str | None = None) -> None:
    if not isinstance(sql, str) or not sql.lstrip().upper().startswith("SELECT "):
        return
    if "filter_fixture_" not in sql or "LIMIT 0" in sql.upper():
        return
    directory = Path("/workspace/tributo-work/trace")
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "event_id": uuid.uuid4().hex,
        "sql": sql,
        "parameters": parameters,
        "pid": os.getpid(),
        "stage": "real_driver_execute_succeeded",
        "query_id": query_id,
    }
    with (directory / f"{os.getpid()}.jsonl").open("a") as output:
        output.write(json.dumps(record, default=str) + "\n")


def setup_sql_observers() -> None:
    """Observe public driver calls while preserving their actual results/errors."""
    import clickhouse_connect
    import pymysql
    from adbc_driver_flightsql import dbapi as flight

    if getattr(clickhouse_connect.get_client, "tributo_observer", False):
        return
    original_client = clickhouse_connect.get_client

    def client(*args: Any, **kwargs: Any) -> Any:
        connection = original_client(*args, **kwargs)
        original_stream = connection.query_arrow_stream

        def stream(sql: str, *args: Any, **kwargs: Any) -> Any:
            result = original_stream(sql, *args, **kwargs)
            _record(
                sql,
                kwargs.get("parameters"),
                kwargs.get("settings", {}).get("query_id"),
            )
            return result

        connection.query_arrow_stream = stream
        return connection

    client.tributo_observer = True
    clickhouse_connect.get_client = client

    def observe_connection(factory: Any) -> Any:
        def connect(*args: Any, **kwargs: Any) -> Any:
            connection = factory(*args, **kwargs)
            original_cursor = connection.cursor

            def cursor(*args: Any, **kwargs: Any) -> Any:
                result = original_cursor(*args, **kwargs)
                original_execute = result.execute

                def execute(sql: str, *args: Any, **kwargs: Any) -> Any:
                    executed = original_execute(sql, *args, **kwargs)
                    _record(sql, args[0] if args else kwargs.get("parameters"))
                    return executed

                result.execute = execute
                return result

            connection.cursor = cursor
            return connection

        return connect

    pymysql.connect = observe_connection(pymysql.connect)
    flight.connect = observe_connection(flight.connect)


def execution_records(table: str) -> list[dict[str, Any]]:
    """Collect successful driver calls for one unique synthetic table."""
    records = []
    for path in Path("/workspace/tributo-work/trace").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if table in record["sql"]:
                records.append(record)
    return records

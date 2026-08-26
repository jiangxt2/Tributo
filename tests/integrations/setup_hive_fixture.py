"""Create the deterministic Hive integration fixture through HiveServer2."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

HS2_CONNECT_ATTEMPTS = 15
HS2_CONNECT_RETRY_SECONDS = 2.0


def split_statements(source: str) -> tuple[str, ...]:
    """Split the controlled fixture on semicolons outside SQL strings."""

    statements: list[str] = []
    buffer: list[str] = []
    in_string = False
    index = 0
    while index < len(source):
        character = source[index]
        buffer.append(character)
        if character == "'":
            if in_string and index + 1 < len(source) and source[index + 1] == "'":
                buffer.append(source[index + 1])
                index += 1
            else:
                in_string = not in_string
        elif character == ";" and not in_string:
            statement = "".join(buffer[:-1]).strip()
            if statement:
                statements.append(statement)
            buffer.clear()
        index += 1

    if in_string:
        raise ValueError("Hive fixture SQL contains an unterminated string literal")
    if "".join(buffer).strip():
        raise ValueError("Hive fixture SQL must terminate every statement")
    return tuple(statements)


def execute_statements(
    client: Any,
    session: Any,
    statements: tuple[str, ...],
    *,
    request_types: Any,
    require_success: Callable[..., None],
) -> None:
    """Execute every fixture statement and close its operation handle."""

    for statement in statements:
        response = client.ExecuteStatement(
            request_types.TExecuteStatementReq(
                sessionHandle=session.value,
                statement=statement,
                runAsync=False,
                queryTimeout=120,
            )
        )
        require_success(response.status, action="fixture ExecuteStatement")
        if response.operationHandle is None:
            raise RuntimeError("Hive fixture operation omitted its handle")
        closed = client.CloseOperation(
            request_types.TCloseOperationReq(operationHandle=response.operationHandle)
        )
        require_success(closed.status, action="fixture CloseOperation")


def load_fixture(script: Path, *, host: str, port: int) -> None:
    """Load one SQL fixture over a single native HiveServer2 session."""

    from ray_hive import HiveConnectionOptions
    from ray_hive._credentials import HiveCredentials
    from ray_hive._hs2.client import NativeHiveRPC, _require_success, ttypes
    from ray_hive._hs2.protocol import SessionHandle

    connection = HiveConnectionOptions(host=host, port=port, rpc_timeout=180.0)
    rpc: Any | None = None
    session: SessionHandle | None = None
    for attempt in range(HS2_CONNECT_ATTEMPTS):
        try:
            rpc = NativeHiveRPC.connect(connection)
            session, _ = rpc.open_session(connection, HiveCredentials(username="hive"))
            break
        except Exception:
            if rpc is not None:
                try:
                    rpc.close_transport()
                except Exception:
                    pass
            rpc = None
            session = None
            if attempt + 1 == HS2_CONNECT_ATTEMPTS:
                raise
            time.sleep(HS2_CONNECT_RETRY_SECONDS)
    if rpc is None or session is None:
        raise RuntimeError("HiveServer2 fixture session was not established")
    try:
        # ray-hive has no public DDL fixture API; this test-only initializer is
        # pinned to ray-hive==1.0 and does not participate in production reads.
        execute_statements(
            rpc.client,
            session,
            split_statements(script.read_text(encoding="utf-8")),
            request_types=ttypes,
            require_success=_require_success,
        )
    finally:
        try:
            rpc.close_session(session)
        finally:
            rpc.close_transport()


def main() -> None:
    """Load the fixture at the requested HiveServer2 endpoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("--host", default="hiveserver2")
    parser.add_argument("--port", type=int, default=10000)
    arguments = parser.parse_args()
    load_fixture(arguments.script, host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()

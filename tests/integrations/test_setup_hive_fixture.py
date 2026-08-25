"""Unit tests for the Hive integration fixture loader."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tests.integrations.setup_hive_fixture import execute_statements, split_statements


class _ExecuteRequest:
    def __init__(self, **kwargs: Any) -> None:
        self.statement = kwargs["statement"]


class _CloseRequest:
    def __init__(self, **kwargs: Any) -> None:
        self.operationHandle = kwargs["operationHandle"]


_REQUEST_TYPES = SimpleNamespace(
    TExecuteStatementReq=_ExecuteRequest,
    TCloseOperationReq=_CloseRequest,
)


def _require_success(_status: Any, *, action: str) -> None:
    assert action.startswith("fixture ")


class _Client:
    def __init__(self, handle: object | None = object()) -> None:
        self.handle = handle
        self.statements: list[str] = []
        self.closed: list[object] = []

    def ExecuteStatement(self, request: Any) -> Any:
        self.statements.append(request.statement)
        return SimpleNamespace(status=object(), operationHandle=self.handle)

    def CloseOperation(self, request: Any) -> Any:
        self.closed.append(request.operationHandle)
        return SimpleNamespace(status=object())


def test_split_statements_preserves_quoted_semicolons() -> None:
    assert split_statements("INSERT INTO t VALUES ('a;''b');\nSELECT * FROM t;\n") == (
        "INSERT INTO t VALUES ('a;''b')",
        "SELECT * FROM t",
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("SELECT 'unterminated;", "unterminated string literal"),
        ("SELECT 1", "terminate every statement"),
    ],
)
def test_split_statements_rejects_incomplete_sql(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        split_statements(source)


def test_execute_statements_closes_each_operation() -> None:
    client = _Client()

    execute_statements(
        client,
        SimpleNamespace(value="session"),
        ("SELECT 1", "SELECT 2"),
        request_types=_REQUEST_TYPES,
        require_success=_require_success,
    )

    assert client.statements == ["SELECT 1", "SELECT 2"]
    assert client.closed == [client.handle, client.handle]


def test_execute_statements_rejects_missing_handle() -> None:
    with pytest.raises(RuntimeError, match="omitted its handle"):
        execute_statements(
            _Client(handle=None),
            SimpleNamespace(value="session"),
            ("SELECT 1",),
            request_types=_REQUEST_TYPES,
            require_success=_require_success,
        )

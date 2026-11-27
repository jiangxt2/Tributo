"""Identity helpers for trusted ClickHouse query reads."""

from __future__ import annotations

from typing import Any, Mapping

from tributo.data.refs import digest


def query_params_digest(params: Mapping[str, Any]) -> str:
    """Include scalar types in a credential-free query parameter digest."""
    return digest(
        {
            key: (type(value).__module__, type(value).__qualname__, value)
            for key, value in params.items()
        }
    )

"""Tests for credential-safe inference diagnostics."""

from __future__ import annotations

import pytest

from tributo.inference._credential_safety import safe_exception_summary


@pytest.mark.parametrize(
    ("message", "secret"),
    [
        (
            "permission denied: Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
            "eyJhbGciOiJIUzI1NiJ9.payload.signature",
        ),
        (
            "permission denied: Authorization=Basic dXNlcjpzZWNyZXQ=",
            "dXNlcjpzZWNyZXQ=",
        ),
        (
            'permission denied: {"access_token": "quoted-secret"}',
            "quoted-secret",
        ),
        (
            "permission denied: Bearer opaque-bearer-token",
            "opaque-bearer-token",
        ),
        (
            "permission denied: eyJ0eXAiOiJKV1QifQ.payload.signature",
            "eyJ0eXAiOiJKV1QifQ.payload.signature",
        ),
    ],
)
def test_exception_summary_redacts_authentication_values(
    message: str, secret: str
) -> None:
    summary = safe_exception_summary(RuntimeError(message))

    assert "permission denied" in summary
    assert "<redacted>" in summary
    assert secret not in summary


def test_exception_summary_intentionally_redacts_machine_local_paths() -> None:
    local_path = "/Users/example/Library/Caches/tributo/model.bin"

    summary = safe_exception_summary(
        RuntimeError(f"download failed while reading {local_path}")
    )

    assert "download failed" in summary
    assert "<local-path>" in summary
    assert local_path not in summary

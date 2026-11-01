"""Opt-in, role validation and real SDK environment conflict contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from tributo._common.reporting.config import load_config
from tributo.observability import ReportingError, SenderConfig, reporting_session


@pytest.mark.parametrize("value", ["true", "false", "", "yes", "2", "-1"])
def test_enabled_is_strict(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIBUTO_LOG_REPORTING_ENABLED", value)
    with pytest.raises(ReportingError, match="config.enabled"):
        load_config()


def test_disabled_ignores_bad_config_and_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIBUTO_LOG_REPORTING_CONFIG", "/missing")
    monkeypatch.setenv("TRIBUTO_LOG_REPORTING_MODE", "invalid")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    with reporting_session() as session:
        assert session.snapshot() == {
            "state": "disabled",
            "role": "owner",
            "enabled": False,
            "diagnostics": {},
        }


def test_import_and_disabled_start_do_not_import_sdk_or_start_threads() -> None:
    code = """
import sys, threading
from tributo.observability import reporting_session
before = set(sys.modules), set(threading.enumerate())
with reporting_session(): pass
assert not any(n.startswith('opentelemetry') for n in set(sys.modules)-before[0])
assert set(threading.enumerate()) == before[1]
assert 'tributo._common.reporting.sender' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


@pytest.mark.parametrize(
    "options",
    [
        {"endpoint": "http://user:secret@localhost/v1/logs"},
        {"endpoint": "http://localhost/v1/logs?password=secret"},
        {"send_timeout": float("inf")},
        {"send_timeout": True},
        {"batch_size": 3, "queue_size": 2},
        {"queue_size": True},
        {"namespaces": ("root",)},
        {"namespaces": ("opentelemetry",)},
        {"endpoint": "http://remote.example/v1/logs"},
    ],
)
def test_unsafe_or_unbounded_sender_is_rejected(options: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SenderConfig.model_validate(options)


def test_invalid_enabled_file_errors_are_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"sender": {"endpoint": "http://user:secret@localhost/v1/logs"}})
    )
    monkeypatch.setenv("TRIBUTO_LOG_REPORTING_ENABLED", "1")
    monkeypatch.setenv("TRIBUTO_LOG_REPORTING_CONFIG", str(path))
    with pytest.raises(ReportingError) as error:
        load_config()
    assert str(error.value) == "reporting.config.invalid"

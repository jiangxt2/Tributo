"""Contract tests for engine-neutral write target planning."""

from __future__ import annotations

from typing import Any

import pytest

from tributo.data.contracts.modes import WriteMode
from tributo.data.writing import (
    GenericWriteTargetProvider,
    LogicalWritePlan,
    WriteCapabilityError,
    WriteRequest,
    WriteTargetRegistry,
)


def _request(**overrides: Any) -> WriteRequest:
    values: dict[str, Any] = {
        "engine": "ray",
        "target_kind": "parquet",
        "target": "/tmp/output",
        "mode": WriteMode.OVERWRITE,
    }
    values.update(overrides)
    return WriteRequest(**values)


def test_builtin_target_provider_creates_immutable_plan() -> None:
    request = _request(options={"compression": "zstd"})
    plan = GenericWriteTargetProvider("parquet").plan(request)

    assert isinstance(plan, LogicalWritePlan)
    assert plan.engine_id == "tributo.ray_data"
    assert plan.provider_id == "tributo.target.parquet"
    assert plan.request_digest == request.request_digest
    assert dict(plan.options) == {"compression": "zstd"}
    mutable_options: Any = plan.options
    with pytest.raises(TypeError):
        mutable_options["compression"] = "snappy"


def test_target_registry_resolves_builtin_formats() -> None:
    registry = WriteTargetRegistry()
    resolved = registry.resolve(_request())

    assert resolved.target_kind == "parquet"
    assert resolved.factory().provider_id == "tributo.target.parquet"


def test_target_registry_fails_closed_for_unknown_format() -> None:
    registry = WriteTargetRegistry()

    with pytest.raises(WriteCapabilityError, match="No write target provider"):
        registry.resolve(_request(target_kind="unknown"))


def test_generic_target_provider_rejects_mismatched_request() -> None:
    with pytest.raises(ValueError, match="does not match"):
        GenericWriteTargetProvider("csv").plan(_request())


def test_logical_plan_rejects_non_hex_digest() -> None:
    with pytest.raises(ValueError, match="SHA-256 hex"):
        LogicalWritePlan(
            plan_version=1,
            provider_id="tributo.target.parquet",
            request_digest="z" * 64,
            engine_id="ray",
            target_kind="parquet",
            target="/tmp/output",
            mode=WriteMode.OVERWRITE,
            options={},
            runtime_options={},
        )

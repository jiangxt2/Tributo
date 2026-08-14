"""Registry tests for write binding capability and dependency validation."""

from __future__ import annotations

import importlib.metadata
from typing import Any

import pytest

from tributo.data.writing import (
    WriteBindingRegistry,
    WriteCapability,
    WriteCapabilityError,
    WriteDescriptor,
    WriteMode,
    WriteRequest,
)
from tributo.data.writing.builtins import _register_if_available


def _descriptor(
    *,
    binding_id: str = "test.ray.parquet",
    mode: WriteMode = WriteMode.OVERWRITE,
    supported_options: frozenset[str] = frozenset(),
) -> WriteDescriptor:
    return WriteDescriptor(
        engine_id="ray",
        target_kind="parquet",
        binding_id=binding_id,
        engine_version_spec="==2.55.1",
        binding_distribution="test-binding",
        binding_distribution_version="1.0.0",
        capabilities=WriteCapability(
            supported_modes=frozenset({mode}),
            supported_options=supported_options,
        ),
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


def _versions(name: str) -> str:
    return {"ray": "2.55.1", "test-binding": "1.0.0"}[name]


def test_registry_resolves_single_matching_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version", _versions
    )
    registry = WriteBindingRegistry()
    registry.register(_descriptor(), lambda: object())

    resolved = registry.resolve(_request())

    assert resolved.descriptor.binding_id == "test.ray.parquet"


def test_registry_requires_explicit_binding_when_candidates_are_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version", _versions
    )
    registry = WriteBindingRegistry()
    registry.register(_descriptor(), lambda: object())
    registry.register(_descriptor(binding_id="test.ray.parquet.alt"), lambda: object())

    with pytest.raises(WriteCapabilityError, match="Multiple write bindings"):
        registry.resolve(_request())


def test_registry_resolves_top_level_binding_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version", _versions
    )
    registry = WriteBindingRegistry()
    registry.register(_descriptor(), lambda: object())
    registry.register(_descriptor(binding_id="test.ray.parquet.alt"), lambda: object())

    resolved = registry.resolve(_request(binding_id="test.ray.parquet.alt"))

    assert resolved.descriptor.binding_id == "test.ray.parquet.alt"


def test_registry_rejects_unsupported_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version", _versions
    )
    registry = WriteBindingRegistry()
    registry.register(
        _descriptor(supported_options=frozenset({"compression"})), lambda: object()
    )

    with pytest.raises(
        WriteCapabilityError, match=r"write option\(s\): partition_cols"
    ):
        registry.resolve(_request(options={"partition_cols": ["date"]}))


def test_registry_accepts_declared_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version", _versions
    )
    registry = WriteBindingRegistry()
    registry.register(
        _descriptor(supported_options=frozenset({"compression"})), lambda: object()
    )

    resolved = registry.resolve(_request(options={"compression": "zstd"}))

    assert resolved.descriptor.binding_id == "test.ray.parquet"


def test_registry_fails_closed_when_engine_version_is_incompatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version", _versions
    )

    with pytest.raises(WriteCapabilityError, match="does not satisfy"):
        WriteBindingRegistry().register(
            _descriptor().model_copy(update={"engine_version_spec": "==2.54.0"}),
            lambda: object(),
        )


def test_registry_fails_closed_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> str:
        if name == "missing-binding":
            raise importlib.metadata.PackageNotFoundError(name)
        return {"ray": "2.55.1", "test-binding": "1.0.0"}[name]

    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version", missing
    )
    descriptor = _descriptor().model_copy(
        update={"dependency_distributions": ("missing-binding",)}
    )

    with pytest.raises(WriteCapabilityError, match="missing-binding"):
        WriteBindingRegistry().register(descriptor, lambda: object())


def test_builtin_registration_isolates_incompatible_optional_binding(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    descriptor = _descriptor().model_copy(update={"engine_version_spec": "==2.54.0"})

    class IncompatibleBinding:
        _descriptor = descriptor

    compatible_descriptor = _descriptor(binding_id="test.ray.parquet.compatible")

    class CompatibleBinding:
        _descriptor = compatible_descriptor

    monkeypatch.setattr(
        "tributo.data.writing.builtins._distribution_available", lambda _: True
    )
    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version", _versions
    )
    registry = WriteBindingRegistry()

    with caplog.at_level("WARNING", logger="tributo.data.writing.builtins"):
        _register_if_available(registry, IncompatibleBinding)
        _register_if_available(registry, CompatibleBinding)

    assert "test.ray.parquet" in caplog.text
    assert "WriteCapabilityError" in caplog.text
    assert (
        registry.resolve(
            _request(binding_id="test.ray.parquet.compatible")
        ).descriptor.binding_id
        == "test.ray.parquet.compatible"
    )

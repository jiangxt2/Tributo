"""Provider contract tests: ResolvedSource immutability/redaction + DatasetHandle lifecycle."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, MutableMapping, cast

import pytest

from tributo.data.provider import (
    DatasetHandle,
    DataSourceProvider,
    ResolvedSource,
)


class _MockHandle(DatasetHandle):
    """Recording handle: counts releases, optional read failure."""

    def __init__(self, fail_read: bool = False) -> None:
        super().__init__()
        self.fail_read = fail_read
        self.release_count = 0
        self.read_count = 0

    def _read(self) -> Any:
        self.read_count += 1
        if self.fail_read:
            raise ValueError("boom")
        return "dataset"

    def _release(self) -> None:
        self.release_count += 1


class _FailingReleaseHandle(DatasetHandle):
    """Release always raises — lifecycle must stay idempotent and not mask."""

    def __init__(self, fail_read: bool = False) -> None:
        super().__init__()
        self.fail_read = fail_read
        self.release_attempts = 0

    def _read(self) -> Any:
        if self.fail_read:
            raise ValueError("read boom")
        return "dataset"

    def _release(self) -> None:
        self.release_attempts += 1
        raise RuntimeError("release boom")


class _MockProvider(DataSourceProvider):
    provider_id = "tributo.mock"
    aliases = frozenset({"mock"})

    def normalize(self, source: object) -> ResolvedSource:
        return ResolvedSource(provider_id=self.provider_id, canonical_uri="mock://x")

    def open(self, resolved: ResolvedSource) -> DatasetHandle:
        return _MockHandle()


class TestResolvedSource:
    """Deep immutability and credential-safe repr."""

    def test_options_deep_frozen(self) -> None:
        resolved = ResolvedSource(
            provider_id="tributo.mock",
            canonical_uri="mock://x",
            identity_options={"columns": ["a"], "nested": {"k": 1}},
            runtime_options={"password": "p@ss"},
        )
        opts = cast(MutableMapping[str, Any], resolved.identity_options)
        with pytest.raises(TypeError):
            opts["columns"] = ["b"]
        nested = cast(MutableMapping[str, Any], opts["nested"])
        with pytest.raises(TypeError):
            nested["k"] = 2
        rt = cast(MutableMapping[str, Any], resolved.runtime_options)
        with pytest.raises(TypeError):
            rt["password"] = "x"

    def test_list_values_frozen(self) -> None:
        resolved = ResolvedSource(
            provider_id="tributo.mock",
            canonical_uri="mock://x",
            identity_options={"columns": ["a", "b"]},
        )
        assert isinstance(resolved.identity_options["columns"], tuple)
        assert resolved.identity_options["columns"] == ("a", "b")

    def test_tuple_values_are_deep_frozen(self) -> None:
        resolved = ResolvedSource(
            provider_id="tributo.mock",
            canonical_uri="mock://x",
            identity_options={"nested": ({"key": "value"},)},
        )
        nested = cast(MutableMapping[str, Any], resolved.identity_options["nested"][0])
        with pytest.raises(TypeError):
            nested["key"] = "changed"

    def test_repr_hides_runtime_values(self) -> None:
        resolved = ResolvedSource(
            provider_id="tributo.clickhouse",
            canonical_uri="ch://db.example/analytics",
            identity_options={"columns": ["a"]},
            runtime_options={"password": "s3cr3t", "user": "admin"},
        )
        text = repr(resolved)
        assert "s3cr3t" not in text
        assert "admin" not in text
        # Keys may appear (they are not secrets); values never do.
        assert "password" in text
        assert "identity_options" in text

    def test_frozen_dataclass(self) -> None:
        resolved = ResolvedSource(provider_id="tributo.mock", canonical_uri="mock://x")
        with pytest.raises(FrozenInstanceError):
            resolved.__setattr__("provider_id", "tributo.other")

    def test_ref_id_with_nested_identity_options(self) -> None:
        # Nested options are deep-frozen (MappingProxyType) — ref_id must
        # still serialize them and stay stable.
        resolved = ResolvedSource(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/a.parquet",
            identity_options={"partition": {"year": 2026, "month": 7}},
        )
        ref1 = resolved.ref_id()
        assert len(ref1) == 64
        assert resolved.ref_id() == ref1

    def test_identity_credential_rejected_before_freezing(self) -> None:
        with pytest.raises(ValueError, match="credential field"):
            ResolvedSource(
                provider_id="tributo.parquet",
                canonical_uri="s3://bkt/a.parquet",
                identity_options={"s3": {"secret_access_key": "secret"}},
            )


class TestDataSourceProvider:
    """Stable contract shape."""

    def test_abstract(self) -> None:
        with pytest.raises(TypeError):
            cast(Any, DataSourceProvider)()

    def test_identity_metadata(self) -> None:
        assert _MockProvider.provider_id == "tributo.mock"
        assert _MockProvider.aliases == frozenset({"mock"})

    def test_normalize_open_roundtrip(self) -> None:
        provider = _MockProvider()
        resolved = provider.normalize(object())
        handle = provider.open(resolved)
        assert isinstance(handle, DatasetHandle)


class TestDatasetHandleLifecycle:
    """close idempotency, post-close access, failure paths, context manager."""

    def test_normal_close(self) -> None:
        handle = _MockHandle()
        assert handle.to_ray_dataset() == "dataset"
        handle.close()
        handle.close()  # idempotent
        assert handle.release_count == 1

    def test_close_then_read_raises(self) -> None:
        handle = _MockHandle()
        handle.close()
        with pytest.raises(RuntimeError, match="already released"):
            handle.to_ray_dataset()

    def test_read_twice_raises(self) -> None:
        handle = _MockHandle()
        handle.to_ray_dataset()
        with pytest.raises(RuntimeError, match="already released"):
            handle.to_ray_dataset()

    def test_failed_read_then_close(self) -> None:
        # _read raised: resources are released by the finally, the handle
        # is closed, and further reads are rejected.
        handle = _MockHandle(fail_read=True)
        with pytest.raises(ValueError, match="boom"):
            handle.to_ray_dataset()
        assert handle.release_count == 1
        with pytest.raises(RuntimeError, match="already released"):
            handle.to_ray_dataset()
        handle.close()
        handle.close()
        assert handle.release_count == 1
        assert handle.read_count == 1

    def test_context_manager_normal(self) -> None:
        with _MockHandle() as handle:
            assert handle.to_ray_dataset() == "dataset"
        assert handle.release_count == 1

    def test_context_manager_on_exception(self) -> None:
        with pytest.raises(ValueError, match="boom"):
            with _MockHandle(fail_read=True) as handle:
                handle.to_ray_dataset()
        assert handle.release_count == 1

    def test_release_called_exactly_once(self) -> None:
        handle = _MockHandle()
        handle.to_ray_dataset()
        handle.close()
        assert handle.release_count == 1

    def test_failing_release_does_not_mask_read(self) -> None:
        handle = _FailingReleaseHandle(fail_read=True)
        with pytest.raises(ValueError, match="read boom"):
            handle.to_ray_dataset()
        # Handle is closed despite the release failure.
        with pytest.raises(RuntimeError, match="already released"):
            handle.to_ray_dataset()
        handle.close()
        assert handle.release_attempts == 1

    def test_failing_release_still_returns_dataset(self) -> None:
        handle = _FailingReleaseHandle()
        assert handle.to_ray_dataset() == "dataset"
        handle.close()
        handle.close()
        assert handle.release_attempts == 1
        assert handle._release_error == "RuntimeError"

    def test_failing_release_log_hides_exception_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        handle = _FailingReleaseHandle()
        with caplog.at_level("WARNING"):
            handle.close()
        assert "release boom" not in caplog.text
        assert "RuntimeError" in caplog.text

"""Tests for the generic ``Registry`` base class.

Covers thread safety, error messages, unregister, and PluginAwareRegistry
lazy discovery.
"""

from __future__ import annotations

import threading

import pytest

from tributo._common.registry import PluginAwareRegistry, Registry
from tributo.exceptions import JobConfigurationError


class TestRegistry:
    """Unit tests for Registry[K, V]."""

    def test_register_and_get(self) -> None:
        r: Registry[str, int] = Registry(name="counter")
        r.register("a", 1)
        assert r.get("a") == 1

    def test_list_returns_sorted_keys(self) -> None:
        r: Registry[str, str] = Registry(name="tag")
        r.register("z", "last")
        r.register("a", "first")
        r.register("m", "middle")
        assert r.list() == ["a", "m", "z"]

    def test_contains(self) -> None:
        r: Registry[str, int] = Registry(name="value")
        r.register("x", 42)
        assert r.contains("x") is True
        assert r.contains("y") is False

    def test_get_unknown_raises(self) -> None:
        r: Registry[str, int] = Registry(name="item")
        with pytest.raises(JobConfigurationError, match="Unknown item: 'missing'"):
            r.get("missing")

    def test_register_duplicate_raises(self) -> None:
        r: Registry[str, int] = Registry(name="item")
        r.register("dup", 1)
        with pytest.raises(JobConfigurationError, match="already registered"):
            r.register("dup", 2)

    def test_unregister_removes_key(self) -> None:
        r: Registry[str, str] = Registry(name="item")
        r.register("tmp", "val")
        r.unregister("tmp")
        assert r.contains("tmp") is False
        with pytest.raises(JobConfigurationError):
            r.get("tmp")

    def test_unregister_missing_is_idempotent(self) -> None:
        r: Registry[str, int] = Registry(name="item")
        # Should not raise
        r.unregister("nonexistent")

    def test_error_message_includes_available_keys(self) -> None:
        r: Registry[str, int] = Registry(name="trainer")
        r.register("xgboost", 1)
        r.register("dnn", 2)
        with pytest.raises(JobConfigurationError) as exc_info:
            r.get("unknown")
        msg = str(exc_info.value)
        assert "unknown" in msg
        assert "xgboost" in msg
        assert "dnn" in msg

    def test_thread_safety_register(self) -> None:
        r: Registry[str, int] = Registry(name="threaded")
        errors: list[Exception] = []

        def register_range(start: int) -> None:
            for i in range(start, start + 100):
                try:
                    r.register(str(i), i * 10)
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=register_range, args=(0,))
        t2 = threading.Thread(target=register_range, args=(100,))
        t3 = threading.Thread(target=register_range, args=(200,))
        for t in (t1, t2, t3):
            t.start()
        for t in (t1, t2, t3):
            t.join()

        assert not errors, f"Unexpected errors: {errors}"
        assert len(r.list()) == 300


class TestPluginAwareRegistry:
    """Unit tests for lazy plugin discovery."""

    def test_discovery_triggered_on_get(self) -> None:
        called: list[int] = []

        def discover() -> list[tuple[str, int]]:
            called.append(1)
            return [("plugin_a", 99)]

        r: PluginAwareRegistry[str, int] = PluginAwareRegistry(
            name="plugin", discover=discover
        )
        r.register("builtin", 1)

        # Before first read, discover hasn't been called
        assert not called

        # get triggers discovery
        assert r.get("plugin_a") == 99
        assert called == [1]

        # Second get does NOT re-trigger
        assert r.get("plugin_a") == 99
        assert called == [1]

    def test_discovery_triggered_on_list(self) -> None:
        called: list[int] = []

        def discover() -> list[tuple[str, int]]:
            called.append(1)
            return [("ext", 42)]

        r: PluginAwareRegistry[str, int] = PluginAwareRegistry(
            name="plugin", discover=discover
        )
        r.register("core", 1)
        keys = r.list()
        assert "core" in keys
        assert "ext" in keys
        assert called == [1]

    def test_discovery_duplicate_is_skipped(self) -> None:
        def discover() -> list[tuple[str, int]]:
            return [("dupe", 99)]

        r: PluginAwareRegistry[str, int] = PluginAwareRegistry(
            name="plugin", discover=discover
        )
        r.register("dupe", 1)  # built-in registers first
        r.list()  # triggers discovery — should not replace built-in
        assert r.get("dupe") == 1  # built-in value preserved

    def test_discovery_none_is_noop(self) -> None:
        r: PluginAwareRegistry[str, int] = PluginAwareRegistry(
            name="plugin", discover=None
        )
        r.register("a", 1)
        assert r.list() == ["a"]

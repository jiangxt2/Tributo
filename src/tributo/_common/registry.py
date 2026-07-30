"""Generic thread-safe registry for plugin-style extensibility.

Eliminates the duplicated ``dict + threading.Lock`` pattern shared by
``training/registry.py``, ``data/registry.py``, and ``embeddings/registry.py``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Generic, TypeVar

from tributo.exceptions import JobConfigurationError

K = TypeVar("K", bound=str)
V = TypeVar("V")

logger = logging.getLogger(__name__)


class Registry(Generic[K, V]):
    """Thread-safe generic registry.

    Args:
        name: Human-readable item name used in error messages
            (e.g. ``"trainer"``, ``"connector"``).
    """

    def __init__(self, name: str = "item") -> None:
        self._name = name
        self._store: dict[K, V] = {}
        self._lock = threading.Lock()

    # -- internal (must precede public methods that use `list` in annotations) --

    def _sorted_keys(self) -> list[K]:
        """Sorted keys, called inside the lock."""
        return sorted(self._store)

    # -- write path -----------------------------------------------------------

    def register(self, key: K, value: V) -> None:
        """Register *value* under *key*.

        Raises:
            JobConfigurationError: If *key* is already registered.
        """
        with self._lock:
            if key in self._store:
                raise JobConfigurationError(
                    f"{self._name.capitalize()} {key!r} already registered. "
                    f"Available: {self._sorted_keys()}"
                )
            self._store[key] = value

    def unregister(self, key: K) -> None:
        """Remove *key* from the registry (idempotent).

        Silently succeeds if *key* is not registered — useful for test
        teardown and hot-reload scenarios.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]

    # -- read path ------------------------------------------------------------

    def get(self, key: K) -> V:
        """Return the value registered under *key*.

        Raises:
            JobConfigurationError: If *key* is not found.
        """
        with self._lock:
            if key not in self._store:
                raise JobConfigurationError(
                    f"Unknown {self._name}: {key!r}. Available: {self._sorted_keys()}"
                )
            return self._store[key]

    def list(self) -> list[K]:
        """Return sorted list of all registered keys."""
        with self._lock:
            return self._sorted_keys()

    def contains(self, key: K) -> bool:
        """Return ``True`` if *key* is registered."""
        with self._lock:
            return key in self._store

    def snapshot(self) -> dict[K, V]:
        """Return a shallow copy of the key→value mapping under the lock.

        The returned dict is a brand-new dict — the caller cannot mutate
        ``_store`` through it.  Value integrity is the responsibility of
        the value type (e.g. deeply-immutable ``AlgorithmSpec``).
        """
        with self._lock:
            return dict(self._store)


class PluginAwareRegistry(Registry[K, V]):
    """Registry with lazy ``entry_points`` plugin discovery.

    Plugin discovery is deferred until the first read operation (``get``,
    ``list``, or ``contains``), so that importing the registry module does
    not eagerly scan all installed packages.

    Args:
        name: Human-readable item name for error messages.
        discover: Callable that returns an iterable of ``(key, value)``
            pairs discovered from ``entry_points``.  Called at most once.
    """

    def __init__(
        self,
        name: str = "item",
        discover: Callable[[], list[tuple[K, V]]] | None = None,
    ) -> None:
        super().__init__(name)
        self._discover = discover
        self._discovered = False

    def _ensure_discovered(self) -> None:
        """Trigger plugin discovery on first access (double-checked locking).

        Writes directly to ``self._store`` inside the lock rather than
        calling ``self.register()``, because ``register()`` acquires the
        same lock (``threading.Lock`` is not reentrant).

        Sets ``self._discovered = True`` only AFTER ``discover()`` succeeds,
        so that transient errors (network timeout, corrupt package) do not
        permanently block discovery for the process lifetime.
        """
        if self._discovered:
            return
        with self._lock:
            if self._discovered:
                return
            if self._discover is None:
                self._discovered = True
                return
            try:
                pairs = self._discover()
            except Exception:
                logger.warning(
                    "Plugin discovery for %s registry failed; "
                    "will retry on next access.",
                    self._name,
                    exc_info=True,
                )
                return
            self._discovered = True
            for key, value in pairs:
                if key in self._store:
                    logger.debug(
                        "%s %r from plugin already registered; skipping.",
                        self._name.capitalize(),
                        key,
                    )
                    continue
                self._store[key] = value

    def unregister(self, key: K) -> None:
        """Remove *key* from the registry (idempotent).

        Triggers ``_ensure_discovered()`` first so that plugin keys not yet
        in the store are discovered before attempting removal.
        """
        self._ensure_discovered()
        super().unregister(key)

    # -- read-path overrides that trigger lazy discovery -----------------------

    def get(self, key: K) -> V:
        self._ensure_discovered()
        return super().get(key)

    def list(self) -> list[K]:
        self._ensure_discovered()
        return super().list()

    def contains(self, key: K) -> bool:
        self._ensure_discovered()
        return super().contains(key)

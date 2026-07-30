"""Deeply-immutable containers with pickle- and JSON-safe serialization.

``FrozenDict`` is a ``dict`` subclass that blocks all mutation paths so
that config values retrieved from ``AlgorithmSpec`` cannot be mutated
back into the Registry.  Because it is a ``dict`` subclass it is
natively supported by ``json``, ``pickle``, and ``ray.cloudpickle``.

``deep_freeze()`` / ``deep_thaw()`` recursively convert between
``FrozenDict`` / ``tuple`` and standard mutable Python containers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# FrozenDict
# ---------------------------------------------------------------------------


class FrozenDict(dict[str, Any]):
    """Read-only dict for configuration data.

    Inherits from ``dict`` for JSON / pickle compatibility but overrides
    every mutation method to raise ``TypeError``.  Recursively freezes
    all nested values after construction.

    Accepts the same constructor arguments as ``dict`` (mapping, iterable
    of pairs, or keyword arguments).
    """

    def __init__(
        self,
        __mapping_or_iterable: Mapping[str, Any]
        | Iterable[tuple[str, Any]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        # Build via dict.__init__ first, then freeze all values in-place.
        if __mapping_or_iterable is not None:
            if isinstance(__mapping_or_iterable, Mapping):
                if not all(isinstance(k, str) for k in __mapping_or_iterable):
                    raise TypeError("configuration mapping keys must be strings")
                super().__init__((k, v) for k, v in __mapping_or_iterable.items())
            else:
                # Pair-mode: materialise to list so that one-shot iterators
                # are not consumed by the key-type check below.
                pairs = list(__mapping_or_iterable)
                if not all(isinstance(k, str) for k, _ in pairs):
                    raise TypeError("configuration mapping keys must be strings")
                super().__init__(pairs)
        if kwargs:
            super().__init__(kwargs)
        # Freeze every value in-place.
        for key in list(self):
            dict.__setitem__(self, key, deep_freeze(self[key]))

    # -- blocked mutation methods --------------------------------------------

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    def __delitem__(self, key: str) -> None:
        raise TypeError("FrozenDict is immutable")

    def clear(self) -> None:
        raise TypeError("FrozenDict is immutable")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("FrozenDict is immutable")

    def popitem(self) -> Any:
        raise TypeError("FrozenDict is immutable")

    def setdefault(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("FrozenDict is immutable")

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        raise TypeError("FrozenDict is immutable")

    # -- hash / equality -----------------------------------------------------

    def __hash__(self) -> int:
        # All values must also be hashable — this holds because deep_freeze
        # only produces FrozenDict, tuple, primitives, and Enum values.
        return hash(tuple(sorted(self.items())))

    # -- pickle support ------------------------------------------------------

    def __reduce__(
        self,
    ) -> tuple[type[FrozenDict], tuple[dict[str, Any]]]:
        """Pickle reconstructs via constructor with a plain dict.

        Without this, pickle's default reconstruction for dict subclasses
        calls ``__setitem__`` / ``update`` which are blocked.
        """
        return (FrozenDict, (dict(self),))

    # -- shallow copy --------------------------------------------------------

    def copy(self) -> dict[str, Any]:  # type: ignore[override]
        """Return a mutable shallow copy."""
        return dict(self)


# ---------------------------------------------------------------------------
# freeze / thaw
# ---------------------------------------------------------------------------

_PRIMITIVES = (str, int, float, bool, type(None))


def deep_freeze(value: Any) -> Any:
    """Recursively freeze *value* into an immutable representation.

    * ``Mapping`` → ``FrozenDict`` (keys must be ``str``)
    * ``list`` / ``tuple`` → ``tuple``
    * ``None``, ``str``, ``int``, ``float``, ``bool``, ``Enum`` → pass through
    * anything else → ``TypeError``

    Note: ``freeze→thaw`` is not a strict inverse for tuple inputs —
    ``deep_thaw(deep_freeze((1, 2)))`` returns ``[1, 2]`` because thaw
    converts all tuples to lists (it cannot distinguish source type).
    """
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("configuration mapping keys must be strings")
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(v) for v in value)
    if value is None or isinstance(value, _PRIMITIVES) or isinstance(value, Enum):
        return value
    raise TypeError(f"unsupported mutable/config value: {type(value).__name__}")


def deep_thaw(value: Any) -> Any:
    """Recursively thaw *value* into standard mutable Python containers.

    * ``Mapping`` (incl. ``FrozenDict``) → ``dict``
    * ``tuple`` → ``list``
    * ``None``, ``str``, ``int``, ``float``, ``bool``, ``Enum`` → pass through
    * anything else → ``TypeError``
    """
    if isinstance(value, Mapping):
        return {k: deep_thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [deep_thaw(v) for v in value]
    if value is None or isinstance(value, _PRIMITIVES) or isinstance(value, Enum):
        return value
    raise TypeError(f"unsupported frozen/config value: {type(value).__name__}")

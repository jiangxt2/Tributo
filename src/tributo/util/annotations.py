"""API stability annotations.

Provides ``@PublicAPI`` and ``@DeveloperAPI`` decorators for documenting
the stability contract of public symbols, inspired by Ray's
``ray.util.annotations`` module.

Usage::

    from tributo.util.annotations import PublicAPI, DeveloperAPI, Stability

    @PublicAPI(stability=Stability.STABLE)
    def my_func():
        ...

    @PublicAPI  # equivalent to @PublicAPI(stability=Stability.BETA)
    def beta_func():
        ...

    @DeveloperAPI
    def _internal_func():
        ...

Runtime query::

    from tributo.util.annotations import get_stability
    level = get_stability(SomeClass)  # → "stable" | "beta" | "alpha" | None
"""

from __future__ import annotations

import types
from typing import Any, Callable, Optional, TypeVar, Union, overload

_CallableOrClass = TypeVar("_CallableOrClass", bound=Union[Callable[..., Any], type])

_STABILITY_ATTR = "_tributo_stability"


class Stability:
    """Stability levels for Tributo public API.

    - ``STABLE``: Backward-compatible within the same major version.
    - ``BETA``: API may change with deprecation notice (≥ 2 minor versions).
    - ``ALPHA``: Under active development; may change without notice.
    """

    STABLE: str = "stable"
    BETA: str = "beta"
    ALPHA: str = "alpha"


@overload
def PublicAPI(stability: str) -> Callable[[_CallableOrClass], _CallableOrClass]: ...
@overload
def PublicAPI(stability: _CallableOrClass) -> _CallableOrClass: ...
def PublicAPI(
    stability: str | _CallableOrClass = Stability.BETA,
) -> _CallableOrClass | Callable[[_CallableOrClass], _CallableOrClass]:
    """Mark a function, class, or method as part of the Tributo public API.

    Can be used with or without parentheses::

        @PublicAPI(stability=Stability.BETA)
        def foo(): ...

        @PublicAPI
        def bar(): ...   # equivalent to stability=Stability.BETA

    Args:
        stability: One of ``Stability.STABLE``, ``Stability.BETA``, or
            ``Stability.ALPHA``.  When called without parentheses and
            *stability* is a callable, defaults to ``Stability.BETA``.

    Returns:
        A decorator that tags the object with the given stability level,
        or the decorated object when used without parentheses.
    """
    # Support @PublicAPI without parentheses.
    if callable(stability):
        setattr(stability, _STABILITY_ATTR, Stability.BETA)
        return stability

    if stability not in (Stability.STABLE, Stability.BETA, Stability.ALPHA):
        raise ValueError(
            f"Invalid stability '{stability}'. "
            f"Expected one of: {Stability.STABLE!r}, {Stability.BETA!r}, {Stability.ALPHA!r}"
        )

    def _decorator(obj: _CallableOrClass) -> _CallableOrClass:
        setattr(obj, _STABILITY_ATTR, stability)
        return obj

    return _decorator


def DeveloperAPI(obj: _CallableOrClass) -> _CallableOrClass:
    """Mark a function, class, or method as a developer-facing (internal) API.

    Developer API symbols are **not** covered by the Tributo stability
    contract and may change without notice between patch releases.

    Args:
        obj: A function, class, or method to mark as developer API.

    Returns:
        The same *obj*, tagged with ``_tributo_stability = "developer"``.
    """
    setattr(obj, _STABILITY_ATTR, "developer")
    return obj


def get_stability(obj: Any) -> Optional[str]:
    """Return the stability level of *obj*, or ``None`` if unannotated.

    Handles classes, functions, and bound methods.  For bound methods
    the attribute is read from the underlying ``__func__``.

    Args:
        obj: A class, function, or bound method.

    Returns:
        ``"stable"``, ``"beta"``, ``"alpha"``, ``"developer"``, or ``None``.
    """
    # Unwrap bound methods to access decorator-set attributes on __func__.
    if isinstance(obj, types.MethodType):
        obj = obj.__func__

    cls = obj if isinstance(obj, type) else type(obj)
    level: Optional[str] = getattr(cls, _STABILITY_ATTR, None)
    if level is None and not isinstance(obj, type):
        level = getattr(obj, _STABILITY_ATTR, None)
    return level


def is_public_api(obj: Any) -> bool:
    """Return ``True`` if *obj* carries a public API stability annotation.

    Args:
        obj: A class, function, or bound method to check.

    Returns:
        ``True`` if *obj* is annotated as ``stable``, ``beta``, or ``alpha``.
    """
    level = get_stability(obj)
    return level is not None and level != "developer"

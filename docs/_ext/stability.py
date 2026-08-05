"""Render Tributo API stability annotations in autodoc output."""

from __future__ import annotations

from typing import Any

_STABILITY_CONTENT: dict[str, tuple[str, str]] = {
    "stable": (
        "Stable",
        "Backward-compatible within the current major version.",
    ),
    "beta": (
        "Beta",
        "May change with a documented deprecation notice.",
    ),
    "alpha": (
        "Alpha",
        "May change without notice while the contract is being validated.",
    ),
    "developer": (
        "Internal",
        "Not part of the supported public API contract.",
    ),
}

_TOP_LEVEL_AUTODOC_TYPES = frozenset({"class", "exception", "function", "data"})


def get_object_stability(obj: Any) -> str:
    """Return the public stability level or ``developer`` when unannotated."""
    from tributo.util.annotations import get_stability

    level = get_stability(obj)
    if level in _STABILITY_CONTENT:
        return level
    return "developer"


def render_stability_lines(level: str) -> list[str]:
    """Return reStructuredText lines for one stability callout."""
    normalized = level if level in _STABILITY_CONTENT else "developer"
    label, description = _STABILITY_CONTENT[normalized]
    return [
        "",
        f".. admonition:: {label} API",
        (f"   :class: tributo-stability tributo-stability-{normalized}"),
        "",
        f"   {description}",
        "",
    ]


def add_stability_to_docstring(
    app: Any,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    lines: list[str],
) -> None:
    """Add a stability callout to top-level autodoc objects."""
    del app, name, options
    if what not in _TOP_LEVEL_AUTODOC_TYPES:
        return
    lines[:0] = render_stability_lines(get_object_stability(obj))


def setup(app: Any) -> dict[str, bool]:
    """Register the autodoc event handler."""
    app.connect("autodoc-process-docstring", add_stability_to_docstring)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

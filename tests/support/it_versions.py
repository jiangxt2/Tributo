"""Single parser for the pinned model-export IT component contract."""

from __future__ import annotations

from pathlib import Path

COMPONENT_VERSIONS_FILE = (
    Path(__file__).parents[1] / "integrations" / "component-versions.env"
)


def load_it_component_versions(path: Path = COMPONENT_VERSIONS_FILE) -> dict[str, str]:
    """Load the checked-in KEY=VALUE version contract without shell expansion."""
    versions: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"Invalid component version at {path}:{line_number}")
        if key in versions:
            raise ValueError(f"Duplicate component version key {key!r}")
        versions[key] = value
    return versions


__all__ = ["COMPONENT_VERSIONS_FILE", "load_it_component_versions"]

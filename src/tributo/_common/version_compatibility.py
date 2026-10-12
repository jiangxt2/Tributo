"""Core version matching shared by plugin and algorithm contracts."""

from __future__ import annotations

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
def tributo_version_matches(version: str | Version, version_spec: str) -> bool:
    """Match Core development versions without changing declared bounds.

    Development releases are explicitly selected Core checkouts, so they may
    satisfy a plugin's compatible range. Formal releases and release candidates
    retain the specifier's default prerelease policy. The full version is kept
    for bounds, exact pins, exclusions, and evidence identity.
    """
    parsed = Version(version) if isinstance(version, str) else version
    return SpecifierSet(version_spec).contains(
        parsed, prereleases=True if parsed.is_devrelease else None
    )

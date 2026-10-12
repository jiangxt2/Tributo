"""Core development-version ranges preserve plugin compatibility boundaries."""

from __future__ import annotations

import pytest
from packaging.specifiers import InvalidSpecifier
from packaging.version import InvalidVersion, Version

from tributo._common.version_compatibility import tributo_version_matches


@pytest.mark.parametrize("parsed", [False, True])
@pytest.mark.parametrize(
    ("version", "version_spec", "accepted"),
    [
        ("1.1.0", ">=1,<2", True),
        ("1.1.0", "==1.1.0", True),
        ("1.1.0", ">=1,<2,!=1.1.0", False),
        ("1.2.0.dev0", ">=1,<2", True),
        ("1.2.0.dev7", ">=1,<2", True),
        ("1.2.0.dev0", ">=1,<1.2", False),
        ("1.2.0.dev0", ">=1.2", False),
        ("1.2.0.dev0", "==1.1.0", False),
        ("1.2.0.dev0", ">=1,<2,!=1.2.*", False),
        ("1.2.0.dev0", ">=1.2.0.dev0,<2", True),
        ("1.2.0.dev0", "==1.2.0.dev0", True),
        ("1.2.0rc1", ">=1,<2", False),
        ("1.2.0b1", ">=1,<2", False),
        ("1.2.0rc1", ">=1.2.0rc1,<2", True),
        ("1.2.0", ">=1,<2", True),
        ("2.0.0.dev0", ">=1,<2", False),
        ("2.0.0", ">=1,<2", False),
    ],
)
def test_core_version_range(
    version: str, version_spec: str, accepted: bool, parsed: bool
) -> None:
    candidate = Version(version) if parsed else version

    assert tributo_version_matches(candidate, version_spec) is accepted


def test_invalid_core_version_is_rejected() -> None:
    with pytest.raises(InvalidVersion):
        tributo_version_matches("unknown", ">=1,<2")


def test_invalid_core_version_range_is_rejected() -> None:
    with pytest.raises(InvalidSpecifier):
        tributo_version_matches("1.2.0.dev0", "<")

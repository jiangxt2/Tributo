"""Worker dependency checks share Core's development-version policy."""

from __future__ import annotations

import importlib.metadata

import pytest

from tributo.algorithms.api import AlgorithmDependencyError
from tributo.algorithms.core.worker import _actual_environment_versions


@pytest.mark.parametrize(
    ("name", "version", "constraint", "accepted"),
    [
        ("tributo", "1.3.0.dev0", ">=1.1", True),
        ("TrIbUtO", "1.3.0.dev0", ">=1.1", True),
        ("tributo", "1.2.0", ">=1.1", True),
        ("tributo", "1.3.0rc1", ">=1.1", False),
        ("tributo", "1.3.0rc1", ">=1.3.0rc1", True),
        ("tributo", "1.0.0", ">=1.1", False),
        ("tributo", "1.3.0.dev0", ">=1.3", False),
        ("tributo", "1.3.0.dev0", "<1.2", False),
        ("tributo", "1.3.0.dev0", "==1.3.0.dev0", True),
        ("tributo", "1.3.0.dev0", ">=1.1,!=1.3.0.dev0", False),
        ("other-library", "1.3.0.dev0", ">=1.1", False),
        ("other-library", "1.3.0.dev0", "==1.3.0.dev0", True),
    ],
)
def test_dependency_version_policy(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    version: str,
    constraint: str,
    accepted: bool,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _: version)
    dependencies = (f"{name}{constraint}",)
    if accepted:
        assert (
            _actual_environment_versions(">=3.12,<3.14", dependencies)[name] == version
        )
    else:
        with pytest.raises(AlgorithmDependencyError, match="does not satisfy"):
            _actual_environment_versions(">=3.12,<3.14", dependencies)


def test_missing_dependency_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(AlgorithmDependencyError, match="not installed: absent"):
        _actual_environment_versions(">=3.12,<3.14", ("absent>=1",))


def test_inactive_marker_does_not_resolve_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(name: str) -> str:
        raise AssertionError(f"inactive marker resolved {name}")

    monkeypatch.setattr(importlib.metadata, "version", unexpected)
    versions = _actual_environment_versions(
        ">=3.12,<3.14", ('absent>=1; python_version < "0"',)
    )
    assert set(versions) == {"python"}


def test_python_version_constraint_is_not_relaxed() -> None:
    with pytest.raises(AlgorithmDependencyError, match="Worker Python"):
        _actual_environment_versions("<3", ())

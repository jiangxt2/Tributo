"""Tests for the offline algorithm Bundle builder."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

from tools.build_algorithm_bundle import BundleBuildError, build_bundle
from tributo._common.algorithm_distribution import prepare_algorithm_distribution
from tributo.algorithms import AlgorithmArtifact, ArtifactDistributionMode, ImageProfile


def _write_wheel(
    directory: Path,
    package: str,
    version: str = "1.0.0",
    *,
    requires: tuple[str, ...] = (),
    plugin_name: str | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{package.replace('-', '_')}-{version}-py3-none-any.whl"
    path = directory / filename
    dist_info = f"{package.replace('-', '_')}-{version}.dist-info"
    metadata = "\n".join(
        (
            "Metadata-Version: 2.3",
            f"Name: {package}",
            f"Version: {version}",
            *(f"Requires-Dist: {item}" for item in requires),
            "",
        )
    )
    wheel = (
        "Wheel-Version: 1.0\n"
        "Generator: tributo-tests\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{package.replace('-', '_')}/__init__.py", "")
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", wheel)
        if plugin_name is not None:
            archive.writestr(
                f"{dist_info}/entry_points.txt",
                f"[tributo.algorithms]\n{plugin_name} = {package}:DESCRIPTOR\n",
            )
    return path


def _profile() -> ImageProfile:
    return ImageProfile(
        profile_id="cpu.test",
        image_uri="tributo:test",
        image_digest="a" * 64,
        python_spec=(
            f">={sys.version_info.major}.{sys.version_info.minor},"
            f"<{sys.version_info.major}.{sys.version_info.minor + 1}"
        ),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        wheel_tags=("py3-none-any",),
        installed_distributions={"pip": "24.3.1"},
    )


def test_builder_creates_a_bundle_accepted_by_preflight(tmp_path: Path) -> None:
    algorithm = _write_wheel(
        tmp_path / "wheels",
        "demo-algorithm",
        requires=("demo-dependency==1.0.0",),
        plugin_name="demo_algorithm",
    )
    dependency = _write_wheel(tmp_path / "wheels", "demo-dependency")
    bundle = build_bundle(
        algorithm_wheel=algorithm,
        dependency_wheels=(dependency,),
        output=tmp_path / "bundle",
        algorithm_id="demo.algorithm",
        plugin_names=("demo_algorithm",),
        profile_ids=("cpu.test",),
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["network"] is False
    assert manifest["wheel"] == f"wheelhouse/{algorithm.name}"
    assert manifest["wheel_requires"]["demo-algorithm"] == ["demo-dependency==1.0.0"]
    assert manifest["wheel_entry_points"] == ["demo_algorithm"]
    assert (bundle / "algorithm.whl").read_bytes() == algorithm.read_bytes()
    assert hashlib.sha256((bundle / "algorithm.whl").read_bytes()).hexdigest() == next(
        item["sha256"] for item in manifest["files"] if item["path"] == "algorithm.whl"
    )

    prepared = prepare_algorithm_distribution(
        AlgorithmArtifact(
            source=str(bundle),
            mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
        ),
        _profile(),
    )
    assert prepared.package_name == "demo-algorithm"
    assert prepared.package_version == "1.0.0"


def test_builder_rejects_duplicate_distribution(tmp_path: Path) -> None:
    algorithm = _write_wheel(tmp_path / "wheels", "demo-algorithm")
    with pytest.raises(BundleBuildError, match="distinct"):
        build_bundle(
            algorithm_wheel=algorithm,
            dependency_wheels=(algorithm,),
            output=tmp_path / "bundle",
            algorithm_id="demo.algorithm",
        )


def test_builder_rejects_plugin_not_declared_by_wheel(tmp_path: Path) -> None:
    algorithm = _write_wheel(tmp_path / "wheels", "demo-algorithm")
    with pytest.raises(BundleBuildError, match="absent from algorithm Wheel"):
        build_bundle(
            algorithm_wheel=algorithm,
            dependency_wheels=(),
            output=tmp_path / "bundle",
            algorithm_id="demo.algorithm",
            plugin_names=("missing_plugin",),
        )


def test_builder_enforces_total_bundle_size(tmp_path: Path) -> None:
    algorithm = _write_wheel(tmp_path / "wheels", "demo-algorithm")
    with pytest.raises(BundleBuildError, match="max_bundle_size"):
        build_bundle(
            algorithm_wheel=algorithm,
            dependency_wheels=(),
            output=tmp_path / "bundle",
            algorithm_id="demo.algorithm",
            max_bundle_size=1,
        )

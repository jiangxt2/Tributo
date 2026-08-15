"""Build a digest-recorded offline algorithm Bundle from Wheels."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import shutil
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version


class BundleBuildError(ValueError):
    """The supplied Wheels cannot form a valid offline Bundle."""


def _wheel_metadata(path: Path) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    try:
        filename_name, filename_version, _build, _tags = parse_wheel_filename(path.name)
    except InvalidWheelFilename as exc:
        raise BundleBuildError(f"invalid Wheel filename: {path.name!r}") from exc
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise BundleBuildError(
                    f"Wheel must contain exactly one METADATA file: {path.name!r}"
                )
            metadata = Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8")
            )
            entry_point_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/entry_points.txt")
            ]
            entry_points: tuple[str, ...] = ()
            if len(entry_point_names) > 1:
                raise BundleBuildError(
                    f"Wheel must contain at most one entry_points.txt file: {path.name!r}"
                )
            if entry_point_names:
                parser = configparser.ConfigParser()
                parser.read_string(archive.read(entry_point_names[0]).decode("utf-8"))
                if parser.has_section("tributo.algorithms"):
                    entry_points = tuple(sorted(parser["tributo.algorithms"].keys()))
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, configparser.Error) as exc:
        raise BundleBuildError(f"cannot inspect Wheel: {path}") from exc
    package_name = canonicalize_name(metadata.get("Name", ""))
    try:
        package_version = str(Version(metadata.get("Version", "")))
    except InvalidVersion as exc:
        raise BundleBuildError(
            f"Wheel metadata has an invalid version: {path.name!r}"
        ) from exc
    if package_name != canonicalize_name(str(filename_name)):
        raise BundleBuildError(f"Wheel name disagrees with filename: {path.name!r}")
    if package_version != str(filename_version):
        raise BundleBuildError(f"Wheel version disagrees with filename: {path.name!r}")
    dependencies: list[str] = []
    for dependency in metadata.get_all("Requires-Dist", ()) or ():
        try:
            requirement = Requirement(dependency)
        except (InvalidRequirement, TypeError) as exc:
            raise BundleBuildError(
                f"Wheel has an invalid Requires-Dist entry: {path.name!r}"
            ) from exc
        if requirement.url is not None:
            raise BundleBuildError(
                f"Wheel has a URL Requires-Dist entry: {path.name!r}"
            )
        dependencies.append(str(requirement))
    return package_name, package_version, tuple(sorted(dependencies)), entry_points


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BundleBuildError(f"Bundle artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {
        "path": path.relative_to(root).as_posix(),
        "size": size,
        "sha256": digest.hexdigest(),
    }


def build_bundle(
    *,
    algorithm_wheel: Path,
    dependency_wheels: tuple[Path, ...],
    output: Path,
    algorithm_id: str,
    plugin_names: tuple[str, ...] = (),
    profile_ids: tuple[str, ...] = (),
    python_spec: str = ">=3.12,<3.14",
    ray_version: str = "2.55.1",
    tributo_version_spec: str = ">=1,<2",
    max_bundle_size: int = 4 * 1024 * 1024 * 1024,
) -> Path:
    """Build one Bundle and return its output directory."""
    sources = (algorithm_wheel, *dependency_wheels)
    if len({path.resolve() for path in sources}) != len(sources):
        raise BundleBuildError("algorithm and dependency Wheels must be distinct")
    if output.exists() and any(output.iterdir()):
        raise BundleBuildError(f"Bundle output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    wheelhouse = output / "wheelhouse"
    wheelhouse.mkdir()

    wheel_info: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...]]] = {}
    copied: list[Path] = []
    for source in sources:
        if not source.is_file() or source.suffix != ".whl":
            raise BundleBuildError(f"Wheel source is not a regular .whl file: {source}")
        info = _wheel_metadata(source)
        if info[0] in wheel_info:
            raise BundleBuildError(f"duplicate distribution in Bundle: {info[0]!r}")
        wheel_info[info[0]] = info
        destination = wheelhouse / source.name
        shutil.copy2(source, destination)
        copied.append(destination)

    (
        algorithm_name,
        algorithm_version,
        _algorithm_dependencies,
        algorithm_entry_points,
    ) = _wheel_metadata(algorithm_wheel)
    if algorithm_name not in wheel_info:
        raise BundleBuildError("algorithm Wheel was not copied into the Wheelhouse")
    algorithm_target = wheelhouse / algorithm_wheel.name
    shutil.copy2(algorithm_target, output / "algorithm.whl")
    if (
        _file_record(output, output / "algorithm.whl")["sha256"]
        != _file_record(output, algorithm_target)["sha256"]
    ):
        raise BundleBuildError("root algorithm.whl and Wheelhouse Wheel differ")
    if not set(plugin_names).issubset(algorithm_entry_points):
        missing = sorted(set(plugin_names) - set(algorithm_entry_points))
        raise BundleBuildError(
            f"plugin names are absent from algorithm Wheel entry points: {missing}"
        )
    requirements = output / "requirements.lock"
    requirements.write_text(
        "\n".join(
            (
                "--no-index",
                "--find-links ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/wheelhouse",
                f"${{RAY_RUNTIME_ENV_CREATE_WORKING_DIR}}/wheelhouse/{algorithm_wheel.name}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    files = [
        _file_record(output, output / "algorithm.whl"),
        _file_record(output, requirements),
    ]
    files.extend(_file_record(output, path) for path in sorted(copied))
    manifest = {
        "schema_version": 1,
        "algorithm_id": algorithm_id,
        "package_name": algorithm_name,
        "package_version": algorithm_version,
        "wheel": f"wheelhouse/{algorithm_wheel.name}",
        "requirements": "requirements.lock",
        "wheelhouse": "wheelhouse",
        "files": files,
        "plugin_names": list(plugin_names),
        "python_spec": python_spec,
        "ray_version": ray_version,
        "tributo_version_spec": tributo_version_spec,
        "profile_ids": list(profile_ids),
        "wheel_entry_points": list(algorithm_entry_points),
        "network": False,
        "requirements_entries": [
            "--no-index",
            "--find-links ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/wheelhouse",
            f"${{RAY_RUNTIME_ENV_CREATE_WORKING_DIR}}/wheelhouse/{algorithm_wheel.name}",
        ],
        "wheel_requires": {
            package: list(info[2]) for package, info in sorted(wheel_info.items())
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle_size = (
        sum(item["size"] for item in files) + (output / "manifest.json").stat().st_size
    )
    if bundle_size > max_bundle_size:
        raise BundleBuildError(
            f"Bundle size {bundle_size} exceeds max_bundle_size={max_bundle_size}"
        )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm-wheel", type=Path, required=True)
    parser.add_argument("--dependency-wheel", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm-id", required=True)
    parser.add_argument("--plugin-name", action="append", default=[])
    parser.add_argument("--profile-id", action="append", default=[])
    parser.add_argument("--python-spec", default=">=3.12,<3.14")
    parser.add_argument("--ray-version", default="2.55.1")
    parser.add_argument("--tributo-version-spec", default=">=1,<2")
    parser.add_argument("--max-bundle-size", type=int, default=4 * 1024 * 1024 * 1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_bundle(
        algorithm_wheel=args.algorithm_wheel,
        dependency_wheels=tuple(args.dependency_wheel),
        output=args.output,
        algorithm_id=args.algorithm_id,
        plugin_names=tuple(args.plugin_name),
        profile_ids=tuple(args.profile_id),
        python_spec=args.python_spec,
        ray_version=args.ray_version,
        tributo_version_spec=args.tributo_version_spec,
        max_bundle_size=args.max_bundle_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

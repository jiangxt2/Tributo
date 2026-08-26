"""Fail-closed validation and Ray runtime wiring for user algorithm artifacts."""

from __future__ import annotations

import configparser
import hashlib
import importlib.metadata
import json
import logging
import sys
import zipfile
from collections import deque
from dataclasses import dataclass, replace
from email.parser import Parser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import Tag, parse_tag
from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

from tributo.algorithms.api.artifacts import (
    AlgorithmArtifact,
    AlgorithmBundleManifest,
    AlgorithmDistributionReceipt,
    ArtifactDistributionMode,
    ImageProfile,
)
from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import DeveloperAPI

logger = logging.getLogger(__name__)

SUPPORTED_RAY_VERSION = "2.55.1"
DEFAULT_MAX_BUNDLE_SIZE = 4 * 1024 * 1024 * 1024
_RUNTIME_WORKING_DIR = "${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}"
_ALLOWED_PIP_INSTALL_OPTIONS = (
    "--disable-pip-version-check",
    "--no-cache-dir",
)


@dataclass(frozen=True)
class _WheelMetadata:
    path: Path | None
    package_name: str
    package_version: str
    requires_dist: tuple[str, ...]
    entry_points: tuple[str, ...]
    tags: frozenset[Tag]
    digest: str
    size: int


@dataclass(frozen=True)
class PreparedAlgorithmDistribution:
    """Internal result shared by preflight and runtime_env construction."""

    artifact: AlgorithmArtifact
    receipt: AlgorithmDistributionReceipt
    package_name: str
    package_version: str
    plugin_names: tuple[str, ...]
    requirements_name: str | None = None
    wheelhouse_name: str | None = None
    pip_check: bool = True
    dependency_names: tuple[str, ...] = ()


def _configuration_error(message: str) -> JobConfigurationError:
    return JobConfigurationError(f"algorithm artifact preflight failed: {message}")


def _hash_file(path: Path, *, max_size: int) -> tuple[str, int]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _configuration_error(f"cannot stat artifact file {path}") from exc
    if not path.is_file():
        raise _configuration_error(f"artifact path is not a regular file: {path}")
    if size > max_size:
        raise _configuration_error(
            f"artifact file {path.name!r} exceeds max_file_size={max_size}"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise _configuration_error(f"cannot read artifact file {path}") from exc
    return digest.hexdigest(), size


def _canonical_bundle_digest(
    manifest_digest: str,
    files: tuple[tuple[str, str, int], ...],
) -> str:
    payload = {
        "manifest_sha256": manifest_digest,
        "files": [
            {"path": path, "sha256": digest, "size": size}
            for path, digest, size in files
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_member_name(name: str) -> None:
    if name.startswith("/") or "\\" in name:
        raise _configuration_error(f"Wheel contains unsafe member path: {name!r}")
    if any(part == ".." for part in Path(name).parts):
        raise _configuration_error(f"Wheel contains traversal member path: {name!r}")


def _wheel_metadata(path: Path, *, max_size: int) -> _WheelMetadata:
    digest, size = _hash_file(path, max_size=max_size)
    if path.suffix != ".whl":
        raise _configuration_error(
            f"algorithm artifact is not a .whl file: {path.name}"
        )
    try:
        filename_name, filename_version, _build, tags = parse_wheel_filename(path.name)
    except InvalidWheelFilename as exc:
        raise _configuration_error(f"invalid Wheel filename: {path.name!r}") from exc

    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            for member in members:
                _validate_member_name(member)
            metadata_members = [
                member for member in members if member.endswith(".dist-info/METADATA")
            ]
            wheel_members = [
                member for member in members if member.endswith(".dist-info/WHEEL")
            ]
            entry_point_members = [
                member
                for member in members
                if member.endswith(".dist-info/entry_points.txt")
            ]
            if len(metadata_members) != 1 or len(wheel_members) != 1:
                raise _configuration_error(
                    f"Wheel {path.name!r} must contain exactly one METADATA and WHEEL"
                )
            metadata = Parser().parsestr(
                archive.read(metadata_members[0]).decode("utf-8")
            )
            package_name = canonicalize_name(metadata.get("Name", ""))
            package_version = str(Version(metadata.get("Version", "")))
            if not package_name or not package_version:
                raise _configuration_error(
                    f"Wheel {path.name!r} has incomplete Core Metadata"
                )
            if package_name != canonicalize_name(str(filename_name)):
                raise _configuration_error(
                    f"Wheel metadata name {package_name!r} disagrees with filename"
                )
            if package_version != str(filename_version):
                raise _configuration_error(
                    f"Wheel metadata version {package_version!r} disagrees with filename"
                )
            requires_dist = tuple(metadata.get_all("Requires-Dist", ()) or ())
            for dependency in requires_dist:
                try:
                    requirement = Requirement(dependency)
                except (InvalidRequirement, TypeError) as exc:
                    raise _configuration_error(
                        f"Wheel {path.name!r} contains invalid Requires-Dist"
                    ) from exc
                if requirement.url is not None:
                    raise _configuration_error(
                        f"Wheel {path.name!r} contains a URL Requires-Dist"
                    )
            wheel_text = archive.read(wheel_members[0]).decode("utf-8")
            wheel_tags = set().union(
                *(
                    parse_tag(line.partition(":")[2].strip())
                    for line in wheel_text.splitlines()
                    if line.startswith("Tag:")
                )
            )
            if wheel_tags and not wheel_tags.intersection(tags):
                raise _configuration_error(
                    f"Wheel {path.name!r} has inconsistent WHEEL tags"
                )
            entry_points: list[str] = []
            if entry_point_members:
                parser = configparser.ConfigParser()
                parser.read_string(archive.read(entry_point_members[0]).decode("utf-8"))
                if parser.has_section("tributo.algorithms"):
                    entry_points = sorted(parser["tributo.algorithms"].keys())
    except zipfile.BadZipFile as exc:
        raise _configuration_error(
            f"Wheel is not a valid ZIP archive: {path.name}"
        ) from exc
    except configparser.Error as exc:
        raise _configuration_error(
            f"Wheel entry-point metadata is invalid: {path.name}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise _configuration_error(f"Wheel metadata is not UTF-8: {path.name}") from exc
    except (InvalidVersion, ValueError) as exc:
        raise _configuration_error(f"Wheel metadata is invalid: {path.name}") from exc

    return _WheelMetadata(
        path=path,
        package_name=package_name,
        package_version=package_version,
        requires_dist=tuple(sorted(requires_dist)),
        entry_points=tuple(entry_points),
        tags=frozenset(tags),
        digest=digest,
        size=size,
    )


def _remote_wheel_metadata(artifact: AlgorithmArtifact) -> _WheelMetadata:
    if artifact.package_name is None or artifact.package_version is None:
        raise _configuration_error(
            "remote artifact metadata requires package_name and package_version"
        )
    if artifact.sha256 is None:
        raise _configuration_error("remote artifact metadata requires sha256")
    if not artifact.wheel_tags:
        raise _configuration_error(
            "remote py_modules artifacts require immutable Wheel tags"
        )
    try:
        tags = frozenset(
            tag for tag_text in artifact.wheel_tags for tag in parse_tag(tag_text)
        )
    except ValueError as exc:
        raise _configuration_error("remote artifact Wheel tags are invalid") from exc
    return _WheelMetadata(
        path=None,
        package_name=artifact.package_name,
        package_version=artifact.package_version,
        requires_dist=artifact.requires_dist,
        entry_points=artifact.plugin_names,
        tags=tags,
        digest=artifact.sha256,
        size=0,
    )


def _profile_version_check(profile: ImageProfile) -> None:
    if Version(profile.ray_version) != Version(SUPPORTED_RAY_VERSION):
        raise _configuration_error(
            f"image Profile Ray version {profile.ray_version!r} is not supported; "
            f"expected {SUPPORTED_RAY_VERSION!r}"
        )
    current_python = Version(
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if current_python not in SpecifierSet(profile.python_spec):
        raise _configuration_error(
            f"submitting Python {current_python} does not satisfy image Profile "
            f"constraint {profile.python_spec!r}"
        )
    target_python = Version(profile.python_version)
    if target_python not in SpecifierSet(profile.python_spec):
        raise _configuration_error(
            f"image Profile target Python {target_python} does not satisfy its "
            f"constraint {profile.python_spec!r}"
        )


def _current_tributo_version() -> Version:
    try:
        return Version(importlib.metadata.version("tributo"))
    except importlib.metadata.PackageNotFoundError:
        # Source checkouts may not have an installed distribution.  The root
        # package has the same source-checkout fallback as the public API.
        try:
            import tributo

            return Version(tributo.__version__)
        except (ImportError, InvalidVersion) as exc:
            raise _configuration_error(
                "cannot determine the running Tributo version"
            ) from exc


def _validate_bundle_compatibility(
    manifest: AlgorithmBundleManifest,
    profile: ImageProfile,
) -> None:
    if manifest.profile_ids and profile.profile_id not in manifest.profile_ids:
        raise _configuration_error(
            f"Bundle does not support image Profile {profile.profile_id!r}"
        )
    if Version(profile.ray_version) != Version(manifest.ray_version):
        raise _configuration_error("Bundle and image Profile Ray versions disagree")
    if profile.algorithm_ids and manifest.algorithm_id not in profile.algorithm_ids:
        raise _configuration_error(
            f"Bundle algorithm {manifest.algorithm_id!r} is not allowed by image "
            f"Profile {profile.profile_id!r}"
        )
    if _current_tributo_version() not in SpecifierSet(manifest.tributo_version_spec):
        raise _configuration_error(
            "Bundle Tributo version constraint excludes the running Tributo version"
        )
    current_python = Version(
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if not SpecifierSet(manifest.python_spec).contains(
        current_python,
        prereleases=True,
    ):
        raise _configuration_error(
            "Bundle Python constraint excludes the submitting Python"
        )


def _normalized_requirements(dependencies: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for dependency in dependencies:
        try:
            requirement = Requirement(dependency)
        except (InvalidRequirement, TypeError) as exc:
            raise _configuration_error(
                f"invalid Wheel dependency requirement: {dependency!r}"
            ) from exc
        if requirement.url is not None:
            raise _configuration_error(
                "Wheel dependencies must not contain URL requirements"
            )
        normalized.append(str(requirement))
    return tuple(sorted(normalized))


def _profile_distribution(
    profile: ImageProfile,
    requirement: Requirement,
) -> bool:
    version = profile.installed_distributions.get(canonicalize_name(requirement.name))
    if version is None:
        return False
    return Version(version) in requirement.specifier


def _local_source_path(artifact: AlgorithmArtifact) -> Path | None:
    parsed = urlsplit(artifact.source)
    if parsed.scheme in {"https", "s3"}:
        return None
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser().resolve()
    return Path(artifact.source).expanduser().resolve()


def _check_wheel_tags(metadata: _WheelMetadata, profile: ImageProfile) -> None:
    if not profile.wheel_tags:
        raise _configuration_error(
            f"image Profile {profile.profile_id!r} must declare supported Wheel tags"
        )
    if not metadata.tags:
        raise _configuration_error(
            f"Wheel {metadata.package_name!r} does not declare any platform tags"
        )
    supported = set(profile.wheel_tags)
    actual = {str(tag) for tag in metadata.tags}
    if not supported.intersection(actual):
        raise _configuration_error(
            f"Wheel {metadata.package_name!r} has no tag compatible with "
            f"image Profile {profile.profile_id!r}"
        )


def _validate_plugin_selection(
    selected: tuple[str, ...],
    available: tuple[str, ...],
    *,
    context: str,
) -> None:
    missing = sorted(set(selected) - set(available))
    if missing:
        raise _configuration_error(
            f"{context} plugin_names are absent from Wheel entry points: {missing}"
        )


def _validate_declared_requirement(
    dependency: str,
    *,
    local_wheels: dict[str, _WheelMetadata],
    profile: ImageProfile,
    context: str,
) -> _WheelMetadata | None:
    try:
        requirement = Requirement(dependency)
    except (InvalidRequirement, TypeError) as exc:
        raise _configuration_error(
            f"invalid {context} requirement: {dependency!r}"
        ) from exc
    if requirement.url is not None:
        raise _configuration_error(f"{context} must not contain URL requirements")
    if requirement.marker is not None and not requirement.marker.evaluate(
        _profile_marker_environment(profile)
    ):
        return None
    name = canonicalize_name(requirement.name)
    local = local_wheels.get(name)
    if local is not None:
        if Version(local.package_version) not in requirement.specifier:
            raise _configuration_error(
                f"local Wheel {local.package_name}=={local.package_version} does not "
                f"satisfy {dependency!r}"
            )
        return local
    if _profile_distribution(profile, requirement):
        return None
    raise _configuration_error(
        f"{context} dependency {dependency!r} is absent from the image Profile "
        "and local Wheelhouse"
    )


def _profile_marker_environment(profile: ImageProfile) -> dict[str, str]:
    """Evaluate PEP 508 markers for the selected image, never the submitter."""
    environment = {key: str(value) for key, value in default_environment().items()}
    target_python = Version(profile.python_version)
    environment.update(
        {
            "python_version": f"{target_python.major}.{target_python.minor}",
            "python_full_version": str(target_python),
            "sys_platform": profile.sys_platform,
            "platform_machine": profile.platform_machine,
            "platform_python_implementation": "CPython",
            "implementation_name": "cpython",
        }
    )
    if profile.sys_platform == "linux":
        environment.update({"platform_system": "Linux", "os_name": "posix"})
    elif profile.sys_platform == "darwin":
        environment.update({"platform_system": "Darwin", "os_name": "posix"})
    elif profile.sys_platform == "win32":
        environment.update({"platform_system": "Windows", "os_name": "nt"})
    return environment


def _resolve_requirement_wheel(
    line: str,
    root: Path,
    wheelhouse: Path,
    *,
    max_size: int,
) -> _WheelMetadata | None:
    if not line.lower().endswith(".whl"):
        return None
    if line.startswith(_RUNTIME_WORKING_DIR):
        relative = line[len(_RUNTIME_WORKING_DIR) :].lstrip("/")
        candidate = (root / relative).resolve()
    else:
        candidate = (root / line).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise _configuration_error(
            f"requirements Wheel path escapes the Bundle: {line!r}"
        ) from exc
    try:
        candidate.relative_to(wheelhouse.resolve())
    except ValueError as exc:
        raise _configuration_error(
            f"requirements Wheel path is outside the Wheelhouse: {line!r}"
        ) from exc
    return _wheel_metadata(candidate, max_size=max_size)


def _parse_requirements(
    path: Path,
    root: Path,
    wheelhouse: Path,
    *,
    max_size: int,
) -> tuple[tuple[str, ...], tuple[_WheelMetadata, ...]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise _configuration_error(f"cannot read requirements file {path}") from exc
    requirements: list[str] = []
    wheel_entries: list[_WheelMetadata] = []
    has_no_index = False
    expected_find_links = (
        f"{_RUNTIME_WORKING_DIR}/{wheelhouse.relative_to(root).as_posix()}"
    )
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "--no-index":
            has_no_index = True
            continue
        if line.startswith("--find-links"):
            _, separator, value = line.partition("=")
            if not separator:
                parts = line.split(None, 1)
                value = parts[1].strip() if len(parts) == 2 else ""
            if value != expected_find_links:
                raise _configuration_error(
                    "requirements --find-links must point to the Bundle Wheelhouse"
                )
            continue
        if line.startswith(
            ("-r", "-c", "--requirement", "--constraint")
        ) or line.startswith(("-e", "--editable")):
            raise _configuration_error(
                "nested requirements, constraints, and editable installs are forbidden"
            )
        if line.startswith("-"):
            raise _configuration_error(f"pip option is not allow-listed: {line!r}")
        if any(
            scheme in line.lower()
            for scheme in ("http://", "https://", "git+", "svn+", "hg+")
        ):
            raise _configuration_error(
                "requirements must not contain remote URLs or VCS references"
            )
        wheel = _resolve_requirement_wheel(
            line,
            root,
            wheelhouse,
            max_size=max_size,
        )
        if wheel is not None:
            wheel_entries.append(wheel)
            continue
        try:
            requirement = Requirement(line)
        except (InvalidRequirement, TypeError) as exc:
            raise _configuration_error(
                f"invalid requirements.lock entry: {line!r}"
            ) from exc
        if requirement.url is not None:
            raise _configuration_error(
                "requirements must not contain direct URL requirements"
            )
        requirements.append(str(requirement))
    if not has_no_index:
        raise _configuration_error("requirements.lock must contain --no-index")
    return tuple(requirements), tuple(wheel_entries)


def _validate_dependency_closure(
    metadata: _WheelMetadata,
    local_wheels: dict[str, _WheelMetadata],
    requirements: tuple[str, ...],
    wheel_entries: tuple[_WheelMetadata, ...],
    declared_dependencies: tuple[str, ...],
    profile: ImageProfile,
) -> None:
    """Validate every dependency reachable from the algorithm and lock file."""
    requested_names: set[str] = set()
    for wheel in wheel_entries:
        requested_names.add(wheel.package_name)

    for dependency in requirements:
        try:
            requested_names.add(canonicalize_name(Requirement(dependency).name))
        except (InvalidRequirement, TypeError) as exc:
            raise _configuration_error(
                f"invalid requirements.lock entry: {dependency!r}"
            ) from exc

    if metadata.package_name not in requested_names:
        raise _configuration_error(
            "requirements.lock must install the algorithm Wheel explicitly"
        )

    queue: deque[tuple[str, str]] = deque()
    queue.append((metadata.package_name, "algorithm Wheel"))
    for dependency in declared_dependencies:
        selected = _validate_declared_requirement(
            dependency,
            local_wheels=local_wheels,
            profile=profile,
            context="EnvironmentSpec",
        )
        if selected is not None:
            queue.append((selected.package_name, "EnvironmentSpec"))
    for dependency in requirements:
        selected = _validate_declared_requirement(
            dependency,
            local_wheels=local_wheels,
            profile=profile,
            context="requirements.lock",
        )
        if selected is not None:
            queue.append((selected.package_name, "requirements.lock"))

    visited: set[str] = set()
    while queue:
        package_name, context = queue.popleft()
        if package_name in visited:
            continue
        visited.add(package_name)
        selected_wheel = local_wheels.get(package_name)
        if selected_wheel is None:
            continue
        for dependency in selected_wheel.requires_dist:
            requirement = Requirement(dependency)
            if requirement.marker is not None and not requirement.marker.evaluate(
                _profile_marker_environment(profile)
            ):
                continue
            selected = _validate_declared_requirement(
                str(requirement),
                local_wheels=local_wheels,
                profile=profile,
                context=f"{context} dependency",
            )
            if selected is not None:
                queue.append((selected.package_name, context))


def _load_manifest(
    artifact: AlgorithmArtifact,
    root: Path,
) -> tuple[AlgorithmBundleManifest, str, tuple[tuple[str, str, int], ...]]:
    manifest_path = root / artifact.manifest_name
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = AlgorithmBundleManifest.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise _configuration_error(f"invalid Bundle manifest {manifest_path}") from exc
    if manifest.requirements != artifact.requirements_name:
        raise _configuration_error(
            "manifest requirements path is inconsistent with artifact configuration"
        )
    if manifest.wheelhouse != artifact.wheelhouse_name:
        raise _configuration_error(
            "manifest Wheelhouse path is inconsistent with artifact configuration"
        )
    if artifact.plugin_names and tuple(artifact.plugin_names) != tuple(
        manifest.plugin_names
    ):
        raise _configuration_error("artifact and manifest plugin_names disagree")
    if (
        artifact.package_name
        and canonicalize_name(artifact.package_name) != manifest.package_name
    ):
        raise _configuration_error("artifact and manifest package_name disagree")
    if (
        artifact.package_version
        and str(Version(artifact.package_version)) != manifest.package_version
    ):
        raise _configuration_error("artifact and manifest package_version disagree")
    manifest_files: list[tuple[str, str, int]] = []
    for item in manifest.files:
        path = (root / item.path).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise _configuration_error(
                f"manifest path escapes Bundle: {item.path!r}"
            ) from exc
        digest, size = _hash_file(path, max_size=artifact.max_file_size)
        if size != item.size or digest != item.sha256:
            raise _configuration_error(
                f"manifest digest or size mismatch: {item.path!r}"
            )
        manifest_files.append((item.path, digest, size))
    recorded_paths = {path for path, _digest, _size in manifest_files}
    actual_paths: set[str] = set()
    try:
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise _configuration_error(
                    f"Bundle must not contain symbolic links: {candidate}"
                )
            if candidate.is_file():
                actual_paths.add(candidate.relative_to(root).as_posix())
    except OSError as exc:
        raise _configuration_error(
            f"cannot enumerate Bundle files under {root}"
        ) from exc
    expected_paths = recorded_paths | {artifact.manifest_name}
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise _configuration_error(
            f"Bundle file inventory disagrees with manifest (missing={missing}, "
            f"extra={extra})"
        )
    manifest_digest, _manifest_size = _hash_file(
        manifest_path,
        max_size=artifact.max_file_size,
    )
    total_size = sum(size for _path, _digest, size in manifest_files) + _manifest_size
    if total_size > artifact.max_bundle_size:
        raise _configuration_error(
            f"Bundle size {total_size} exceeds max_bundle_size={artifact.max_bundle_size}"
        )
    return (
        manifest,
        _canonical_bundle_digest(manifest_digest, tuple(manifest_files)),
        tuple(manifest_files),
    )


def _remote_manifest(artifact: AlgorithmArtifact) -> AlgorithmBundleManifest:
    if artifact.manifest is None:
        raise _configuration_error(
            "remote offline Wheelhouse artifacts require an attested manifest"
        )
    try:
        manifest = AlgorithmBundleManifest.model_validate(artifact.manifest)
    except (TypeError, ValueError) as exc:
        raise _configuration_error("remote Bundle manifest is invalid") from exc
    if manifest.requirements != artifact.requirements_name:
        raise _configuration_error(
            "manifest requirements path is inconsistent with artifact configuration"
        )
    if manifest.wheelhouse != artifact.wheelhouse_name:
        raise _configuration_error(
            "manifest Wheelhouse path is inconsistent with artifact configuration"
        )
    if artifact.package_name is None or artifact.package_version is None:
        raise _configuration_error(
            "remote offline metadata requires package_name and package_version"
        )
    if artifact.package_name != manifest.package_name:
        raise _configuration_error("artifact and manifest package_name disagree")
    if artifact.package_version != manifest.package_version:
        raise _configuration_error("artifact and manifest package_version disagree")
    if artifact.plugin_names and tuple(artifact.plugin_names) != tuple(
        manifest.plugin_names
    ):
        raise _configuration_error("artifact and manifest plugin_names disagree")
    total_size = 0
    for item in manifest.files:
        if item.size > artifact.max_file_size:
            raise _configuration_error(
                f"remote Bundle file {item.path!r} exceeds max_file_size="
                f"{artifact.max_file_size}"
            )
        total_size += item.size
    if total_size > artifact.max_bundle_size:
        raise _configuration_error(
            f"remote Bundle size {total_size} exceeds "
            f"max_bundle_size={artifact.max_bundle_size}"
        )
    return manifest


def _remote_manifest_wheel(
    path: str,
    item: Any,
    manifest: AlgorithmBundleManifest,
    *,
    entry_points: tuple[str, ...] = (),
) -> _WheelMetadata:
    try:
        filename_name, filename_version, _build, tags = parse_wheel_filename(
            Path(path).name
        )
    except InvalidWheelFilename as exc:
        raise _configuration_error(f"invalid remote Wheel filename: {path!r}") from exc
    package_name = canonicalize_name(str(filename_name))
    package_version = str(filename_version)
    dependencies = manifest.wheel_requires.get(package_name)
    if dependencies is None:
        raise _configuration_error(
            f"remote manifest lacks Requires-Dist attestation for {package_name!r}"
        )
    return _WheelMetadata(
        path=None,
        package_name=package_name,
        package_version=package_version,
        requires_dist=tuple(dependencies),
        entry_points=entry_points,
        tags=frozenset(tags),
        digest=item.sha256,
        size=item.size,
    )


def _remote_requirement_wheel(
    line: str,
    manifest: AlgorithmBundleManifest,
    wheels_by_path: dict[str, _WheelMetadata],
) -> _WheelMetadata | None:
    if not line.lower().endswith(".whl"):
        return None
    if line.startswith(_RUNTIME_WORKING_DIR):
        relative = line[len(_RUNTIME_WORKING_DIR) :].lstrip("/")
    else:
        relative = line
    if (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or any(part == ".." for part in Path(relative).parts)
    ):
        raise _configuration_error(
            f"remote requirements Wheel path is unsafe: {line!r}"
        )
    wheelhouse_prefix = manifest.wheelhouse.rstrip("/") + "/"
    if not relative.startswith(wheelhouse_prefix):
        raise _configuration_error(
            f"requirements Wheel path is outside the Bundle Wheelhouse: {line!r}"
        )
    wheel = wheels_by_path.get(relative)
    if wheel is None:
        raise _configuration_error(
            f"requirements Wheel is not recorded in the remote manifest: {relative!r}"
        )
    return wheel


def _parse_remote_requirements(
    manifest: AlgorithmBundleManifest,
    wheels_by_path: dict[str, _WheelMetadata],
) -> tuple[tuple[str, ...], tuple[_WheelMetadata, ...]]:
    if not manifest.requirements_entries:
        raise _configuration_error(
            "remote Bundle manifest must attest requirements_entries"
        )
    requirements: list[str] = []
    wheel_entries: list[_WheelMetadata] = []
    has_no_index = False
    expected_find_links = f"{_RUNTIME_WORKING_DIR}/{manifest.wheelhouse}"
    for raw_line in manifest.requirements_entries:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "--no-index":
            has_no_index = True
            continue
        if line.startswith("--find-links"):
            _, separator, value = line.partition("=")
            if not separator:
                parts = line.split(None, 1)
                value = parts[1].strip() if len(parts) == 2 else ""
            if value != expected_find_links:
                raise _configuration_error(
                    "remote requirements --find-links must point to the Bundle Wheelhouse"
                )
            continue
        if line.startswith(
            ("-r", "-c", "--requirement", "--constraint")
        ) or line.startswith(("-e", "--editable")):
            raise _configuration_error(
                "nested requirements, constraints, and editable installs are forbidden"
            )
        if line.startswith("-"):
            raise _configuration_error(f"pip option is not allow-listed: {line!r}")
        if any(
            scheme in line.lower()
            for scheme in ("http://", "https://", "git+", "svn+", "hg+")
        ):
            raise _configuration_error(
                "requirements must not contain remote URLs or VCS references"
            )
        wheel = _remote_requirement_wheel(line, manifest, wheels_by_path)
        if wheel is not None:
            wheel_entries.append(wheel)
            continue
        try:
            requirement = Requirement(line)
        except (InvalidRequirement, TypeError) as exc:
            raise _configuration_error(
                f"invalid remote requirements.lock entry: {line!r}"
            ) from exc
        if requirement.url is not None:
            raise _configuration_error(
                "requirements must not contain direct URL requirements"
            )
        requirements.append(str(requirement))
    if not has_no_index:
        raise _configuration_error("remote requirements must contain --no-index")
    return tuple(requirements), tuple(wheel_entries)


def _validate_remote_offline_bundle(
    artifact: AlgorithmArtifact,
    profile: ImageProfile,
    declared_dependencies: tuple[str, ...],
) -> tuple[_WheelMetadata, str, tuple[str, ...], str, str]:
    manifest = _remote_manifest(artifact)
    _validate_bundle_compatibility(manifest, profile)
    file_records = {item.path: item for item in manifest.files}
    root_record = file_records.get(manifest.wheel)
    if root_record is None:
        raise _configuration_error(
            "remote manifest does not record the algorithm Wheel"
        )
    wheelhouse_prefix = manifest.wheelhouse.rstrip("/") + "/"
    wheels_by_path: dict[str, _WheelMetadata] = {}
    wheelhouse_metadata: dict[str, _WheelMetadata] = {}
    for path, item in file_records.items():
        if not path.startswith(wheelhouse_prefix) or not path.endswith(".whl"):
            continue
        wheel = _remote_manifest_wheel(path, item, manifest)
        _check_wheel_tags(wheel, profile)
        if wheel.package_name in wheelhouse_metadata:
            raise _configuration_error(
                f"remote Wheelhouse contains duplicate distribution {wheel.package_name!r}"
            )
        wheels_by_path[path] = wheel
        wheelhouse_metadata[wheel.package_name] = wheel
    if not wheelhouse_metadata:
        raise _configuration_error("remote manifest records no Wheelhouse Wheels")

    try:
        root_wheel = _remote_manifest_wheel(
            manifest.wheel,
            root_record,
            manifest,
            entry_points=tuple(
                manifest.wheel_entry_points
                or artifact.plugin_names
                or manifest.plugin_names
            ),
        )
    except JobConfigurationError:
        # The documented Bundle layout may use a friendly root alias such as
        # ``algorithm.whl`` while the Wheelhouse contains the real PEP 427
        # filename.  The duplicate is still required to have the same digest.
        if manifest.wheel != "algorithm.whl":
            raise
        candidates = [
            wheel
            for wheel in wheelhouse_metadata.values()
            if wheel.package_name == manifest.package_name
            and wheel.package_version == manifest.package_version
        ]
        if len(candidates) != 1:
            raise _configuration_error(
                "remote manifest algorithm.whl alias does not identify one Wheelhouse Wheel"
            ) from None
        candidate = candidates[0]
        if candidate.digest != root_record.sha256:
            raise _configuration_error(
                "remote root Wheel and Wheelhouse Wheel have different digests"
            ) from None
        root_wheel = replace(
            candidate,
            entry_points=tuple(
                manifest.wheel_entry_points
                or artifact.plugin_names
                or manifest.plugin_names
            ),
            digest=root_record.sha256,
            size=root_record.size,
        )
    if root_wheel.package_name != manifest.package_name:
        raise _configuration_error(
            "remote manifest package identity does not match algorithm Wheel"
        )
    if root_wheel.package_version != manifest.package_version:
        raise _configuration_error(
            "remote manifest package version does not match algorithm Wheel"
        )
    _check_wheel_tags(root_wheel, profile)
    if artifact.requires_dist and _normalized_requirements(artifact.requires_dist) != (
        root_wheel.requires_dist
    ):
        raise _configuration_error(
            "artifact requires_dist disagrees with remote manifest wheel_requires"
        )
    wheelhouse_root = wheelhouse_metadata.get(root_wheel.package_name)
    if wheelhouse_root is None:
        raise _configuration_error("remote Wheelhouse must contain the algorithm Wheel")
    if wheelhouse_root.digest != root_wheel.digest:
        raise _configuration_error(
            "remote root Wheel and Wheelhouse Wheel have different digests"
        )
    root_alias = file_records.get("algorithm.whl")
    if root_alias is not None and manifest.wheel != "algorithm.whl":
        if root_alias.sha256 != root_wheel.digest or root_alias.size != root_wheel.size:
            raise _configuration_error(
                "remote root algorithm.whl and Wheelhouse Wheel have different digests"
            )
    wheelhouse_metadata[root_wheel.package_name] = root_wheel
    local_wheels = dict(wheelhouse_metadata)
    expected_wheel_names = set(local_wheels)
    if set(manifest.wheel_requires) != expected_wheel_names:
        raise _configuration_error(
            "remote manifest wheel_requires must cover exactly every Wheelhouse distribution"
        )

    requirements, wheel_entries = _parse_remote_requirements(
        manifest,
        wheels_by_path,
    )
    available_plugins = tuple(
        manifest.wheel_entry_points or artifact.plugin_names or manifest.plugin_names
    )
    plugin_names = tuple(
        artifact.plugin_names or manifest.plugin_names or manifest.wheel_entry_points
    )
    _validate_plugin_selection(
        plugin_names,
        available_plugins,
        context="remote manifest",
    )
    if len(plugin_names) > 1:
        raise _configuration_error(
            "one Job may activate only one algorithm entry point"
        )
    _validate_dependency_closure(
        root_wheel,
        local_wheels,
        requirements,
        wheel_entries,
        declared_dependencies,
        profile,
    )
    if artifact.sha256 is None:
        raise _configuration_error("remote offline Bundle requires sha256")
    return (
        root_wheel,
        artifact.sha256,
        plugin_names,
        manifest.requirements,
        manifest.wheelhouse,
    )


def _validate_offline_bundle(
    artifact: AlgorithmArtifact,
    profile: ImageProfile,
    declared_dependencies: tuple[str, ...],
) -> tuple[_WheelMetadata, str, tuple[str, ...], str, str]:
    root = _local_source_path(artifact)
    if root is None or not root.is_dir():
        raise _configuration_error(
            "offline Wheelhouse source must be an existing directory"
        )
    manifest, bundle_digest, manifest_files = _load_manifest(artifact, root)
    _validate_bundle_compatibility(manifest, profile)
    wheel_path = (root / manifest.wheel).resolve()
    requirements_path = (root / manifest.requirements).resolve()
    wheelhouse = (root / manifest.wheelhouse).resolve()
    if not wheelhouse.is_dir():
        raise _configuration_error("Bundle Wheelhouse directory does not exist")
    try:
        wheel_path.relative_to(root.resolve())
        requirements_path.relative_to(root.resolve())
        wheelhouse.relative_to(root.resolve())
    except ValueError as exc:
        raise _configuration_error(
            "Bundle paths must remain inside the Bundle root"
        ) from exc
    metadata = _wheel_metadata(wheel_path, max_size=artifact.max_file_size)
    if (
        metadata.package_name != manifest.package_name
        or metadata.package_version != manifest.package_version
    ):
        raise _configuration_error(
            "manifest package identity does not match algorithm Wheel"
        )
    manifest_file_records = {
        path: (digest, size) for path, digest, size in manifest_files
    }
    root_alias = manifest_file_records.get("algorithm.whl")
    if root_alias is not None and manifest.wheel != "algorithm.whl":
        if root_alias[0] != metadata.digest or root_alias[1] != metadata.size:
            raise _configuration_error(
                "local root algorithm.whl and Wheelhouse Wheel have different digests"
            )
    _check_wheel_tags(metadata, profile)
    wheelhouse_metadata: dict[str, _WheelMetadata] = {}
    for candidate in sorted(wheelhouse.glob("*.whl")):
        wheel = _wheel_metadata(candidate, max_size=artifact.max_file_size)
        if wheel.package_name in wheelhouse_metadata:
            raise _configuration_error(
                f"Wheelhouse contains duplicate distribution {wheel.package_name!r}"
            )
        _check_wheel_tags(wheel, profile)
        wheelhouse_metadata[wheel.package_name] = wheel
    requirements, wheel_entries = _parse_requirements(
        requirements_path,
        root,
        wheelhouse,
        max_size=artifact.max_file_size,
    )
    for wheel in wheel_entries:
        existing = wheelhouse_metadata.get(wheel.package_name)
        if existing is not None and existing.digest != wheel.digest:
            raise _configuration_error(
                f"requirements Wheel disagrees with Wheelhouse index: {wheel.package_name!r}"
            )
        wheelhouse_metadata[wheel.package_name] = wheel
    local_wheels = dict(wheelhouse_metadata)
    wheelhouse_algorithm = wheelhouse_metadata.get(metadata.package_name)
    if wheelhouse_algorithm is None:
        raise _configuration_error("Bundle Wheelhouse must contain the algorithm Wheel")
    if wheelhouse_algorithm.digest != metadata.digest:
        raise _configuration_error(
            "local root algorithm.whl and Wheelhouse Wheel have different digests"
        )
    local_wheels[metadata.package_name] = metadata

    if manifest.wheel_entry_points and set(manifest.wheel_entry_points) != set(
        metadata.entry_points
    ):
        raise _configuration_error(
            "manifest wheel_entry_points disagree with Wheel entry points"
        )

    _validate_dependency_closure(
        metadata,
        local_wheels,
        requirements,
        wheel_entries,
        declared_dependencies,
        profile,
    )
    plugin_names = tuple(manifest.plugin_names or metadata.entry_points)
    _validate_plugin_selection(
        plugin_names,
        metadata.entry_points,
        context="local Bundle",
    )
    if artifact.sha256 is not None and artifact.sha256 != bundle_digest:
        raise _configuration_error("artifact sha256 does not match the Bundle digest")
    return (
        metadata,
        bundle_digest,
        plugin_names,
        manifest.requirements,
        manifest.wheelhouse,
    )


def _validate_py_modules_artifact(
    artifact: AlgorithmArtifact,
    profile: ImageProfile,
) -> tuple[_WheelMetadata, tuple[str, ...]]:
    path = _local_source_path(artifact)
    metadata = (
        _remote_wheel_metadata(artifact)
        if path is None
        else _wheel_metadata(path, max_size=artifact.max_file_size)
    )
    if metadata.requires_dist:
        raise _configuration_error(
            "image + py_modules Wheel must be code-only and contain no Requires-Dist"
        )
    _check_wheel_tags(metadata, profile)
    if artifact.package_name and artifact.package_name != metadata.package_name:
        raise _configuration_error(
            "artifact package_name does not match Wheel metadata"
        )
    if (
        artifact.package_version
        and artifact.package_version != metadata.package_version
    ):
        raise _configuration_error(
            "artifact package_version does not match Wheel metadata"
        )
    if artifact.sha256 is not None and artifact.sha256 != metadata.digest:
        raise _configuration_error("artifact sha256 does not match the Wheel digest")
    if artifact.wheel_tags and artifact.is_remote is False:
        declared_tags = frozenset(
            tag for tag_text in artifact.wheel_tags for tag in parse_tag(tag_text)
        )
        if not declared_tags.intersection(metadata.tags):
            raise _configuration_error(
                "artifact wheel_tags do not match the local Wheel metadata"
            )
    _validate_plugin_selection(
        tuple(artifact.plugin_names),
        metadata.entry_points,
        context="artifact",
    )
    for dependency in artifact.requires_dist:
        _validate_declared_requirement(
            dependency,
            local_wheels={},
            profile=profile,
            context="remote artifact metadata",
        )
    plugin_names = tuple(artifact.plugin_names or metadata.entry_points)
    if len(plugin_names) > 1:
        raise _configuration_error(
            "one Job may activate only one algorithm entry point"
        )
    return metadata, plugin_names


@DeveloperAPI
def prepare_algorithm_distribution(
    artifact: AlgorithmArtifact,
    image_profile: ImageProfile,
    *,
    declared_dependencies: tuple[str, ...] = (),
) -> PreparedAlgorithmDistribution:
    """Validate an artifact against one selected image Profile.

    This function performs no package installation and never contacts a
    package index.  For local artifacts it verifies Wheel metadata, Bundle
    file digests, requirements safety, and the complete local dependency
    closure.  Remote artifacts must carry equivalent immutable metadata in
    ``AlgorithmArtifact``; remote offline Bundles additionally carry an
    attested manifest with requirements and dependency metadata.
    """
    if not isinstance(artifact, AlgorithmArtifact):
        raise TypeError("artifact must be an AlgorithmArtifact")
    if not isinstance(image_profile, ImageProfile):
        raise TypeError("image_profile must be an ImageProfile")
    _profile_version_check(image_profile)
    if artifact.mode is ArtifactDistributionMode.OFFLINE_WHEELHOUSE:
        if not image_profile.allow_offline_pip:
            raise _configuration_error(
                f"image Profile {image_profile.profile_id!r} disallows offline pip"
            )
        if "pip" not in image_profile.installed_distributions:
            raise _configuration_error(
                f"image Profile {image_profile.profile_id!r} must declare pip "
                "for Ray runtime_env.pip"
            )
        if artifact.is_remote:
            metadata, digest, plugin_names, requirements_name, wheelhouse_name = (
                _validate_remote_offline_bundle(
                    artifact,
                    image_profile,
                    declared_dependencies,
                )
            )
        else:
            metadata, digest, plugin_names, requirements_name, wheelhouse_name = (
                _validate_offline_bundle(
                    artifact,
                    image_profile,
                    declared_dependencies,
                )
            )
    else:
        metadata, plugin_names = _validate_py_modules_artifact(artifact, image_profile)
        digest = metadata.digest
        requirements_name = None
        wheelhouse_name = None
    if artifact.mode is ArtifactDistributionMode.IMAGE_PY_MODULES:
        for dependency in declared_dependencies:
            _validate_declared_requirement(
                dependency,
                local_wheels={metadata.package_name: metadata},
                profile=image_profile,
                context="EnvironmentSpec",
            )
    warnings = tuple(
        f"approved image pip-check baseline: {item}"
        for item in image_profile.pip_check_baseline
    )
    pip_check = not bool(image_profile.pip_check_baseline)
    checks = (
        "artifact_digest",
        "image_profile",
        "python_compatibility",
        "ray_compatibility",
        "wheel_tags",
        "dependency_closure",
        "offline_network_policy",
        "plugin_identity",
    ) + (
        (
            ("pip_check_full_environment",)
            if pip_check
            else ("pip_check_skipped_approved_baseline",)
        )
        if artifact.mode is ArtifactDistributionMode.OFFLINE_WHEELHOUSE
        else ()
    )
    receipt = AlgorithmDistributionReceipt(
        mode=artifact.mode,
        artifact_uri=artifact.source,
        artifact_sha256=digest,
        image_profile_id=image_profile.profile_id,
        image_digest=image_profile.image_digest,
        tributo_version=str(_current_tributo_version()),
        ray_version=image_profile.ray_version,
        python_version=image_profile.python_version,
        package_name=metadata.package_name,
        package_version=metadata.package_version,
        plugin_names=plugin_names,
        checks=checks,
        warnings=warnings,
    )
    return PreparedAlgorithmDistribution(
        artifact=artifact,
        receipt=receipt,
        package_name=metadata.package_name,
        package_version=metadata.package_version,
        plugin_names=plugin_names,
        requirements_name=requirements_name,
        wheelhouse_name=wheelhouse_name,
        pip_check=pip_check,
        dependency_names=tuple(
            canonicalize_name(Requirement(item).name) for item in declared_dependencies
        ),
    )


@DeveloperAPI
def algorithm_runtime_env_patch(
    prepared: PreparedAlgorithmDistribution,
    *,
    existing_env_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build only the Ray runtime_env fields owned by an algorithm artifact."""
    artifact = prepared.artifact
    env_vars = dict(existing_env_vars or {})
    identity = {
        "TRIBUTO_ALGORITHM_DISTRIBUTION_MODE": artifact.mode.value,
        "TRIBUTO_ALGORITHM_ARTIFACT_SHA256": prepared.receipt.artifact_sha256,
        "TRIBUTO_ALGORITHM_PACKAGE": prepared.package_name,
        "TRIBUTO_ALGORITHM_PACKAGE_VERSION": prepared.package_version,
        "TRIBUTO_IMAGE_PROFILE_ID": prepared.receipt.image_profile_id,
        "TRIBUTO_IMAGE_DIGEST": prepared.receipt.image_digest,
        "TRIBUTO_ALGORITHM_PREFLIGHT_RECEIPT": prepared.receipt.model_dump_json(),
    }
    for key, value in identity.items():
        existing = env_vars.get(key)
        if existing is not None and existing != value:
            raise _configuration_error(
                f"artifact identity conflicts with existing environment variable {key}"
            )
        env_vars[key] = value
    if prepared.plugin_names:
        # A Wheel may rely on exporter/flavor/validator entry points supplied
        # by its declared algorithm dependencies.  Selecting only the root
        # distribution would make those runtime capabilities disappear.
        dependency_distributions: list[str] = []
        dependency_distributions.extend(prepared.dependency_names)
        distributions = tuple(
            dict.fromkeys((prepared.package_name, *dependency_distributions))
        )
        requested = ",".join(f"distribution:{name}" for name in distributions)
        existing_plugins = env_vars.get("TRIBUTO_PLUGINS")
        if existing_plugins and existing_plugins != requested:
            raise _configuration_error(
                "artifact plugin_names conflict with the existing TRIBUTO_PLUGINS filter"
            )
        env_vars["TRIBUTO_PLUGINS"] = requested
    patch: dict[str, Any] = {"env_vars": env_vars}
    if artifact.mode is ArtifactDistributionMode.IMAGE_PY_MODULES:
        patch["py_modules"] = [artifact.source]
    else:
        patch["working_dir"] = artifact.source
        patch["excludes"] = []
        requirements = prepared.requirements_name or artifact.requirements_name
        patch["pip"] = {
            "packages": [f"-r {_RUNTIME_WORKING_DIR}/{requirements}"],
            "pip_check": prepared.pip_check,
            "pip_install_options": list(_ALLOWED_PIP_INSTALL_OPTIONS),
        }
    return patch


__all__ = [
    "algorithm_runtime_env_patch",
    "prepare_algorithm_distribution",
]

"""Immutable contracts for algorithm artifacts and image profiles.

The models in this module deliberately describe *what* a Job is allowed to
use.  Filesystem inspection, Wheel metadata parsing, and dependency closure
validation live in :mod:`tributo._common.algorithm_distribution` so that a
descriptor can remain importable without touching the submitting host.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import parse_tag
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator

from tributo._common.config import StrictConfigModel
from tributo.util.annotations import PublicAPI

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_RELATIVE_PATH = re.compile(r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$)).+$")


@PublicAPI(stability="alpha")
class ArtifactDistributionMode(str, Enum):
    """How a Job receives an algorithm package."""

    IMAGE_PY_MODULES = "image_py_modules"
    OFFLINE_WHEELHOUSE = "offline_wheelhouse"


def _validate_relative_path(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or _RELATIVE_PATH.fullmatch(value) is None
    ):
        raise ValueError(f"{field_name} must be a relative POSIX path")
    return value


def _validate_uri_without_credentials(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    value = value.strip()
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not contain query or fragment data")
    return value


@PublicAPI(stability="alpha")
class AlgorithmArtifact(StrictConfigModel):
    """A user-provided algorithm Wheel or offline dependency Bundle.

    ``source`` may be a local path, an immutable HTTPS/S3 Wheel URI, or (for
    offline mode) a local Bundle directory or immutable HTTPS/S3 ZIP Bundle
    archive. Remote sources must provide the immutable digest and package
    metadata because the submitting process is not allowed to resolve or
    inspect them through an online package index.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(..., min_length=1)
    mode: ArtifactDistributionMode = ArtifactDistributionMode.IMAGE_PY_MODULES
    sha256: str | None = None
    package_name: str | None = None
    package_version: str | None = None
    requires_dist: tuple[str, ...] = ()
    plugin_names: tuple[str, ...] = ()
    wheel_tags: tuple[str, ...] = ()
    manifest_name: str = "manifest.json"
    requirements_name: str = "requirements.lock"
    wheelhouse_name: str = "wheelhouse"
    max_file_size: int = Field(default=1024 * 1024 * 1024, gt=0)
    max_bundle_size: int = Field(default=4 * 1024 * 1024 * 1024, gt=0)
    # Remote offline Bundles cannot be inspected by the submitting process.
    # The caller therefore supplies the same manifest metadata that is signed
    # or otherwise attested by the internal artifact service.
    manifest: dict[str, Any] | None = None

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _validate_uri_without_credentials(value, "source")

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is not None and _DIGEST.fullmatch(value) is None:
            raise ValueError("sha256 must be a lower-case SHA-256 digest")
        return value

    @field_validator("package_name")
    @classmethod
    def validate_package_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return canonicalize_name(value)
        except Exception as exc:
            raise ValueError("package_name must be a valid distribution name") from exc

    @field_validator("package_version")
    @classmethod
    def validate_package_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return str(Version(value))
        except InvalidVersion as exc:
            raise ValueError("package_version must be a valid PEP 440 version") from exc

    @field_validator("requires_dist")
    @classmethod
    def validate_requires_dist(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        from packaging.requirements import InvalidRequirement, Requirement

        normalized: list[str] = []
        for dependency in value:
            try:
                requirement = Requirement(dependency)
            except (InvalidRequirement, TypeError) as exc:
                raise ValueError(
                    f"invalid requires_dist entry: {dependency!r}"
                ) from exc
            if requirement.url is not None:
                raise ValueError("requires_dist must not contain URL requirements")
            normalized.append(str(requirement))
        if len(set(normalized)) != len(normalized):
            raise ValueError("requires_dist must not contain duplicates")
        return tuple(sorted(normalized))

    @field_validator("plugin_names")
    @classmethod
    def validate_plugin_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        names = tuple(str(item).strip() for item in value)
        if any(not _IDENTIFIER.fullmatch(item) for item in names):
            raise ValueError("plugin_names must contain lower-case entry-point names")
        if len(set(names)) != len(names):
            raise ValueError("plugin_names must not contain duplicates")
        return tuple(sorted(names))

    @field_validator("wheel_tags")
    @classmethod
    def validate_wheel_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for tag_text in value:
            if not isinstance(tag_text, str) or not tag_text.strip():
                raise ValueError("wheel_tags must contain non-empty tag strings")
            try:
                tags = parse_tag(tag_text.strip())
            except ValueError as exc:
                raise ValueError(f"invalid Wheel tag: {tag_text!r}") from exc
            if len(tags) != 1:
                raise ValueError("wheel_tags entries must contain exactly one tag")
            normalized.append(str(next(iter(tags))))
        if len(set(normalized)) != len(normalized):
            raise ValueError("wheel_tags must not contain duplicates")
        return tuple(sorted(normalized))

    @field_validator("manifest_name", "requirements_name", "wheelhouse_name")
    @classmethod
    def validate_paths(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "artifact path"
        return _validate_relative_path(value, field_name)

    @model_validator(mode="after")
    def validate_remote_contract(self) -> AlgorithmArtifact:
        parsed = urlsplit(self.source)
        is_remote = parsed.scheme in {"https", "s3"}
        if is_remote and self.sha256 is None:
            raise ValueError("remote algorithm artifacts require sha256")
        if is_remote and self.mode is ArtifactDistributionMode.OFFLINE_WHEELHOUSE:
            if self.manifest is None:
                raise ValueError(
                    "remote offline Wheelhouse artifacts require an attested manifest"
                )
        if is_remote and (self.package_name is None or self.package_version is None):
            raise ValueError(
                "remote algorithm artifacts require package_name and package_version"
            )
        if (
            is_remote
            and self.mode is ArtifactDistributionMode.OFFLINE_WHEELHOUSE
            and not self.source.lower().endswith(".zip")
        ):
            raise ValueError(
                "remote offline Wheelhouse sources must be ZIP Bundle archives"
            )
        if self.mode is ArtifactDistributionMode.OFFLINE_WHEELHOUSE:
            if self.source.startswith("file://"):
                raise ValueError(
                    "offline Wheelhouse source must be a local directory path"
                )
        return self

    @property
    def is_remote(self) -> bool:
        """Whether the source is an immutable remote URI."""
        return urlsplit(self.source).scheme in {"https", "s3"}


@PublicAPI(stability="alpha")
class ImageProfile(StrictConfigModel):
    """An immutable description of the Ray image selected by the platform."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(..., min_length=1)
    image_uri: str = Field(..., min_length=1)
    image_digest: str
    ray_version: str = "2.55.1"
    python_spec: str = ">=3.12,<3.14"
    python_version: str = "3.12"
    sys_platform: str = "linux"
    platform_machine: str = "x86_64"
    wheel_tags: tuple[str, ...] = ()
    installed_distributions: dict[str, str] = Field(default_factory=dict)
    allow_offline_pip: bool = True
    pip_check_baseline: tuple[str, ...] = ()
    algorithm_ids: tuple[str, ...] = ()

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("profile_id must be a lower-case namespaced identifier")
        return value

    @field_validator("image_uri")
    @classmethod
    def validate_image_uri(cls, value: str) -> str:
        return _validate_uri_without_credentials(value, "image_uri")

    @field_validator("image_digest")
    @classmethod
    def validate_image_digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("image_digest must be a lower-case SHA-256 digest")
        return value

    @field_validator("ray_version")
    @classmethod
    def validate_ray_version(cls, value: str) -> str:
        try:
            return str(Version(value))
        except InvalidVersion as exc:
            raise ValueError("ray_version must be a valid PEP 440 version") from exc

    @field_validator("python_spec")
    @classmethod
    def validate_python_spec(cls, value: str) -> str:
        try:
            SpecifierSet(value)
        except (InvalidSpecifier, TypeError) as exc:
            raise ValueError("python_spec must be a valid version specifier") from exc
        return value

    @field_validator("python_version")
    @classmethod
    def validate_python_version(cls, value: str) -> str:
        try:
            Version(value)
        except InvalidVersion as exc:
            raise ValueError("python_version must be a valid PEP 440 version") from exc
        return value

    @field_validator("sys_platform", "platform_machine")
    @classmethod
    def validate_platform_values(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("platform marker values must be non-empty")
        return value

    @field_validator("installed_distributions")
    @classmethod
    def validate_installed_distributions(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for name, version in value.items():
            try:
                normalized[canonicalize_name(name)] = str(Version(version))
            except (InvalidVersion, TypeError) as exc:
                raise ValueError(
                    f"invalid installed distribution {name!r}={version!r}"
                ) from exc
        return dict(sorted(normalized.items()))

    @field_validator("algorithm_ids")
    @classmethod
    def validate_algorithm_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _IDENTIFIER.fullmatch(item) for item in value):
            raise ValueError(
                "algorithm_ids must contain lower-case namespaced identifiers"
            )
        if len(set(value)) != len(value):
            raise ValueError("algorithm_ids must not contain duplicates")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_target_python(self) -> ImageProfile:
        if Version(self.python_version) not in SpecifierSet(self.python_spec):
            raise ValueError("python_version must satisfy the ImageProfile python_spec")
        return self


@PublicAPI(stability="alpha")
class ArtifactFile(StrictConfigModel):
    """One manifest-recorded file in an offline algorithm Bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size: int = Field(..., ge=0)
    sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_path(value, "path")

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("sha256 must be a lower-case SHA-256 digest")
        return value


@PublicAPI(stability="alpha")
class AlgorithmBundleManifest(StrictConfigModel):
    """Manifest contract for a locally inspectable offline Wheelhouse."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)
    algorithm_id: str
    package_name: str
    package_version: str
    wheel: str = "algorithm.whl"
    requirements: str = "requirements.lock"
    wheelhouse: str = "wheelhouse"
    files: tuple[ArtifactFile, ...]
    plugin_names: tuple[str, ...] = ()
    wheel_entry_points: tuple[str, ...] = ()
    python_spec: str = ">=3.12,<3.14"
    ray_version: str = "2.55.1"
    tributo_version_spec: str = ">=1,<2"
    profile_ids: tuple[str, ...] = ()
    network: bool = False
    # These fields are used for remote, preflight-only attestations.  Local
    # Bundles remain authoritative from their on-disk requirements and Wheel
    # metadata; the fields are optional there for backwards compatibility.
    requirements_entries: tuple[str, ...] = ()
    wheel_requires: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("algorithm_id", "package_name")
    @classmethod
    def validate_names(cls, value: str, info: ValidationInfo) -> str:
        if not value or (
            info.field_name == "algorithm_id" and _IDENTIFIER.fullmatch(value) is None
        ):
            raise ValueError(f"invalid {info.field_name}")
        if info.field_name == "package_name":
            return canonicalize_name(value)
        return value

    @field_validator("package_version", "ray_version")
    @classmethod
    def validate_versions(cls, value: str) -> str:
        try:
            return str(Version(value))
        except InvalidVersion as exc:
            raise ValueError(
                "manifest version must be a valid PEP 440 version"
            ) from exc

    @field_validator("python_spec", "tributo_version_spec")
    @classmethod
    def validate_specs(cls, value: str) -> str:
        try:
            SpecifierSet(value)
        except (InvalidSpecifier, TypeError) as exc:
            raise ValueError("manifest version constraint is invalid") from exc
        return value

    @field_validator("wheel", "requirements", "wheelhouse")
    @classmethod
    def validate_manifest_paths(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "manifest path"
        return _validate_relative_path(value, field_name)

    @field_validator("plugin_names", "wheel_entry_points", "profile_ids")
    @classmethod
    def validate_manifest_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _IDENTIFIER.fullmatch(item) for item in value):
            raise ValueError(
                "manifest identifiers must be lower-case namespaced values"
            )
        if len(set(value)) != len(value):
            raise ValueError("manifest identifiers must not contain duplicates")
        return tuple(sorted(value))

    @field_validator("requirements_entries")
    @classmethod
    def validate_requirement_entries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(
                "manifest requirements_entries must contain non-empty lines"
            )
        return tuple(item.rstrip() for item in value)

    @field_validator("wheel_requires")
    @classmethod
    def validate_wheel_requires(
        cls, value: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        from packaging.requirements import InvalidRequirement, Requirement

        normalized: dict[str, tuple[str, ...]] = {}
        for name, dependencies in value.items():
            package_name = canonicalize_name(name)
            rendered: list[str] = []
            for dependency in dependencies:
                try:
                    requirement = Requirement(dependency)
                except (InvalidRequirement, TypeError) as exc:
                    raise ValueError(
                        f"manifest wheel_requires contains an invalid requirement: "
                        f"{dependency!r}"
                    ) from exc
                if requirement.url is not None:
                    raise ValueError(
                        "manifest wheel_requires must not contain URL requirements"
                    )
                rendered.append(str(requirement))
            if package_name in normalized:
                raise ValueError(
                    f"manifest wheel_requires contains duplicate package {package_name!r}"
                )
            if len(set(rendered)) != len(rendered):
                raise ValueError(
                    f"manifest wheel_requires duplicates dependencies for {package_name!r}"
                )
            normalized[package_name] = tuple(sorted(rendered))
        return dict(sorted(normalized.items()))

    @model_validator(mode="after")
    def validate_manifest(self) -> AlgorithmBundleManifest:
        paths = {item.path for item in self.files}
        if len(paths) != len(self.files):
            raise ValueError("manifest files must not contain duplicate paths")
        required = {self.wheel, self.requirements}
        if not required.issubset(paths):
            raise ValueError(
                "manifest must record the algorithm Wheel and requirements file"
            )
        if self.network:
            raise ValueError("algorithm Bundles must declare network=false")
        return self


@PublicAPI(stability="alpha")
class AlgorithmDistributionReceipt(StrictConfigModel):
    """Credential-free evidence recorded for one prepared Job environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ArtifactDistributionMode
    artifact_uri: str
    artifact_sha256: str
    image_profile_id: str
    image_digest: str
    tributo_version: str
    ray_version: str
    python_version: str
    package_name: str
    package_version: str
    plugin_names: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @field_validator("artifact_uri")
    @classmethod
    def validate_artifact_uri(cls, value: str) -> str:
        return _validate_uri_without_credentials(value, "artifact_uri")

    @field_validator("artifact_sha256", "image_digest")
    @classmethod
    def validate_receipt_digests(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("receipt digests must be lower-case SHA-256 values")
        return value

    @field_validator("package_name")
    @classmethod
    def validate_receipt_package(cls, value: str) -> str:
        return canonicalize_name(value)

    @field_validator("tributo_version", "ray_version", "python_version")
    @classmethod
    def validate_receipt_versions(cls, value: str) -> str:
        try:
            return str(Version(value))
        except InvalidVersion as exc:
            raise ValueError("receipt versions must be valid PEP 440 versions") from exc

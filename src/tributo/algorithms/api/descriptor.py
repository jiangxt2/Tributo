"""Narrow, immutable descriptor SPI for trusted distributed algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from tributo.algorithms.api.distribution import (
    DistributionStrategy,
    ExecutionProfile,
)
from tributo.algorithms.api.errors import AlgorithmConfigurationError
from tributo.algorithms.api.models import AlgorithmRegistration, ExecutionMode
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class DistributedAlgorithmDescriptor:
    """Atomically expose one formal registration and support evidence."""

    registration: AlgorithmRegistration
    package_name: str
    package_version: str
    tributo_version_spec: str
    stability: Literal["alpha", "beta", "stable"] = "alpha"
    tested: bool = False
    supported: bool = False
    validated_execution_profiles: tuple[ExecutionProfile, ...] = field(
        default_factory=tuple
    )
    limitations: tuple[str, ...] = field(default_factory=tuple)
    api_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.api_version, int)
            or isinstance(self.api_version, bool)
            or self.api_version != 1
        ):
            raise AlgorithmConfigurationError(
                f"unsupported distributed descriptor api_version: {self.api_version}"
            )
        if self.registration.distribution_spec is None:
            raise AlgorithmConfigurationError(
                "distributed descriptors require a DistributionSpec"
            )
        try:
            package_name = canonicalize_name(self.package_name)
            package_version = str(Version(self.package_version))
            tributo_version = SpecifierSet(self.tributo_version_spec)
        except (InvalidSpecifier, InvalidVersion, TypeError) as exc:
            raise AlgorithmConfigurationError(
                "distributed descriptor package/version metadata is invalid"
            ) from exc
        if not package_name or not str(tributo_version):
            raise AlgorithmConfigurationError(
                "distributed descriptor requires a package name, exact package "
                "version, and non-empty Tributo compatibility range"
            )
        object.__setattr__(self, "package_name", package_name)
        object.__setattr__(self, "package_version", package_version)
        object.__setattr__(self, "tributo_version_spec", str(tributo_version))
        expected_mode = {
            DistributionStrategy.RAY_TRAIN_COLLECTIVE: ExecutionMode.COLLECTIVE,
            DistributionStrategy.FRAMEWORK_NATIVE: ExecutionMode.FRAMEWORK_NATIVE,
            DistributionStrategy.RAY_MAP_REDUCE: ExecutionMode.MAP_REDUCE,
        }[self.registration.distribution_spec.strategy]
        if self.registration.implementation.execution_mode is not expected_mode:
            raise AlgorithmConfigurationError(
                "distributed descriptor strategy and implementation mode disagree"
            )
        if self.supported and not self.tested:
            raise AlgorithmConfigurationError(
                "supported distributed descriptors must be tested"
            )
        try:
            validated_profiles = tuple(
                sorted(
                    {
                        ExecutionProfile(profile)
                        for profile in self.validated_execution_profiles
                    },
                    key=str,
                )
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                "validated execution profiles must be local or kubernetes"
            ) from exc
        compatible_profiles = set(
            self.registration.distribution_spec.supported_execution_profiles
        )
        if not set(validated_profiles).issubset(compatible_profiles):
            raise AlgorithmConfigurationError(
                "validated execution profiles must be compatible with DistributionSpec"
            )
        object.__setattr__(
            self,
            "validated_execution_profiles",
            validated_profiles,
        )
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def name(self) -> str:
        """Return the canonical algorithm identity."""
        return self.registration.spec.name


__all__ = ["DistributedAlgorithmDescriptor"]

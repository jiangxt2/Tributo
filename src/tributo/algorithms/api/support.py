"""Trusted support evidence for installed distributed algorithm Wheels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from tributo.algorithms.api.descriptor import DistributedAlgorithmDescriptor
from tributo.algorithms.api.distribution import (
    DistributionStrategy,
    ExecutionProfile,
)
from tributo.algorithms.api.errors import AlgorithmConfigurationError
from tributo.util.annotations import PublicAPI


def _digest(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AlgorithmConfigurationError(f"{name} must be a lower-case SHA-256")
    return value


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise AlgorithmConfigurationError(f"{name} must be a non-empty string")
    return value


@PublicAPI(stability="alpha")
class SupportTier(str, Enum):
    """Governance tier assigned by trusted evidence, never by a Wheel."""

    COMMUNITY = "community"
    VERIFIED = "verified"
    OFFICIAL = "official"


@PublicAPI(stability="alpha")
class DistributedSemantics(str, Enum):
    """Observed relation between Ray work and the resulting model."""

    SINGLE_WORKER = "single_worker"
    TRIAL_PARALLEL = "trial_parallel"
    ESTIMATOR_INTERNAL_PARALLEL = "estimator_internal_parallel"
    SINGLE_MODEL_DISTRIBUTED = "single_model_distributed"


def descriptor_distributed_semantics(
    descriptor: DistributedAlgorithmDescriptor,
) -> DistributedSemantics:
    """Derive the only execution semantic a descriptor may be awarded."""
    distribution = descriptor.registration.distribution_spec
    if distribution is None:
        return DistributedSemantics.SINGLE_WORKER
    if distribution.strategy is DistributionStrategy.RAY_JOBLIB_ESTIMATOR:
        return DistributedSemantics.ESTIMATOR_INTERNAL_PARALLEL
    return DistributedSemantics.SINGLE_MODEL_DISTRIBUTED


def descriptor_contract_digests(
    descriptor: DistributedAlgorithmDescriptor,
) -> tuple[str, ...]:
    """Return the stable ordered contract digest set for evidence matching."""
    bindings = descriptor.registration.contract_bindings
    if bindings is None:
        return ()
    return tuple(
        sorted(
            binding.schema_digest
            for binding in (
                bindings.config,
                bindings.input,
                bindings.output,
                *((bindings.coverage,) if bindings.coverage is not None else ()),
            )
        )
    )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class AlgorithmSupportEvidence:
    """One immutable support decision for one exact installed implementation."""

    algorithm_id: str
    implementation_id: str
    distribution: str
    package_version: str
    wheel_sha256: str
    descriptor_api_version: int
    contract_digests: tuple[str, ...]
    distributed_semantics: DistributedSemantics
    execution_profile: ExecutionProfile
    tributo_version: str
    ray_version: str
    python_version: str
    image_profile: str
    framework_version: str
    hardware_profile: str
    issuer: str
    source_commit: str
    gate: str
    result_reference: str
    issued_at: datetime
    support_tier: SupportTier = SupportTier.VERIFIED
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    revocation_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "algorithm_id",
            "implementation_id",
            "tributo_version",
            "ray_version",
            "python_version",
            "image_profile",
            "framework_version",
            "hardware_profile",
            "issuer",
            "source_commit",
            "gate",
            "result_reference",
        ):
            _non_empty(getattr(self, name), name)
        distribution = canonicalize_name(self.distribution)
        if not distribution:
            raise AlgorithmConfigurationError("distribution must be non-empty")
        try:
            package_version = str(Version(self.package_version))
            semantics = DistributedSemantics(self.distributed_semantics)
            profile = ExecutionProfile(self.execution_profile)
            tier = SupportTier(self.support_tier)
        except (InvalidVersion, TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                "support evidence contains invalid version or enum metadata"
            ) from exc
        if tier is SupportTier.COMMUNITY:
            raise AlgorithmConfigurationError(
                "Community status does not require trusted support evidence"
            )
        if (
            not isinstance(self.descriptor_api_version, int)
            or isinstance(self.descriptor_api_version, bool)
            or self.descriptor_api_version < 1
        ):
            raise AlgorithmConfigurationError(
                "descriptor_api_version must be a positive integer"
            )
        contract_digests = tuple(sorted(self.contract_digests))
        if not contract_digests:
            raise AlgorithmConfigurationError(
                "support evidence must bind executable contract digests"
            )
        _digest(self.wheel_sha256, "wheel_sha256")
        for contract_digest in contract_digests:
            _digest(contract_digest, "contract_digest")
        for name in ("issued_at", "expires_at", "revoked_at"):
            timestamp = getattr(self, name)
            if timestamp is not None and (
                timestamp.tzinfo is None or timestamp.utcoffset() is None
            ):
                raise AlgorithmConfigurationError(
                    f"{name} must include a timezone offset"
                )
        if self.expires_at is not None and self.expires_at <= self.issued_at:
            raise AlgorithmConfigurationError("expires_at must follow issued_at")
        if self.revoked_at is not None and not self.revocation_reason:
            raise AlgorithmConfigurationError(
                "revoked evidence requires revocation_reason"
            )
        if self.revoked_at is None and self.revocation_reason is not None:
            raise AlgorithmConfigurationError("revocation_reason requires revoked_at")
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "package_version", package_version)
        object.__setattr__(self, "contract_digests", contract_digests)
        object.__setattr__(self, "distributed_semantics", semantics)
        object.__setattr__(self, "execution_profile", profile)
        object.__setattr__(self, "support_tier", tier)

    @property
    def evidence_id(self) -> str:
        """Return a deterministic, credential-free evidence identity."""
        payload = {
            "algorithm_id": self.algorithm_id,
            "implementation_id": self.implementation_id,
            "distribution": self.distribution,
            "package_version": self.package_version,
            "wheel_sha256": self.wheel_sha256,
            "descriptor_api_version": self.descriptor_api_version,
            "contract_digests": self.contract_digests,
            "distributed_semantics": self.distributed_semantics.value,
            "execution_profile": self.execution_profile.value,
            "issuer": self.issuer,
            "issued_at": self.issued_at.isoformat(),
            "gate": self.gate,
            "result_reference": self.result_reference,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether evidence is issued, unexpired, and not revoked."""
        current = now or datetime.now(timezone.utc)
        return (
            current >= self.issued_at
            and (self.expires_at is None or current < self.expires_at)
            and self.revoked_at is None
        )

    def matches(
        self,
        descriptor: DistributedAlgorithmDescriptor,
        *,
        wheel_sha256: str,
    ) -> bool:
        """Match every immutable package, contract, profile, and semantic key."""
        distribution = descriptor.registration.distribution_spec
        environment = descriptor.registration.environment
        ray_requirements = [
            requirement
            for requirement in environment.dependencies
            if canonicalize_name(Requirement(requirement).name) == "ray"
        ]
        ray_matches = (
            self.ray_version == "2.55.1"
            if not ray_requirements
            else any(
                Version(self.ray_version) in Requirement(requirement).specifier
                for requirement in ray_requirements
            )
        )
        return (
            self.algorithm_id == descriptor.name
            and self.implementation_id
            == descriptor.registration.implementation.implementation_id
            and self.distribution == descriptor.package_name
            and self.package_version == descriptor.package_version
            and self.wheel_sha256 == wheel_sha256
            and self.descriptor_api_version == descriptor.api_version
            and self.contract_digests == descriptor_contract_digests(descriptor)
            and self.distributed_semantics
            is descriptor_distributed_semantics(descriptor)
            and distribution is not None
            and self.execution_profile in distribution.supported_execution_profiles
            and Version(self.tributo_version)
            in SpecifierSet(descriptor.tributo_version_spec)
            and Version(self.python_version) in SpecifierSet(environment.python)
            and ray_matches
        )


@PublicAPI(stability="alpha")
class AlgorithmSupportEvidenceRegistry:
    """Resolve support only from evidence issued by explicit trust roots."""

    def __init__(
        self,
        evidence: tuple[AlgorithmSupportEvidence, ...] = (),
        *,
        trusted_issuers: tuple[str, ...] = (),
    ) -> None:
        self._evidence = tuple(evidence)
        self._trusted_issuers = frozenset(trusted_issuers)

    def resolve(
        self,
        descriptor: DistributedAlgorithmDescriptor,
        *,
        wheel_sha256: str | None,
        now: datetime | None = None,
    ) -> tuple[AlgorithmSupportEvidence, ...]:
        """Return all active exact matches; missing Wheel digest fails closed."""
        if wheel_sha256 is None:
            return ()
        _digest(wheel_sha256, "wheel_sha256")
        return tuple(
            sorted(
                (
                    item
                    for item in self._evidence
                    if item.issuer in self._trusted_issuers
                    and item.is_active(now=now)
                    and item.matches(descriptor, wheel_sha256=wheel_sha256)
                ),
                key=lambda item: (
                    0 if item.support_tier is SupportTier.OFFICIAL else 1,
                    item.execution_profile.value,
                    item.evidence_id,
                ),
            )
        )


__all__ = [
    "AlgorithmSupportEvidence",
    "AlgorithmSupportEvidenceRegistry",
    "DistributedSemantics",
    "SupportTier",
]

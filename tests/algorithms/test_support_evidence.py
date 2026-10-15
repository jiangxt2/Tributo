"""Trusted support evidence overlay tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tests.support.runtime_versions import RAY_VERSION
from tributo.algorithms.api import (
    AlgorithmSupportEvidence,
    AlgorithmSupportEvidenceRegistry,
    ContractBinding,
    ContractBindingSet,
    DistributedAlgorithmDescriptor,
    DistributedSemantics,
    ExecutionProfile,
    QualifiedReference,
    SupportTier,
)
from tributo.training.registry import TrainingAlgorithmRegistry

from .conftest import map_reduce_registration

_WHEEL_DIGEST = "e" * 64


@pytest.mark.parametrize(
    ("ray_version", "accepted"),
    [("2.59.0", True), ("2.55.1", False), ("3.0.0.dev0", False)],
)
def test_default_runtime_evidence_requires_audited_ray(
    ray_version: str, accepted: bool
) -> None:
    descriptor = _descriptor()
    evidence = replace(_evidence(descriptor), ray_version=ray_version)

    assert evidence.matches(descriptor, wheel_sha256=_WHEEL_DIGEST) is accepted


def test_explicit_algorithm_runtime_requirement_is_not_widened() -> None:
    descriptor = _descriptor()
    environment = replace(
        descriptor.registration.environment, dependencies=("ray==2.55.1",)
    )
    descriptor = replace(
        descriptor,
        registration=replace(descriptor.registration, environment=environment),
    )

    assert (
        _evidence(descriptor).matches(descriptor, wheel_sha256=_WHEEL_DIGEST) is False
    )
    historical = replace(_evidence(descriptor), ray_version="2.55.1")
    assert historical.matches(descriptor, wheel_sha256=_WHEEL_DIGEST) is True


@pytest.mark.parametrize(
    ("core_version", "version_spec", "accepted"),
    [
        ("1.1.0", ">=1,<2", True),
        ("1.2.0.dev0", ">=1,<2", True),
        ("1.2.0.dev0", ">=1,<1.2", False),
        ("1.2.0.dev0", ">=1.2", False),
        ("1.2.0.dev0", "==1.1.0", False),
        ("1.2.0.dev0", ">=1,<2,!=1.2.*", False),
        ("1.2.0rc1", ">=1,<2", False),
        ("2.0.0.dev0", ">=1,<2", False),
    ],
)
def test_support_evidence_core_development_version_range(
    core_version: str, version_spec: str, accepted: bool
) -> None:
    descriptor = replace(_descriptor(), tributo_version_spec=version_spec)
    evidence = replace(_evidence(descriptor), tributo_version=core_version)

    assert evidence.matches(descriptor, wheel_sha256=_WHEEL_DIGEST) is accepted
    assert evidence.tributo_version == core_version
    assert evidence.matches(descriptor, wheel_sha256="f" * 64) is False


def _binding(contract_id: str, digest: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            "tests.support.portable_contracts:ConfigValidator"
        ),
    )


def _descriptor() -> DistributedAlgorithmDescriptor:
    registration = map_reduce_registration()
    spec = registration.spec
    registration = replace(
        registration,
        contract_bindings=ContractBindingSet(
            config=_binding(spec.config_contract_ref, "a"),
            input=_binding(spec.input_contract_ref, "b"),
            output=_binding(spec.output_contract_ref, "c"),
            coverage=_binding("tests.support.coverage.v1", "d"),
        ),
    )
    return DistributedAlgorithmDescriptor(
        registration=registration,
        package_name="tests-support-algorithm",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
        tested=True,
        supported=True,
        validated_execution_profiles=(ExecutionProfile.CLUSTER,),
        api_version=2,
    )


def _evidence(
    descriptor: DistributedAlgorithmDescriptor,
    *,
    issuer: str = "tributo-release",
    issued_at: datetime | None = None,
    expires_at: datetime | None = None,
    revoked_at: datetime | None = None,
    revocation_reason: str | None = None,
) -> AlgorithmSupportEvidence:
    contracts = descriptor.registration.contract_bindings
    assert contracts is not None
    return AlgorithmSupportEvidence(
        algorithm_id=descriptor.name,
        implementation_id=descriptor.registration.implementation.implementation_id,
        distribution=descriptor.package_name,
        package_version=descriptor.package_version,
        wheel_sha256=_WHEEL_DIGEST,
        descriptor_api_version=descriptor.api_version,
        contract_digests=tuple(
            binding.schema_digest
            for binding in (
                contracts.config,
                contracts.input,
                contracts.output,
                contracts.coverage,
            )
            if binding is not None
        ),
        distributed_semantics=DistributedSemantics.SINGLE_MODEL_DISTRIBUTED,
        execution_profile=ExecutionProfile.CLUSTER,
        tributo_version="1.0.0",
        ray_version=RAY_VERSION,
        python_version="3.12",
        image_profile="tests.cpu",
        framework_version="tests-1.0",
        hardware_profile="cpu-two-node",
        issuer=issuer,
        source_commit="source-commit",
        gate="docker-multi-node",
        result_reference="tests/logs/support-evidence.json",
        issued_at=issued_at or datetime.now(timezone.utc) - timedelta(minutes=1),
        expires_at=expires_at,
        revoked_at=revoked_at,
        revocation_reason=revocation_reason,
        support_tier=SupportTier.OFFICIAL,
    )


def _training_registry(
    descriptor: DistributedAlgorithmDescriptor,
    evidence: AlgorithmSupportEvidenceRegistry | None = None,
    *,
    wheel_digest: str | None = None,
) -> TrainingAlgorithmRegistry:
    registry = TrainingAlgorithmRegistry(
        support_evidence=evidence,
        installed_wheel_digests=(
            {descriptor.package_name: wheel_digest}
            if wheel_digest is not None
            else None
        ),
    )
    registry._execution_registry.register(descriptor.registration)
    registry._distributed_descriptors = {
        descriptor.registration.implementation.implementation_id: descriptor
    }
    registry._bootstrapped = True
    return registry


def test_descriptor_cannot_self_grant_tested_or_supported() -> None:
    descriptor = _descriptor()

    (record,) = _training_registry(descriptor).record_snapshot()

    assert descriptor.tested is True
    assert descriptor.supported is True
    assert record.tested is False
    assert record.supported is False
    assert record.validated_execution_profiles == ()


def test_non_torch_evidence_id_keeps_the_pre_torch_payload() -> None:
    descriptor = _descriptor()
    issued_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evidence = _evidence(descriptor, issued_at=issued_at)
    expected = {
        "algorithm_id": evidence.algorithm_id,
        "implementation_id": evidence.implementation_id,
        "distribution": evidence.distribution,
        "package_version": evidence.package_version,
        "wheel_sha256": evidence.wheel_sha256,
        "descriptor_api_version": evidence.descriptor_api_version,
        "contract_digests": evidence.contract_digests,
        "distributed_semantics": evidence.distributed_semantics.value,
        "execution_profile": evidence.execution_profile.value,
        "issuer": evidence.issuer,
        "issued_at": issued_at.isoformat(),
        "gate": evidence.gate,
        "result_reference": evidence.result_reference,
    }

    assert (
        evidence.evidence_id
        == hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_exact_trusted_wheel_evidence_grants_support() -> None:
    descriptor = _descriptor()
    overlay = AlgorithmSupportEvidenceRegistry(
        (_evidence(descriptor),),
        trusted_issuers=("tributo-release",),
    )

    (record,) = _training_registry(
        descriptor,
        overlay,
        wheel_digest=_WHEEL_DIGEST,
    ).record_snapshot()

    assert record.tested is True
    assert record.supported is True
    assert record.validated_execution_profiles == ("cluster",)


def test_wrong_digest_untrusted_expired_and_revoked_evidence_fail_closed() -> None:
    descriptor = _descriptor()
    now = datetime.now(timezone.utc)
    expired = _evidence(
        descriptor,
        issued_at=now - timedelta(days=2),
        expires_at=now - timedelta(days=1),
    )
    revoked = _evidence(
        descriptor,
        revoked_at=now - timedelta(seconds=1),
        revocation_reason="invalidated gate",
    )

    for overlay, digest in (
        (
            AlgorithmSupportEvidenceRegistry(
                (_evidence(descriptor),),
                trusted_issuers=("another-issuer",),
            ),
            _WHEEL_DIGEST,
        ),
        (
            AlgorithmSupportEvidenceRegistry(
                (_evidence(descriptor),),
                trusted_issuers=("tributo-release",),
            ),
            "f" * 64,
        ),
        (
            AlgorithmSupportEvidenceRegistry(
                (expired, revoked),
                trusted_issuers=("tributo-release",),
            ),
            _WHEEL_DIGEST,
        ),
    ):
        (record,) = _training_registry(
            descriptor,
            overlay,
            wheel_digest=digest,
        ).record_snapshot()
        assert record.supported is False
        assert record.validated_execution_profiles == ()

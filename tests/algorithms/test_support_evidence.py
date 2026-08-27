"""Trusted support evidence overlay tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

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
        ray_version="2.55.1",
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

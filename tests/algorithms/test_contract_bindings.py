"""Tests for executable algorithm contract bindings."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmOperation,
    AlgorithmRegistration,
    ContractBinding,
    ContractBindingSet,
    DistributedAlgorithmDescriptor,
    InputBinding,
    InputBindingSet,
    InputCoverageContract,
    QualifiedReference,
    WorkerExecutionResult,
)
from tributo.algorithms.core import AlgorithmPlanner, AlgorithmRegistrationRegistry
from tributo.algorithms.input import (
    FakeInputInvocation,
    FakeInputResolver,
    FakeTabularPayload,
)
from tributo.algorithms.spi import (
    InputExecutionContext,
    RuntimeExecutionEnvelope,
    WorkerInputPayloadSet,
)

from .conftest import (
    dispatcher_for,
    function_registration,
    map_reduce_registration,
    request_for,
)


def _binding(
    contract_id: str,
    digest_character: str,
    validator: str,
) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest_character * 64,
        validator_ref=QualifiedReference.parse(
            f"tests.support.portable_contracts:{validator}"
        ),
    )


def _contracts(registration: AlgorithmRegistration) -> ContractBindingSet:
    spec = registration.spec
    return ContractBindingSet(
        config=_binding(spec.config_contract_ref, "a", "ConfigValidator"),
        input=_binding(spec.input_contract_ref, "b", "InputValidator"),
        output=_binding(spec.output_contract_ref, "c", "OutputValidator"),
        coverage=_binding("test.coverage.complete.v1", "d", "CoverageValidator"),
    )


def test_contract_binding_requires_canonical_metadata() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="schema_digest"):
        ContractBinding(
            contract_id="vendor.config.v1",
            schema_version=1,
            schema_digest="invalid",
            validator_ref=QualifiedReference.parse("tests.support:validator"),
        )


def test_input_coverage_contract_proves_partition_and_dimensions() -> None:
    contract = InputCoverageContract(dimensions=("treated", "control"))
    contract.validate(
        (
            {
                "shard_id": "shard-a",
                "rows_processed": 3,
                "input_rows": {
                    "coverage.treated": 2,
                    "coverage.control": 1,
                },
            },
            {
                "shard_id": "shard-b",
                "rows_processed": 3,
                "input_rows": {
                    "coverage.treated": 1,
                    "coverage.control": 2,
                },
            },
        ),
        expected_rows=6,
    )


def test_input_coverage_contract_rejects_duplicate_shards() -> None:
    contract = InputCoverageContract()
    with pytest.raises(AlgorithmExecutionError):
        contract.validate(
            (
                {"shard_id": "same", "rows_processed": 1, "input_rows": {}},
                {"shard_id": "same", "rows_processed": 1, "input_rows": {}},
            ),
            expected_rows=2,
        )


def test_registration_rejects_contract_identity_drift() -> None:
    registration = function_registration()
    contracts = _contracts(registration)

    with pytest.raises(AlgorithmConfigurationError, match="identities"):
        replace(
            registration,
            contract_bindings=replace(
                contracts,
                config=replace(contracts.config, contract_id="vendor.other.config.v1"),
            ),
        )


def test_planner_loads_selected_contracts_and_normalizes_config() -> None:
    registration = function_registration()
    registration = replace(
        registration,
        contract_bindings=_contracts(registration),
    )
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = FakeInputResolver()

    plan = AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
        request_for("external_function", AlgorithmOperation.FIT)
    )

    assert dict(plan.algorithm_config) == {"threshold": 0.5}
    assert plan.contract_bindings == registration.contract_bindings
    assert plan.to_dict()["contract_bindings"]["config"]["schema_digest"] == ("a" * 64)


def test_planner_rejects_validator_schema_digest_drift_before_input_access() -> None:
    registration = function_registration()
    contracts = _contracts(registration)
    registration = replace(
        registration,
        contract_bindings=replace(
            contracts,
            config=replace(
                contracts.config,
                validator_ref=QualifiedReference.parse(
                    "tests.support.portable_contracts:WrongDigestValidator"
                ),
            ),
        ),
    )
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = FakeInputResolver()

    with pytest.raises(AlgorithmConfigurationError, match="schema digest mismatch"):
        AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
            request_for("external_function", AlgorithmOperation.FIT)
        )


class _SuccessfulRuntime:
    def __init__(self) -> None:
        self.payload: object | None = None

    @property
    def runtime_id(self) -> str:
        return "tributo.ray_task"

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        self.payload = envelope.input_payloads[0]
        return WorkerExecutionResult(
            execution=AlgorithmExecutionResult(status="succeeded"),
            actual_versions={},
        )


def test_coordinator_enforces_selected_output_contract() -> None:
    registration = function_registration()
    contracts = _contracts(registration)
    registration = replace(
        registration,
        contract_bindings=replace(
            contracts,
            output=replace(
                contracts.output,
                validator_ref=QualifiedReference.parse(
                    "tests.support.portable_contracts:RejectingOutputValidator"
                ),
            ),
        ),
    )

    with pytest.raises(AlgorithmConfigurationError, match="validation failed"):
        dispatcher_for(registration, _SuccessfulRuntime()).execute(
            request_for("external_function", AlgorithmOperation.FIT),
            InputExecutionContext(
                {
                    "binary-fixture": FakeInputInvocation(
                        FakeTabularPayload(
                            {
                                "x0": (0.0, 1.0),
                                "x1": (1.0, 0.0),
                                "label": (0, 1),
                            }
                        )
                    )
                }
            ),
        )


def test_coordinator_combines_role_payloads_and_provenance() -> None:
    registration = function_registration()
    runtime = _SuccessfulRuntime()
    request = request_for("external_function", AlgorithmOperation.FIT)
    train = request.input_binding
    assert isinstance(train, InputBinding)
    validation = replace(
        train,
        name="validation",
        reference="validation-fixture",
    )
    request = replace(
        request,
        input_binding=InputBindingSet(
            bindings=(train, validation),
            primary_role="train",
        ),
    )
    columns = {
        "x0": (0.0, 1.0),
        "x1": (1.0, 0.0),
        "label": (0, 1),
    }

    result = dispatcher_for(registration, runtime).execute(
        request,
        InputExecutionContext(
            {
                "binary-fixture": FakeInputInvocation(FakeTabularPayload(columns)),
                "validation-fixture": FakeInputInvocation(FakeTabularPayload(columns)),
            }
        ),
    )

    assert isinstance(runtime.payload, WorkerInputPayloadSet)
    assert tuple(item.input_name for item in runtime.payload.payloads) == (
        "train",
        "validation",
    )
    assert set(result.input_provenance) == {"train", "validation"}


def test_descriptor_api_v2_requires_executable_contract_bindings() -> None:
    registration = map_reduce_registration()
    descriptor = DistributedAlgorithmDescriptor(
        registration=registration,
        package_name="tests-contract-algorithm",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
    )

    with pytest.raises(AlgorithmConfigurationError, match="ContractBindingSet"):
        replace(descriptor, api_version=2)

    descriptor = replace(
        descriptor,
        registration=replace(
            registration,
            contract_bindings=_contracts(registration),
        ),
        api_version=2,
    )

    assert descriptor.api_version == 2

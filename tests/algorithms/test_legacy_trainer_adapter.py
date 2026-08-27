"""Conformance tests for the Worker-only legacy Trainer compatibility path."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest

from tributo.algorithms.api import (
    AlgorithmOperation,
    AlgorithmRegistration,
    BackendInputCompatibility,
    EnvironmentSpec,
    ExecutionMode,
    ImplementationDescriptor,
    QualifiedReference,
    RuntimeBinding,
    RuntimeTopology,
)
from tributo.algorithms.core import AlgorithmPlanner, AlgorithmRegistrationRegistry
from tributo.algorithms.core.worker import worker_bootstrap
from tributo.algorithms.input import FakeInputResolver, FakeInputRuntimeAdapter
from tributo.algorithms.input.fake import FakeInputInvocation, FakeTabularPayload
from tributo.algorithms.spi import ExecutionEnvelope, InputExecutionContext
from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
    BUILTIN_LEGACY_DESCRIPTORS,
    LegacyTrainerDescriptor,
)

from .conftest import make_spec, request_for

_PROBE_MODULE = "tests.support.legacy_trainers"
_INPUT_ADAPTER = QualifiedReference.parse(
    "tests.support.legacy_trainers:prepare_native_input"
)


class _ProbeResolver(FakeInputResolver):
    def describe(self, binding, context):
        descriptor = super().describe(binding, context)
        return replace(
            descriptor,
            compatible_worker_input_adapter_refs=(
                *descriptor.compatible_worker_input_adapter_refs,
                str(_INPUT_ADAPTER),
            ),
        )


def _registration() -> AlgorithmRegistration:
    compatibility = BackendInputCompatibility(
        accepted_input_views=("materialized_tabular",),
        accepted_ingestion_engines=("tributo.fake_tabular",),
        required_input_capabilities=("materializable",),
        supported_explicit_adapters=(_INPUT_ADAPTER,),
        distribution_policy=(RuntimeTopology.SINGLE_WORKER,),
    )
    return AlgorithmRegistration(
        spec=make_spec(
            "legacy_probe",
            operations=("fit",),
            mode=ExecutionMode.LEGACY_TRAINER,
        ),
        implementation=ImplementationDescriptor(
            implementation_id="tests.legacy_probe",
            version="1.0.0",
            execution_mode=ExecutionMode.LEGACY_TRAINER,
            implementation_ref=QualifiedReference.parse(
                "tests.support.legacy_trainers:ProbeLegacyTrainer"
            ),
            executable_factory_ref=QualifiedReference.parse(
                "tributo.integrations.algorithm_runtimes.legacy_trainer:create_executable"
            ),
            operations=(AlgorithmOperation.FIT,),
            input_compatibility=compatibility,
            allowed_config_keys=("metric", "fail", "nonfinite", "resume"),
        ),
        environment=EnvironmentSpec(environment_id="tests.legacy_probe"),
        runtime=RuntimeBinding(
            runtime_id="tributo.ray_task",
            worker_input_adapter_ref=_INPUT_ADAPTER,
            num_cpus=0,
        ),
        is_default=True,
    )


def _envelope(config: dict[str, object]) -> ExecutionEnvelope:
    registry = AlgorithmRegistrationRegistry()
    registry.register(_registration())
    resolver = _ProbeResolver()
    plan = AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
        request_for("legacy_probe", AlgorithmOperation.FIT, config=config)
    )
    invocation = FakeInputInvocation(
        FakeTabularPayload(
            columns_by_name={
                "x0": (0.0, 1.0),
                "x1": (1.0, 0.0),
                "label": (0, 1),
            }
        )
    )
    lease = resolver.open(
        plan.input_binding,
        plan.input_descriptor,
        context=InputExecutionContext(values={"binary-fixture": invocation}),
    )
    payload = FakeInputRuntimeAdapter().bind(lease, plan).payload
    return ExecutionEnvelope(plan=plan, input_payload=payload)


def test_core_has_no_builtin_legacy_trainer_descriptors() -> None:
    assert BUILTIN_LEGACY_DESCRIPTORS == ()
    script = """
import sys
import tributo.training
from tributo.integrations.algorithm_runtimes.legacy_descriptors import BUILTIN_LEGACY_DESCRIPTORS
assert BUILTIN_LEGACY_DESCRIPTORS == ()
assert "TuneRunner" in tributo.training.__all__
assert "submit_training_job" in tributo.training.__all__
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_legacy_descriptor_cannot_claim_alpha_stability() -> None:
    registration = _registration()
    descriptor = LegacyTrainerDescriptor(
        registration=registration,
        trainer_ref=registration.implementation.implementation_ref,
        config_model_ref=QualifiedReference.parse(
            "tests.support.legacy_trainers:ProbeLegacyConfig"
        ),
        limitations=(),
    )
    with pytest.raises(ValueError, match="must remain Beta"):
        replace(descriptor, stability="alpha")


def test_worker_bootstrap_delays_trainer_load_and_normalizes_result() -> None:
    sys.modules.pop(_PROBE_MODULE, None)
    envelope = _envelope({"metric": 0.25})
    assert _PROBE_MODULE not in sys.modules

    result = worker_bootstrap(envelope, {"worker_id": "unit-worker"})

    assert _PROBE_MODULE in sys.modules
    assert result.execution.status == "succeeded"
    assert result.execution.metrics == {"loss": 0.25, "nested": {"epochs": 1}}
    assert result.execution.outputs == {
        "adapter": "legacy_trainer",
        "checkpoint_available": True,
        "delivery_performed": False,
    }
    assert result.execution.artifacts == ()


def test_worker_bootstrap_classifies_config_and_execution_failures() -> None:
    invalid = worker_bootstrap(_envelope({"metric": "invalid"}))
    assert invalid.execution.status == "failed"
    assert invalid.execution.failure_category.value == "validation"

    failed = worker_bootstrap(_envelope({"fail": True}))
    assert failed.execution.status == "failed"
    assert failed.execution.failure_category.value == "execution"
    assert "do-not-leak" not in (failed.execution.error_message or "")


def test_worker_bootstrap_rejects_non_finite_legacy_metrics() -> None:
    result = worker_bootstrap(_envelope({"nonfinite": True}))
    assert result.execution.status == "failed"
    assert result.execution.failure_category.value == "execution"
    assert "not finite" in (result.execution.error_message or "")


def test_worker_bootstrap_preserves_resume_config_without_serializing_checkpoint() -> (
    None
):
    result = worker_bootstrap(_envelope({"resume": {"attempt": 3}}))

    assert result.execution.status == "succeeded"
    assert result.execution.metrics["resume_attempt"] == 3
    assert result.execution.outputs["checkpoint_available"] is True
    assert result.execution.artifacts == ()

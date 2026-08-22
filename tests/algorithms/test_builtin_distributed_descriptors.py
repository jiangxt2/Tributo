"""Conformance tests for every first-party distributed algorithm descriptor."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    DistributedAlgorithmDescriptor,
    DistributionStrategy,
    ExecutionMode,
    ExecutionProfile,
    MapReducePolicy,
    ResolvedAlgorithmPlan,
    RuntimeTopology,
)
from tributo.algorithms.builtin import (
    DNN_DESCRIPTOR,
    MULTINOMIAL_NB_DESCRIPTOR,
    PU_DESCRIPTOR,
    XGBOOST_DESCRIPTOR,
    DistributedDNN,
    DistributedMultinomialNB,
    DistributedPU,
    DistributedXGBoost,
)
from tributo.algorithms.builtin.torch_collective import DNNTrainingRecipe
from tributo.algorithms.spi import (
    CollectiveAlgorithm,
    FrameworkNativeAlgorithm,
    MapReduceAlgorithm,
    TorchTrainingRecipe,
)
from tributo.training.registry import TrainingAlgorithmRegistry


def test_first_party_descriptors_cover_all_builtin_training_algorithms() -> None:
    descriptors = (
        DNN_DESCRIPTOR,
        PU_DESCRIPTOR,
        XGBOOST_DESCRIPTOR,
        MULTINOMIAL_NB_DESCRIPTOR,
    )

    assert {descriptor.name for descriptor in descriptors} == {
        "dnn",
        "pu",
        "xgboost",
        "multinomial_nb",
    }
    for descriptor in descriptors:
        registration = descriptor.registration
        distribution = registration.distribution_spec
        assert distribution is not None
        assert distribution.supported_execution_profiles == (
            ExecutionProfile.CLUSTER,
            ExecutionProfile.LOCAL,
        )
        assert distribution.supported_worker_range.contains(1)
        assert distribution.supported_worker_range.contains(2)
        assert descriptor.tested is True
        assert descriptor.supported is True
        assert descriptor.validated_execution_profiles == (
            ExecutionProfile.CLUSTER,
            ExecutionProfile.LOCAL,
        )
        assert registration.is_default is True
        assert registration.implementation.exporter_ref is not None
        assert registration.implementation.flavor_id is not None


def test_builtin_implementations_inherit_the_declared_strategy_interfaces() -> None:
    assert issubclass(DNNTrainingRecipe, TorchTrainingRecipe)
    assert issubclass(DistributedDNN, CollectiveAlgorithm)
    assert issubclass(DistributedPU, CollectiveAlgorithm)
    assert issubclass(DistributedMultinomialNB, MapReduceAlgorithm)
    assert issubclass(DistributedXGBoost, FrameworkNativeAlgorithm)

    assert (
        DNN_DESCRIPTOR.registration.implementation.execution_mode
        is ExecutionMode.COLLECTIVE
    )
    assert str(DNN_DESCRIPTOR.registration.implementation.implementation_ref).endswith(
        ":DNNTrainingRecipe"
    )
    assert str(
        DNN_DESCRIPTOR.registration.implementation.executable_factory_ref
    ).endswith(":create_torch_recipe_algorithm")
    assert (
        PU_DESCRIPTOR.registration.implementation.execution_mode
        is ExecutionMode.COLLECTIVE
    )
    assert (
        MULTINOMIAL_NB_DESCRIPTOR.registration.implementation.execution_mode
        is ExecutionMode.MAP_REDUCE
    )
    assert (
        XGBOOST_DESCRIPTOR.registration.implementation.execution_mode
        is ExecutionMode.FRAMEWORK_NATIVE
    )


@pytest.mark.parametrize("descriptor", (DNN_DESCRIPTOR, PU_DESCRIPTOR))
def test_torch_collective_hard_dependencies_match_legacy_onnx_fallback(
    descriptor: DistributedAlgorithmDescriptor,
) -> None:
    dependencies = descriptor.registration.environment.dependencies

    assert "onnx>=1.16.0" in dependencies
    assert "onnxruntime>=1.20.0" in dependencies
    assert "torch>=2.5.0" in dependencies
    assert all(not dependency.startswith("onnxscript") for dependency in dependencies)


def test_registry_atomically_publishes_native_and_compatibility_implementations() -> (
    None
):
    registry = TrainingAlgorithmRegistry()
    registrations = registry.execution_registry().snapshot()
    by_name: dict[str, list[object]] = {}
    for registration in registrations:
        by_name.setdefault(registration.spec.name, []).append(registration)

    assert len(by_name["dnn"]) == 2
    assert len(by_name["pu"]) == 2
    assert len(by_name["xgboost"]) == 2
    assert len(by_name["multinomial_nb"]) == 1
    assert {
        registration.distribution_spec.strategy
        for registrations_for_name in by_name.values()
        for registration in registrations_for_name
        if registration.distribution_spec is not None
    } == {
        DistributionStrategy.RAY_TRAIN_COLLECTIVE,
        DistributionStrategy.FRAMEWORK_NATIVE,
        DistributionStrategy.RAY_MAP_REDUCE,
    }


def test_formal_topologies_do_not_reuse_legacy_data_parallel_meaning() -> None:
    assert {
        DNN_DESCRIPTOR.registration.distribution_spec.strategy.value,
        PU_DESCRIPTOR.registration.distribution_spec.strategy.value,
        XGBOOST_DESCRIPTOR.registration.distribution_spec.strategy.value,
        MULTINOMIAL_NB_DESCRIPTOR.registration.distribution_spec.strategy.value,
    } == {
        RuntimeTopology.RAY_TRAIN_COLLECTIVE.value,
        RuntimeTopology.FRAMEWORK_NATIVE.value,
        RuntimeTopology.RAY_MAP_REDUCE.value,
    }


def _adapter_plan(
    *,
    config: dict[str, object],
    worker_count: int = 2,
    resume_from: str | None = None,
) -> ResolvedAlgorithmPlan:
    return cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            algorithm_config=config,
            runtime=SimpleNamespace(
                worker_count=worker_count,
                num_gpus=0,
                resume_from=resume_from,
            ),
        ),
    )


@pytest.mark.parametrize("implementation", (DistributedDNN, DistributedPU))
def test_collective_adapters_reject_unsynchronized_batch_norm(
    implementation: type[DistributedDNN] | type[DistributedPU],
) -> None:
    with pytest.raises(AlgorithmConfigurationError, match="BatchNorm"):
        implementation(_adapter_plan(config={"model": {"use_batch_norm": True}}))


def test_formal_dnn_rejects_nnpu_before_selecting_the_worker_loop() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="canonical PU"):
        DistributedDNN(
            _adapter_plan(
                config={
                    "loss": {"type": "nnpu"},
                    "pu_learning": {"enabled": True, "class_prior": 0.2},
                }
            )
        )


def test_framework_native_xgboost_rejects_unproved_multi_worker_resume() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="multi-worker XGBoost"):
        DistributedXGBoost(_adapter_plan(config={}, resume_from="/checkpoint"))


@pytest.mark.parametrize(
    "implementation", (DistributedDNN, DistributedPU, DistributedXGBoost)
)
def test_formal_adapters_reject_ungated_worker_retries(
    implementation: type[DistributedDNN]
    | type[DistributedPU]
    | type[DistributedXGBoost],
) -> None:
    with pytest.raises(AlgorithmConfigurationError, match="failure"):
        implementation(_adapter_plan(config={"ray": {"max_failures": 1}}))


def test_map_reduce_descriptors_disable_retry_for_single_pass_shards() -> None:
    for descriptor in (MULTINOMIAL_NB_DESCRIPTOR,):
        policy = descriptor.registration.distribution_spec.policy
        assert isinstance(policy, MapReducePolicy)
        assert policy.max_retries == 0


@pytest.mark.parametrize("api_version", (True, "1"))
def test_distributed_descriptor_requires_strict_integer_api_version(
    api_version: object,
) -> None:
    with pytest.raises(AlgorithmConfigurationError, match="api_version"):
        replace(
            DNN_DESCRIPTOR,
            api_version=cast(int, api_version),
        )

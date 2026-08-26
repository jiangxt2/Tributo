"""Tests for trainer registry capability validation (T2).

Covers the capability invariants enforced at registration time:
``trainer_cls`` must be a ``BaseTrainer`` subclass, and
``execution_kind`` must match the trainer class hierarchy.
"""

from __future__ import annotations

from typing import Any

import pytest

from tributo.exceptions import JobConfigurationError
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    Capability,
    ExecutionKind,
)
from tributo.training.base import BaseTrainer
from tributo.training.causal_estimator import (
    BaseCausalEstimator,
    CausalEffect,
    CausalGraph,
)
from tributo.training.registry import _registry, register
from tributo.training.tune_config import TuneSearchConfig
from tributo.training.tune_runner import TuneRunner
from tributo.training.tune_space import SearchParamSpec, SearchSpaceSpec


class MockTrainer(BaseTrainer):
    """Minimal ``BaseTrainer`` subclass for testing."""

    def setup(self) -> None:
        pass

    def training_loop(self) -> dict[str, Any]:
        return {}

    def export_model(self, checkpoint: Any, output_path: str) -> None:
        pass


class NotATrainer:
    """A plain class that is not a ``BaseTrainer`` subclass."""


class _MinimalCausal(BaseCausalEstimator):
    """``BaseCausalEstimator`` subclass with only the abstract hooks."""

    def identify(
        self,
        data: Any,
        treatment: str,
        outcome: str,
        **kwargs: Any,
    ) -> CausalGraph:
        return CausalGraph(treatment=treatment, outcome=outcome)

    def estimate(self, data: Any, causal_graph: CausalGraph) -> CausalEffect:
        return CausalEffect(method="test.ols")


def _spec(name: str = "test-spec", **overrides: Any) -> AlgorithmSpec:
    kwargs: dict[str, Any] = {
        "trainer_cls": MockTrainer,
        "execution_kind": ExecutionKind.TRAIN,
        "capabilities": (Capability.TUNABLE,),
    }
    kwargs.update(overrides)
    return AlgorithmSpec(name=name, **kwargs)


class TestDefaultCapabilities:
    """Capabilities are not assumed: an undeclared spec is empty."""

    def test_capabilities_default_to_empty(self) -> None:
        spec = AlgorithmSpec(name="test", trainer_cls=MockTrainer)
        assert spec.capabilities == ()


class TestRegisterValidation:
    """Registration rejects specs whose capability declarations are
    inconsistent with their trainer class."""

    def test_rejects_non_class_trainer_cls(self) -> None:
        bad_cls: Any = "xgboost"
        with pytest.raises(TypeError, match="trainer_cls must be a class"):
            register(_spec(trainer_cls=bad_cls))

    def test_rejects_non_trainer_class(self) -> None:
        with pytest.raises(TypeError, match="BaseTrainer subclass"):
            register(_spec(trainer_cls=NotATrainer))

    def test_rejects_estimate_without_causal_class(self) -> None:
        with pytest.raises(TypeError, match="ESTIMATE requires"):
            register(_spec(execution_kind=ExecutionKind.ESTIMATE))

    def test_rejects_causal_class_outside_estimate(self) -> None:
        with pytest.raises(TypeError, match="ESTIMATE lifecycle"):
            register(_spec(trainer_cls=_MinimalCausal))

    def test_registers_valid_spec(self) -> None:
        register(_spec(name="valid-registration-test"))
        try:
            assert _registry.contains("valid-registration-test")
        finally:
            _registry.unregister("valid-registration-test")

    def test_rejected_spec_leaves_no_entry(self) -> None:
        bad_cls: Any = "xgboost"
        with pytest.raises(TypeError):
            register(_spec(name="should-not-register", trainer_cls=bad_cls))
        assert not _registry.contains("should-not-register")


class TestBuiltinSpecs:
    """Built-in algorithms declare their capabilities explicitly."""

    def test_builtins_declare_capabilities(self) -> None:
        from tributo.training.registry import get_trainer

        try:
            xgb = get_trainer("xgboost")
        except JobConfigurationError:
            pytest.skip("official algorithm wheels are not installed in the dev suite")
        assert xgb.capabilities == (
            Capability.TUNABLE,
            Capability.EXPORTABLE,
            Capability.DISTRIBUTED,
        )

        pytest.importorskip("torch")
        try:
            pu = get_trainer("pu")
            dnn = get_trainer("dnn")
        except JobConfigurationError:
            pytest.skip("official algorithm wheels are not installed in the dev suite")
        assert pu.capabilities == (
            Capability.TUNABLE,
            Capability.EXPORTABLE,
            Capability.DISTRIBUTED,
        )
        assert dnn.capabilities == (
            Capability.TUNABLE,
            Capability.EXPORTABLE,
            Capability.DISTRIBUTED,
        )


class TestTuneRunnerCapabilityGate:
    """TuneRunner rejects algorithms that do not declare TUNABLE."""

    def test_tune_rejects_missing_tunable(self) -> None:
        spec = _spec(name="no-tunable", capabilities=())
        tune_config = TuneSearchConfig(
            metric="loss",
            mode="min",
            num_samples=1,
            search_alg="random",
            scheduler="fifo",
        )
        search_space = SearchSpaceSpec(
            parameters=(
                SearchParamSpec(
                    path="training.learning_rate",
                    kind="uniform",
                    lower=0.001,
                    upper=0.1,
                ),
            )
        )
        runner = TuneRunner(spec, tune_config, search_space, {})
        with pytest.raises(JobConfigurationError, match="Capability.TUNABLE"):
            runner.run({}, "/tmp/tune-reject-test")

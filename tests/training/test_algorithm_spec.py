"""Tests for AlgorithmSpec, DataContract, ResourceHints, and TrainerSpec alias."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    DataContract,
    ProblemType,
    ResourceHints,
)
from tributo.training.base import TrainerSpec

# ---------------------------------------------------------------------------
# TrainerSpec alias
# ---------------------------------------------------------------------------


class TestTrainerSpecAlias:
    """TrainerSpec is a pure alias for AlgorithmSpec in Phase 1."""

    def test_is_identical(self) -> None:
        assert TrainerSpec is AlgorithmSpec

    def test_instantiation_via_alias(self) -> None:
        """Legacy code uses TrainerSpec(name=..., trainer_cls=...)."""
        spec = TrainerSpec(name="xgboost", trainer_cls=type("FakeTrainer", (), {}))
        assert isinstance(spec, AlgorithmSpec)
        assert spec.name == "xgboost"

    def test_instantiation_via_algorithm_spec(self) -> None:
        spec = AlgorithmSpec(name="dnn", trainer_cls=type("FakeTrainer", (), {}))
        assert isinstance(spec, TrainerSpec)  # works both ways
        assert spec.name == "dnn"

    def test_default_fields(self) -> None:
        spec = AlgorithmSpec(name="test", trainer_cls=type("Fake", (), {}))
        assert spec.default_config == {}
        assert spec.supported_tasks == ("train",)
        assert spec.version == "0.1.0"
        assert spec.problem_types == ()
        assert spec.data_modality == ()
        assert spec.tags == ()
        assert spec.extras_group is None
        assert spec.input_schema is None
        assert spec.output_schema is None
        assert spec.config_model is None

    def test_frozen(self) -> None:
        spec = AlgorithmSpec(name="test", trainer_cls=type("Fake", (), {}))
        with pytest.raises(FrozenInstanceError):
            spec.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DataContract
# ---------------------------------------------------------------------------


class TestDataContract:
    def test_defaults(self) -> None:
        dc = DataContract()
        assert dc.columns == {}
        assert dc.sparse == ()
        assert dc.dense == ()
        assert dc.min_rows is None

    def test_frozen(self) -> None:
        dc = DataContract(columns={"label": "int"})
        with pytest.raises(FrozenInstanceError):
            dc.columns = {}  # type: ignore[misc]

    def test_serializable(self) -> None:
        dc = DataContract(
            columns={"feat_0": "float32", "label": "int64"},
            sparse=["feat_"],
            min_rows=1000,
        )
        d = asdict(dc)
        assert json.dumps(d)  # does not raise


# ---------------------------------------------------------------------------
# ResourceHints
# ---------------------------------------------------------------------------


class TestResourceHints:
    def test_defaults(self) -> None:
        rh = ResourceHints()
        assert rh.gpu_required is False
        assert rh.min_memory_gb == 2
        assert rh.min_cpus == 1

    def test_gpu_heavy(self) -> None:
        rh = ResourceHints(gpu_required=True, min_memory_gb=16, min_cpus=4)
        assert rh.gpu_required is True


# ---------------------------------------------------------------------------
# ProblemType
# ---------------------------------------------------------------------------


class TestProblemType:
    def test_all_values_are_strings(self) -> None:
        for pt in ProblemType:
            assert isinstance(pt.value, str)

    def test_common_values(self) -> None:
        assert ProblemType.BINARY_CLASSIFICATION.value == "binary_classification"
        assert ProblemType.REGRESSION.value == "regression"


# ---------------------------------------------------------------------------
# Full AlgorithmSpec
# ---------------------------------------------------------------------------


class TestFullAlgorithmSpec:
    def test_with_all_fields(self) -> None:
        """Exercise all AlgorithmSpec field groups."""
        spec = AlgorithmSpec(
            name="deepfm",
            trainer_cls=type("DeepFMTrainer", (), {}),
            default_config={"embedding_dim": 16},
            supported_tasks=["train", "predict"],
            version="1.2.3",
            resource_hints=ResourceHints(gpu_required=True),
            extras_group="training",
            problem_types=[
                ProblemType.BINARY_CLASSIFICATION,
                ProblemType.RANKING,
            ],
            data_modality=["tabular"],
            tags=["deep-learning", "ctr"],
            input_schema=DataContract(columns={"user_id": "int64"}),
            output_schema=DataContract(columns={"prediction": "float32"}),
        )
        assert len(spec.problem_types) == 2
        assert ProblemType.BINARY_CLASSIFICATION in spec.problem_types
        assert spec.resource_hints.gpu_required is True
        assert spec.input_schema is not None
        assert spec.input_schema.columns == {"user_id": "int64"}

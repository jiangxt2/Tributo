"""Core boundary tests after first-party algorithm Wheel migration."""

from __future__ import annotations

import importlib.util

import tributo.algorithms.builtin as builtin
from tributo._bootstrap import (
    first_party_export_plugins,
    first_party_model_flavors,
    first_party_source_providers,
)
from tributo.training.registry import TrainingAlgorithmRegistry


def test_core_builtin_namespace_contains_no_production_algorithms() -> None:
    assert builtin.__all__ == []
    for module in (
        "tributo.algorithms.builtin.multinomial_nb",
        "tributo.algorithms.builtin.torch_collective",
        "tributo.algorithms.builtin.x_learner",
        "tributo.algorithms.builtin.xgboost_native",
    ):
        assert importlib.util.find_spec(module) is None


def test_core_composition_root_contains_only_algorithm_neutral_plugins() -> None:
    exporters, validators = first_party_export_plugins()
    assert {item.exporter_id for item in exporters} == {
        "hf-onnx-v1",
        "onnx-quantizer-v1",
        "torch-export-v1",
        "torch-onnx-v1",
        "torch-safetensors-v1",
    }
    assert {item.validator_id for item in validators} == {"onnx-runtime-v1"}
    assert {item.provider_id for item in first_party_source_providers()} == {
        "ray-torch-v1"
    }
    assert {item.flavor_id for item in first_party_model_flavors()} == {
        "onnx-runtime-v1"
    }


def test_registry_has_no_core_owned_algorithm_descriptor_fallback() -> None:
    records = TrainingAlgorithmRegistry().record_snapshot()
    assert all(
        not str(registration.implementation.implementation_ref).startswith(
            "tributo.algorithms.builtin"
        )
        for record in records
        for registration in record.registrations
    )

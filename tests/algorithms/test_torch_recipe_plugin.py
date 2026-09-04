"""Isolation checks for the out-of-tree Torch recipe package fixture."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from tributo.algorithms import TorchRecipe
from tributo.algorithms.api import ExecutionProfile, ResultPolicy


def _fixture_source() -> Path:
    return (
        Path(__file__).parents[1] / "fixtures" / "torch_recipe_algorithm_plugin" / "src"
    )


def test_out_of_tree_recipe_uses_no_builtin_or_worker_loop() -> None:
    module_path = (
        _fixture_source() / "tributo_test_torch_recipe_algorithm" / "__init__.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "tributo.algorithms.builtin" not in source
    assert "train_loop_per_worker" not in source
    assert "ray.train" not in source
    assert "BundleExportService" not in source


def test_out_of_tree_recipe_descriptor_uses_public_lowering_contract() -> None:
    module_name = "tributo_test_torch_recipe_algorithm"
    sys.path.insert(0, str(_fixture_source()))
    try:
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
    finally:
        sys.path.remove(str(_fixture_source()))

    assert issubclass(module.ThirdPartyBinaryLinearRecipe, TorchRecipe)
    distribution = module.DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.result_policy is ResultPolicy.BUNDLE_REQUIRED
    assert distribution.supported_execution_profiles == (
        ExecutionProfile.CLUSTER,
        ExecutionProfile.LOCAL,
    )
    assert module.DESCRIPTOR.registration.implementation.flavor_id == (
        "onnx-runtime-v1"
    )

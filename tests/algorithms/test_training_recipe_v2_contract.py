"""Regression tests for removal of the legacy Torch RecipeV2 contract."""

from __future__ import annotations

import pytest

from tributo.algorithms import AlgorithmBuilder


def test_legacy_recipe_v2_is_not_exported() -> None:
    import tributo.algorithms as algorithms

    assert not hasattr(algorithms, "TrainingRecipeV2")
    assert not hasattr(AlgorithmBuilder, "from_training_recipe_v2")


def test_legacy_recipe_v2_module_is_not_importable() -> None:
    with pytest.raises(ImportError):
        __import__("tributo.integrations.algorithm_runtimes.torch_recipe")

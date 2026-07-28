"""Tests for embeddings/registry.py."""

from __future__ import annotations

import pytest

from tributo.embeddings.registry import (
    ModelSpec,
    get_spec,
    list_models,
    register,
)
from tributo.exceptions import JobConfigurationError


def test_list_models_includes_bge():
    models = list_models()
    assert "bge-small-zh" in models


def test_get_spec_bge_small_zh():
    spec = get_spec("bge-small-zh")
    assert spec.name == "bge-small-zh-v1.5"
    assert spec.hf_model_id == "BAAI/bge-small-zh-v1.5"
    assert spec.dim == 512
    assert spec.pooling == "cls"
    assert spec.normalize is True
    assert spec.max_length == 512


def test_get_spec_unknown_raises():
    with pytest.raises(JobConfigurationError):
        get_spec("nonexistent-model")


def test_register_custom_model():
    custom = ModelSpec(
        name="custom-test",
        hf_model_id="org/custom",
        dim=128,
        pooling="mean",
        normalize=False,
    )
    register(custom)
    assert "custom-test" in list_models()
    assert get_spec("custom-test").dim == 128


def test_register_duplicate_raises():
    custom = ModelSpec(
        name="dup-test",
        hf_model_id="org/dup",
        dim=64,
    )
    register(custom)
    with pytest.raises(JobConfigurationError):
        register(custom)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))

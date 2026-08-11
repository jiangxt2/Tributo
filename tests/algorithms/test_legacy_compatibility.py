"""Fail-closed boundaries between portable registrations and legacy runners."""

from __future__ import annotations

import pytest

from tributo.exceptions import JobConfigurationError
from tributo.pipeline.core import Pipeline, PipelineStep
from tributo.training.algorithm_spec import AlgorithmSpec
from tributo.training.registry import _registry


def test_legacy_pipeline_rejects_portable_registration() -> None:
    name = "tests.portable-pipeline"
    _registry.register(
        name,
        AlgorithmSpec(
            name=name,
            trainer_cls=None,
            operations=("fit",),
        ),
    )
    try:
        with pytest.raises(JobConfigurationError, match="portable execution path"):
            Pipeline([PipelineStep(name="portable", algorithm=name)]).run({})
    finally:
        _registry.unregister(name)


def test_legacy_pipeline_identifies_incomplete_legacy_spec() -> None:
    name = "tests.incomplete-legacy-pipeline"
    _registry.register(name, AlgorithmSpec(name=name, trainer_cls=None))
    try:
        with pytest.raises(JobConfigurationError, match="missing.*trainer_cls"):
            Pipeline([PipelineStep(name="incomplete", algorithm=name)]).run({})
    finally:
        _registry.unregister(name)

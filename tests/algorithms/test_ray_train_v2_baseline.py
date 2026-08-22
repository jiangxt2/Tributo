"""Fixed-version facts for Tributo's default Ray Train execution path."""

from __future__ import annotations

import inspect

import pytest


def test_ray_2551_defaults_to_train_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    import ray
    from ray.train.v2._internal.constants import is_v2_enabled

    monkeypatch.delenv("RAY_TRAIN_V2_ENABLED", raising=False)

    assert ray.__version__ == "2.55.1"
    assert is_v2_enabled() is True
    assert "env_bool(V2_ENABLED_ENV_VAR, True)" in inspect.getsource(is_v2_enabled)


def test_v2_reuses_public_data_config_and_rejects_legacy_initial_checkpoint() -> None:
    from ray.train.v2.api.data_parallel_trainer import DataParallelTrainer

    from tributo.integrations.algorithm_runtimes.collective import (
        RayTrainCollectiveRuntime,
    )

    source = inspect.getsource(DataParallelTrainer.__init__)

    assert "self.data_config = dataset_config or DataConfig()" in source
    assert "resume_from_checkpoint is not None" in source
    assert "raise DeprecationWarning" in source
    assert "resume_from_checkpoint" not in inspect.getsource(
        RayTrainCollectiveRuntime.execute
    )
